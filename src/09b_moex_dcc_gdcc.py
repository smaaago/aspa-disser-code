"""Локальный (отраслевой) уровень, этап 2: DCC / ADCC / cDCC + кластерная GDCC.

Зеркало 04_dcc_adcc.py и 04b_gdcc_clustered.py на 7 отраслевых индексах
Мосбиржи. Кластерная гипотеза: экспортные сектора (нефтегаз, металлы, химия)
против внутренних (финансы, потребительский, энергетика, транспорт) -> K = 2;
K = 3 оценивается как альтернатива. Правдоподобие всюду многомерное t.

Рабочая рекурсия GDCC — как в 04b: диагональный случай обобщённой DCC
Каппьелло–Энгла–Шеппарда с адамаровым интерсептом (11' - aa' - bb') ⊙ Q_bar;
его положительная полуопределённость при разнородных параметрах не
гарантирована (предупреждение Хафнера–Франсеса), поэтому в конце скрипта —
PD-диагностика траектории и робастность со скалярным интерсептом H-F
(1 - ā² - b̄²) Q_bar, положительно определённым по построению.

Outputs:
  output/tables/table18_moex_dcc_family.csv    — model comparison (AIC/BIC)
  output/tables/table18b_moex_lr_test.txt      — LR ADCC vs DCC
  output/tables/table18c_moex_clusters.csv     — cluster assignment (hier/spectral)
  output/tables/table18d_moex_gdcc_params.csv  — best GDCC cluster parameters
  output/tables/table18e_moex_gdcc_recursion_check.csv — PD-диагностика + H-F
  output/R_moex_{dcc,adcc,cdcc,gdcc}.npy       — correlation tensors
  output/mean_sector_corr.csv                  — mean pairwise sector correlation
  output/figures/fig15_moex_corr.png           — dendrogram + mean correlation path
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

OUT = Path("output")
TAB = OUT / "tables"
FIG = OUT / "figures"

Z = pd.read_csv(OUT / "std_residuals_moex.csv", parse_dates=["Datetime"]).set_index("Datetime")
cols = list(Z.columns)
Zv = Z.values.astype(float)
T, N = Zv.shape
print(f"T={T}, N={N} ({cols})")


def mvt_loglike(R, z, nu):
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0 or not np.isfinite(logdet):
        return -1e10
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return -1e10
    quad = float(z @ Rinv @ z)
    p = R.shape[0]
    c = gammaln((nu + p) / 2.0) - gammaln(nu / 2.0) - 0.5 * p * np.log(np.pi * (nu - 2.0))
    return c - 0.5 * logdet - ((nu + p) / 2.0) * np.log(1.0 + quad / (nu - 2.0))


def filter_dcc(z, params, model="dcc", return_R=False):
    T, N = z.shape
    a, b = params[0], params[1]
    g = params[2] if model == "adcc" else 0.0
    nu = params[-1]
    if a < 1e-6 or b < 1e-6 or a + b + 0.5 * g >= 0.999 or nu <= 4.5:
        return 1e10 if not return_R else None
    Q_bar = z.T @ z / T
    if model == "adcc":
        zneg = np.where(z < 0, z, 0.0)
        N_bar = zneg.T @ zneg / T
        intercept = Q_bar - a * Q_bar - b * Q_bar - g * N_bar
    else:
        intercept = (1 - a - b) * Q_bar
    Q = Q_bar.copy()
    Rs = np.empty((T, N, N)) if return_R else None
    loglike = 0.0
    for t in range(T):
        if t == 0:
            qd = np.sqrt(np.diag(Q_bar))
            R = Q_bar / np.outer(qd, qd)
            R = (R + R.T) / 2.0
        else:
            zt = z[t - 1]
            if model == "adcc":
                zn = np.where(zt < 0, zt, 0.0)
                Q = intercept + a * np.outer(zt, zt) + g * np.outer(zn, zn) + b * Q
            elif model == "cdcc":
                qd = np.sqrt(np.diag(Q))
                zs = qd * zt
                Q = intercept + a * np.outer(zs, zs) + b * Q
            else:
                Q = intercept + a * np.outer(zt, zt) + b * Q
            qd = np.sqrt(np.diag(Q))
            R = Q / np.outer(qd, qd)
            R = (R + R.T) / 2.0
        if return_R:
            Rs[t] = R
        loglike += mvt_loglike(R, z[t], nu)
    return (-loglike, Rs) if return_R else -loglike


def fit_dcc(z, model="dcc"):
    if model == "adcc":
        x0, bounds = [0.03, 0.95, 0.02, 8.0], [(1e-4, 0.5), (1e-4, 0.999), (1e-4, 0.5), (4.5, 60.0)]
    else:
        x0, bounds = [0.05, 0.93, 8.0], [(1e-4, 0.5), (1e-4, 0.999), (4.5, 60.0)]
    return minimize(lambda p: filter_dcc(z, p, model=model), x0,
                    method="L-BFGS-B", bounds=bounds, options=dict(maxiter=200))


# ============ scalar DCC family ============
results = {}
for m in ["dcc", "adcc", "cdcc"]:
    print(f"\nFitting {m.upper()}...")
    r = fit_dcc(Zv, model=m)
    print(f"  converged={r.success}, neg-loglike={r.fun:.2f}, params={np.round(r.x, 4)}")
    results[m] = r

rows = []
for m, r in results.items():
    if m == "adcc":
        a, b, g, nu = r.x
    else:
        (a, b, nu), g = r.x, np.nan
    rows.append({"model": m.upper(), "a": a, "b": b, "g": g, "nu_mvt": nu,
                 "neg_loglike": r.fun, "params_count": len(r.x)})
parm = pd.DataFrame(rows).set_index("model")
parm["AIC"] = 2 * parm["params_count"] + 2 * parm["neg_loglike"]
parm["BIC"] = parm["params_count"] * np.log(T) + 2 * parm["neg_loglike"]

LR = 2 * (-results["adcc"].fun + results["dcc"].fun)
p_lr = 1 - chi2.cdf(LR, df=1)
with open(TAB / "table18b_moex_lr_test.txt", "w") as f:
    f.write(f"LR test ADCC vs DCC (sectors): LR={LR:.2f}, df=1, p={p_lr:.4g}\n")
print(f"\nLR ADCC vs DCC: {LR:.2f}, p={p_lr:.4g}")

for m in ["dcc", "adcc", "cdcc"]:
    _, Rs_m = filter_dcc(Zv, results[m].x, model=m, return_R=True)
    np.save(OUT / f"R_moex_{m}.npy", Rs_m)

# ============ clustering ============
R_emp = np.corrcoef(Zv, rowvar=False)
np.fill_diagonal(R_emp, 1.0)
D = np.sqrt(np.clip(2 * (1 - R_emp), 0, None))
np.fill_diagonal(D, 0.0)
linkage_mat = linkage(squareform(D, checks=False), method="average")

clu = {"asset": cols}
for K in (2, 3):
    clu[f"hier_K{K}"] = fcluster(linkage_mat, t=K, criterion="maxclust")
    spc = SpectralClustering(n_clusters=K, affinity="precomputed",
                             assign_labels="kmeans", random_state=2026)
    clu[f"spec_K{K}"] = spc.fit_predict(np.abs(R_emp)) + 1
ca = pd.DataFrame(clu).set_index("asset")
ca.to_csv(TAB / "table18c_moex_clusters.csv")
print("\nCluster assignments:\n", ca)


# ============ GDCC (кластерное ограничение, адамарова форма CES) ============
def filter_gdcc(z, a_vec, b_vec, nu, return_R=False):
    T, N = z.shape
    A, B = np.outer(a_vec, a_vec), np.outer(b_vec, b_vec)
    if float(np.max(a_vec ** 2 + b_vec ** 2)) >= 0.999 or np.min(a_vec) <= 0 \
            or np.min(b_vec) <= 0 or nu <= 4.5:
        return (1e10, None) if return_R else 1e10
    Q_bar = z.T @ z / T
    intercept = (np.ones((N, N)) - A - B) * Q_bar
    Q = Q_bar.copy()
    Rs = np.empty((T, N, N)) if return_R else None
    loglike = 0.0
    for t in range(T):
        if t == 0:
            qd = np.sqrt(np.diag(Q_bar))
            R = Q_bar / np.outer(qd, qd)
        else:
            zt = z[t - 1]
            Q = intercept + A * np.outer(zt, zt) + B * Q
            qd = np.sqrt(np.diag(Q))
            R = Q / np.outer(qd, qd)
        R = (R + R.T) / 2.0
        if return_R:
            Rs[t] = R
        loglike += mvt_loglike(R, z[t], nu)
    return (-loglike, Rs) if return_R else -loglike


def fit_gdcc_clustered(z, labels):
    labels = np.asarray(labels)
    K = len(np.unique(labels))

    def to_full(params):
        return params[:K][labels - 1], params[K:2 * K][labels - 1], params[-1]

    def neg_ll(params):
        a_vec, b_vec, nu = to_full(params)
        return filter_gdcc(z, a_vec, b_vec, nu)

    x0 = np.concatenate([np.full(K, 0.10), np.full(K, 0.95), [9.0]])
    bounds = [(1e-3, 0.6)] * K + [(1e-3, 0.999)] * K + [(4.5, 60.0)]
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=300))
    return res, to_full(res.x)


gdcc_fits = {}
for key in ["hier_K2", "spec_K2", "hier_K3", "spec_K3"]:
    labels = ca[key].values
    K = len(np.unique(labels))
    res, (a_v, b_v, nu) = fit_gdcc_clustered(Zv, labels)
    n_par = 2 * K + 1
    aic = 2 * n_par + 2 * res.fun
    bic = n_par * np.log(T) + 2 * res.fun
    gdcc_fits[key] = dict(res=res, a=a_v, b=b_v, nu=nu, labels=labels,
                          n_par=n_par, aic=aic, bic=bic)
    parm.loc[f"GDCC_{key}"] = [a_v.mean(), b_v.mean(), np.nan, nu,
                               res.fun, n_par, aic, bic]
    print(f"GDCC {key}: neg_ll={res.fun:.2f}, BIC={bic:.1f}")

parm.to_csv(TAB / "table18_moex_dcc_family.csv")
print("\n=== Sectoral model comparison ===")
print(parm.round(2).sort_values("BIC"))

best_key = min(gdcc_fits, key=lambda k: gdcc_fits[k]["bic"])
gf = gdcc_fits[best_key]
print(f"\nBest GDCC variant: {best_key}")
_, Rs_gdcc = filter_gdcc(Zv, gf["a"], gf["b"], gf["nu"], return_R=True)
np.save(OUT / "R_moex_gdcc.npy", Rs_gdcc)

Kb = len(np.unique(gf["labels"]))
best_pars = pd.DataFrame({
    "cluster": range(1, Kb + 1),
    "a": [gf["a"][gf["labels"] == k][0] for k in range(1, Kb + 1)],
    "b": [gf["b"][gf["labels"] == k][0] for k in range(1, Kb + 1)],
    "members": [", ".join(cols[i].upper() for i in range(N) if gf["labels"][i] == k)
                for k in range(1, Kb + 1)]})
best_pars["nu_mvt"] = gf["nu"]
best_pars["variant"] = best_key
best_pars.to_csv(TAB / "table18d_moex_gdcc_params.csv", index=False)
print(best_pars)

# ============ mean pairwise sector correlation (best scalar model = cDCC) ============
_, Rs_c = filter_dcc(Zv, results["cdcc"].x, model="cdcc", return_R=True)
iu = np.triu_indices(N, 1)
mean_corr = np.array([Rs_c[t][iu].mean() for t in range(T)])
mc = pd.DataFrame({"mean_sector_corr": mean_corr}, index=Z.index)
mc.to_csv(OUT / "mean_sector_corr.csv")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6),
                       gridspec_kw={"width_ratios": [1, 1.6]})
dendrogram(linkage_mat, labels=[c.upper() for c in cols],
           color_threshold=linkage_mat[-1, 2] * 0.99,
           above_threshold_color="grey", ax=ax[0])
ax[0].set_title("(а) Кластеризация секторов (расстояние Мантеньи)", fontsize=10)
ax[0].tick_params(axis="x", labelsize=7, rotation=45)
ax[1].plot(mc.index, mc["mean_sector_corr"], color="navy", lw=0.7)
ax[1].plot(mc.index, mc["mean_sector_corr"].rolling(60).mean(), color="red", lw=1.5,
           label="60-дн. среднее")
for a0, b0, c0 in [("2008-09-15", "2009-06-30", "red"),
                   ("2020-02-20", "2020-06-30", "purple"),
                   ("2022-02-21", "2022-06-30", "orange")]:
    ax[1].axvspan(pd.Timestamp(a0), pd.Timestamp(b0), alpha=0.12, color=c0)
ax[1].set_title("(б) Средняя попарная корреляция 7 отраслевых индексов (cDCC-t)", fontsize=10)
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig15_moex_corr.png", dpi=200)
plt.close()
print("\nfig15_moex_corr.png saved.")


# ============ PD-диагностика рекурсии и скалярный интерсепт H-F ============
def pd_diagnostics(z, a_vec, b_vec, scalar_intercept=False):
    """Минимальные собственные числа интерсепта и всех Q_t на траектории."""
    T, N = z.shape
    A, B = np.outer(a_vec, a_vec), np.outer(b_vec, b_vec)
    Q_bar = z.T @ z / T
    if scalar_intercept:
        c0 = 1.0 - a_vec.mean() ** 2 - b_vec.mean() ** 2
        intercept = c0 * Q_bar
    else:
        intercept = (np.ones((N, N)) - A - B) * Q_bar
    lam_int = float(np.linalg.eigvalsh(intercept)[0])
    Q = Q_bar.copy()
    lam_traj = np.inf
    for t in range(1, T):
        zt = z[t - 1]
        Q = intercept + A * np.outer(zt, zt) + B * Q
        lam_traj = min(lam_traj, float(np.linalg.eigvalsh(Q)[0]))
    return lam_int, lam_traj


def filter_gdcc_hf(z, a_vec, b_vec, nu):
    """GDCC со скалярным интерсептом Хафнера–Франсеса (1 - ā² - b̄²) Q_bar."""
    T, N = z.shape
    A, B = np.outer(a_vec, a_vec), np.outer(b_vec, b_vec)
    c0 = 1.0 - a_vec.mean() ** 2 - b_vec.mean() ** 2
    if float(np.max(a_vec ** 2 + b_vec ** 2)) >= 0.999 or c0 <= 1e-6 \
            or np.min(a_vec) <= 0 or np.min(b_vec) <= 0 or nu <= 4.5:
        return 1e10
    Q_bar = z.T @ z / T
    intercept = c0 * Q_bar
    Q = Q_bar.copy()
    loglike = 0.0
    for t in range(T):
        if t == 0:
            qd = np.sqrt(np.diag(Q_bar))
            R = Q_bar / np.outer(qd, qd)
        else:
            zt = z[t - 1]
            Q = intercept + A * np.outer(zt, zt) + B * Q
            qd = np.sqrt(np.diag(Q))
            R = Q / np.outer(qd, qd)
        R = (R + R.T) / 2.0
        loglike += mvt_loglike(R, z[t], nu)
    return -loglike


def fit_gdcc_hf(z, labels):
    labels = np.asarray(labels)
    K = len(np.unique(labels))

    def to_full(params):
        return params[:K][labels - 1], params[K:2 * K][labels - 1], params[-1]

    res = minimize(lambda p: filter_gdcc_hf(z, *to_full(p)),
                   np.concatenate([np.full(K, 0.10), np.full(K, 0.95), [9.0]]),
                   method="L-BFGS-B",
                   bounds=[(1e-3, 0.6)] * K + [(1e-3, 0.999)] * K + [(4.5, 60.0)],
                   options=dict(maxiter=300))
    return res, to_full(res.x)


print("\n=== Диагностика рекурсии GDCC (PD-проверка, вариант H-F) ===")
check_rows = []
for key, gfv in gdcc_fits.items():
    Kv = len(np.unique(gfv["labels"]))
    lam_int, lam_traj = pd_diagnostics(Zv, gfv["a"], gfv["b"])
    check_rows.append({"variant": f"hadamard_{key}",
                       "a_clusters": "; ".join(f"{gfv['a'][gfv['labels'] == k][0]:.4f}" for k in range(1, Kv + 1)),
                       "b_clusters": "; ".join(f"{gfv['b'][gfv['labels'] == k][0]:.4f}" for k in range(1, Kv + 1)),
                       "nu": gfv["nu"], "neg_loglike": gfv["res"].fun, "BIC": gfv["bic"],
                       "lam_min_intercept": lam_int, "lam_min_Q_traj": lam_traj})
    res_hf, (a_hf, b_hf, nu_hf) = fit_gdcc_hf(Zv, gfv["labels"])
    n_par = 2 * Kv + 1
    bic_hf = n_par * np.log(T) + 2 * res_hf.fun
    lam_int2, lam_traj2 = pd_diagnostics(Zv, a_hf, b_hf, scalar_intercept=True)
    check_rows.append({"variant": f"hf_scalar_{key}",
                       "a_clusters": "; ".join(f"{a_hf[gfv['labels'] == k][0]:.4f}" for k in range(1, Kv + 1)),
                       "b_clusters": "; ".join(f"{b_hf[gfv['labels'] == k][0]:.4f}" for k in range(1, Kv + 1)),
                       "nu": nu_hf, "neg_loglike": res_hf.fun, "BIC": bic_hf,
                       "lam_min_intercept": lam_int2, "lam_min_Q_traj": lam_traj2})
check = pd.DataFrame(check_rows)
check.to_csv(TAB / "table18e_moex_gdcc_recursion_check.csv", index=False)
print(check.round(6).to_string(index=False))
