"""Кластерно-регуляризованная обобщённая DCC (этап 2 методики, § 2.3.1).

Рабочая рекурсия — диагональный случай обобщённой асимметричной DCC
Каппьелло–Энгла–Шеппарда (2006) без члена асимметрии:

    Q_t = (Q_bar - aa' ⊙ Q_bar - bb' ⊙ Q_bar)
          + (aa') ⊙ (z_{t-1} z_{t-1}') + (bb') ⊙ Q_{t-1},

где ⊙ — поэлементное произведение, a, b — N-векторы с ограничением равенства
внутри кластеров: a_i = a_{c(i)}, b_i = b_{c(i)}. Интерсепт
(11' - aa' - bb') ⊙ Q_bar даёт точное таргетирование Q_bar в среднем, но, как
предупреждают Хафнер и Франсес (2009) для матриц этого вида, при разнородных
параметрах он НЕ обязан быть положительно полуопределённым, поэтому:
  (а) в правдоподобии любая не положительно определённая R_t получает штраф
      −1e10 — принятая оценка заведомо даёт PD-траекторию на выборке;
  (б) по итогам оценивания выполняется прямая диагностика: минимальные
      собственные числа интерсепта и всех Q_t траектории (секция 5);
  (в) как робастность там же оценивается вариант с каноническим скалярным
      интерсептом Хафнера–Франсеса (1 - ā² - b̄²) Q_bar, положительно
      определённым по построению (ā, b̄ — средние параметров по активам).

Кластеризация работает на эмпирической корреляционной матрице
стандартизированных остатков двумя независимыми методами:
  - иерархическая (average linkage на метрике Мантеньи d = sqrt(2 (1 - rho)));
  - спектральная (лапласиан графа |rho|, K-средних на собственных векторах).

Правдоподобие многомерное t; (a_c, b_c, nu) оптимизируются совместно.

Outputs:
  output/tables/table5c_gdcc_clusters.csv        — cluster assignment
  output/tables/table5d_gdcc_params.csv          — fitted cluster parameters
  output/tables/table5e_gdcc_recursion_check.csv — PD-диагностика + вариант H-F
  output/figures/fig9_corr_dendrogram.png        — dendrogram
  output/R_gdcc.npy                              — full dynamic correlation tensor
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
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

Z = pd.read_csv(OUT / "std_residuals.csv", parse_dates=["Datetime"]).set_index("Datetime")
cols = list(Z.columns)
Zv = Z.values.astype(float)
T, N = Zv.shape
print(f"Standardised residuals: T={T}, N={N}")

# ============================================================
# 1. Build empirical correlation, then two clusterings
# ============================================================
R_emp = np.corrcoef(Zv, rowvar=False)
np.fill_diagonal(R_emp, 1.0)

# Mantegna distance: d_ij = sqrt(2 (1 - rho_ij))
D = np.sqrt(np.clip(2 * (1 - R_emp), 0, None))
np.fill_diagonal(D, 0.0)
Dcond = squareform(D, checks=False)

# Hierarchical average linkage
linkage_mat = linkage(Dcond, method="average")
# Cut at K = 3 clusters (matches FMSV factor count for content consistency)
K_CLUST = 3
hc_labels = fcluster(linkage_mat, t=K_CLUST, criterion="maxclust")

# Spectral clustering on similarity matrix |rho|
sim = np.abs(R_emp)
spc = SpectralClustering(n_clusters=K_CLUST, affinity="precomputed",
                         assign_labels="kmeans", random_state=2026)
sp_labels = spc.fit_predict(sim) + 1  # 1-indexed

print("\nCluster assignment:")
ca = pd.DataFrame({"asset": cols,
                   "hierarchical": hc_labels,
                   "spectral": sp_labels}).set_index("asset")
print(ca)
ca.to_csv(TAB / "table5c_gdcc_clusters.csv")

# Pretty dendrogram with cluster colors
fig, ax = plt.subplots(figsize=(11, 5.5))
dn = dendrogram(linkage_mat, labels=[c.upper() for c in cols],
                color_threshold=linkage_mat[-K_CLUST + 1, 2],
                above_threshold_color="grey", ax=ax)
ax.set_title("Иерархическая кластеризация 15 рынков (расстояние Мантеньи, average linkage)",
             fontsize=11)
ax.set_ylabel("Distance")
plt.tight_layout()
plt.savefig(FIG / "fig9_corr_dendrogram.png", dpi=200)
plt.close()
print("Dendrogram saved.")

# ============================================================
# 2. GDCC likelihood with cluster-constrained parameters
# ============================================================

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


def filter_gdcc(z, a_vec, b_vec, nu, return_R=False):
    """Hafner-Franses GDCC with vectorised a_vec, b_vec (one per asset)."""
    T, N = z.shape
    A = np.outer(a_vec, a_vec)
    B = np.outer(b_vec, b_vec)
    # Stationarity (element-wise per Hafner-Franses): max_i (a_i^2 + b_i^2) < 1
    max_diag = float(np.max(a_vec ** 2 + b_vec ** 2))
    if max_diag >= 0.999:
        return (1e10, None) if return_R else 1e10
    if np.min(a_vec) <= 0 or np.min(b_vec) <= 0:
        return (1e10, None) if return_R else 1e10
    if nu <= 4.5:
        return (1e10, None) if return_R else 1e10

    Q_bar = z.T @ z / T
    # Intercept matrix
    ones = np.ones((N, N))
    Mavg = ones - A - B
    intercept = Mavg * Q_bar  # Hadamard

    Q = Q_bar.copy()
    Rs = np.empty((T, N, N)) if return_R else None
    loglike = 0.0
    for t in range(T):
        if t == 0:
            qd = np.sqrt(np.diag(Q_bar))
            R = Q_bar / np.outer(qd, qd)
        else:
            zt = z[t - 1]
            outer = np.outer(zt, zt)
            Q = intercept + A * outer + B * Q
            qd = np.sqrt(np.diag(Q))
            R = Q / np.outer(qd, qd)
        R = (R + R.T) / 2.0
        if return_R:
            Rs[t] = R
        ll = mvt_loglike(R, z[t], nu)
        loglike += ll
    return (-loglike, Rs) if return_R else -loglike


def fit_gdcc_clustered(z, labels):
    """Fit GDCC with a_i = a_{c(i)}, b_i = b_{c(i)}."""
    labels = np.asarray(labels)
    K = len(np.unique(labels))
    # Parameter vector: (a_1, ..., a_K, b_1, ..., b_K, nu)
    def to_full(params):
        ac = params[:K]
        bc = params[K:2 * K]
        nu = params[-1]
        a_vec = ac[labels - 1]
        b_vec = bc[labels - 1]
        return a_vec, b_vec, nu

    def neg_ll(params):
        a_vec, b_vec, nu = to_full(params)
        return filter_gdcc(z, a_vec, b_vec, nu, return_R=False)

    x0 = np.concatenate([np.full(K, 0.10), np.full(K, 0.95), [9.0]])
    bounds = ([(1e-3, 0.6)] * K + [(1e-3, 0.999)] * K + [(4.5, 60.0)])
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=300))
    return res, to_full(res.x)


print("\n=== GDCC with hierarchical clustering ===")
res_h, (a_h, b_h, nu_h) = fit_gdcc_clustered(Zv, hc_labels)
print(f"converged={res_h.success}; neg_loglike={res_h.fun:.2f}")
print("Cluster-level a:", [f"{a_h[hc_labels == k][0]:.4f}" for k in range(1, K_CLUST + 1)])
print("Cluster-level b:", [f"{b_h[hc_labels == k][0]:.4f}" for k in range(1, K_CLUST + 1)])
print(f"nu={nu_h:.3f}")

print("\n=== GDCC with spectral clustering ===")
res_s, (a_s, b_s, nu_s) = fit_gdcc_clustered(Zv, sp_labels)
print(f"converged={res_s.success}; neg_loglike={res_s.fun:.2f}")
print("Cluster-level a:", [f"{a_s[sp_labels == k][0]:.4f}" for k in range(1, K_CLUST + 1)])
print("Cluster-level b:", [f"{b_s[sp_labels == k][0]:.4f}" for k in range(1, K_CLUST + 1)])
print(f"nu={nu_s:.3f}")

# ============================================================
# 3. Build comparison table with prior DCC/ADCC/cDCC
# ============================================================
prev = pd.read_csv(TAB / "table5_dcc_models.csv").set_index("model")
n_obs = T

def info_crit(neg_ll, n_par, n):
    return 2 * n_par + 2 * neg_ll, n_par * np.log(n) + 2 * neg_ll

n_par_gdcc = 2 * K_CLUST + 1
aic_h, bic_h = info_crit(res_h.fun, n_par_gdcc, n_obs)
aic_s, bic_s = info_crit(res_s.fun, n_par_gdcc, n_obs)

cmp = prev.copy()
cmp.loc["GDCC_hier"] = [a_h.mean(), b_h.mean(), np.nan, nu_h,
                       res_h.fun, n_par_gdcc, aic_h, bic_h]
cmp.loc["GDCC_spec"] = [a_s.mean(), b_s.mean(), np.nan, nu_s,
                       res_s.fun, n_par_gdcc, aic_s, bic_s]
cmp.to_csv(TAB / "table5_dcc_family_full.csv")
print("\n=== Extended model comparison ===")
print(cmp.round(2))

# ============================================================
# 4. Save dynamic correlation tensor for the best clustering
# ============================================================
best = "hier" if res_h.fun <= res_s.fun else "spec"
print(f"\nBest GDCC clustering: {best}")
if best == "hier":
    a_vec, b_vec, nu = a_h, b_h, nu_h
    labels = hc_labels
else:
    a_vec, b_vec, nu = a_s, b_s, nu_s
    labels = sp_labels

_, Rs_gdcc = filter_gdcc(Zv, a_vec, b_vec, nu, return_R=True)
np.save(OUT / "R_gdcc.npy", Rs_gdcc)
print(f"Saved R_gdcc.npy with shape {Rs_gdcc.shape}")

# Save the cluster-level params for the best
best_pars = pd.DataFrame({"cluster": list(range(1, K_CLUST + 1)),
                          "a": [a_vec[labels == k][0] for k in range(1, K_CLUST + 1)],
                          "b": [b_vec[labels == k][0] for k in range(1, K_CLUST + 1)],
                          "members": [", ".join([cols[i].upper() for i in range(N)
                                                  if labels[i] == k])
                                       for k in range(1, K_CLUST + 1)]})
best_pars["nu_mvt"] = nu
best_pars.to_csv(TAB / "table5d_gdcc_params.csv", index=False)
print("\n=== Best GDCC cluster-level parameters ===")
print(best_pars)

# ============================================================
# 5. Проверка рекурсии: PD-диагностика и скалярный интерсепт Хафнера–Франсеса
# ============================================================

def pd_diagnostics(z, a_vec, b_vec, scalar_intercept=False):
    """Минимальные собственные числа интерсепта и всех Q_t на траектории."""
    T, N = z.shape
    A = np.outer(a_vec, a_vec)
    B = np.outer(b_vec, b_vec)
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


def filter_gdcc_hf(z, a_vec, b_vec, nu, return_R=False):
    """GDCC со скалярным интерсептом Хафнера–Франсеса (1 - ā² - b̄²) Q_bar:
    при положительном скаляре и PD Q_bar каждое слагаемое рекурсии PSD,
    так что Q_t положительно определена по построению."""
    T, N = z.shape
    A = np.outer(a_vec, a_vec)
    B = np.outer(b_vec, b_vec)
    max_diag = float(np.max(a_vec ** 2 + b_vec ** 2))
    c0 = 1.0 - a_vec.mean() ** 2 - b_vec.mean() ** 2
    if max_diag >= 0.999 or c0 <= 1e-6:
        return (1e10, None) if return_R else 1e10
    if np.min(a_vec) <= 0 or np.min(b_vec) <= 0 or nu <= 4.5:
        return (1e10, None) if return_R else 1e10
    Q_bar = z.T @ z / T
    intercept = c0 * Q_bar
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


def fit_gdcc_hf(z, labels):
    labels = np.asarray(labels)
    K = len(np.unique(labels))

    def to_full(params):
        return params[:K][labels - 1], params[K:2 * K][labels - 1], params[-1]

    def neg_ll(params):
        a_vec, b_vec, nu = to_full(params)
        return filter_gdcc_hf(z, a_vec, b_vec, nu)

    x0 = np.concatenate([np.full(K, 0.10), np.full(K, 0.95), [9.0]])
    bounds = [(1e-3, 0.6)] * K + [(1e-3, 0.999)] * K + [(4.5, 60.0)]
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=300))
    return res, to_full(res.x)


print("\n=== GDCC со скалярным интерсептом Хафнера–Франсеса (робастность) ===")
res_h_hf, (a_h_hf, b_h_hf, nu_h_hf) = fit_gdcc_hf(Zv, hc_labels)
res_s_hf, (a_s_hf, b_s_hf, nu_s_hf) = fit_gdcc_hf(Zv, sp_labels)
print(f"hier: neg_ll={res_h_hf.fun:.2f}; spec: neg_ll={res_s_hf.fun:.2f}")

check_rows = []
for tag, labels_v, a_v, b_v, nu_v, res_v, scalar in [
        ("hadamard_hier", hc_labels, a_h, b_h, nu_h, res_h, False),
        ("hadamard_spec", sp_labels, a_s, b_s, nu_s, res_s, False),
        ("hf_scalar_hier", hc_labels, a_h_hf, b_h_hf, nu_h_hf, res_h_hf, True),
        ("hf_scalar_spec", sp_labels, a_s_hf, b_s_hf, nu_s_hf, res_s_hf, True)]:
    lam_int, lam_traj = pd_diagnostics(Zv, a_v, b_v, scalar_intercept=scalar)
    _, bic_v = info_crit(res_v.fun, n_par_gdcc, n_obs)
    check_rows.append({
        "variant": tag,
        "a_clusters": "; ".join(f"{a_v[labels_v == k][0]:.4f}" for k in range(1, K_CLUST + 1)),
        "b_clusters": "; ".join(f"{b_v[labels_v == k][0]:.4f}" for k in range(1, K_CLUST + 1)),
        "nu": nu_v, "neg_loglike": res_v.fun, "BIC": bic_v,
        "lam_min_intercept": lam_int, "lam_min_Q_traj": lam_traj})
check = pd.DataFrame(check_rows)
check.to_csv(TAB / "table5e_gdcc_recursion_check.csv", index=False)
print("\n=== Диагностика рекурсии GDCC (PD-проверка, вариант H-F) ===")
print(check.round(6).to_string(index=False))
