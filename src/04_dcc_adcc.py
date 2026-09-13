"""DCC / ADCC / cDCC-GARCH with multivariate-t errors.

Stage 1 (univariate) is done in 03_univariate_garch.py -> std_residuals.csv.
Stage 2 here estimates the correlation dynamic.

Models:
- DCC (Engle 2002): Q_t = (1 - a - b) Q_bar + a z_{t-1} z_{t-1}' + b Q_{t-1}
- ADCC (Cappiello, Engle, Sheppard 2006): plus asymmetric term in negative shocks
- cDCC (Aielli 2013): bias-corrected DCC

Likelihood: multivariate t with nu_mvt degrees of freedom.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
import warnings
warnings.filterwarnings("ignore")

RES = Path("output/std_residuals.csv")
TAB = Path("output/tables")
FIG = Path("output/figures")
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

Z = pd.read_csv(RES, parse_dates=["Datetime"]).set_index("Datetime")
cols = list(Z.columns)
Zv = Z.values.astype(float)
T, N = Zv.shape
print(f"T={T}, N={N}")

# Pre-compute neg part for ADCC
Zneg = np.where(Zv < 0, Zv, 0.0)


def mvt_loglike(R, z, nu):
    """Multivariate-t log-density up to a constant; computed per-time-step.
    R is the correlation matrix, z the standardized residual (length N)."""
    # log f(z) for mvt with degrees nu, scale R (so cov = R*nu/(nu-2))
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0 or not np.isfinite(logdet):
        return -1e10
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return -1e10
    quad = float(z @ Rinv @ z)
    # log density
    p = R.shape[0]
    c = gammaln((nu + p) / 2.0) - gammaln(nu / 2.0) - 0.5 * p * np.log(np.pi * (nu - 2.0))
    return c - 0.5 * logdet - ((nu + p) / 2.0) * np.log(1.0 + quad / (nu - 2.0))


def filter_dcc(z, params, model="dcc", return_R=False):
    """Run the DCC/ADCC/cDCC filter and return -loglike (for minimiser)."""
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
            R = Q_bar / np.sqrt(np.outer(np.diag(Q_bar), np.diag(Q_bar)))
            R = (R + R.T) / 2.0
        else:
            zt = z[t - 1]
            outer = np.outer(zt, zt)
            if model == "adcc":
                zn = np.where(zt < 0, zt, 0.0)
                Q = intercept + a * outer + g * np.outer(zn, zn) + b * Q
            elif model == "cdcc":
                # cDCC: rescale shocks by Q_diag^{1/2} (Aielli 2013)
                qd = np.sqrt(np.diag(Q))
                zt_star = qd * zt
                Q = intercept + a * np.outer(zt_star, zt_star) + b * Q
            else:
                Q = intercept + a * outer + b * Q
            qd = np.sqrt(np.diag(Q))
            R = Q / np.outer(qd, qd)
            R = (R + R.T) / 2.0
        if return_R:
            Rs[t] = R
        ll = mvt_loglike(R, z[t], nu)
        loglike += ll
    return (-loglike, Rs) if return_R else -loglike


def fit_dcc(z, model="dcc"):
    if model == "adcc":
        x0 = [0.03, 0.95, 0.02, 8.0]
        bounds = [(1e-4, 0.5), (1e-4, 0.999), (1e-4, 0.5), (4.5, 60.0)]
    else:
        x0 = [0.05, 0.93, 8.0]
        bounds = [(1e-4, 0.5), (1e-4, 0.999), (4.5, 60.0)]
    res = minimize(lambda p: filter_dcc(z, p, model=model),
                   x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=200))
    return res


results = {}
for m in ["dcc", "adcc", "cdcc"]:
    print(f"\nFitting {m.upper()}...")
    r = fit_dcc(Zv, model=m)
    print(f"  converged={r.success}, neg-loglike={r.fun:.2f}")
    print(f"  params={r.x}")
    results[m] = r

rows = []
for m, r in results.items():
    if m == "adcc":
        a, b, g, nu = r.x
    else:
        a, b, nu = r.x; g = np.nan
    rows.append({"model": m.upper(), "a": a, "b": b, "g": g, "nu_mvt": nu,
                 "neg_loglike": r.fun, "params_count": len(r.x)})
parm = pd.DataFrame(rows).set_index("model")
n_obs = T
parm["AIC"] = 2 * parm["params_count"] + 2 * parm["neg_loglike"]
parm["BIC"] = parm["params_count"] * np.log(n_obs) + 2 * parm["neg_loglike"]
parm.to_csv(TAB / "table5_dcc_models.csv")
print("\n=== DCC family comparison ===")
print(parm.round(4))

# Save Rs tensors for all three models for the VaR comparison
for m in ["dcc", "adcc", "cdcc"]:
    _, Rs_m = filter_dcc(Zv, results[m].x, model=m, return_R=True)
    np.save(f"output/R_{m}.npy", Rs_m)
    print(f"Saved R_{m}.npy")
# Pull dynamic correlations from best model (cDCC by BIC)
best_name = parm["BIC"].idxmin().lower()
print(f"\nBest model by BIC: {best_name.upper()}")
neg_ll, Rs = filter_dcc(Zv, results[best_name].x, model=best_name, return_R=True)

# LR tests for model nesting
ll_dcc = -results["dcc"].fun
ll_adcc = -results["adcc"].fun
LR_adcc_vs_dcc = 2 * (ll_adcc - ll_dcc)
from scipy.stats import chi2 as _chi2
p_lr = 1 - _chi2.cdf(LR_adcc_vs_dcc, df=1)
print(f"\nLR test ADCC vs DCC: LR={LR_adcc_vs_dcc:.2f}, df=1, p={p_lr:.4g}")
with open(TAB / "table5b_lr_test.txt", "w") as f:
    f.write(f"LR test ADCC vs DCC: LR={LR_adcc_vs_dcc:.2f}, df=1, p={p_lr:.4g}\n")

# Build long-format correlation series for RTSI vs key markets
key_markets = ["spx", "dax", "hsi", "ssec"]
idx_rtsi = cols.index("rtsi")
corr_paths = {}
for c in key_markets:
    j = cols.index(c)
    corr_paths[f"rtsi_vs_{c}"] = Rs[:, idx_rtsi, j]
cp = pd.DataFrame(corr_paths, index=Z.index)
cp.to_csv("output/dyn_corr_rtsi.csv")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))
for c in cp.columns:
    ax.plot(cp.index, cp[c], lw=0.8, label=c.replace("_", " ").upper())
ax.axvspan(pd.Timestamp("2008-09-15"), pd.Timestamp("2009-06-30"),
           alpha=0.1, color="red", label="GFC")
ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-30"),
           alpha=0.1, color="purple", label="COVID")
ax.axvspan(pd.Timestamp("2022-02-21"), pd.Timestamp("2022-06-30"),
           alpha=0.1, color="orange", label="2022 SVO")
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.set_title(f"Динамические корреляции RTSI с ключевыми рынками ({best_name.upper()}-GARCH-t)")
ax.set_ylabel("Условная корреляция")
ax.legend(loc="lower right", ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig3_dyn_corr_rtsi.png", dpi=200)
plt.close()

# Time-averaged "Russia-world" correlation
rest = [j for j, c in enumerate(cols) if c != "rtsi"]
mean_rtsi_world = np.array([Rs[t, idx_rtsi, rest].mean() for t in range(T)])
mw = pd.DataFrame({"mean_rtsi_world_corr": mean_rtsi_world}, index=Z.index)
mw.to_csv("output/mean_rtsi_world_corr.csv")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(Z.index, mean_rtsi_world, color="navy", lw=0.7)
# 60-day moving average
ma = pd.Series(mean_rtsi_world).rolling(60).mean()
ax.plot(Z.index, ma, color="red", lw=1.6, label="60-day MA")
ax.axvspan(pd.Timestamp("2008-09-15"), pd.Timestamp("2009-06-30"),
           alpha=0.1, color="red")
ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-30"),
           alpha=0.1, color="purple")
ax.axvspan(pd.Timestamp("2022-02-21"), pd.Timestamp("2022-06-30"),
           alpha=0.1, color="orange")
ax.set_title(f"Средняя корреляция RTSI с 14 мировыми индексами ({best_name.upper()}-GARCH-t)")
ax.set_ylabel("Средняя условная корреляция")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "fig4_mean_rtsi_world.png", dpi=200)
plt.close()

# Save full correlation tensor for later use
np.save("output/R_dynamic.npy", Rs)
print("Done. Saved Rs tensor with shape", Rs.shape)
