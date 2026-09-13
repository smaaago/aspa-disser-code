"""Робастность к обработке пропусков (§ 3.8.1): EM-импутация против нулевого заполнения.

Календарные пропуски глобальной панели (нулевые заполнения объединённого
календаря) замещаются условными ожиданиями EM-алгоритма для многомерного
нормального приближения; окно остановки Мосбиржи (25.02.2022–24.03.2022)
для RTSI НЕ импутируется — информация в нём отсутствует, оно остаётся
исключением протокола. Поверх точечной EM-импутации строится multiple
imputation (5 розыгрышей из условного распределения) для оценки разброса.

Для базовой и каждой импутированной панели повторяется цепочка этапов 1 и 3
методики: одномерные фильтры лучших спецификаций (table3_best_garch.csv,
масштаб и распределение — как в 03_univariate_garch.py) -> лог-волатильности
-> связность DY (полная выборка VAR(2), подпериоды, скользящее окно 250/10
c VAR(1), GFEVD H = 10). Сопоставляются: сигмы рядов, TCI полной выборки,
подпериодные TCI, траектории скользящего TCI.

Выход: output/tables/table24_imputation_compare.csv,
       output/figures/fig19_imputation_tci.png.
Запуск из корня: .venv/bin/python src/12_imputation_robustness.py  (~5-8 мин)
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260727)

DATA = Path("data/log_rets.csv")
TAB = Path("output/tables")
FIG = Path("output/figures")

SCALE = 100.0
N_MI = 5
CLOSURE = slice("2022-02-25", "2022-03-24")

PERIODS = {
    "Pre-GFC": ("2007-09-19", "2008-08-31"),
    "GFC": ("2008-09-01", "2009-06-30"),
    "Calm": ("2009-07-01", "2019-12-31"),
    "COVID": ("2020-02-20", "2020-06-30"),
    "Inter": ("2020-07-01", "2022-01-31"),
    "2022 shock": ("2022-02-21", "2022-12-30"),
    "Recent": ("2023-01-01", "2026-06-30"),
}

# ---------------------------------------------------------------- DY (как в 05)
def fit_var(Y, p=2):
    res = VAR(np.asarray(Y)).fit(p)
    return list(np.asarray(res.coefs)), np.asarray(res.sigma_u)


def fevd_generalised(B_list, Sigma, H=10):
    P = len(B_list)
    N = Sigma.shape[0]
    A = [np.eye(N)]
    for h in range(1, H):
        acc = np.zeros((N, N))
        for p in range(min(h, P)):
            acc += B_list[p] @ A[h - 1 - p]
        A.append(acc)
    sd = np.sqrt(np.diag(Sigma))
    num = np.zeros((N, N))
    den = np.zeros(N)
    for h in range(H):
        AS = A[h] @ Sigma
        num += (AS / sd[None, :]) ** 2
        den += np.einsum("ij,jk,ik->i", A[h], Sigma, A[h])
    theta = num / den[:, None]
    theta /= theta.sum(axis=1, keepdims=True)
    return theta


def tci_of(theta):
    N = theta.shape[0]
    return 100.0 * (theta.sum() - np.trace(theta)) / N


# ---------------------------------------------------------------- EM-импутация
def em_impute(df, mask, n_iter=30, tol=1e-9):
    """mask=True -> значение считается пропуском и переоценивается."""
    X = df.values.copy()
    M = mask.values
    patterns = {}
    for i in range(len(X)):
        key = tuple(np.where(M[i])[0])
        if key:
            patterns.setdefault(key, []).append(i)
    prev = None
    for it in range(n_iter):
        mu = X.mean(axis=0)
        S = np.cov(X, rowvar=False)
        for mis, rows in patterns.items():
            mis = list(mis)
            obs = [j for j in range(X.shape[1]) if j not in mis]
            Soo = S[np.ix_(obs, obs)]
            Smo = S[np.ix_(mis, obs)]
            W = Smo @ np.linalg.inv(Soo)
            for i in rows:
                X[i, mis] = mu[mis] + W @ (X[i, obs] - mu[obs])
        crit = np.sum((mu - prev) ** 2) if prev is not None else np.inf
        prev = mu
        if crit < tol:
            break
    return pd.DataFrame(X, index=df.index, columns=df.columns), patterns


def mi_draw(df_em, patterns):
    X = df_em.values.copy()
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False)
    for mis, rows in patterns.items():
        mis = list(mis)
        obs = [j for j in range(X.shape[1]) if j not in mis]
        Soo_inv = np.linalg.inv(S[np.ix_(obs, obs)])
        Smo = S[np.ix_(mis, obs)]
        cond_cov = S[np.ix_(mis, mis)] - Smo @ Soo_inv @ Smo.T
        L = np.linalg.cholesky(cond_cov + 1e-14 * np.eye(len(mis)))
        for i in rows:
            cm = mu[mis] + Smo @ Soo_inv @ (X[i, obs] - mu[obs])
            X[i, mis] = cm + L @ rng.standard_normal(len(mis))
    return pd.DataFrame(X, index=df_em.index, columns=df_em.columns)


# ---------------------------------------------------------------- этап 1 -> 3
BEST = pd.read_csv(TAB / "table3_best_garch.csv").set_index("series")["best_spec"]
VOLKW = {"GARCH": dict(vol="GARCH", p=1, q=1, o=0),
         "GJR": dict(vol="GARCH", p=1, q=1, o=1),
         "EGARCH": dict(vol="EGARCH", p=1, q=1, o=1)}


def cond_vols(panel):
    out = {}
    for c in panel.columns:
        am = arch_model(panel[c] * SCALE, mean="Constant", dist="skewt", **VOLKW[BEST[c]])
        res = am.fit(disp="off", show_warning=False)
        out[c] = res.conditional_volatility / SCALE
    return pd.DataFrame(out, index=panel.index)


def dy_bundle(logvol, label):
    y = logvol.dropna()
    theta = fevd_generalised(*fit_var(y, p=2), H=10)
    full = tci_of(theta)
    sub = {}
    for name, (a, b) in PERIODS.items():
        s = y.loc[a:b]
        if len(s) < 40:
            continue
        p_var = 2 if len(s) > 150 else 1
        sub[name] = tci_of(fevd_generalised(*fit_var(s, p=p_var), H=10))
    roll = {}
    arr = y.values
    idx = y.index
    for end in range(250, len(y), 10):
        w = arr[end - 250:end]
        try:
            roll[idx[end - 1]] = tci_of(fevd_generalised(*fit_var(w, p=1), H=10))
        except Exception:
            pass
    print(f"  [{label}] TCI full = {full:.1f}; rolling точек = {len(roll)}")
    return full, sub, pd.Series(roll)


# ================================================================ прогон
df = pd.read_csv(DATA, parse_dates=["Datetime"]).set_index("Datetime")
mask = df == 0.0
mask.loc[CLOSURE, "rtsi"] = False  # окно остановки не импутируем
print(f"Импутируемых значений: {int(mask.values.sum())} "
      f"({100 * mask.values.mean():.2f}% панели); "
      f"окно остановки Мосбиржи оставлено исключением")

df_em, patterns = em_impute(df, mask)

print("Этап 1 (базовая панель: готовые фильтры cond_vol.csv)")
base_vol = pd.read_csv("output/cond_vol.csv", parse_dates=["Datetime"]).set_index("Datetime")
print("Этап 1 (EM-панель: перефит лучших спецификаций)")
em_vol = cond_vols(df_em)

print("Этап 3: связность")
b_full, b_sub, b_roll = dy_bundle(np.log(base_vol), "нулевое заполнение")
e_full, e_sub, e_roll = dy_bundle(np.log(em_vol), "EM-импутация")

mi_fulls, mi_recents = [], []
for k in range(N_MI):
    print(f"MI-розыгрыш {k + 1}/{N_MI}")
    dk = mi_draw(df_em, patterns)
    vk = cond_vols(dk)
    yk = np.log(vk).dropna()
    mi_fulls.append(tci_of(fevd_generalised(*fit_var(yk, p=2), H=10)))
    s = yk.loc[PERIODS["Recent"][0]:PERIODS["Recent"][1]]
    mi_recents.append(tci_of(fevd_generalised(*fit_var(s, p=2), H=10)))

# ---------------------------------------------------------------- сводка
rows = []
for c in df.columns:
    rows.append({"metric": f"sigma_{c}", "zero_fill": df[c].std(),
                 "imputed": df_em[c].std(),
                 "delta_pct": 100 * (df_em[c].std() / df[c].std() - 1)})
rows.append({"metric": "TCI_full", "zero_fill": b_full, "imputed": e_full,
             "delta_pct": e_full - b_full})
for name in PERIODS:
    if name in b_sub and name in e_sub:
        rows.append({"metric": f"TCI_{name}", "zero_fill": b_sub[name],
                     "imputed": e_sub[name], "delta_pct": e_sub[name] - b_sub[name]})
common = b_roll.index.intersection(e_roll.index)
corr_traj = b_roll[common].corr(e_roll[common])
maxdiff = (b_roll[common] - e_roll[common]).abs().max()
rows.append({"metric": "rolling_corr_traj", "zero_fill": np.nan, "imputed": corr_traj,
             "delta_pct": np.nan})
rows.append({"metric": "rolling_max_absdiff_pp", "zero_fill": np.nan, "imputed": maxdiff,
             "delta_pct": np.nan})
rows.append({"metric": "MI_TCI_full_range", "zero_fill": min(mi_fulls),
             "imputed": max(mi_fulls), "delta_pct": np.nan})
rows.append({"metric": "MI_TCI_Recent_range", "zero_fill": min(mi_recents),
             "imputed": max(mi_recents), "delta_pct": np.nan})
out = pd.DataFrame(rows)
out.to_csv(TAB / "table24_imputation_compare.csv", index=False)
print(out.round(3).to_string(index=False))

# ---------------------------------------------------------------- рисунок
fig, ax = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(b_roll.index, b_roll.values, color="navy", lw=1.2, label="нулевое заполнение (базовая спецификация)")
ax[0].plot(e_roll.index, e_roll.values, color="#b2182b", lw=1.0, ls="--", label="EM-импутация")
ax[0].set_ylabel("TCI, %")
ax[0].legend(fontsize=9, loc="lower left")
ax[0].set_title("(а) Скользящий TCI глобального уровня при двух способах обработки пропусков", fontsize=10)
d = (b_roll[common] - e_roll[common])
ax[1].plot(common, d.values, color="#4d4d4d", lw=0.9)
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_ylabel("разность, п.п.")
ax[1].set_title("(б) Разность траекторий (нулевое заполнение − импутация)", fontsize=10)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig19_imputation_tci.png", dpi=200)
plt.close()
print("fig19_imputation_tci.png done")
