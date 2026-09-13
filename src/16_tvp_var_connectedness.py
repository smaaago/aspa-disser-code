"""TVP-VAR-оценка динамики связности без скользящего окна (§ 3.8.5).

Полная формулировка Antonakakis–Chatziantoniou–Gabauer (2020): VAR(1) с
изменяющимися во времени коэффициентами, оценённый калмановским фильтром с
забывающими факторами Koop–Korobilis (2014), поверх — та же обобщённая
декомпозиция дисперсии прогноза (GFEVD, H = 10), что и в основном стеке
(05/09c). Применяется зеркально к обеим панелям лог-условных волатильностей:
глобальной (15 индексов) и локальной (7 секторов).

Конфигурация: kappa1 = 0.99 (забывание ковариации состояния), kappa2 = 0.96
(EWMA-затухание ковариации ошибок) — значения ACG-2020; инициализация — OLS
VAR(1) на первых 250 наблюдениях (длина скользящего окна основного протокола
§ 2.3.3), так что траектории двух механизмов стартуют в одной точке выборки.
Чувствительность: повторный прогон с kappa2 = 0.99.

Выход: output/tables/table28_tvp_global_daily.csv   — дневной TCI/NET RTSI (TVP)
       output/tables/table28b_tvp_local_daily.csv   — дневной локальный TCI (TVP)
       output/tables/table28c_tvp_summary.csv       — сводка по подпериодам + тесты
       output/figures/fig23_tvp_var.png
Запуск из корня: .venv/bin/python src/16_tvp_var_connectedness.py  (~1-2 мин)
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

from common import TAB, FIG, PERIODS_GLOBAL

warnings.filterwarnings("ignore")

W0 = 250          # длина обучающего отрезка = окну основного протокола
KAPPA1 = 0.99
KAPPA2 = 0.96
H = 10


def fevd_generalised_var1(B1, Sigma, H=10):
    """GFEVD (Pesaran–Shin) для VAR(1) — идентична конструкции 05/09c."""
    n = B1.shape[0]
    M = np.empty((H, n, n))
    M[0] = np.eye(n)
    for h in range(1, H):
        M[h] = B1 @ M[h - 1]
    sigma_diag = np.diag(Sigma).copy()
    sigma_diag[sigma_diag <= 0] = 1e-12
    num = np.zeros((n, n))
    den = np.zeros((n, n))
    for h in range(H):
        num += (M[h] @ Sigma) ** 2 / sigma_diag[None, :]
        den += np.diag(M[h] @ Sigma @ M[h].T)[:, None] * np.ones((1, n))
    theta = num / den
    return theta / theta.sum(axis=1, keepdims=True)


def tvp_var_tci(y, kappa1=KAPPA1, kappa2=KAPPA2, net_col=None):
    """TVP-VAR(1) с константой: фильтр Koop–Korobilis, ежедневные TCI (и NET
    выбранного узла). Состояние — коэффициенты, слож. по уравнениям; шаг t
    использует x_t = [1, y_{t-1}]."""
    Y = np.asarray(y)
    T, n = Y.shape
    k = n + 1
    m = n * k

    # --- инициализация: OLS VAR(1) на первых W0 наблюдениях
    Ytr, Xtr = Y[1:W0], np.column_stack([np.ones(W0 - 1), Y[:W0 - 1]])
    coef, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)      # k×n
    Bmat = coef.T                                          # n×k, строка = уравнение
    resid = Ytr - Xtr @ coef
    Sigma = resid.T @ resid / (W0 - 1 - k)
    XtX_inv = np.linalg.inv(Xtr.T @ Xtr + 1e-8 * np.eye(k))
    P = np.kron(Sigma, XtX_inv)                            # ковариация состояния
    beta = Bmat.reshape(-1)

    dates, tci, net = [], [], []
    j_net = None if net_col is None else list(y.columns).index(net_col)
    for t in range(W0, T):
        x = np.concatenate([[1.0], Y[t - 1]])
        P = P / kappa1                                     # забывание
        Bmat = beta.reshape(n, k)
        e = Y[t] - Bmat @ x
        ZP = np.stack([x @ P[i * k:(i + 1) * k, :] for i in range(n)])  # n×m
        F = np.stack([ZP[:, j * k:(j + 1) * k] @ x for j in range(n)]).T + Sigma
        F = 0.5 * (F + F.T) + 1e-10 * np.eye(n)
        K = np.linalg.solve(F, ZP).T                       # m×n = P Z' F^{-1}
        beta = beta + K @ e
        P = P - K @ ZP
        P = 0.5 * (P + P.T)
        Bmat = beta.reshape(n, k)
        eps = Y[t] - Bmat @ x
        Sigma = kappa2 * Sigma + (1 - kappa2) * np.outer(eps, eps)

        theta = fevd_generalised_var1(Bmat[:, 1:], Sigma, H=H)
        tci.append((theta.sum() - np.trace(theta)) / n * 100)
        if j_net is not None:
            to_ = theta[:, j_net].sum() - theta[j_net, j_net]
            fr_ = theta[j_net, :].sum() - theta[j_net, j_net]
            net.append((to_ - fr_) * 100)
        dates.append(y.index[t])
    out = pd.DataFrame({"TCI_tvp": tci}, index=pd.DatetimeIndex(dates))
    if j_net is not None:
        out["NET_tvp"] = net
    return out


PERIODS = {k: v for k, v in PERIODS_GLOBAL.items() if not k.startswith("Pre-GFC")}

# ------------------------------------------------------------- глобальный уровень
vol_g = pd.read_csv("output/cond_vol.csv", parse_dates=["Datetime"]).set_index("Datetime")
y_g = np.log(vol_g.clip(lower=1e-6))
print("Глобальная панель:", y_g.shape)
tvp_g = tvp_var_tci(y_g, net_col="rtsi")
tvp_g.to_csv(TAB / "table28_tvp_global_daily.csv")

# чувствительность к kappa2
tvp_g_s = tvp_var_tci(y_g, kappa2=0.99)
corr_kappa = tvp_g["TCI_tvp"].corr(tvp_g_s["TCI_tvp"])
print(f"Чувствительность kappa2 0.96 vs 0.99: corr = {corr_kappa:.3f}")

# ------------------------------------------------------------- локальный уровень
vol_l = pd.read_csv("output/cond_vol_moex.csv", parse_dates=["Datetime"]).set_index("Datetime")
y_l = np.log(vol_l.clip(lower=1e-6))
print("Локальная панель:", y_l.shape)
tvp_l = tvp_var_tci(y_l)
tvp_l = tvp_l.rename(columns={"TCI_tvp": "TCI_int_tvp"})
tvp_l.to_csv(TAB / "table28b_tvp_local_daily.csv")
tvp_l_s = tvp_var_tci(y_l, kappa2=0.99).rename(columns={"TCI_tvp": "TCI_int_tvp"})

# ------------------------------------------------------------- скользящие траектории
roll_g = pd.read_csv(TAB / "table8_dy_rolling.csv")
roll_g = roll_g.set_index(pd.to_datetime(roll_g.iloc[:, 0])).iloc[:, 1:]
roll_l = pd.read_csv(TAB / "table8b_moex_dy_rolling.csv")
roll_l = roll_l.set_index(pd.to_datetime(roll_l.iloc[:, 0])).iloc[:, 1:]

# корреляции траекторий на сетке скользящей оценки (шаг 10 дней)
cmp_g = pd.DataFrame({"tvp": tvp_g["TCI_tvp"], "roll": roll_g["TCI"]}).dropna()
cmp_l = pd.DataFrame({"tvp": tvp_l["TCI_int_tvp"], "roll": roll_l["TCI_int"]}).dropna()
corr_g, corr_l = cmp_g["tvp"].corr(cmp_g["roll"]), cmp_l["tvp"].corr(cmp_l["roll"])
print(f"corr(TVP, окно): глобальный {corr_g:.3f} (n={len(cmp_g)}), локальный {corr_l:.3f} (n={len(cmp_l)})")

# ------------------------------------------------------------- мост и спред (TVP)
grid = pd.date_range(max(tvp_g.index.min(), tvp_l.index.min()),
                     min(tvp_g.index.max(), tvp_l.index.max()), freq="B")
bridge_tvp = pd.DataFrame({
    "TCI_int_tvp": tvp_l["TCI_int_tvp"].reindex(grid, method="ffill"),
    "TCI_ext_tvp": tvp_g["TCI_tvp"].reindex(grid, method="ffill"),
}).dropna()
bridge_tvp["spread_tvp"] = bridge_tvp["TCI_int_tvp"] - bridge_tvp["TCI_ext_tvp"]
spread_k99 = (tvp_l_s["TCI_int_tvp"].reindex(grid, method="ffill")
              - tvp_g_s["TCI_tvp"].reindex(grid, method="ffill")).dropna()
print(f"Спред (kappa2=0.99): шок-2022 {spread_k99.loc['2022-02-21':'2022-12-30'].mean():+.1f}, "
      f"Recent {spread_k99.loc['2023-01-01':].mean():+.1f} п.п.")

bridge_roll = pd.read_csv(TAB / "table22b_tci_bridge_daily.csv",
                          parse_dates=["Datetime"]).set_index("Datetime")
corr_spread = bridge_tvp["spread_tvp"].corr(
    bridge_roll["spread_int_ext"].reindex(bridge_tvp.index, method="ffill"))
print(f"corr(спред TVP, спред окно) на дневной сетке: {corr_spread:.3f}")

# первый устойчивый переход спреда в положительную область в 2022 г.
s22 = bridge_tvp.loc["2022-01-01":, "spread_tvp"]
run, first_pos = 0, None
for d, v in s22.items():
    run = run + 1 if v > 0 else 0
    if run >= 21 and first_pos is None:            # месяц торговых дней подряд
        first_pos = d - pd.tseries.offsets.BDay(20)
        break
print(f"Первый устойчивый (>=21 дн.) положительный спред TVP: {first_pos.date() if first_pos is not None else '—'}")

# HAC-инференция (после эндогенного разрыва § 3.8.4): сдвиг режима и пост-среднее
sp_all = bridge_tvp["spread_tvp"]
D22 = (sp_all.index >= "2022-02-01").astype(float)
shift_ols = sm.OLS(sp_all.values, sm.add_constant(D22)).fit(
    cov_type="HAC", cov_kwds={"maxlags": 250})
shift, t_shift, p_shift = shift_ols.params[1], shift_ols.tvalues[1], shift_ols.pvalues[1]
pre_mean = sp_all[sp_all.index < "2022-02-01"].mean()
post = sp_all.loc["2022-02-01":].values
hac = sm.OLS(post, np.ones(len(post))).fit(cov_type="HAC", cov_kwds={"maxlags": 250})
mean_post, t_post, p_post = post.mean(), hac.tvalues[0], hac.pvalues[0] / 2
print(f"[HAC] спред TVP: среднее до 01.02.2022 {pre_mean:+.1f} п.п.; сдвиг режима {shift:+.1f} п.п., "
      f"t = {t_shift:.2f}, p = {p_shift:.2e}")
print(f"[HAC] средний спред TVP после 01.02.2022: {mean_post:+.1f} п.п., t = {t_post:.2f}, p(one-sided) = {p_post:.2e}")

# исторический минимум глобального TCI (TVP)
tci_min_date = tvp_g["TCI_tvp"].idxmin()
print(f"Минимум глобального TCI (TVP) за всю историю: {tvp_g['TCI_tvp'].min():.1f}% "
      f"({tci_min_date.date()}), попадает в Recent: {tci_min_date >= pd.Timestamp('2023-01-01')}")

# ------------------------------------------------------------- сводка по подпериодам
rows = []
for name, (a, b) in PERIODS.items():
    tg, rg = tvp_g.loc[a:b, "TCI_tvp"], roll_g.loc[a:b, "TCI"]
    tl, rl = tvp_l.loc[a:b, "TCI_int_tvp"], roll_l.loc[a:b, "TCI_int"]
    sp_t = bridge_tvp.loc[a:b, "spread_tvp"]
    sp_r = bridge_roll.loc[a:b, "spread_int_ext"]
    rows.append({"period": name,
                 "TCI_glob_tvp": tg.mean(), "TCI_glob_roll": rg.mean(),
                 "TCI_loc_tvp": tl.mean() if len(tl) else np.nan,
                 "TCI_loc_roll": rl.mean() if len(rl) else np.nan,
                 "spread_tvp": sp_t.mean() if len(sp_t) else np.nan,
                 "spread_roll": sp_r.mean() if len(sp_r) else np.nan,
                 "TCI_glob_tvp_k99": tvp_g_s.loc[a:b, "TCI_tvp"].mean(),
                 "TCI_loc_tvp_k99": tvp_l_s.loc[a:b, "TCI_int_tvp"].mean()})
summ = pd.DataFrame(rows)
extra = pd.DataFrame([
    {"period": "corr(TVP, окно), глобальный", "TCI_glob_tvp": corr_g},
    {"period": "corr(TVP, окно), локальный", "TCI_glob_tvp": corr_l},
    {"period": "corr спредов (дневная сетка)", "TCI_glob_tvp": corr_spread},
    {"period": "corr kappa2 0.96 vs 0.99", "TCI_glob_tvp": corr_kappa},
    {"period": "TVP min Recent (глоб.)", "TCI_glob_tvp": tvp_g.loc["2023-01-01":, "TCI_tvp"].min()},
    {"period": "HAC regime shift/t/p",
     "TCI_glob_tvp": shift, "TCI_glob_roll": t_shift, "TCI_loc_tvp": p_shift},
    {"period": "HAC post-2022 mean/t/p",
     "TCI_glob_tvp": mean_post, "TCI_glob_roll": t_post, "TCI_loc_tvp": p_post},
])
pd.concat([summ, extra]).to_csv(TAB / "table28c_tvp_summary.csv", index=False)
print("\n=== Подпериодные средние (TVP vs окно) ===")
print(summ.round(1).to_string(index=False))
print(f"NET RTSI (TVP), Recent: {tvp_g.loc['2023-01-01':, 'NET_tvp'].mean():+.1f} п.п.")

# ------------------------------------------------------------- рисунок
fig, ax = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True)
for a0, b0, c0 in [("2008-09-15", "2009-06-30", "red"),
                   ("2020-02-20", "2020-06-30", "purple"),
                   ("2022-02-21", "2022-06-30", "orange")]:
    for axx in ax:
        axx.axvspan(pd.Timestamp(a0), pd.Timestamp(b0), alpha=0.12, color=c0)
ax[0].plot(roll_g.index, roll_g["TCI"], color="#9db4d0", lw=0.9, label="скользящее окно 250 дн.")
ax[0].plot(tvp_g.index, tvp_g["TCI_tvp"], color="navy", lw=1.1, label="TVP-VAR (фильтр Калмана)")
ax[0].set_ylabel("TCI, %")
ax[0].legend(fontsize=9, loc="lower left")
ax[0].set_title("(а) Глобальный уровень: TCI по TVP-VAR против скользящего окна", fontsize=10)
ax[1].plot(roll_l.index, roll_l["TCI_int"], color="#d9a3a3", lw=0.9, label="скользящее окно 250 дн.")
ax[1].plot(tvp_l.index, tvp_l["TCI_int_tvp"], color="#b2182b", lw=1.1, label="TVP-VAR (фильтр Калмана)")
ax[1].set_ylabel("TCI, %")
ax[1].legend(fontsize=9, loc="lower left")
ax[1].set_title("(б) Локальный уровень: TCI по TVP-VAR против скользящего окна", fontsize=10)
ax[2].plot(bridge_roll.index, bridge_roll["spread_int_ext"], color="#9dbf9d", lw=0.9,
           label="спред (окно)")
ax[2].plot(bridge_tvp.index, bridge_tvp["spread_tvp"], color="darkgreen", lw=1.1,
           label="спред (TVP-VAR)")
ax[2].axhline(0, color="grey", lw=0.6, ls="--")
ax[2].set_ylabel("TCI_лок − TCI_глоб, п.п.")
ax[2].legend(fontsize=9, loc="lower left")
ax[2].set_title("(в) Спред двух уровней: TVP-VAR против скользящего окна", fontsize=10)
for axx in ax:
    axx.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "fig23_tvp_var.png", dpi=200)
plt.close()
print("fig23_tvp_var.png done")
