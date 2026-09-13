"""Локальный уровень, этап 3: связность отраслевых индексов + мост двух уровней.

Зеркало 05_connectedness.py: тот же модуль common.py, те же параметры протокола
(VAR(2) на полной выборке и цельных подпериодах, GFEVD H = 10, скользящее окно
250/10 с VAR(1), полосы Баруника–Кржехлика), другая панель — 7 отраслевых
индексов Мосбиржи. Далее локальная скользящая траектория TCI сводится с
глобальной (table8_dy_rolling.csv) на общей дневной сетке и строится спред
«локальный − глобальный» — проверка гипотезы о переориентации связности.

Выходы: table19_moex_dy_full, table20_moex_dy_periods, table21_moex_bk_*,
table8b_moex_dy_rolling, table22_tci_bridge, table22b_tci_bridge_daily,
fig13_moex_dy, fig14_tci_bridge.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (OUT, TAB, FIG, PERIODS_LOCAL, BANDS, BAND_LABELS, load_log_vol,
                    fit_var, fevd_generalised, fevd_bk, tci_of, to_from_net,
                    spillover_table, rolling_connectedness, shade)

warnings.filterwarnings("ignore")

H = 10
NICE = {"moexog": "Нефтегаз", "moexfn": "Финансы", "moexmm": "Металлы",
        "moexcn": "Потребит.", "moexeu": "Энергетика", "moexch": "Химия",
        "moextn": "Транспорт"}

y = load_log_vol(OUT / "cond_vol_moex.csv")
cols = list(y.columns)
N = len(cols)
print("Sectoral vol:", y.shape, y.index.min().date(), "->", y.index.max().date())

# ===== полная выборка =====
B_list, Sigma = fit_var(y, p=2)
theta = fevd_generalised(B_list, Sigma, H=H)
full_table = spillover_table(theta, cols)
full_table.to_csv(TAB / "table19_moex_dy_full.csv")
TCI_full = tci_of(theta)
print(f"\nInternal TCI (full sample): {TCI_full:.2f}%")
print(full_table.round(1))

# ===== подпериоды (сетка та же, что на глобальном уровне) =====
idx_og = cols.index("moexog")
sub_rows = []
for name, (a, b) in PERIODS_LOCAL.items():
    sub = y.loc[a:b]
    if len(sub) < 40:
        continue
    B, S = fit_var(sub, p=2 if len(sub) > 150 else 1)
    th = fevd_generalised(B, S, H=H)
    to_og, from_og, net_og = to_from_net(th, idx_og)
    sub_rows.append({"period": name, "n_obs": len(sub), "TCI_int": tci_of(th),
                     "OG_TO": to_og, "OG_FROM": from_og, "OG_NET": net_og})
period_tbl = pd.DataFrame(sub_rows)
period_tbl.to_csv(TAB / "table20_moex_dy_periods.csv", index=False)
print("\n=== Internal spillovers by subperiod ===")
print(period_tbl.round(2))

# ===== скользящая локальная связность =====
roll_int = rolling_connectedness(y, win=250, step=10, p=1, H=H)[["TCI"]]
roll_int = roll_int.rename(columns={"TCI": "TCI_int"})
roll_int.to_csv(TAB / "table8b_moex_dy_rolling.csv")

# ===== частотные полосы =====
bk = fevd_bk(B_list, Sigma, bands=BANDS)
for b, M in bk.items():
    print(f"BK {BAND_LABELS[b]}: {tci_of(M):.2f} п.п.")
    dfbk = pd.DataFrame(M * 100, index=cols, columns=cols)
    dfbk["FROM"] = dfbk.sum(axis=1) - np.diag(M * 100)
    dfbk.to_csv(TAB / f"table21_moex_bk_{BAND_LABELS[b].replace(' ', '_').replace('>', 'gt')}.csv")

# ===== мост: локальный против глобального TCI =====
# Скользящие сетки двух панелей (каждый 10-й торговый день своей панели) почти
# не совпадают, поэтому обе траектории приводятся к общей рабочей сетке с
# протяжкой последнего значения.
roll_ext = pd.read_csv(TAB / "table8_dy_rolling.csv")
roll_ext = roll_ext.set_index(pd.to_datetime(roll_ext.iloc[:, 0])).iloc[:, 1:]
roll_ext.index.name = "Datetime"
grid = pd.date_range(max(roll_int.index.min(), roll_ext.index.min()),
                     min(roll_int.index.max(), roll_ext.index.max()), freq="B")
bridge = pd.DataFrame({
    "TCI_int": roll_int["TCI_int"].reindex(grid, method="ffill"),
    "TCI_ext": roll_ext["TCI"].reindex(grid, method="ffill"),
}).dropna()
bridge.index.name = "Datetime"
bridge["spread_int_ext"] = bridge["TCI_int"] - bridge["TCI_ext"]
bridge.to_csv(TAB / "table22b_tci_bridge_daily.csv")

br_rows = []
for name, (a, b) in PERIODS_LOCAL.items():
    w = bridge.loc[a:b]
    if len(w) < 5:
        continue
    br_rows.append({"period": name, "TCI_int_mean": w["TCI_int"].mean(),
                    "TCI_ext_mean": w["TCI_ext"].mean(),
                    "spread_mean": w["spread_int_ext"].mean(),
                    "corr_int_ext": w["TCI_int"].corr(w["TCI_ext"])})
br = pd.DataFrame(br_rows)
br.loc[len(br)] = {"period": "FULL", "TCI_int_mean": bridge["TCI_int"].mean(),
                   "TCI_ext_mean": bridge["TCI_ext"].mean(),
                   "spread_mean": bridge["spread_int_ext"].mean(),
                   "corr_int_ext": bridge["TCI_int"].corr(bridge["TCI_ext"])}
br.to_csv(TAB / "table22_tci_bridge.csv", index=False)
print("\n=== TCI bridge (internal vs external) ===")
print(br.round(2))

# ===== рисунки =====
nice = [NICE[c] for c in cols]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
im = ax[0].imshow(theta * 100, cmap="YlOrRd", vmin=0)
ax[0].set_xticks(range(N), nice, rotation=45, ha="right", fontsize=8)
ax[0].set_yticks(range(N), nice, fontsize=8)
for i in range(N):
    for j in range(N):
        v = theta[i, j] * 100
        ax[0].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                   color="white" if v > 30 else "black")
ax[0].set_title(f"(а) Матрица переливов между секторами, % (TCI = {TCI_full:.1f}%)", fontsize=10)
plt.colorbar(im, ax=ax[0], fraction=0.046)

net = full_table.loc["NET"].iloc[:-1].astype(float)
ax[1].barh(range(N), net.values,
           color=["#b2182b" if v > 0 else "#2166ac" for v in net])
ax[1].set_yticks(range(N), nice, fontsize=8)
ax[1].axvline(0, color="grey", lw=0.7)
ax[1].set_title("(б) Чистая позиция NET по секторам, п.п.", fontsize=10)
ax[1].invert_yaxis()
plt.tight_layout()
plt.savefig(FIG / "fig13_moex_dy.png", dpi=200)
plt.close()

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(bridge.index, bridge["TCI_int"], color="#b2182b", lw=1.1,
           label="Локальный уровень (7 секторов МосБиржи)")
ax[0].plot(bridge.index, bridge["TCI_ext"], color="navy", lw=1.1,
           label="Глобальный уровень (15 мировых индексов)")
ax[0].set_ylabel("TCI, %")
ax[0].legend(fontsize=9, loc="lower left")
ax[0].set_title("Связность двух уровней: локальная (секторальная) против глобальной")
ax[1].plot(bridge.index, bridge["spread_int_ext"], color="darkgreen", lw=1.0)
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_ylabel("TCI_int − TCI_ext, п.п.")
for a_ in ax:
    shade(a_)
plt.tight_layout()
plt.savefig(FIG / "fig14_tci_bridge.png", dpi=200)
plt.close()
print("\nfig13_moex_dy.png, fig14_tci_bridge.png saved.")
