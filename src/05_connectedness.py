"""Глобальный уровень, этап 3: направленная связность Диболда–Йылмаза.

Вход — условные волатильности одномерного этапа (03), логарифмируются и
подаются в VAR; поверх — обобщённая декомпозиция дисперсии прогноза (GFEVD,
H = 10) и её частотное разложение по Барунику–Кржехлику. Динамика отслеживается
скользящим окном 250 дней с шагом 10 и VAR(1) внутри окна (§ 2.3.3); полная
TVP-VAR-альтернатива без окон реализована в 16_tvp_var_connectedness.py.

Выходы: table6_dy_full, table7_dy_periods, table8_dy_rolling, table9_bk_*,
fig5_rolling_tci.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (TAB, FIG, PERIODS_GLOBAL, BANDS, BAND_LABELS, load_log_vol,
                    fit_var, fevd_generalised, fevd_bk, tci_of, to_from_net,
                    spillover_table, rolling_connectedness, shade, OUT)

warnings.filterwarnings("ignore")

H = 10
y = load_log_vol(OUT / "cond_vol.csv")
cols = list(y.columns)
N = len(cols)
print("Vol shape:", y.shape, "range:", y.index.min().date(), y.index.max().date())

# ===== полная выборка =====
B_list, Sigma = fit_var(y, p=2)
theta = fevd_generalised(B_list, Sigma, H=H)
full_table = spillover_table(theta, cols)
full_table.to_csv(TAB / "table6_dy_full.csv")
print("\n=== Static full-sample spillover (%) ===")
print(full_table.round(1))
print(f"\nTCI (full sample): {tci_of(theta):.2f}%")

# ===== подпериоды =====
idx_r = cols.index("rtsi")
sub_rows = []
for name, (a, b) in PERIODS_GLOBAL.items():
    sub = y.loc[a:b]
    if len(sub) < 40:
        continue
    B, S = fit_var(sub, p=2 if len(sub) > 150 else 1)
    th = fevd_generalised(B, S, H=H)
    to_, from_, net_ = to_from_net(th, idx_r)
    sub_rows.append({"period": name, "n_obs": len(sub), "TCI": tci_of(th),
                     "Russia_TO": to_, "Russia_FROM": from_, "Russia_NET": net_})
period_tbl = pd.DataFrame(sub_rows)
period_tbl.to_csv(TAB / "table7_dy_periods.csv", index=False)
print("\n=== Spillovers by subperiod ===")
print(period_tbl.round(2))

# ===== скользящее окно =====
roll = rolling_connectedness(y, node="rtsi", win=250, step=10, p=1, H=H)
roll = roll.rename(columns={"NET": "RTSI_NET"})
roll.to_csv(TAB / "table8_dy_rolling.csv")

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(roll.index, roll["TCI"], color="navy", lw=0.9)
ax[0].set_ylabel("TCI, %")
ax[0].set_title("Dynamic Total Connectedness Index (250-дневное окно, VAR(1), GFEVD H=10)")
ax[1].plot(roll.index, roll["RTSI_NET"], color="darkred", lw=0.9)
ax[1].axhline(0, color="grey", lw=0.5)
ax[1].set_ylabel("NET (RTSI), %")
ax[1].set_title("Чистый перелив для RTSI: положительный — источник, отрицательный — приёмник")
for a_ in ax:
    shade(a_)
plt.tight_layout()
plt.savefig(FIG / "fig5_rolling_tci.png", dpi=200)
plt.close()

# ===== частотная декомпозиция Баруника–Кржехлика =====
bk = fevd_bk(B_list, Sigma, bands=BANDS)
print("\n=== Baruník-Křehlík bands (full sample) ===")
for b, M in bk.items():
    print(f"  {BAND_LABELS[b]}: within-band TCI contribution = {tci_of(M):.2f}%")
    dfbk = pd.DataFrame(M * 100, index=cols, columns=cols)
    dfbk["FROM"] = dfbk.sum(axis=1) - np.diag(M * 100)
    dfbk.to_csv(TAB / f"table9_bk_{BAND_LABELS[b].replace(' ', '_').replace('>', 'gt')}.csv")
print("Done. BK saved.")
