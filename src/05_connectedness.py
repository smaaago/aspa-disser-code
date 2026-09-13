"""Глобальный уровень, этап 3: направленная связность Диболда–Йылмаза.

Вход — условные волатильности одномерного этапа (03), логарифмируются и
подаются в VAR; поверх — обобщённая декомпозиция дисперсии прогноза (GFEVD,
H = 10) и её частотное разложение по Барунику–Кржехлику. Динамика отслеживается
скользящим окном 250 дней с шагом 10 и VAR(1) внутри окна (§ 2.3.3); полная
TVP-VAR-альтернатива без окон реализована в 16_tvp_var_connectedness.py.

Выходы: table6_dy_full, table7_dy_periods, table8_dy_rolling, table9_bk_*,
table9b_tci_horizon (сходимость TCI по горизонту к спектральному пределу),
table9c_bk_boundary (чувствительность к границе полосы), fig5_rolling_tci.
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

# ===== контроль конструкции BK (§ 3.4.4) =====
# (а) сходимость: сумма частотных полос равна связности спектральной
# (H -> inf) таблицы GFEVD, а конечно-горизонтный TCI монотонно приближается
# к ней с ростом горизонта прогноза
hor_rows = [{"H": h, "TCI": tci_of(fevd_generalised(B_list, Sigma, H=h))}
            for h in (10, 50, 200, 1600)]
hor_rows.append({"H": "sum_of_bands",
                 "TCI": sum(tci_of(M) for M in bk.values())})
pd.DataFrame(hor_rows).to_csv(TAB / "table9b_tci_horizon.csv", index=False)
print("\n=== TCI(H) convergence ===")
for r in hor_rows:
    print(f"  H={r['H']}: {r['TCI']:.2f}")

# (б) чувствительность к границе полосы: отсечка 20 / 10 (протокол § 2.3.3) /
# 5 торговых дней
bound_rows = []
for lab, wc in [("20d", np.pi / 10), ("10d", np.pi / 5), ("5d", 2 * np.pi / 5)]:
    bk_b = fevd_bk(B_list, Sigma, bands=[(0.0, wc), (wc, np.pi)])
    bound_rows.append({"boundary": lab,
                       "Long": tci_of(bk_b[(0.0, wc)]),
                       "Short": tci_of(bk_b[(wc, np.pi)])})
pd.DataFrame(bound_rows).to_csv(TAB / "table9c_bk_boundary.csv", index=False)
print("=== BK boundary sensitivity ===")
for r in bound_rows:
    print(f"  cutoff {r['boundary']}: Long={r['Long']:.2f}, Short={r['Short']:.2f}")
print("Done. BK saved.")
