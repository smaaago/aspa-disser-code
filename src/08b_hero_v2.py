"""Refined hero figure (v2) — includes FMSV results in the capital panel
and adds a comparative violations bar.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

OUT = Path("output")
TAB = OUT / "tables"
FIG = OUT / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

mw = pd.read_csv(OUT / "mean_rtsi_world_corr.csv",
                 parse_dates=["Datetime"]).set_index("Datetime")
roll = pd.read_csv(TAB / "table8_dy_rolling.csv",
                   parse_dates=[0], index_col=0)
cap = pd.read_csv(TAB / "table16_capital_path_full.csv",
                  parse_dates=[0], index_col=0)
period_bt = pd.read_csv(TAB / "table15_var_periods_full.csv")

# --- Hero figure ---
fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(3, 6, height_ratios=[1, 1, 0.85], hspace=0.50, wspace=2.0)

# (a) TCI dynamic
ax1 = fig.add_subplot(gs[0, :3])
ax1.fill_between(roll.index, roll["TCI"], 50, alpha=0.20, color="#003049")
ax1.plot(roll.index, roll["TCI"], color="#003049", lw=1.4)
ax1.axhline(50, ls=":", color="grey", lw=0.6)
for vs, ve, c in [("2008-09-15", "2009-06-30", "#d62828"),
                  ("2020-02-20", "2020-06-30", "#7209b7"),
                  ("2022-02-21", "2022-06-30", "#f77f00")]:
    ax1.axvspan(pd.Timestamp(vs), pd.Timestamp(ve), alpha=0.13, color=c)
ax1.annotate("ГФК\n2008–09", (pd.Timestamp("2009-01-01"), 88),
             ha="center", fontsize=9, color="#7a0014")
ax1.annotate("COVID-19\n2020", (pd.Timestamp("2020-04-20"), 88),
             ha="center", fontsize=9, color="#43006e")
ax1.annotate("Шок 2022\n→ деглобализация", (pd.Timestamp("2023-05-01"), 60),
             ha="center", fontsize=9, color="#a14d00")
ax1.set_title("(а) Индекс связности TCI — впервые ниже 60% после 2022",
              loc="left", fontweight="bold")
ax1.set_ylabel("TCI, %")
ax1.set_ylim(45, 92)

# (b) RTSI net
ax2 = fig.add_subplot(gs[0, 3:])
ax2.fill_between(roll.index, roll["RTSI_NET"], 0,
                 where=roll["RTSI_NET"] >= 0, alpha=0.5, color="#3a7d44",
                 interpolate=True, label="источник")
ax2.fill_between(roll.index, roll["RTSI_NET"], 0,
                 where=roll["RTSI_NET"] < 0, alpha=0.5, color="#c33149",
                 interpolate=True, label="приёмник")
ax2.plot(roll.index, roll["RTSI_NET"], color="black", lw=0.5)
ax2.axhline(0, color="black", lw=0.5)
for vs, ve, c in [("2008-09-15", "2009-06-30", "#d62828"),
                  ("2020-02-20", "2020-06-30", "#7209b7"),
                  ("2022-02-21", "2022-06-30", "#f77f00")]:
    ax2.axvspan(pd.Timestamp(vs), pd.Timestamp(ve), alpha=0.13, color=c)
ax2.set_title("(б) Чистая позиция RTSI: коридор сжимается с 2023 г.",
              loc="left", fontweight="bold")
ax2.set_ylabel("NET, %")
ax2.legend(loc="upper left", frameon=False)

# (c) Mean corr RTSI-world
ax3 = fig.add_subplot(gs[1, :3])
ax3.plot(mw.index, mw["mean_rtsi_world_corr"], color="#999", lw=0.4, alpha=0.4)
ma = mw["mean_rtsi_world_corr"].rolling(120).mean()
ax3.plot(mw.index, ma, color="#283618", lw=1.6)
mean_pre = mw.loc[:"2021-12-31"]["mean_rtsi_world_corr"].mean()
mean_post = mw.loc["2023-01-01":]["mean_rtsi_world_corr"].mean()
ax3.axhline(mean_pre, ls="--", color="#003049", lw=0.7,
            label=f"среднее 2007–2021: {mean_pre:.2f}")
ax3.axhline(mean_post, ls="--", color="#a40000", lw=0.7,
            label=f"среднее 2023–2026: {mean_post:.2f}")
for vs, ve, c in [("2008-09-15", "2009-06-30", "#d62828"),
                  ("2020-02-20", "2020-06-30", "#7209b7"),
                  ("2022-02-21", "2022-06-30", "#f77f00")]:
    ax3.axvspan(pd.Timestamp(vs), pd.Timestamp(ve), alpha=0.13, color=c)
ax3.set_title("(в) Средняя корреляция RTSI ↔ 14 рынков (cDCC, 120-d MA)",
              loc="left", fontweight="bold")
ax3.set_ylabel("ρ")
ax3.legend(loc="lower left", frameon=False)

# (d) Capital — emphasise FMSV savings
ax4 = fig.add_subplot(gs[1, 3:])
ax4.plot(cap.index, cap["M0_EWMA"] / 1e9, color="#aaaaaa", lw=0.7, label="M0 EWMA (baseline)")
ax4.plot(cap.index, cap["M3_cDCC_t"] / 1e9, color="#003049", lw=0.7, label="M3 cDCC-t")
ax4.plot(cap.index, cap["M5_FMSV"] / 1e9, color="#a40000", lw=0.9, label="M5 FMSV")
ax4.fill_between(cap.index, cap["M5_FMSV"] / 1e9, cap["M0_EWMA"] / 1e9,
                 where=cap["M0_EWMA"] > cap["M5_FMSV"],
                 alpha=0.18, color="#3a7d44", label="экономия FMSV vs EWMA")
for vs, ve, c in [("2008-09-15", "2009-06-30", "#d62828"),
                  ("2020-02-20", "2020-06-30", "#7209b7"),
                  ("2022-02-21", "2022-06-30", "#f77f00")]:
    ax4.axvspan(pd.Timestamp(vs), pd.Timestamp(ve), alpha=0.10, color=c)
ax4.set_title("(г) Требуемый капитал: FMSV экономит ~98 млн руб. на 10 млрд нотионала",
              loc="left", fontweight="bold")
ax4.set_ylabel("млрд руб.")
ax4.legend(loc="upper right", frameon=False)

# (e) Violations bar by period
ax5 = fig.add_subplot(gs[2, :])
piv = period_bt.pivot(index="period", columns="model", values="viol_%")
piv = piv.reindex(["GFC", "Calm", "COVID", "Inter", "2022 shock", "Recent"])
piv = piv[["M0_EWMA", "M3_cDCC_t", "M4_GDCC_t", "M5_FMSV"]]
piv.plot(kind="bar", ax=ax5,
         color=["#aaaaaa", "#003049", "#f77f00", "#a40000"], width=0.8)
ax5.axhline(1.0, color="grey", ls=":", lw=0.7)
ax5.set_xticklabels(piv.index, rotation=0)
ax5.set_ylabel("Доля пробоев VaR(99%), %")
ax5.set_title("(д) Частота пробоев VaR(99%) по подпериодам:  ожидаемый уровень — пунктирная линия 1%",
              loc="left", fontweight="bold")
ax5.legend(loc="upper right", frameon=False, ncol=4)

fig.suptitle("Структурный сдвиг 2022–2026 и прикладной выигрыш FMSV: интегральный обзор результатов",
             y=0.985, fontsize=15, fontweight="bold")
plt.savefig(FIG / "fig10_deglobalization_hero.png", dpi=200, bbox_inches="tight")
plt.close()
print("Hero v2 saved.")

# ---- Additional figure: GDCC cluster dendrogram + parameter table
ca = pd.read_csv(TAB / "table5c_gdcc_clusters.csv")
gp = pd.read_csv(TAB / "table5d_gdcc_params.csv")
print("\nGDCC clusters:")
print(ca)
print(gp)
