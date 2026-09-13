"""Сводные рисунки глобального уровня.

fig11_connectedness_heatmap.png — тепловая карта полной матрицы направленных
    переливов Диболда–Йылмаза (печатный Рис. 10);
fig12_corr_three_regimes.png — условная корреляционная матрица в трёх режимах:
    стабильная полоса 2018–2021, шок 2022, новая фаза 2023–2026 (печатный Рис. 9).

Интегральный обзорный рисунок (fig10) строится в 08b_hero_v2.py — он включает
результаты FMSV и потому идёт после прикладного блока.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

OUT = Path("output")
TAB = OUT / "tables"
FIG = OUT / "figures"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

cols = ["rtsi", "asx", "bel", "bist", "bovespa", "cac", "dax", "ftse",
        "hsi", "nifty", "nikkei", "smi", "spx", "ssec", "tsx"]

# ============================================================
# Connectedness heatmap
# ============================================================
full = pd.read_csv(TAB / "table6_dy_full.csv", index_col=0)
M = full.iloc[:15, :15].values
labels = [c.upper() for c in cols]
fig, ax = plt.subplots(figsize=(11, 9))
cmap = LinearSegmentedColormap.from_list("dy", ["white", "#fdc500", "#d62828"])
im = ax.imshow(M, cmap=cmap, aspect="equal", vmin=0, vmax=20)
ax.set_xticks(np.arange(15)); ax.set_yticks(np.arange(15))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Источник шока (j)", fontweight="bold")
ax.set_ylabel("Цель (i)", fontweight="bold")
for i in range(15):
    for j in range(15):
        v = M[i, j]
        color = "white" if v > 12 else "black"
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                color=color, fontsize=7.5)
cbar = plt.colorbar(im, ax=ax, fraction=0.045)
cbar.set_label("Объяснённая доля дисперсии, %")
ax.set_title("Матрица направленных переливов волатильности по Diebold–Yilmaz, "
             "полная выборка (GFEVD, H = 10)\n"
             "TCI = 68,6%; европейский кластер (CAC, DAX, BEL, FTSE, SMI) — "
             "сильнейший источник; RTSI — приёмник",
             loc="left", fontsize=11)
plt.tight_layout()
plt.savefig(FIG / "fig11_connectedness_heatmap.png", dpi=200)
plt.close()
print("Heatmap saved.")

# ============================================================
# Correlation panel — pre 2022, 2022, post 2022 (using cDCC tensor)
# ============================================================
Rs = np.load(OUT / "R_cdcc.npy")
dates = pd.read_csv(OUT / "log_rets_clean.csv", parse_dates=["Datetime"])["Datetime"]

windows = [("2018-01-01", "2021-12-31", "Стабильный 2018–2021"),
           ("2022-01-01", "2022-12-31", "Шок 2022"),
           ("2023-01-01", "2026-06-30", "Новая равновесная фаза 2023–2026")]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for ax, (s, e, name) in zip(axes, windows):
    mask = (dates >= s) & (dates <= e)
    R_avg = Rs[mask.values].mean(axis=0)
    cmap = LinearSegmentedColormap.from_list("c", ["#1a4d80", "white", "#a52828"])
    im = ax.imshow(R_avg, cmap=cmap, vmin=-0.2, vmax=0.8)
    ax.set_xticks(np.arange(15)); ax.set_yticks(np.arange(15))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    avg_off = (R_avg.sum() - np.trace(R_avg)) / (15 * 14)
    ax.set_title(f"{name}\nсредняя кросс-корреляция = {avg_off:.2f}",
                 fontsize=10, loc="left", fontweight="bold")
cbar = fig.colorbar(im, ax=axes, fraction=0.025, shrink=0.85, pad=0.02)
cbar.set_label("ρ")
fig.suptitle("Условная корреляционная матрица (cDCC-GARCH-t) в трёх режимах",
             fontsize=13, fontweight="bold", y=1.02)
plt.savefig(FIG / "fig12_corr_three_regimes.png", dpi=200, bbox_inches="tight")
plt.close()
print("Three-regime panel saved.")
