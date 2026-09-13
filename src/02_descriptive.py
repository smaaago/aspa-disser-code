"""Дескриптивный анализ глобальной панели (§ 2.2).

Вход: output/log_rets_clean.csv — лог-доходности 15 индексов (§ 2.1).
Выход: table1_desc.csv и table2_diag.csv (печатные Таблицы 3–4),
fig1_returns_timeseries.png и fig2_densities.png (печатные Рисунки 4–5).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import warnings
warnings.filterwarnings("ignore")

DATA = Path("output/log_rets_clean.csv")
TAB = Path("output/tables")
FIG = Path("output/figures")
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["Datetime"]).set_index("Datetime")
cols = list(df.columns)
print("Series:", cols)
print("Sample:", df.shape)

# ===== Описательные статистики (печатная Таблица 3) =====
desc = pd.DataFrame(index=cols)
desc["mean"] = df.mean()
desc["median"] = df.median()
desc["std"] = df.std()
desc["min"] = df.min()
desc["max"] = df.max()
desc["skew"] = df.skew()
desc["kurtosis"] = df.kurtosis()  # избыточный эксцесс (по Фишеру)
desc.to_csv(TAB / "table1_desc.csv")
print("\n=== Table 1: descriptive statistics ===")
print(desc.round(6))

# ===== Тесты предпосылок: ADF, KPSS, ARCH-LM(10), Льюнг—Бокс по доходностям
# и их квадратам (печатная Таблица 4) =====
rows = []
for c in cols:
    x = df[c].values
    adf = adfuller(x, autolag="AIC")
    adf_stat = adf[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat = kpss(x, regression="c", nlags="auto")[0]
    arch_lm = het_arch(x, nlags=10)[0]
    lb_r = acorr_ljungbox(x, lags=[20], return_df=True)["lb_stat"].values[0]
    lb_r2 = acorr_ljungbox(x**2, lags=[20], return_df=True)["lb_stat"].values[0]
    rows.append({"ADF": adf_stat, "KPSS": kpss_stat,
                 "ARCH-LM": arch_lm, "LB-returns": lb_r, "LB-squares": lb_r2})
diag = pd.DataFrame(rows, index=cols)
diag.to_csv(TAB / "table2_diag.csv")
print("\n=== Table 2: diagnostics ===")
print(diag.round(3))

# ===== Траектории доходностей, панель 5×3 (печатный Рисунок 4) =====
fig, axes = plt.subplots(5, 3, figsize=(15, 14), sharex=True)
for ax, c in zip(axes.flat, cols):
    ax.plot(df.index, df[c], lw=0.4, color="black")
    ax.set_title(c.upper(), fontsize=9)
    ax.axhline(0, color="grey", lw=0.4, ls="--")
plt.tight_layout()
plt.savefig(FIG / "fig1_returns_timeseries.png", dpi=200)
plt.close()

# ===== Эмпирические плотности против нормальной (печатный Рисунок 5) =====
fig, axes = plt.subplots(5, 3, figsize=(15, 14))
xx = np.linspace(-0.1, 0.1, 400)
for ax, c in zip(axes.flat, cols):
    x = df[c].values
    x_clip = np.clip(x, -0.1, 0.1)
    ax.hist(x_clip, bins=120, density=True, alpha=0.5, color="steelblue")
    mu, sd = x.mean(), x.std()
    ax.plot(xx, stats.norm.pdf(xx, mu, sd), color="red", lw=1.2, label="N(μ,σ²)")
    ax.set_title(c.upper(), fontsize=9)
    ax.set_xlim(-0.08, 0.08)
plt.tight_layout()
plt.savefig(FIG / "fig2_densities.png", dpi=200)
plt.close()

print("\nFigures written.")
