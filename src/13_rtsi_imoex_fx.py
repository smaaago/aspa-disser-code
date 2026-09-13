"""Пара (RTSI, IMOEX) и валютная компонента (§ 3.8.2).

Одна и та же корзина в двух номинациях связана тождеством дневных
лог-доходностей r_RTSI = r_IMOEX − r_FX, где r_FX — курсовой фактор
пересчёта, использованный биржей при расчёте индексов. Поэтому валютный
фактор извлекается НЕЯВНО и точно: r_FX := r_IMOEX − r_RTSI; внешняя
котировка USD/RUB (Yahoo) служит только санитарной проверкой (её фиксинг
несинхронен времени расчёта индексов, из-за чего прямая регрессия на неё
занижает валютный коэффициент). На собственной сетке Московской биржи:
  (1) санитарная сверка неявного фактора с внешним курсом;
  (2) скользящая (250 дн.) декомпозиция дисперсии RTSI на фондовую
      (IMOEX) и валютную составляющие с симметричным разнесением ковариации;
  (3) скользящая корреляция фондовой и валютной компонент;
  (4) робастность деглобализации: средняя скользящая корреляция
      с 14 мировыми индексами для долларового RTSI и рублёвого IMOEX.

Выход: output/tables/table25_fx_decomposition.csv,
       output/figures/fig20_rtsi_imoex_fx.png.
Запуск из корня: .venv/bin/python src/13_rtsi_imoex_fx.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TAB = Path("output/tables")
FIG = Path("output/figures")

PERIODS = {
    "GFC": ("2008-09-01", "2009-06-30"),
    "Calm": ("2009-07-01", "2019-12-31"),
    "COVID": ("2020-02-20", "2020-06-30"),
    "Inter": ("2020-07-01", "2022-01-31"),
    "2022 shock": ("2022-02-21", "2022-12-30"),
    "Recent": ("2023-01-01", "2026-06-30"),
}

# ------------------------------------------------- данные на сетке Мосбиржи
mx = pd.read_csv("data/moex_log_rets.csv", parse_dates=["Datetime"]).set_index("Datetime")
pair = mx[["imoex", "rtsi"]].dropna()
pair = pair.loc["2007-09-19":"2026-06-30"]

# неявный валютный фактор: r_FX = r_IMOEX − r_RTSI (тождество корзины)
pair["fx"] = pair["imoex"] - pair["rtsi"]
print(f"Сетка Мосбиржи: {pair.index.min().date()} -> {pair.index.max().date()}, T = {len(pair)}")

# ------------------------------------------------- (1) санитарная сверка
usd = pd.read_csv("data/usdrub.csv", parse_dates=["Datetime"]).set_index("Datetime")["usdrub"]
lvl = np.log(usd).reindex(pair.index.union(usd.index)).ffill().reindex(pair.index)
fx_ext = lvl.diff()
both = pd.concat([pair["fx"], fx_ext.rename("ext")], axis=1).dropna()
c_daily = both["fx"].corr(both["ext"])
c_weekly = both.resample("W").sum().apply(lambda s: s).corr().iloc[0, 1]
cum_int = pair["fx"].sum()
cum_ext = fx_ext.sum()
print(f"[санити] corr(неявный fx, внешний USD/RUB): дневная {c_daily:.3f}, недельная {c_weekly:.3f}")
print(f"[санити] накопленный лог-ход за период: неявный {cum_int:.3f}, внешний {cum_ext:.3f}")

# ------------------------------------------------- (2)-(3) скользящая декомпозиция
W = 250
var_i = pair["imoex"].rolling(W).var()
var_x = pair["fx"].rolling(W).var()
cov_ix = pair["imoex"].rolling(W).cov(pair["fx"])
var_r_implied = var_i + var_x - 2 * cov_ix
share_fx = (var_x - cov_ix) / var_r_implied
share_eq = (var_i - cov_ix) / var_r_implied
corr_ix = pair["imoex"].rolling(W).corr(pair["fx"])

# ------------------------------------------------- (4) корреляция с миром
gl = pd.read_csv("data/log_rets.csv", parse_dates=["Datetime"]).set_index("Datetime")
world = [c for c in gl.columns if c != "rtsi"]
im_on_grid = mx["imoex"].reindex(gl.index).fillna(0.0)  # то же правило заполнения, что у RTSI
roll_rtsi = pd.DataFrame({c: gl["rtsi"].rolling(W).corr(gl[c]) for c in world}).mean(axis=1)
roll_imx = pd.DataFrame({c: im_on_grid.rolling(W).corr(gl[c]) for c in world}).mean(axis=1)

# ------------------------------------------------- подпериодные сводки
rows = []
for name, (a, b) in PERIODS.items():
    seg = pair.loc[a:b]
    vi, vx = seg["imoex"].var(), seg["fx"].var()
    cix = seg["imoex"].cov(seg["fx"])
    vr = vi + vx - 2 * cix
    rows.append({
        "period": name,
        "share_fx_pct": 100 * (vx - cix) / vr,
        "share_eq_pct": 100 * (vi - cix) / vr,
        "corr_imoex_fx": seg["imoex"].corr(seg["fx"]),
        "mean_worldcorr_rtsi": roll_rtsi.loc[a:b].mean(),
        "mean_worldcorr_imoex": roll_imx.loc[a:b].mean(),
    })
out = pd.DataFrame(rows)
out.to_csv(TAB / "table25_fx_decomposition.csv", index=False)
print(out.round(3).to_string(index=False))

# ------------------------------------------------- рисунок
fig, ax = plt.subplots(3, 1, figsize=(12, 8.6), sharex=True)
ax[0].plot(share_fx.index, 100 * share_fx.values, color="#b2182b", lw=1.1,
           label="валютная составляющая")
ax[0].plot(share_eq.index, 100 * share_eq.values, color="navy", lw=1.1,
           label="фондовая составляющая")
ax[0].axhline(50, color="grey", lw=0.6, ls="--")
ax[0].set_ylabel("доля дисперсии RTSI, %")
ax[0].legend(fontsize=9, loc="upper left")
ax[0].set_title("(а) Скользящая декомпозиция дисперсии RTSI на фондовую и валютную составляющие (окно 250 дней)", fontsize=10)

ax[1].plot(corr_ix.index, corr_ix.values, color="#4d4d4d", lw=1.1)
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_ylabel("корреляция")
ax[1].set_title("(б) Скользящая корреляция фондовой (IMOEX) и валютной (USD/RUB) компонент", fontsize=10)

ax[2].plot(roll_rtsi.index, roll_rtsi.values, color="navy", lw=1.1, label="RTSI (долларовый)")
ax[2].plot(roll_imx.index, roll_imx.values, color="#b2182b", lw=1.1, label="IMOEX (рублёвый)")
ax[2].axhline(0, color="grey", lw=0.6)
ax[2].set_ylabel("средняя корреляция")
ax[2].legend(fontsize=9, loc="upper right")
ax[2].set_title("(в) Средняя скользящая корреляция с 14 мировыми индексами: долларовая и рублёвая номинации", fontsize=10)

for a0, b0, c0 in [("2008-09-15", "2009-06-30", "red"),
                   ("2014-03-01", "2015-03-31", "orange"),
                   ("2020-02-20", "2020-06-30", "purple"),
                   ("2022-02-21", "2022-06-30", "orange")]:
    for axx in ax:
        axx.axvspan(pd.Timestamp(a0), pd.Timestamp(b0), alpha=0.10, color=c0)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig20_rtsi_imoex_fx.png", dpi=200)
plt.close()
print("fig20_rtsi_imoex_fx.png done")
