"""Товарные факторы в системе связности (§ 3.8.3): Brent и золото.

Ряды: фьючерс Brent (BZ=F, история с 30.07.2007 покрывает выборку) и золото
(GC=F). Газовые бенчмарки НЕ включаются по критерию глубины истории § 1.3:
ликвидная история TTF начинается в октябре 2017 г. и не покрывает ни ГФК,
ни COVID (Henry Hub длиннее, но нерелевантен для российского канала);
дополнительно экспортные цены газа исторически следовали нефтяной индексации,
то есть нефтяной канал частично содержит газовый.

Конструкция: тот же одномерный этап (выбор GARCH/GJR/EGARCH-skew-t по BIC),
затем связность DY на расширенных системах — глобальной (15 + Brent + золото,
17 узлов) и локальной (7 секторов + Brent + золото, 9 узлов): полная выборка
VAR(2), подпериоды, скользящие вклады Brent в FROM-связность RTSI и
нефтегазового сектора (окно 250, шаг 10, VAR(1), H = 10).

Выход: data/commodity_log_rets.csv (кэш котировок),
       output/tables/table26_commodities.csv,
       output/figures/fig21_commodities.png.
Запуск из корня: .venv/bin/python src/14_commodities.py  (~3-5 мин)
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arch import arch_model

from common import DATA, TAB, FIG, PERIODS_SHORT as PERIODS, fit_var, fevd_generalised, tci_of

warnings.filterwarnings("ignore")

SCALE = 100.0


def net_of(theta, i):
    return 100.0 * (theta[:, i].sum() - theta[i, i] - (theta[i, :].sum() - theta[i, i]))


# ------------------------------------------------- котировки (кэш)
def load_commodities():
    f = DATA / "commodity_log_rets.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["Datetime"]).set_index("Datetime")
    import yfinance as yf
    out = {}
    for col, tick in [("brent", "BZ=F"), ("gold", "GC=F")]:
        px = yf.download(tick, start="2007-01-01", auto_adjust=False, progress=False)["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        px = px.dropna()
        med = px.rolling(11, center=True, min_periods=3).median()
        px = px[(px - med).abs() / med < 0.25]        # чистка битых котировок
        out[col] = np.log(px).diff()
    df = pd.DataFrame(out).dropna(how="all")
    df.index.name = "Datetime"
    df.to_csv(f)
    print(f"commodities saved: {f} ({df.index.min().date()} -> {df.index.max().date()})")
    return df


com = load_commodities()

# ------------------------------------------------- одномерный этап для товаров
gl = pd.read_csv(DATA / "log_rets.csv", parse_dates=["Datetime"]).set_index("Datetime")
com_grid = com.reindex(gl.index).fillna(0.0).loc[gl.index]  # правило нулевого заполнения § 1.3

VOLKW = {"GARCH": dict(vol="GARCH", p=1, q=1, o=0),
         "GJR": dict(vol="GARCH", p=1, q=1, o=1),
         "EGARCH": dict(vol="EGARCH", p=1, q=1, o=1)}
com_vol = {}
for c in ["brent", "gold"]:
    fits = {}
    for lab, kw in VOLKW.items():
        try:
            fits[lab] = arch_model(com_grid[c] * SCALE, mean="Constant", dist="skewt", **kw).fit(
                disp="off", show_warning=False)
        except Exception:
            pass
    best = min(fits, key=lambda k: fits[k].bic)
    print(f"[{c}] лучшая спецификация: {best} (BIC {fits[best].bic:.0f})")
    com_vol[c] = fits[best].conditional_volatility / SCALE
com_vol = pd.DataFrame(com_vol, index=gl.index)

# ------------------------------------------------- системы лог-волатильностей
base_vol = pd.read_csv("output/cond_vol.csv", parse_dates=["Datetime"]).set_index("Datetime")
y17 = np.log(pd.concat([base_vol, com_vol], axis=1)).dropna()
cols17 = list(y17.columns)
i_rtsi, i_brent, i_gold = cols17.index("rtsi"), cols17.index("brent"), cols17.index("gold")

mvol = pd.read_csv("output/cond_vol_moex.csv", parse_dates=["Datetime"]).set_index("Datetime")
y9 = np.log(pd.concat([mvol, com_vol.reindex(mvol.index)], axis=1)).dropna()
cols9 = list(y9.columns)
j_og, j_brent, j_gold = cols9.index("moexog"), cols9.index("brent"), cols9.index("gold")

# ------------------------------------------------- полная выборка и подпериоды
rows = []
def snapshot(theta, label, cols, idx_map):
    r = {"period": label,
         "TCI": tci_of(theta),
         "NET_brent": net_of(theta, idx_map["brent"]),
         "NET_gold": net_of(theta, idx_map["gold"])}
    if "rtsi" in idx_map:
        r["brent_to_rtsi"] = 100 * theta[idx_map["rtsi"], idx_map["brent"]]
        r["gold_to_rtsi"] = 100 * theta[idx_map["rtsi"], idx_map["gold"]]
    if "og" in idx_map:
        r["brent_to_og"] = 100 * theta[idx_map["og"], idx_map["brent"]]
        r["gold_to_og"] = 100 * theta[idx_map["og"], idx_map["gold"]]
    return r

th = fevd_generalised(*fit_var(y17, p=2), H=10)
rows.append({**snapshot(th, "Full", cols17, {"brent": i_brent, "gold": i_gold, "rtsi": i_rtsi}),
             "system": "global17"})
for name, (a, b) in PERIODS.items():
    s = y17.loc[a:b]
    if len(s) < 60:
        continue
    p_var = 2 if len(s) > 150 else 1
    th = fevd_generalised(*fit_var(s, p=p_var), H=10)
    rows.append({**snapshot(th, name, cols17, {"brent": i_brent, "gold": i_gold, "rtsi": i_rtsi}),
                 "system": "global17"})

th = fevd_generalised(*fit_var(y9, p=2), H=10)
rows.append({**snapshot(th, "Full", cols9, {"brent": j_brent, "gold": j_gold, "og": j_og}),
             "system": "local9"})
for name, (a, b) in PERIODS.items():
    s = y9.loc[a:b]
    if len(s) < 60:
        continue
    p_var = 2 if len(s) > 150 else 1
    th = fevd_generalised(*fit_var(s, p=p_var), H=10)
    rows.append({**snapshot(th, name, cols9, {"brent": j_brent, "gold": j_gold, "og": j_og}),
                 "system": "local9"})

out = pd.DataFrame(rows)
out.to_csv(TAB / "table26_commodities.csv", index=False)
print(out.round(2).to_string(index=False))

# ------------------------------------------------- скользящие вклады Brent
def rolling_contrib(y, src_i, dst_i):
    res = {}
    arr, idx = y.values, y.index
    for end in range(250, len(y), 10):
        w = arr[end - 250:end]
        try:
            th = fevd_generalised(*fit_var(w, p=1), H=10)
            res[idx[end - 1]] = 100 * th[dst_i, src_i]
        except Exception:
            pass
    return pd.Series(res)

roll_g = rolling_contrib(y17, i_brent, i_rtsi)
roll_l = rolling_contrib(y9, j_brent, j_og)
print(f"rolling: brent→RTSI точек {len(roll_g)}, brent→нефтегаз точек {len(roll_l)}")

# ------------------------------------------------- рисунок
sub = out[(out.system == "global17") & (out.period != "Full")]
subl = out[(out.system == "local9") & (out.period != "Full")]
labels = [p for p in sub.period]
x = np.arange(len(labels))

fig, ax = plt.subplots(2, 1, figsize=(12, 7.2),
                       gridspec_kw={"height_ratios": [1, 1.2]})
w = 0.38
ax[0].bar(x - w / 2, sub["brent_to_rtsi"].values, w, color="navy",
          label="Brent → RTSI (глобальная система)")
og_vals = [subl.loc[subl.period == p, "brent_to_og"].values[0] if (subl.period == p).any() else np.nan
           for p in labels]
ax[0].bar(x + w / 2, og_vals, w, color="#b2182b",
          label="Brent → нефтегазовый сектор (локальная система)")
ax[0].set_xticks(x, ["до-ГФК", "ГФК", "Спокойный", "COVID", "Промежут.", "Шок 2022", "Недавний"], fontsize=8.5)
ax[0].set_ylabel("доля дисперсии прогноза, %")
ax[0].legend(fontsize=9)
ax[0].set_title("(а) Вклад шоков Brent в дисперсию ошибки прогноза волатильности по подпериодам (GFEVD H = 10)", fontsize=10)

ax[1].plot(roll_g.index, roll_g.values, color="navy", lw=1.1, label="Brent → RTSI")
ax[1].plot(roll_l.index, roll_l.values, color="#b2182b", lw=1.1, label="Brent → нефтегазовый сектор")
for a0, b0, c0 in [("2008-09-15", "2009-06-30", "red"),
                   ("2020-02-20", "2020-06-30", "purple"),
                   ("2022-02-21", "2022-06-30", "orange")]:
    ax[1].axvspan(pd.Timestamp(a0), pd.Timestamp(b0), alpha=0.10, color=c0)
ax[1].set_ylabel("доля дисперсии прогноза, %")
ax[1].legend(fontsize=9, loc="upper right")
ax[1].set_title("(б) Скользящий вклад шоков Brent (окно 250 дней, шаг 10)", fontsize=10)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig21_commodities.png", dpi=200)
plt.close()
print("fig21_commodities.png done")
