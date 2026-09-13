"""Рисунки для §§ 1.1–1.2 Главы 1 (новая структура от 27.07.2026).

fig16_market_structure.png — структура рынка через индексы:
    (а) веса секторов в IMOEX (агрегация официальных весов бумаг MOEX ISS);
    (б) концентрация корзин отраслевых индексов (вес топ-3 бумаг);
    (в) корреляция дневных доходностей секторов с композитным индексом.
fig17_2014_break.png — режимный разрыв 2014 г. (локальный, санкционно-валютный):
    (а) USD/RUB; (б) годовая волатильность RTSI против медианы 14 мировых;
    (в) скользящая средняя корреляция RTSI с 14 мировыми (безмодельная, 250 дн.).
fig18_two_level_scheme.png — схема двухуровневой постановки исследования.

Сетевые источники кэшируются в data/ (снапшот состава индексов с датой,
ряд USD/RUB); повторный запуск сети не требует.
Запуск из корня: .venv/bin/python src/10_ch1_figures.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NAVY = "#2166ac"
RED = "#b2182b"
GREY = "#4d4d4d"

SECTOR_IDX = {  # живые отраслевые индексы, состав берём из ISS
    "MOEXOG": "Нефть и газ", "MOEXFN": "Финансы", "MOEXMM": "Металлы и добыча",
    "MOEXCN": "Потребительский", "MOEXEU": "Электроэнергетика",
    "MOEXCH": "Химия и нефтехимия", "MOEXTN": "Транспорт",
    "MOEXIT": "Информационные технологии", "MOEXRE": "Строительство",
}
# телекоммуникационный индекс прекращён 20.03.2026 — бумаги сектора замапим вручную
TELECOM_TICKERS = {"MTSS", "RTKM", "RTKMP", "MGTSP"}

INNER7 = ["MOEXOG", "MOEXFN", "MOEXMM", "MOEXCN", "MOEXEU", "MOEXCH", "MOEXTN"]


# ---------------------------------------------------------------- ISS snapshot
def fetch_index_composition(secid: str) -> pd.DataFrame:
    url = (f"https://iss.moex.com/iss/statistics/engines/stock/markets/index/"
           f"analytics/{secid}.json?limit=100&iss.meta=off")
    js = requests.get(url, timeout=30).json()["analytics"]
    df = pd.DataFrame(js["data"], columns=js["columns"])
    return df[["indexid", "tradedate", "ticker", "shortnames", "weight"]]


def load_composition_snapshot() -> pd.DataFrame:
    snap = DATA / "index_composition_snapshot.csv"
    if snap.exists():
        return pd.read_csv(snap)
    frames = [fetch_index_composition("IMOEX")]
    for s in SECTOR_IDX:
        frames.append(fetch_index_composition(s))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(snap, index=False)
    print(f"ISS snapshot saved: {snap} (tradedate={df['tradedate'].iloc[0]})")
    return df


# ---------------------------------------------------------------- USD/RUB
def load_usdrub() -> pd.Series:
    f = DATA / "usdrub.csv"
    if f.exists():
        s = pd.read_csv(f, parse_dates=["Datetime"]).set_index("Datetime")["usdrub"]
        return s
    import yfinance as yf
    px = yf.download("RUB=X", start="2007-01-01", auto_adjust=False, progress=False)
    close = px["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.dropna()
    med = s.rolling(11, center=True, min_periods=3).median()
    s = s[(s - med).abs() / med < 0.20]  # чистка битых котировок Yahoo
    s.index.name = "Datetime"
    s.name = "usdrub"
    s.to_csv(f, header=True)
    print(f"USD/RUB saved: {f} ({s.index.min().date()} -> {s.index.max().date()})")
    return s


# ================================================================ FIG 16
def fig_market_structure():
    comp = load_composition_snapshot()
    snap_date = comp["tradedate"].iloc[0]
    imoex = comp[comp["indexid"] == "IMOEX"].copy()

    ticker2sector = {}
    for s, name in SECTOR_IDX.items():
        for t in comp.loc[comp["indexid"] == s, "ticker"]:
            ticker2sector[t] = name
    for t in TELECOM_TICKERS:
        ticker2sector.setdefault(t, "Телекоммуникации")

    imoex["sector"] = imoex["ticker"].map(ticker2sector).fillna("Прочие")
    w = imoex.groupby("sector")["weight"].sum().sort_values()

    top10 = imoex["weight"].nlargest(10).sum()
    print(f"[fig16] IMOEX {snap_date}: {len(imoex)} бумаг; top-10 weight = {top10:.1f}%")
    print(w.sort_values(ascending=False).round(1))

    # (б) концентрация корзин: вес топ-3 бумаг в каждом из 7 отраслевых
    conc = {}
    for s in INNER7:
        ws = comp.loc[comp["indexid"] == s, "weight"].astype(float)
        conc[SECTOR_IDX[s]] = ws.nlargest(3).sum()
    conc = pd.Series(conc).sort_values()
    print("[fig16] top-3 концентрация корзин:\n", conc.round(1))

    # (в) корреляция дневных доходностей сектора с IMOEX
    rets = pd.read_csv(DATA / "moex_log_rets.csv", parse_dates=["Datetime"]).set_index("Datetime")
    cor = {}
    for s in INNER7:
        c = s.lower()
        sub = rets[[c, "imoex"]].dropna()
        cor[SECTOR_IDX[s]] = sub[c].corr(sub["imoex"])
    cor = pd.Series(cor).sort_values()
    print("[fig16] корреляции с IMOEX:\n", cor.round(3))

    # регрессия IMOEX на 7 секторов — R^2 (число для текста)
    sub = rets[[s.lower() for s in INNER7] + ["imoex"]].dropna()
    X = sub[[s.lower() for s in INNER7]].values
    y = sub["imoex"].values
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    r2 = 1 - ((y - Xc @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"[fig16] R^2 регрессии IMOEX на 7 секторов: {r2:.4f} (n={len(sub)})")

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    ax[0].barh(range(len(w)), w.values, color=NAVY, height=0.62)
    ax[0].set_yticks(range(len(w)), w.index, fontsize=9)
    for i, v in enumerate(w.values):
        ax[0].text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=8, color=GREY)
    ax[0].set_xlim(0, w.max() * 1.16)
    ax[0].set_title(f"(а) Вес секторов в индексе МосБиржи, %\n(по данным MOEX ISS, {snap_date})", fontsize=10)
    ax[0].set_xlabel("вес, %", fontsize=9)

    ax[1].barh(range(len(conc)), conc.values, color=NAVY, height=0.62)
    ax[1].set_yticks(range(len(conc)), conc.index, fontsize=9)
    for i, v in enumerate(conc.values):
        ax[1].text(v + 1.2, i, f"{v:.0f}", va="center", fontsize=8, color=GREY)
    ax[1].set_xlim(0, 108)
    ax[1].set_title("(б) Суммарный вес трёх крупнейших бумаг\nв отраслевом индексе, %", fontsize=10)
    ax[1].set_xlabel("вес топ-3, %", fontsize=9)

    ax[2].barh(range(len(cor)), cor.values, color=NAVY, height=0.62)
    ax[2].set_yticks(range(len(cor)), cor.index, fontsize=9)
    for i, v in enumerate(cor.values):
        ax[2].text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=8, color=GREY)
    ax[2].set_xlim(0, 1.02)
    ax[2].set_title("(в) Корреляция дневных доходностей\nсектора с индексом МосБиржи, 2008–2026", fontsize=10)
    ax[2].set_xlabel("коэффициент корреляции", fontsize=9)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig16_market_structure.png", dpi=200)
    plt.close()
    print("fig16_market_structure.png done")


# ================================================================ FIG 17
def fig_2014_break():
    rets = pd.read_csv(DATA / "log_rets.csv", parse_dates=["Datetime"]).set_index("Datetime")
    world = [c for c in rets.columns if c != "rtsi"]
    usdrub = load_usdrub()

    win = slice("2012-01-01", "2016-12-31")

    # (б) годовая волатильность, окно 63 дня
    vol = rets.rolling(63).std() * np.sqrt(252) * 100
    rtsi_vol = vol["rtsi"]
    world_vol_med = vol[world].median(axis=1)

    # (в) скользящая средняя попарная корреляция RTSI с 14 рынками, 250 дней
    roll = {c: rets["rtsi"].rolling(250).corr(rets[c]) for c in world}
    mean_corr = pd.DataFrame(roll).mean(axis=1)

    pre = mean_corr.loc["2012-01-01":"2013-12-31"].mean()
    post = mean_corr.loc["2015-01-01":"2016-12-31"].mean()
    peak_rtsi = rtsi_vol.loc["2014-10-01":"2015-03-31"].max()
    peak_world = world_vol_med.loc["2014-10-01":"2015-03-31"].max()
    rub_2014 = (usdrub.loc["2014-01-01":"2015-01-31"].max()
                / usdrub.loc["2014-01-01":"2014-06-30"].mean() - 1) * 100
    print(f"[fig17] ср. корреляция 2012–2013: {pre:.3f}; 2015–2016: {post:.3f}")
    print(f"[fig17] пик годовой волатильности X-2014..III-2015: RTSI {peak_rtsi:.0f}%, медиана мира {peak_world:.0f}%")
    print(f"[fig17] ослабление рубля в 2014 к среднему уровню 1П2014: {rub_2014:.0f}%")

    events = [("2014-03-17", "март 2014:\nпервые санкции", 0.97, "right"),
              ("2014-07-31", "июль 2014:\nсекторальные санкции", 0.78, "left"),
              ("2014-12-16", "декабрь 2014:\nвалютный кризис", 0.97, "left")]

    fig, ax = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.15, 1]})
    u = usdrub.loc[win]
    ax[0].plot(u.index, u.values, color="black", lw=1.1)
    ax[0].set_ylabel("руб. за долл. США", fontsize=9)
    ax[0].set_title("(а) Курс USD/RUB", fontsize=10)

    rv, wv = rtsi_vol.loc[win], world_vol_med.loc[win]
    ax[1].plot(rv.index, rv.values, color=RED, lw=1.2, label="RTSI")
    ax[1].plot(wv.index, wv.values, color=NAVY, lw=1.1, label="медиана 14 мировых индексов")
    ax[1].set_ylabel("волатильность, % годовых", fontsize=9)
    ax[1].set_title("(б) Годовая волатильность (скользящее окно 63 дня)", fontsize=10)
    ax[1].legend(fontsize=9, loc="upper left")

    mc = mean_corr.loc[win]
    ax[2].plot(mc.index, mc.values, color=GREY, lw=1.2)
    ax[2].hlines(pre, pd.Timestamp("2012-01-01"), pd.Timestamp("2014-01-01"),
                 colors=NAVY, ls="--", lw=1.1)
    ax[2].hlines(post, pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31"),
                 colors=NAVY, ls="--", lw=1.1)
    ax[2].text(pd.Timestamp("2012-02-01"), pre + 0.012, f"среднее 2012–2013: {pre:.2f}",
               fontsize=8, color=NAVY)
    ax[2].text(pd.Timestamp("2015-02-01"), post + 0.012, f"среднее 2015–2016: {post:.2f}",
               fontsize=8, color=NAVY)
    ax[2].set_ylabel("средняя корреляция", fontsize=9)
    ax[2].set_title("(в) Скользящая средняя корреляция дневных доходностей RTSI\nс 14 мировыми индексами (окно 250 дней)", fontsize=10)

    for d, lab, ytxt, ha in events:
        for a in ax:
            a.axvline(pd.Timestamp(d), color="orange", lw=0.9, alpha=0.75)
        pad = pd.Timedelta(days=6) * (-1 if ha == "right" else 1)
        ax[0].text(pd.Timestamp(d) + pad, ytxt, lab, fontsize=7.5, color="#8a5a00",
                   va="top", ha=ha, transform=ax[0].get_xaxis_transform())
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig17_2014_break.png", dpi=200)
    plt.close()
    print("fig17_2014_break.png done")


# ================================================================ FIG 18
def _rbox(ax, x, y, w, h, fc, ec, lw=1.6, r=0.025):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, lw=lw, zorder=2)
    ax.add_patch(b)
    return b


def fig_two_level_scheme():
    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -------- методический стек (слева)
    _rbox(ax, 0.015, 0.20, 0.24, 0.60, "#f5f5f5", GREY, lw=1.3)
    ax.text(0.135, 0.755, "ЕДИНЫЙ\nМЕТОДИЧЕСКИЙ СТЕК", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=GREY)
    steps = [
        (0.665, "1. Одномерные модели\nусловной волатильности\n(GARCH / GJR / EGARCH, skew-t)"),
        (0.525, "2. Динамические корреляции\n(DCC / ADCC / cDCC +\nавторская кластерная GDCC)"),
        (0.385, "3. Направленная связность\n(Diebold–Yilmaz) и её частотная\nдекомпозиция (Baruník–Křehlík)"),
        (0.26, "4. Валидация: информационные\nкритерии, LR-тесты, бэктесты"),
    ]
    for yy, txt in steps:
        ax.text(0.135, yy, txt, ha="center", va="center", fontsize=8.4, color="black")
    for y0, y1 in [(0.62, 0.575), (0.478, 0.435), (0.345, 0.30)]:
        ax.add_patch(FancyArrowPatch((0.135, y0), (0.135, y1), arrowstyle="-|>",
                                     mutation_scale=11, color=GREY, lw=1.1))

    # -------- глобальный уровень (верх)
    _rbox(ax, 0.30, 0.56, 0.46, 0.385, "#eef3fa", NAVY)
    ax.text(0.53, 0.905, "ГЛОБАЛЬНЫЙ УРОВЕНЬ", ha="center", fontsize=12,
            fontweight="bold", color=NAVY)
    ax.text(0.53, 0.862, "мировая система: 15 фондовых индексов (RTSI + 14 зарубежных)\n19.09.2007 – 30.06.2026,  T = 4893 торговых дня",
            ha="center", va="top", fontsize=8.6, color="black")
    cx, cy, rr = 0.53, 0.695, 0.115
    labels = ["SPX", "TSX", "BOVESPA", "FTSE", "DAX", "CAC", "BEL", "SMI",
              "BIST", "HSI", "SSEC", "NIFTY", "Nikkei", "ASX"]
    for k, lab in enumerate(labels):
        ang = 2 * np.pi * k / len(labels)
        x, y = cx + 1.55 * rr * np.cos(ang), cy + 0.60 * rr * np.sin(ang)
        ax.plot([cx, x], [cy, y], color=NAVY, lw=0.5, alpha=0.35, zorder=3)
        ax.add_patch(Circle((x, y), 0.0105, color=NAVY, alpha=0.85, zorder=4))
        ax.text(x, y - 0.021, lab, ha="center", fontsize=5.6, color=NAVY)
    ax.add_patch(Circle((cx, cy), 0.017, color=RED, zorder=5))
    ax.text(cx, cy + 0.026, "RTSI", ha="center", fontsize=7.5, color=RED, fontweight="bold")
    ax.text(0.53, 0.573, "дополнительно: факторная стохастическая модель (FMSV), приложение VaR/ES и капитал",
            ha="center", fontsize=7.6, color=GREY, style="italic")

    # -------- локальный уровень (низ)
    _rbox(ax, 0.30, 0.055, 0.46, 0.385, "#fdf1ef", RED)
    ax.text(0.53, 0.40, "ЛОКАЛЬНЫЙ УРОВЕНЬ", ha="center", fontsize=12,
            fontweight="bold", color=RED)
    ax.text(0.53, 0.357, "отраслевая система российского рынка: 7 индексов МосБиржи\n09.01.2008 – 30.06.2026,  T = 4616 торговых дней",
            ha="center", va="top", fontsize=8.6, color="black")
    sec = ["Нефть и газ", "Финансы", "Металлы", "Потребит.", "Энергетика", "Химия", "Транспорт"]
    cy2 = 0.175
    for k, lab in enumerate(sec):
        x = 0.345 + 0.37 * k / (len(sec) - 1)
        y = cy2 + (0.035 if k % 2 == 0 else -0.035)
        ax.add_patch(Circle((x, y), 0.0125, color=RED, alpha=0.85, zorder=4))
        ax.text(x, y - 0.030, lab, ha="center", fontsize=6.4, color=RED)
        for k2 in range(k + 1, len(sec)):
            x2 = 0.345 + 0.37 * k2 / (len(sec) - 1)
            y2 = cy2 + (0.035 if k2 % 2 == 0 else -0.035)
            ax.plot([x, x2], [y, y2], color=RED, lw=0.35, alpha=0.16, zorder=3)

    # -------- стрелки стек -> уровни
    for target_y in (0.75, 0.25):
        ax.add_patch(FancyArrowPatch((0.257, 0.50), (0.295, target_y),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=GREY, lw=1.5,
                                     connectionstyle="arc3,rad=" + ("-0.25" if target_y > 0.5 else "0.25")))
    ax.text(0.272, 0.50, "идентичный протокол\nоценивания", ha="center", va="center",
            fontsize=7.2, color=GREY, rotation=90)

    # -------- мост (справа)
    _rbox(ax, 0.795, 0.335, 0.19, 0.33, "#f2f8f2", "darkgreen", lw=1.4)
    ax.text(0.89, 0.615, "МОСТ ДВУХ УРОВНЕЙ", ha="center", fontsize=9.5,
            fontweight="bold", color="darkgreen")
    ax.text(0.89, 0.49, "сопоставление траекторий\nсовокупной связности (TCI)\nна общей временной сетке;\n\nспред «локальный − глобальный»:\nсмена знака = переориентация\nсистемного риска вовнутрь",
            ha="center", va="center", fontsize=7.6, color="black")
    ax.add_patch(FancyArrowPatch((0.765, 0.75), (0.875, 0.67), arrowstyle="-|>",
                                 mutation_scale=13, color="darkgreen", lw=1.4,
                                 connectionstyle="arc3,rad=-0.2"))
    ax.add_patch(FancyArrowPatch((0.765, 0.25), (0.875, 0.33), arrowstyle="-|>",
                                 mutation_scale=13, color="darkgreen", lw=1.4,
                                 connectionstyle="arc3,rad=0.2"))
    ax.text(0.5, 0.012, "Источник: составлено автором.", fontsize=7.5, color=GREY, ha="center")

    plt.tight_layout()
    plt.savefig(FIG / "fig18_two_level_scheme.png", dpi=220)
    plt.close()
    print("fig18_two_level_scheme.png done")


if __name__ == "__main__":
    fig_market_structure()
    fig_2014_break()
    fig_two_level_scheme()
