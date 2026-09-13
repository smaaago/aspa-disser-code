"""Формальные тесты режимного сдвига спреда двух уровней (§ 3.8.4).

Вход — дневная сетка моста (table22b_tci_bridge_daily.csv): траектории
локального и глобального TCI и их спред. Три уровня строгости:

(1) Эндогенная датировка сдвигов среднего в духе Бая–Перрона: сегментация
    наименьших квадратов динамическим программированием, число разрывов
    (0..3) по BIC, минимальная длина сегмента 250 наблюдений (=длине окна
    скользящей оценки).
(2) Инференция о новом режиме с учётом сериальной зависимости скользящих
    оценок: HAC-ошибки Ньюи–Уэста (ширина = длине окна) для сдвиговой
    регрессии и стационарный бутстрап Политиса–Романо (средняя длина блока
    250, 5000 репликаций) для гипотезы «средний спред после сдвига > 0»;
    контроль на прореженных (шаг 250, неперекрывающихся) окнах.
(3) Тест Грегори–Хансена коинтеграции со структурным сдвигом: модели C
    (сдвиг уровня) и C/S (смена режима), ADF-статистика на остатках,
    минимум по датам сдвига в [0.15T, 0.85T]; критические значения
    Gregory, Hansen (1996), m = 1.

Выход: output/tables/table27_spread_tests.csv,
       output/figures/fig22_spread_break.png.
Запуск из корня: .venv/bin/python src/15_spread_break_tests.py  (~1-2 мин)
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260727)

TAB = Path("output/tables")
FIG = Path("output/figures")
MIN_SEG = 250
MAX_BREAKS = 3

br = pd.read_csv(TAB / "table22b_tci_bridge_daily.csv", parse_dates=["Datetime"]).set_index("Datetime")
spread = br["spread_int_ext"].dropna()
loc_tci, glob_tci = br["TCI_int"].dropna(), br["TCI_ext"].dropna()
T = len(spread)
print(f"Спред: {spread.index.min().date()} -> {spread.index.max().date()}, T = {T}")

# ---------------------------------------------------------- (1) Бай–Перрон (DP)
x = spread.values
cs, cs2 = np.concatenate([[0.0], np.cumsum(x)]), np.concatenate([[0.0], np.cumsum(x ** 2)])

def sse(i, j):  # сегмент [i, j)
    n = j - i
    s = cs[j] - cs[i]
    return (cs2[j] - cs2[i]) - s * s / n

def segment(k):
    """Оптимальные k разрывов: DP по SSE, min длина сегмента MIN_SEG."""
    if k == 0:
        return [], sse(0, T)
    D = np.full((k + 1, T + 1), np.inf)
    arg = np.zeros((k + 1, T + 1), dtype=int)
    for j in range(MIN_SEG, T + 1):
        D[0, j] = sse(0, j)
    for m in range(1, k + 1):
        lo = MIN_SEG * (m + 1)
        for j in range(lo, T + 1):
            best, bi = np.inf, -1
            for i in range(MIN_SEG * m, j - MIN_SEG + 1):
                v = D[m - 1, i] + sse(i, j)
                if v < best:
                    best, bi = v, i
            D[m, j], arg[m, j] = best, bi
    breaks, j = [], T
    for m in range(k, 0, -1):
        j = arg[m, j]
        breaks.append(j)
    return sorted(breaks), D[k, T]

results = {}
for k in range(MAX_BREAKS + 1):
    bks, s = segment(k)
    n_par = 2 * k + 1                     # k дат + (k+1) средних − ... (даты как параметры)
    bic = T * np.log(s / T) + n_par * np.log(T)
    results[k] = (bks, s, bic)
    print(f"k={k}: BIC={bic:.1f}, разрывы: {[spread.index[b].date() for b in bks]}")
k_star = min(results, key=lambda k: results[k][2])
breaks = results[k_star][0]
break_dates = [spread.index[b] for b in breaks]
print(f"Выбрано по BIC: k* = {k_star}, даты: {[d.date() for d in break_dates]}")

seg_bounds = [0] + breaks + [T]
seg_means = [(spread.index[a], spread.index[b - 1], x[a:b].mean())
             for a, b in zip(seg_bounds[:-1], seg_bounds[1:])]
for a, b, m in seg_means:
    print(f"  сегмент {a.date()} — {b.date()}: среднее {m:+.1f} п.п.")

# ---------------------------------------------------------- (2) HAC + бутстрап
last_break = breaks[-1]
D2022 = (np.arange(T) >= last_break).astype(float)
X = sm.add_constant(D2022)
ols = sm.OLS(x, X).fit(cov_type="HAC", cov_kwds={"maxlags": 250})
beta, t_beta, p_beta = ols.params[1], ols.tvalues[1], ols.pvalues[1]
print(f"[HAC] сдвиг среднего после последнего разрыва: {beta:+.1f} п.п., t = {t_beta:.2f}, p = {p_beta:.2e}")

post = x[last_break:]
mean_post = post.mean()
se_post = sm.OLS(post, np.ones(len(post))).fit(cov_type="HAC", cov_kwds={"maxlags": 250})
t_post, p_post = se_post.tvalues[0], se_post.pvalues[0] / 2  # односторонняя
print(f"[HAC] средний спред после разрыва: {mean_post:+.1f} п.п., t = {t_post:.2f}, p(one-sided) = {p_post:.2e}")

def stationary_bootstrap(v, n_rep=5000, p=1 / 250):
    n = len(v)
    means = np.empty(n_rep)
    for r in range(n_rep):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        restart = rng.random(n) < p
        for t in range(1, n):
            idx[t] = rng.integers(n) if restart[t] else (idx[t - 1] + 1) % n
        means[r] = v[idx].mean()
    return means

boot = stationary_bootstrap(post - mean_post)
p_boot = float((boot >= mean_post).mean())        # H0: среднее <= 0
print(f"[бутстрап] p(среднее пост-режима <= 0) = {p_boot:.4f} (5000 репликаций)")

thin = spread.iloc[::250]
thin_post = thin[thin.index >= spread.index[last_break]]
print(f"[прореженные окна] точек после разрыва: {len(thin_post)}, все положительны: {(thin_post > 0).all()}, "
      f"значения: {[round(v, 1) for v in thin_post.values]}")

# ---------------------------------------------------------- (3) Грегори–Хансен
yv = loc_tci.reindex(spread.index).values
xv = glob_tci.reindex(spread.index).values
CV = {"C": {"1%": -5.13, "5%": -4.61, "10%": -4.34},
      "C/S": {"1%": -5.47, "5%": -4.95, "10%": -4.68}}

def gh_test(model):
    best = (np.inf, None)
    for b in range(int(0.15 * T), int(0.85 * T)):
        D = (np.arange(T) >= b).astype(float)
        if model == "C":
            Z = np.column_stack([np.ones(T), D, xv])
        else:
            Z = np.column_stack([np.ones(T), D, xv, D * xv])
        beta, *_ = np.linalg.lstsq(Z, yv, rcond=None)
        e = yv - Z @ beta
        try:
            adf = adfuller(e, regression="n", autolag="BIC")[0]
        except Exception:
            continue
        if adf < best[0]:
            best = (adf, b)
    return best

gh_rows = []
for model in ("C", "C/S"):
    stat, b = gh_test(model)
    date = spread.index[b]
    verdict = ("отвержение на 1%" if stat < CV[model]["1%"]
               else "отвержение на 5%" if stat < CV[model]["5%"]
               else "отвержение на 10%" if stat < CV[model]["10%"]
               else "не отвергается")
    print(f"[GH {model}] ADF* = {stat:.2f} при сдвиге {date.date()} (CV 5%: {CV[model]['5%']}) -> {verdict}")
    gh_rows.append({"model": model, "ADF_star": stat, "break_date": str(date.date()),
                    "cv_1pct": CV[model]["1%"], "cv_5pct": CV[model]["5%"], "verdict": verdict})

# ---------------------------------------------------------- сводка + рисунок
rows = [{"test": "BP_k_star", "value": k_star,
         "detail": "; ".join(str(d.date()) for d in break_dates)}]
for i, (a, b, m) in enumerate(seg_means, 1):
    rows.append({"test": f"BP_segment_{i}", "value": round(m, 2),
                 "detail": f"{a.date()}..{b.date()}"})
rows += [
    {"test": "HAC_shift_pp", "value": round(beta, 2), "detail": f"t={t_beta:.2f}, p={p_beta:.2e}"},
    {"test": "HAC_post_mean_pp", "value": round(mean_post, 2), "detail": f"t={t_post:.2f}, one-sided p={p_post:.2e}"},
    {"test": "bootstrap_p_post_leq_0", "value": round(p_boot, 4), "detail": "stationary bootstrap, block 250, 5000 rep"},
    {"test": "thinned_windows_positive", "value": int((thin_post > 0).all()),
     "detail": f"{len(thin_post)} неперекрывающихся окон"},
]
for r in gh_rows:
    rows.append({"test": f"GH_{r['model']}", "value": round(r["ADF_star"], 2),
                 "detail": f"break {r['break_date']}; CV5% {r['cv_5pct']}; {r['verdict']}"})
pd.DataFrame(rows).to_csv(TAB / "table27_spread_tests.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 4.6))
ax.plot(spread.index, x, color="#4d4d4d", lw=0.9, label="спред TCI (локальный − глобальный)")
for a, b, m in seg_means:
    ax.hlines(m, a, b, colors="#b2182b", lw=2.0)
for d in break_dates:
    ax.axvline(d, color="orange", lw=1.2)
    ax.text(d, ax.get_ylim()[1] * 0.92, f" {d.date()}", fontsize=8, color="#8a5a00")
ax.axhline(0, color="grey", lw=0.6, ls="--")
ax.set_ylabel("спред, п.п.")
ax.legend(fontsize=9, loc="lower right")
ax.set_title("Спред связности двух уровней: эндогенно датированные разрывы и режимные средние", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "fig22_spread_break.png", dpi=200)
plt.close()
print("fig22_spread_break.png done")

# ---------------------------------------------------- (4) Поправка размерности
# Сырые уровни TCI систем из 7 и 15 узлов живут на разных шкалах: максимум
# среднего внедиагонального вклада равен (N - 1)/N. Нормированный спред
# использует поправку cTCI = TCI * N/(N - 1) (Chatziantoniou, Gabauer 2021);
# она домножает локальный уровень на 7/6, глобальный на 15/14, то есть
# положительный сырой спред от нормировки может только вырасти, а вот
# отрицательные значения и датировка смены знака требуют прямой проверки.

def bp_breaks(v, min_seg=250, max_breaks=3):
    """Бай-Перрон (та же DP-реализация, что в блоке (1), в функциональной
    обёртке для повторного применения к нормированному спреду)."""
    Tv = len(v)
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v ** 2)])

    def _sse(i, j):
        n = j - i
        s = c1[j] - c1[i]
        return (c2[j] - c2[i]) - s * s / n

    def _seg(k):
        if k == 0:
            return [], _sse(0, Tv)
        D = np.full((k + 1, Tv + 1), np.inf)
        A = np.zeros((k + 1, Tv + 1), dtype=int)
        for j in range(min_seg, Tv + 1):
            D[0, j] = _sse(0, j)
        for m in range(1, k + 1):
            for j in range(min_seg * (m + 1), Tv + 1):
                best, bi = np.inf, -1
                for i in range(min_seg * m, j - min_seg + 1):
                    val = D[m - 1, i] + _sse(i, j)
                    if val < best:
                        best, bi = val, i
                D[m, j], A[m, j] = best, bi
        bks, j = [], Tv
        for m in range(k, 0, -1):
            j = A[m, j]
            bks.append(j)
        return sorted(bks), D[k, Tv]

    res = {}
    for k in range(max_breaks + 1):
        bks, s = _seg(k)
        res[k] = (bks, Tv * np.log(s / Tv) + (2 * k + 1) * np.log(Tv))
    ks = min(res, key=lambda kk: res[kk][1])
    return ks, res[ks][0]

spread_c = (loc_tci * (7 / 6) - glob_tci * (15 / 14)).reindex(spread.index).dropna()
xc = spread_c.values
Tc = len(xc)
kc, breaks_c = bp_breaks(xc)
bdates_c = [spread_c.index[b] for b in breaks_c]
segb = [0] + breaks_c + [Tc]
seg_means_c = [(spread_c.index[a], spread_c.index[b - 1], xc[a:b].mean())
               for a, b in zip(segb[:-1], segb[1:])]
print(f"\n[поправка размерности] k* = {kc}, даты: {[d.date() for d in bdates_c]}")
for a, b, m in seg_means_c:
    print(f"  сегмент {a.date()} — {b.date()}: среднее {m:+.1f} п.п.")

lb_c = breaks_c[-1] if breaks_c else 0
post_c = xc[lb_c:]
hac_c = sm.OLS(post_c, np.ones(len(post_c))).fit(cov_type="HAC", cov_kwds={"maxlags": 250})
p_post_c = hac_c.pvalues[0] / 2
sub_means = {}
for pname, (a, b) in [("Calm", ("2009-07-01", "2019-12-31")),
                      ("2022 shock", ("2022-02-21", "2022-12-30")),
                      ("Recent", ("2023-01-01", "2026-06-30"))]:
    sub_means[pname] = float(spread_c.loc[a:b].mean())
print(f"[поправка размерности] подпериодные средние: Calm {sub_means['Calm']:+.1f}, "
      f"шок-2022 {sub_means['2022 shock']:+.1f}, Recent {sub_means['Recent']:+.1f} п.п.; "
      f"пост-режим {post_c.mean():+.1f} п.п., one-sided p = {p_post_c:.2e}")

pd.DataFrame([
    {"test": "cTCI_BP_k_star", "value": kc,
     "detail": "; ".join(str(d.date()) for d in bdates_c)},
    *[{"test": f"cTCI_segment_{i}", "value": round(m, 2),
       "detail": f"{a.date()}..{b.date()}"}
      for i, (a, b, m) in enumerate(seg_means_c, 1)],
    {"test": "cTCI_post_mean_pp", "value": round(float(post_c.mean()), 2),
     "detail": f"one-sided p = {p_post_c:.2e}"},
    *[{"test": f"cTCI_mean_{k}", "value": round(v, 2), "detail": "subperiod mean"}
      for k, v in sub_means.items()],
]).to_csv(TAB / "table27b_spread_dimension_check.csv", index=False)
print("table27b_spread_dimension_check.csv done")
