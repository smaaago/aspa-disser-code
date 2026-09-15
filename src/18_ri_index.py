"""Единая внешне-внутренняя система и индекс внутренней ориентации риска RI
(§ 3.6.5).

Мотив: спред «локальный − глобальный TCI» сопоставляет две РАЗДЕЛЬНО
нормированные системы разной размерности. Здесь оба уровня помещаются в одну
векторную авторегрессию с общим знаменателем декомпозиции: 7 отраслевых
индексов Мосбиржи + 3 региональных фактора мировой волатильности (Америка,
Европа, Азия — равновзвешенные средние лог-волатильностей рынков региона;
RTSI в систему не входит, поскольку является агрегатом тех же семи секторов).

Для каждого сектора i дисперсия ошибки прогноза раскладывается на
    E_i — вклад внешних (региональных) узлов,
    I_i — вклад остальных секторов,
    own_i — собственный вклад;
индекс внутренней ориентации RI = sum(I) / (sum(I) + sum(E)) лежит в [0, 1]
и имеет общий знаменатель для внешней и внутренней частей — это устраняет
несоизмеримость двух раздельно нормированных TCI. Уровень сырого RI при этом
зависит от состава и числа узлов: при равной силе всех связей 6 внутренних и
3 внешних источника механически дают RI = 2/3, поэтому рядом считается
нормированный на число источников RI_avg = (I/6) / (I/6 + E/G) с нейтральной
точкой 0,5, а чувствительность к представлению внешнего блока проверяется
гранулярностью G = 1 / 3 / 14 узлов (одна равновзвешенная мировая
лог-волатильность / три региона — базовый вариант / все 14 внешних рынков).

Оценки: полная выборка и подпериоды (VAR(2), GFEVD H = 10), скользящее окно
250/10 с VAR(1) — конфигурация протокола § 2.3.3; фильтрационная проверка —
TVP-VAR(1) с калмановским фильтром и забывающими факторами (механика § 3.8.5,
kappa1 = 0,99, kappa2 = 0,96 и 0,99). Режимный сдвиг RI датируется и
тестируется тем же аппаратом, что и спред в § 3.8.4 (Бай-Перрон, HAC,
стационарный бутстрап).

Выходы: output/tables/table31_ri_full.csv, table31b_ri_periods.csv,
        table31c_ri_daily.csv, table31d_ri_breaks.csv,
        table31e_ri_granularity.csv (Прил. Б.8),
        output/figures/fig24_ri_index.png.
Запуск из корня: .venv/bin/python src/18_ri_index.py  (~3-5 мин)
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (OUT, TAB, FIG, PERIODS_LOCAL, load_log_vol, fit_var,
                    fevd_generalised, shade)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260805)

H = 10
SECTORS = ["moexog", "moexfn", "moexmm", "moexcn", "moexeu", "moexch", "moextn"]
REGIONS = {"AMER": ["spx", "tsx", "bovespa"],
           "EUR": ["ftse", "dax", "cac", "smi", "bel", "bist"],
           "ASIA": ["nikkei", "hsi", "ssec", "nifty", "asx"]}
NICE = {"moexog": "Нефтегаз", "moexfn": "Финансы", "moexmm": "Металлы",
        "moexcn": "Потребит.", "moexeu": "Энергетика", "moexch": "Химия",
        "moextn": "Транспорт"}

# ------------------------------------------------------------------ панель
y_loc = load_log_vol(OUT / "cond_vol_moex.csv")[SECTORS]
y_glob = load_log_vol(OUT / "cond_vol.csv")
reg = pd.DataFrame({r: y_glob[mkts].mean(axis=1) for r, mkts in REGIONS.items()})
y = y_loc.join(reg, how="inner").dropna()
cols = list(y.columns)
n_sec = len(SECTORS)
print(f"Единая система: {y.shape[0]} дней x {y.shape[1]} узлов "
      f"({y.index.min().date()} — {y.index.max().date()})")


def decompose(theta):
    """E_i, I_i, own_i по секторам (в п.п., строки нормированы на 100)."""
    th = theta * 100
    E = th[:n_sec, n_sec:].sum(axis=1)
    I = th[:n_sec, :n_sec].sum(axis=1) - np.diag(th[:n_sec, :n_sec])
    own = np.diag(th[:n_sec, :n_sec])
    return E, I, own


def ri_of(theta):
    E, I, _ = decompose(theta)
    return float(I.sum() / (I.sum() + E.sum()))


N_INT = len(SECTORS) - 1  # внутренних источников у каждого сектора


def ri_avg_of(E_sum, I_sum, g_ext):
    """RI, нормированный на число источников: средний внутренний источник
    против среднего внешнего; нейтральная точка 0,5 при любых размерностях."""
    return float((I_sum / N_INT) / (I_sum / N_INT + E_sum / g_ext))


# ------------------------------------------------------- полная выборка
B_list, Sigma = fit_var(y, p=2)
theta_full = fevd_generalised(B_list, Sigma, H=H)
E_f, I_f, own_f = decompose(theta_full)
full_rows = [{"sector": NICE[s], "own": own_f[i], "I_internal": I_f[i],
              "E_external": E_f[i], "RI": I_f[i] / (I_f[i] + E_f[i])}
             for i, s in enumerate(SECTORS)]
full_rows.append({"sector": "Все сектора", "own": own_f.mean(),
                  "I_internal": I_f.sum(), "E_external": E_f.sum(),
                  "RI": ri_of(theta_full),
                  "RI_avg": ri_avg_of(E_f.sum(), I_f.sum(), 3)})
ri_full = pd.DataFrame(full_rows)
ri_full.to_csv(TAB / "table31_ri_full.csv", index=False)
print("\n=== Полная выборка: внешне-внутренняя декомпозиция ===")
print(ri_full.round(3).to_string(index=False))

# ------------------------------------------------------- подпериоды
per_rows = []
for name, (a, b) in PERIODS_LOCAL.items():
    sub = y.loc[a:b]
    if len(sub) < 60:
        continue
    B, S = fit_var(sub, p=2 if len(sub) > 150 else 1)
    th = fevd_generalised(B, S, H=H)
    E, I, _ = decompose(th)
    row = {"period": name, "n_obs": len(sub), "E_sum": E.sum(), "I_sum": I.sum(),
           "RI": I.sum() / (I.sum() + E.sum()),
           "RI_avg": ri_avg_of(E.sum(), I.sum(), 3)}
    for i, s in enumerate(SECTORS):
        row[f"RI_{s}"] = I[i] / (I[i] + E[i])
    per_rows.append(row)
ri_per = pd.DataFrame(per_rows)
ri_per.to_csv(TAB / "table31b_ri_periods.csv", index=False)
print("\n=== Подпериоды: агрегатный RI ===")
print(ri_per[["period", "n_obs", "E_sum", "I_sum", "RI", "RI_avg"]]
      .round(3).to_string(index=False))

# ---------------------------------------- гранулярность внешнего блока (Б.8)
# Уровень сырого RI зависит от представления внешнего блока; проверяются три
# варианта: G = 1 (одна равновзвешенная мировая лог-волатильность), G = 3
# (базовые регионы), G = 14 (все внешние рынки по отдельности).
WORLD_MKTS = [m for ms in REGIONS.values() for m in ms]
gran_variants = {
    1: y_loc.join(pd.DataFrame({"WORLD": y_glob[WORLD_MKTS].mean(axis=1)}),
                  how="inner").dropna(),
    3: y,
    14: y_loc.join(y_glob[WORLD_MKTS], how="inner").dropna(),
}
gran_rows = []
for g, yy in gran_variants.items():
    windows = [("FULL", yy)] + [(name, yy.loc[a:b])
                                for name, (a, b) in PERIODS_LOCAL.items()]
    for name, sub in windows:
        if len(sub) < 60:
            continue
        B, S = fit_var(sub, p=2 if len(sub) > 150 else 1)
        th = fevd_generalised(B, S, H=H)
        E, I, _ = decompose(th)
        gran_rows.append({"G": g, "period": name, "n_obs": len(sub),
                          "E_sum": E.sum(), "I_sum": I.sum(),
                          "RI": I.sum() / (I.sum() + E.sum()),
                          "RI_avg": ri_avg_of(E.sum(), I.sum(), g)})
gran = pd.DataFrame(gran_rows)
gran.to_csv(TAB / "table31e_ri_granularity.csv", index=False)
print("\n=== Гранулярность внешнего блока (G = 1/3/14) ===")
print(gran.round(3).to_string(index=False))

# ------------------------------------------------------- скользящее окно
def rolling_ri(y, win=250, step=10, p=1):
    rows, dates = [], []
    for end in range(win, len(y), step):
        sub = y.iloc[end - win:end]
        try:
            B, S = fit_var(sub, p=p)
            th = fevd_generalised(B, S, H=H)
        except Exception:
            continue
        rows.append(ri_of(th))
        dates.append(y.index[end - 1])
    return pd.Series(rows, index=pd.DatetimeIndex(dates), name="RI_roll")

ri_roll = rolling_ri(y)
print(f"\nСкользящий RI: {len(ri_roll)} точек")

# ------------------------------------------------------- TVP-VAR (механика § 3.8.5)
W0 = 250
KAPPA1 = 0.99


def fevd_generalised_var1(B1, Sigma, H=10):
    n = B1.shape[0]
    M = np.empty((H, n, n))
    M[0] = np.eye(n)
    for h in range(1, H):
        M[h] = B1 @ M[h - 1]
    sd = np.diag(Sigma).copy()
    sd[sd <= 0] = 1e-12
    num = np.zeros((n, n)); den = np.zeros((n, n))
    for h in range(H):
        num += (M[h] @ Sigma) ** 2 / sd[None, :]
        den += np.diag(M[h] @ Sigma @ M[h].T)[:, None] * np.ones((1, n))
    theta = num / den
    return theta / theta.sum(axis=1, keepdims=True)


def tvp_ri(y, kappa1=KAPPA1, kappa2=0.96):
    """TVP-VAR(1): фильтр Koop-Korobilis, ежедневный RI (повторяет 16)."""
    Y = np.asarray(y)
    T, n = Y.shape
    k = n + 1
    Ytr, Xtr = Y[1:W0], np.column_stack([np.ones(W0 - 1), Y[:W0 - 1]])
    coef, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
    resid = Ytr - Xtr @ coef
    Sigma = resid.T @ resid / (W0 - 1 - k)
    P = np.kron(Sigma, np.linalg.inv(Xtr.T @ Xtr + 1e-8 * np.eye(k)))
    beta = coef.T.reshape(-1)
    dates, ri = [], []
    for t in range(W0, T):
        x = np.concatenate([[1.0], Y[t - 1]])
        P = P / kappa1
        Bmat = beta.reshape(n, k)
        e = Y[t] - Bmat @ x
        ZP = np.stack([x @ P[i * k:(i + 1) * k, :] for i in range(n)])
        F = np.stack([ZP[:, j * k:(j + 1) * k] @ x for j in range(n)]).T + Sigma
        F = 0.5 * (F + F.T) + 1e-10 * np.eye(n)
        K = np.linalg.solve(F, ZP).T
        beta = beta + K @ e
        P = P - K @ ZP
        P = 0.5 * (P + P.T)
        Bmat = beta.reshape(n, k)
        eps = Y[t] - Bmat @ x
        Sigma = kappa2 * Sigma + (1 - kappa2) * np.outer(eps, eps)
        theta = fevd_generalised_var1(Bmat[:, 1:], Sigma, H=H)
        ri.append(ri_of(theta))
        dates.append(y.index[t])
    return pd.Series(ri, index=pd.DatetimeIndex(dates))

ri_tvp96 = tvp_ri(y, kappa2=0.96)
ri_tvp99 = tvp_ri(y, kappa2=0.99)
daily = pd.concat([ri_roll, ri_tvp96.rename("RI_tvp_k96"),
                   ri_tvp99.rename("RI_tvp_k99")], axis=1)
daily.to_csv(TAB / "table31c_ri_daily.csv")

# ------------------------------------------------------- разрывы и инференция
x = ri_roll.values
T_r = len(x)
MIN_SEG, MAX_BREAKS = 25, 3          # окно шага 10 дней: 25 точек = 250 дней
cs = np.concatenate([[0.0], np.cumsum(x)])
cs2 = np.concatenate([[0.0], np.cumsum(x ** 2)])

def sse(i, j):
    n = j - i
    s = cs[j] - cs[i]
    return (cs2[j] - cs2[i]) - s * s / n

def segment(k):
    if k == 0:
        return [], sse(0, T_r)
    D = np.full((k + 1, T_r + 1), np.inf)
    arg = np.zeros((k + 1, T_r + 1), dtype=int)
    for j in range(MIN_SEG, T_r + 1):
        D[0, j] = sse(0, j)
    for m in range(1, k + 1):
        for j in range(MIN_SEG * (m + 1), T_r + 1):
            best, bi = np.inf, -1
            for i in range(MIN_SEG * m, j - MIN_SEG + 1):
                v = D[m - 1, i] + sse(i, j)
                if v < best:
                    best, bi = v, i
            D[m, j], arg[m, j] = best, bi
    breaks, j = [], T_r
    for m in range(k, 0, -1):
        j = arg[m, j]
        breaks.append(j)
    return sorted(breaks), D[k, T_r]

results = {}
for k in range(MAX_BREAKS + 1):
    bks, s = segment(k)
    bic = T_r * np.log(s / T_r) + (2 * k + 1) * np.log(T_r)
    results[k] = (bks, s, bic)
k_star = min(results, key=lambda k: results[k][2])
breaks = results[k_star][0]
bdates = [ri_roll.index[b] for b in breaks]
seg_bounds = [0] + breaks + [T_r]
segs = [(ri_roll.index[a], ri_roll.index[b - 1], float(x[a:b].mean()))
        for a, b in zip(seg_bounds[:-1], seg_bounds[1:])]
print(f"\n[Бай-Перрон] k* = {k_star}, даты: {[d.date() for d in bdates]}")
for a, b, m in segs:
    print(f"  сегмент {a.date()} — {b.date()}: средний RI {m:.3f}")

last_break = breaks[-1] if breaks else 0
D2022 = (np.arange(T_r) >= last_break).astype(float)
ols = sm.OLS(x, sm.add_constant(D2022)).fit(cov_type="HAC", cov_kwds={"maxlags": 25})
beta_sh, t_sh, p_sh = ols.params[1], ols.tvalues[1], ols.pvalues[1]
post = x[last_break:]
above = sm.OLS(post - 0.5, np.ones(len(post))).fit(cov_type="HAC", cov_kwds={"maxlags": 25})
t_above, p_above = above.tvalues[0], above.pvalues[0] / 2

def stationary_bootstrap(v, n_rep=5000, p=1 / 25):
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

boot = stationary_bootstrap(post - post.mean())
p_boot = float((boot >= post.mean() - 0.5).mean())
print(f"[HAC] сдвиг RI после последнего разрыва: {beta_sh:+.3f}, t = {t_sh:.2f}, p = {p_sh:.2e}")
print(f"[HAC] RI > 0,5 после разрыва: среднее {post.mean():.3f}, t = {t_above:.2f}, "
      f"p(one-sided) = {p_above:.2e}; бутстрап p = {p_boot:.4f}")

pd.DataFrame({
    "item": ["k_star", "break_dates", "segment_means",
             "post_shift", "post_shift_t", "post_shift_p",
             "post_mean", "post_gt_05_t", "post_gt_05_p", "post_gt_05_p_boot",
             "ri_tvp96_recent", "ri_tvp99_recent"],
    "value": [k_star, "; ".join(str(d.date()) for d in bdates),
              "; ".join(f"{m:.3f}" for _, _, m in segs),
              f"{beta_sh:.4f}", f"{t_sh:.2f}", f"{p_sh:.3e}",
              f"{post.mean():.4f}", f"{t_above:.2f}", f"{p_above:.3e}", f"{p_boot:.4f}",
              f"{ri_tvp96.loc['2023-01-01':].mean():.4f}",
              f"{ri_tvp99.loc['2023-01-01':].mean():.4f}"],
}).to_csv(TAB / "table31d_ri_breaks.csv", index=False)

# ------------------------------------------------------- датировка (контроль)
# (а) Устойчивость эндогенных разрывов к монотонной перешкале индекса:
# та же сегментация на трансформации r/(2−r), спрямляющей сжатие шкалы у
# единицы. Разрыв, переживающий перешкалу, датирован надёжно; смещающийся
# на годы — фрагилен и содержательной нагрузки не несёт.
x_tr = x / (2.0 - x)
cs = np.concatenate([[0.0], np.cumsum(x_tr)])
cs2 = np.concatenate([[0.0], np.cumsum(x_tr ** 2)])
results_tr = {}
for k in range(MAX_BREAKS + 1):
    bks, s = segment(k)
    bic = T_r * np.log(s / T_r) + (2 * k + 1) * np.log(T_r)
    results_tr[k] = (bks, s, bic)
k_star_tr = min(results_tr, key=lambda k: results_tr[k][2])
bdates_tr = [ri_roll.index[b] for b in results_tr[k_star_tr][0]]
print(f"[Бай-Перрон, r/(2-r)] k* = {k_star_tr}, даты: {[d.date() for d in bdates_tr]}")

# (б) Датировка нарастания без окна: у скользящей меры сдвиг в декабре 2021 г.
# означает изменения в данных не позднее осени 2021 г. (трейлинговое окно
# будущего не видит); фильтрационная TVP-траектория проверяет это напрямую.
tvp = ri_tvp96
month_means = {m: float(tvp.loc[m].mean())
               for m in ("2021-10", "2021-11", "2021-12", "2022-01")}
base_tvp = tvp.loc["2021-01-01":"2021-10-31"]
thr = base_tvp.mean() + 2 * base_tvp.std()
ma20 = tvp.rolling(20).mean()
after = ma20.loc["2021-11-01":]
above = after > thr
sustained = above & pd.Series(
    [above.iloc[i:i + 20].all() for i in range(len(above))], index=above.index)
cross_date = sustained.idxmax() if sustained.any() else None
print(f"[TVP-RI] месячные средние: " +
      ", ".join(f"{m}: {v:.3f}" for m, v in month_means.items()))
print(f"[TVP-RI] база 01–10.2021 + 2 SD = {thr:.3f}; устойчивое превышение "
      f"20-дневным средним с {cross_date.date() if cross_date is not None else '—'}")

pd.DataFrame({
    "item": ["k_star_transformed", "break_dates_transformed",
             "tvp_mean_2021_10", "tvp_mean_2021_11", "tvp_mean_2021_12",
             "tvp_mean_2022_01", "tvp_base_plus2sd", "tvp_ma20_cross_date"],
    "value": [k_star_tr, "; ".join(str(d.date()) for d in bdates_tr),
              f"{month_means['2021-10']:.4f}", f"{month_means['2021-11']:.4f}",
              f"{month_means['2021-12']:.4f}", f"{month_means['2022-01']:.4f}",
              f"{thr:.4f}",
              str(cross_date.date()) if cross_date is not None else ""],
}).to_csv(TAB / "table31f_ri_dating.csv", index=False)

# ------------------------------------------------------- рисунок
fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2),
                         gridspec_kw={"height_ratios": [1.35, 1]})
ax = axes[0]
ax.plot(ri_roll.index, ri_roll.values, color="navy", lw=1.0,
        label="Скользящее окно 250/10")
ax.plot(ri_tvp96.index, ri_tvp96.values, color="#d62828", lw=0.8, alpha=0.85,
        label="TVP-VAR ($\\kappa_2 = 0{,}96$)")
ax.axhline(0.5, color="grey", lw=0.8, ls="--")
for d in bdates:
    ax.axvline(d, color="black", lw=0.9, ls=":")
shade(ax)
ax.set_ylabel("RI: доля внутренних источников")
ax.set_title("(а) Индекс внутренней ориентации риска RI$_t$ = I / (I + E), единая система 7 + 3")
ax.legend(loc="lower right", fontsize=9, frameon=False)

ax = axes[1]
calm = ri_per[ri_per["period"].str.startswith("Calm")].iloc[0]
rec = ri_per[ri_per["period"].str.startswith("Recent")].iloc[0]
xpos = np.arange(n_sec)
w = 0.38
ax.bar(xpos - w / 2, [calm[f"RI_{s}"] for s in SECTORS], w,
       color="#669bbc", label="Спокойный (2009–2019)")
ax.bar(xpos + w / 2, [rec[f"RI_{s}"] for s in SECTORS], w,
       color="#d62828", label="Недавний (2023–2026)")
ax.axhline(0.5, color="grey", lw=0.8, ls="--")
ax.set_xticks(xpos, [NICE[s] for s in SECTORS], fontsize=9)
ax.set_ylabel("RI сектора")
ax.set_title("(б) RI по секторам: спокойный период против 2023–2026")
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(FIG / "fig24_ri_index.png", dpi=200)
plt.close()
print("\nfig24_ri_index.png saved. DONE-18")
