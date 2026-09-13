"""Вневыборочные прогнозы условной ковариации GARCH/DCC-семейства (§ 2.3.5).

Протокол: начальное окно 19.09.2007 — 31.12.2014; далее каждые 63 торговых дня
(квартал) весь конвейер переоценивается на расширяющемся окне, используя
только данные до даты переоценки. Переоценке подлежат:
  - одномерные спецификации (GARCH / GJR / EGARCH со скошенным t,
    ПЕРЕВЫБОР лучшей по BIC внутри окна — выбор спецификации тоже часть
    прогнозного правила и не должен видеть будущее);
  - корреляционные модели DCC / ADCC / cDCC и кластерная GDCC, включая
    саму кластеризацию (спектральную, K = 3), которая выучивается заново
    на корреляциях остатков окна;
  - степени свободы t-распределения стандартизированной портфельной
    доходности (по каждой модели отдельно) — для квантилей VaR / ES.

Внутри блока между переоценками параметры зафиксированы («винтажные»), а
фильтры — одномерные GARCH-рекурсии и корреляционные Q-рекурсии — обновляются
ежедневно информацией, доступной на конец предыдущего дня; безусловные
моменты Q_bar и N_bar берутся только по окну оценивания. Прогноз дня t,
таким образом, не использует ни одного наблюдения дня t и позже.

Выходы:
  output/oos_dcc_forecasts.csv — по дням OOS-периода: sigma_p и винтажные nu
                                 каждой модели, дата переоценки;
  output/oos_refit_log.csv     — журнал переоценок (спецификации, параметры,
                                 кластеры, время счёта).

Запуск из корня: .venv/bin/python src/19_oos_forecasts.py  (~15-30 мин, Pool)
"""
from pathlib import Path
import os
import time
import warnings

import numpy as np
import pandas as pd
from multiprocessing import Pool

warnings.filterwarnings("ignore")

OUT = Path("output")
SCALE = 100.0
STEP = 63                      # квартал торговых дней
INIT_END = "2014-12-31"        # конец начального окна оценивания
SPECS = [("GARCH", dict(vol="GARCH", p=1, q=1, o=0)),
         ("GJR", dict(vol="GARCH", p=1, q=1, o=1)),
         ("EGARCH", dict(vol="EGARCH", p=1, q=1, o=1))]
CORR_MODELS = ["dcc", "adcc", "cdcc", "gdcc"]
K_CLUST = 3


# ------------------------------------------------------------------ ковариации
def mvt_loglike_path(Rs_needed=False):
    pass  # (документационная заглушка: правдоподобие ниже, в filter_*)


def _mvt(R, z, nu, gammaln):
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0 or not np.isfinite(logdet):
        return -1e10
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return -1e10
    quad = float(z @ Rinv @ z)
    p = R.shape[0]
    c = gammaln((nu + p) / 2.0) - gammaln(nu / 2.0) - 0.5 * p * np.log(np.pi * (nu - 2.0))
    return c - 0.5 * logdet - ((nu + p) / 2.0) * np.log(1.0 + quad / (nu - 2.0))


def filter_scalar(z, params, model, win_end, ws=None, loglike_to=None):
    """Скалярные DCC/ADCC/cDCC: рекурсия по всему переданному массиву, моменты
    Q_bar/N_bar — только по окну [0, win_end). Если ws задан (T, N — веса,
    умноженные на винтажные сигмы), возвращает и траекторию sigma_p."""
    from scipy.special import gammaln
    T, N = z.shape
    a, b = params[0], params[1]
    g = params[2] if model == "adcc" else 0.0
    nu = params[-1]
    if a < 1e-6 or b < 1e-6 or a + b + 0.5 * g >= 0.999 or nu <= 4.5:
        return (1e10, None)
    zw = z[:win_end]
    Q_bar = zw.T @ zw / win_end
    if model == "adcc":
        zn = np.where(zw < 0, zw, 0.0)
        N_bar = zn.T @ zn / win_end
        intercept = Q_bar - a * Q_bar - b * Q_bar - g * N_bar
    else:
        intercept = (1 - a - b) * Q_bar
    stop = T if loglike_to is None else loglike_to
    Q = Q_bar.copy()
    sp = np.full(T, np.nan) if ws is not None else None
    ll = 0.0
    for t in range(T):
        if t == 0:
            qd = np.sqrt(np.diag(Q_bar))
            R = Q_bar / np.outer(qd, qd)
        else:
            zt = z[t - 1]
            if model == "adcc":
                znt = np.where(zt < 0, zt, 0.0)
                Q = intercept + a * np.outer(zt, zt) + g * np.outer(znt, znt) + b * Q
            elif model == "cdcc":
                qd = np.sqrt(np.diag(Q))
                zs = qd * zt
                Q = intercept + a * np.outer(zs, zs) + b * Q
            else:
                Q = intercept + a * np.outer(zt, zt) + b * Q
            qd = np.sqrt(np.diag(Q))
            R = Q / np.outer(qd, qd)
        R = (R + R.T) / 2.0
        if t < stop:
            ll += _mvt(R, z[t], nu, gammaln)
        if ws is not None:
            v = ws[t]
            sp[t] = np.sqrt(max(float(v @ R @ v), 0.0))
    return -ll, sp


def filter_gdcc(z, a_vec, b_vec, nu, win_end, ws=None, loglike_to=None):
    """Кластерная GDCC (адамарова форма, как в 04b) на винтажных моментах."""
    from scipy.special import gammaln
    T, N = z.shape
    A = np.outer(a_vec, a_vec)
    B = np.outer(b_vec, b_vec)
    if float(np.max(a_vec ** 2 + b_vec ** 2)) >= 0.999 or np.min(a_vec) <= 0 \
            or np.min(b_vec) <= 0 or nu <= 4.5:
        return (1e10, None)
    zw = z[:win_end]
    Q_bar = zw.T @ zw / win_end
    intercept = (np.ones((N, N)) - A - B) * Q_bar
    stop = T if loglike_to is None else loglike_to
    Q = Q_bar.copy()
    sp = np.full(T, np.nan) if ws is not None else None
    ll = 0.0
    for t in range(T):
        if t == 0:
            qd = np.sqrt(np.diag(Q_bar))
            R = Q_bar / np.outer(qd, qd)
        else:
            zt = z[t - 1]
            Q = intercept + A * np.outer(zt, zt) + B * Q
            qd = np.sqrt(np.diag(Q))
            R = Q / np.outer(qd, qd)
        R = (R + R.T) / 2.0
        if t < stop:
            ll += _mvt(R, z[t], nu, gammaln)
        if ws is not None:
            v = ws[t]
            sp[t] = np.sqrt(max(float(v @ R @ v), 0.0))
    return -ll, sp


# ------------------------------------------------------------------- воркер
def do_refit(task):
    """Одна переоценка: окно [0, t_k), блок прогнозов [t_k, t_next)."""
    from arch import arch_model
    from scipy.optimize import minimize
    from scipy.stats import t as tdist
    from sklearn.cluster import SpectralClustering

    k, t_k, t_next, ret_values, dates, cols, W = task
    t_start = time.time()
    T_all = t_next                      # данных дальше конца блока не трогаем
    r = ret_values[:T_all]
    N = len(cols)

    # --- этап 1: одномерные модели (выбор спецификации по BIC внутри окна)
    sig = np.empty((T_all, N))
    spec_chosen = {}
    for j, c in enumerate(cols):
        s_win = pd.Series(r[:t_k, j] * SCALE)
        best = None
        for lab, kw in SPECS:
            try:
                res = arch_model(s_win, mean="Constant", dist="skewt", **kw).fit(
                    disp="off", show_warning=False)
                if best is None or res.bic < best[2]:
                    best = (lab, res.params.values, res.bic, kw)
            except Exception:
                continue
        lab, par, _, kw = best
        spec_chosen[c] = lab
        s_full = pd.Series(r[:, j] * SCALE)
        fixed = arch_model(s_full, mean="Constant", dist="skewt", **kw).fix(par)
        sig[:, j] = fixed.conditional_volatility.values / SCALE

    z = r / np.maximum(sig, 1e-12)
    ws = W[:T_all] * sig                # (w ⊙ sigma), т.к. sigma_p^2 = v' R v

    # --- этап 2: корреляционные модели на остатках окна
    out_sigma = {}
    out_nu = {}
    params_log = {}
    for m in ["dcc", "adcc", "cdcc"]:
        if m == "adcc":
            x0 = [0.03, 0.95, 0.02, 8.0]
            bounds = [(1e-4, 0.5), (1e-4, 0.999), (1e-4, 0.5), (4.5, 60.0)]
        else:
            x0 = [0.05, 0.93, 8.0]
            bounds = [(1e-4, 0.5), (1e-4, 0.999), (4.5, 60.0)]
        res = minimize(lambda p: filter_scalar(z[:t_k], p, m, t_k)[0], x0,
                       method="L-BFGS-B", bounds=bounds, options=dict(maxiter=200))
        _, sp = filter_scalar(z, res.x, m, t_k, ws=ws, loglike_to=0)
        out_sigma[m] = sp
        params_log[m] = np.round(res.x, 5).tolist()
        zp_win = (W[:t_k] * ret_values[:t_k]).sum(axis=1)[30:] / np.maximum(sp[30:t_k], 1e-12)
        out_nu[m] = max(4.5, tdist.fit(zp_win)[0])

    # GDCC: кластеризация только на корреляциях остатков окна
    R_emp = np.corrcoef(z[:t_k], rowvar=False)
    np.fill_diagonal(R_emp, 1.0)
    labels = SpectralClustering(n_clusters=K_CLUST, affinity="precomputed",
                                assign_labels="kmeans", random_state=2026
                                ).fit_predict(np.abs(R_emp)) + 1
    Kc = len(np.unique(labels))

    def to_full(p):
        return p[:Kc][labels - 1], p[Kc:2 * Kc][labels - 1], p[-1]

    res_g = minimize(lambda p: filter_gdcc(z[:t_k], *to_full(p), t_k)[0],
                     np.concatenate([np.full(Kc, 0.10), np.full(Kc, 0.95), [9.0]]),
                     method="L-BFGS-B",
                     bounds=[(1e-3, 0.6)] * Kc + [(1e-3, 0.999)] * Kc + [(4.5, 60.0)],
                     options=dict(maxiter=300))
    a_v, b_v, nu_g = to_full(res_g.x)
    _, sp_g = filter_gdcc(z, a_v, b_v, nu_g, t_k, ws=ws, loglike_to=0)
    out_sigma["gdcc"] = sp_g
    params_log["gdcc"] = np.round(res_g.x, 5).tolist()
    zp_win = (W[:t_k] * ret_values[:t_k]).sum(axis=1)[30:] / np.maximum(sp_g[30:t_k], 1e-12)
    out_nu["gdcc"] = max(4.5, tdist.fit(zp_win)[0])

    block = slice(t_k, t_next)
    rows = pd.DataFrame({"Datetime": dates[block]})
    for m in CORR_MODELS:
        rows[f"sigma_{m}"] = out_sigma[m][block]
        rows[f"nu_{m}"] = out_nu[m]
    rows["refit_date"] = dates[t_k - 1]
    log = {"refit": k, "refit_date": str(pd.Timestamp(dates[t_k - 1]).date()),
           "T_win": t_k, "specs": ";".join(f"{c}:{spec_chosen[c]}" for c in cols),
           "clusters": ";".join(map(str, labels)),
           **{f"params_{m}": str(params_log[m]) for m in CORR_MODELS},
           "sec": round(time.time() - t_start, 1)}
    return rows, log


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    ret = pd.read_csv(OUT / "log_rets_clean.csv", parse_dates=["Datetime"]
                      ).set_index("Datetime")
    res_idx = pd.read_csv(OUT / "std_residuals.csv", parse_dates=["Datetime"]
                          )["Datetime"]
    ret = ret.reindex(res_idx).fillna(0.0)
    cols = list(ret.columns)
    dates = ret.index.values
    T, N = ret.shape

    # Веса модельного портфеля по протоколу § 2.3.5: 70% RTSI + 30% зарубежных
    # позиций (до 21.02.2022 — S&P 500 и DAX, после — SSEC и NIFTY 50)
    W = pd.DataFrame(0.0, index=ret.index, columns=cols)
    switch = pd.Timestamp("2022-02-21")
    pre = ret.index < switch
    W.loc[pre, "rtsi"] = 0.70; W.loc[pre, "spx"] = 0.15; W.loc[pre, "dax"] = 0.15
    W.loc[~pre, "rtsi"] = 0.70; W.loc[~pre, "ssec"] = 0.15; W.loc[~pre, "nifty"] = 0.15
    W = W.values

    t0 = int(np.searchsorted(ret.index.values, np.datetime64(INIT_END), side="right"))
    refits = list(range(t0, T - 1, STEP))
    print(f"T={T}, начальное окно {t0} строк (до {INIT_END}), переоценок: {len(refits)}")

    tasks = []
    for k, t_k in enumerate(refits):
        t_next = min(t_k + STEP, T)
        tasks.append((k, t_k, t_next, ret.values, dates, cols, W))

    test_one = os.environ.get("OOS_TEST_ONE")
    if test_one:
        rows, log = do_refit(tasks[int(test_one)])
        print(log)
        print(rows.head(8).to_string(index=False))
    else:
        with Pool(processes=6) as pool:
            results = pool.map(do_refit, tasks)
        fc = pd.concat([r for r, _ in results], ignore_index=True)
        fc.to_csv(OUT / "oos_dcc_forecasts.csv", index=False)
        logs = pd.DataFrame([l for _, l in results])
        logs.to_csv(OUT / "oos_refit_log.csv", index=False)
        print(f"OOS дней: {len(fc)}; период {fc['Datetime'].min()} — {fc['Datetime'].max()}")
        print(logs[["refit", "refit_date", "T_win", "sec"]].tail(3).to_string(index=False))
        print("DONE-19")
