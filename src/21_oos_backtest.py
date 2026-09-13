"""Вневыборочный бэктест VaR(99%)/ES(97,5%) и симуляция капитала (§ 3.7).

Вход — честные однодневные прогнозы, построенные только на прошлом:
  - output/oos_dcc_forecasts.csv (скрипт 19): sigma_p и винтажные nu для
    DCC/ADCC/cDCC/GDCC, квартальные переоценки на расширяющемся окне;
  - output/oos_fmsv_sdraws_chunk*.bin (скрипт 20): постериорно-предиктивные
    розыгрыши портфельного СКО FMSV, переоценка каждые 10 дней; VaR и ES
    вычисляются как квантиль и хвостовое среднее СМЕСИ нормальных
    распределений по розыгрышам — полный posterior predictive без plug-in;
  - M0 EWMA (lambda = 0,94) считается здесь же: рекурсия не содержит
    оцениваемых параметров и вневыборочна по построению; nu переоценивается
    на той же квартальной сетке, что и у DCC-семейства.

Батарея тестов — как в 07b (Купик, Кристофферсен, официальный базельский
светофор + скользящие 250-дневные зоны, Ачерби-Секели Z1/Z2 на согласованном
уровне 97,5% с симуляционными p-значениями, FZ0-потеря и Диболд-Мариано
против cDCC), но на вневыборочном периоде 2015-01 — 2026-06.

Капитал считается по базельскому правилу IMA с надбавкой светофора
(плюс-фактором) по пробоям за скользящие 250 дней; прежняя форма
mult*max(VaR, avg60) сохранена контрольными колонками. Дополнительно
приводится строка FMSV с рекалибровкой покрытия (множитель c* подобран на
том же периоде до уровня пробоев лучшей DCC-модели — верхняя граница, не
прогноз).

Выходы: output/tables/table30_oos_backtest.csv,
        table30b_oos_periods.csv, table30c_oos_capital.csv,
        output/figures/fig6_var_capital_full.png (заменяет ин-сэмпл версию).
Запуск из корня: .venv/bin/python src/21_oos_backtest.py  (~2-3 мин)
"""
from pathlib import Path
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as tdist, chi2, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT = Path("output")
TAB = OUT / "tables"
FIG = OUT / "figures"
ALPHA = 0.01
TAIL_ES = 0.025
NOTIONAL = 10e9
WACC = 0.13

# ---------------------------------------------------------------- данные
ret = pd.read_csv(OUT / "log_rets_clean.csv", parse_dates=["Datetime"]).set_index("Datetime")
res_idx = pd.read_csv(OUT / "std_residuals.csv", parse_dates=["Datetime"])["Datetime"]
ret = ret.reindex(res_idx).fillna(0.0)
cols = list(ret.columns)

# Веса модельного портфеля по протоколу § 2.3.5: 70% RTSI + 30% зарубежных
# позиций (до 21.02.2022 — S&P 500 и DAX, после — SSEC и NIFTY 50)
W = pd.DataFrame(0.0, index=ret.index, columns=cols)
switch = pd.Timestamp("2022-02-21")
pre = ret.index < switch
W.loc[pre, "rtsi"] = 0.70; W.loc[pre, "spx"] = 0.15; W.loc[pre, "dax"] = 0.15
W.loc[~pre, "rtsi"] = 0.70; W.loc[~pre, "ssec"] = 0.15; W.loc[~pre, "nifty"] = 0.15
port_ret_full = (W.values * ret.values).sum(axis=1)
port = pd.Series(port_ret_full, index=ret.index)

# ---------------------------------------------------------------- прогнозы 19
fc = pd.read_csv(OUT / "oos_dcc_forecasts.csv", parse_dates=["Datetime"]).set_index("Datetime")
oos_idx = fc.index
pr = port.reindex(oos_idx).values
T = len(oos_idx)
print(f"OOS период: {oos_idx.min().date()} — {oos_idx.max().date()}, T = {T}")

# ---------------------------------------------------------------- EWMA (M0)
LAM = 0.94
sig2 = np.zeros(len(port)); sig2[0] = port.iloc[0] ** 2
for t in range(1, len(port)):
    sig2[t] = LAM * sig2[t - 1] + (1 - LAM) * port.iloc[t - 1] ** 2
sd_ewma_full = pd.Series(np.sqrt(sig2), index=port.index)
sd_ewma = sd_ewma_full.reindex(oos_idx).values
# винтажные nu на квартальной сетке 19-го скрипта
nu_ewma = np.empty(T)
refit_dates = fc["refit_date"].values
for rd in np.unique(refit_dates):
    sel = refit_dates == rd
    hist = (port / sd_ewma_full).loc[:pd.Timestamp(rd)].iloc[30:]
    nu_ewma[sel] = max(4.5, tdist.fit(hist.values)[0])

# ---------------------------------------------------------------- FMSV (M5)
chunks = sorted(glob.glob(str(OUT / "oos_fmsv_sdraws_chunk*.bin")))
fmsv_dates, fmsv_rows = [], []
for b in chunks:
    base = b[:-4]
    meta = {l.split()[0]: int(l.split()[1]) for l in open(base + ".meta")
            if l.split()[0] in ("rows", "cols")}
    arr = np.fromfile(b, dtype=np.float64).reshape((meta["rows"], meta["cols"]), order="F")
    dts = pd.to_datetime([l.strip() for l in open(base + ".dates")])
    fmsv_dates.append(pd.Series(range(len(dts)), index=dts))
    fmsv_rows.append(arr)
S_draws = np.vstack(fmsv_rows)
d_all = pd.DatetimeIndex(np.concatenate([d.index.values for d in fmsv_dates]))
order = np.argsort(d_all.values)
S_draws = S_draws[order]
d_all = d_all[order]
assert not d_all.duplicated().any(), "дубли дат в FMSV-чанках"
S = pd.DataFrame(S_draws, index=d_all).reindex(oos_idx).values
have_fmsv = np.isfinite(S).all(axis=1)
print(f"FMSV: розыгрышей {S.shape[1]}, покрыто дней {have_fmsv.sum()}/{T}")


def mixture_var_es(S, alpha):
    """VaR и ES уровня alpha для смеси N(0, s_d^2) по розыгрышам s_d.
    Возвращает положительные величины (величина потерь)."""
    n, D = S.shape
    lo = -14.0 * np.nanmax(S, axis=1)
    hi = np.zeros(n)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        p = norm.cdf(mid[:, None] / S).mean(axis=1)
        too_low = p < alpha
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    x = 0.5 * (lo + hi)
    tail_mass = norm.cdf(x[:, None] / S).mean(axis=1)
    es = (S * norm.pdf(x[:, None] / S)).mean(axis=1) / np.maximum(tail_mass, 1e-12)
    return -x, es


# ---------------------------------------------------------------- VaR/ES
def t_var_es(sd, nu, alpha):
    """Параметрические VaR/ES по стандартизированному t (единичная дисперсия)."""
    scale = np.sqrt((nu - 2.0) / nu)
    tq = tdist.ppf(alpha, nu)
    var_ = -tq * scale * sd
    es_ = (tdist.pdf(tq, nu) / alpha) * ((nu + tq ** 2) / (nu - 1.0)) * scale * sd
    return var_, es_


models = {}
for m, label in [("dcc", "M1_DCC_t"), ("adcc", "M2_ADCC_t"),
                 ("cdcc", "M3_cDCC_t"), ("gdcc", "M4_GDCC_t")]:
    sd = fc[f"sigma_{m}"].values
    nu = fc[f"nu_{m}"].values
    v99, _ = t_var_es(sd, nu, ALPHA)
    v975, e975 = t_var_es(sd, nu, TAIL_ES)
    models[label] = dict(sd=sd, nu=nu, VaR=v99, VaR975=v975, ES=e975)
v99, _ = t_var_es(sd_ewma, nu_ewma, ALPHA)
v975, e975 = t_var_es(sd_ewma, nu_ewma, TAIL_ES)
models = {"M0_EWMA": dict(sd=sd_ewma, nu=nu_ewma, VaR=v99, VaR975=v975, ES=e975), **models}

v99_f, _ = mixture_var_es(S, ALPHA)
v975_f, e975_f = mixture_var_es(S, TAIL_ES)
models["M5_FMSV"] = dict(sd=np.nanmean(S, axis=1), nu=np.full(T, np.nan),
                         VaR=v99_f, VaR975=v975_f, ES=e975_f, mixture=S)

# ---------------------------------------------------------------- тесты
def kupiec_pof(viol, alpha):
    n = len(viol); x = int(viol.sum())
    if x == 0 or x == n:
        return np.nan, np.nan
    pi = x / n
    LR = -2 * (x * np.log(alpha) + (n - x) * np.log(1 - alpha)
               - x * np.log(pi) - (n - x) * np.log(1 - pi))
    return LR, 1 - chi2.cdf(LR, df=1)

def christoffersen_cc(viol, alpha):
    v = viol.astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(v)):
        n00 += (v[i-1] == 0) & (v[i] == 0); n01 += (v[i-1] == 0) & (v[i] == 1)
        n10 += (v[i-1] == 1) & (v[i] == 0); n11 += (v[i-1] == 1) & (v[i] == 1)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1) if (n10 + n11) > 0 else 0
    pi_u = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    LR_ind = (-2 * ((n00 + n10) * np.log(max(1 - pi_u, 1e-12)) + (n01 + n11) * np.log(max(pi_u, 1e-12)))
              + 2 * (n00 * np.log(max(1 - pi01, 1e-10)) + n01 * np.log(max(pi01, 1e-10))
                     + n10 * np.log(max(1 - pi11, 1e-10)) + n11 * np.log(max(pi11, 1e-10))))
    LR_pof, _ = kupiec_pof(viol, alpha)
    cc = (LR_pof if np.isfinite(LR_pof) else 0.0) + LR_ind
    return cc, 1 - chi2.cdf(cc, df=2), LR_ind, 1 - chi2.cdf(max(LR_ind, 0), df=1)

def basel_tl(n_violations, n_obs, alpha=0.01):
    from scipy.stats import binom
    cum = binom.cdf(n_violations, n_obs, alpha)
    return "Green" if cum < 0.95 else ("Yellow" if cum < 0.9999 else "Red")

def as_z(r, var975_, es_, alpha_es=0.025):
    excl = r < -var975_
    n_x = int(excl.sum())
    z1 = (-(r[excl] / es_[excl]).mean() - 1) if n_x else np.nan
    z2 = (-r * excl / es_).sum() / (len(r) * alpha_es) - 1
    return z1, z2, n_x

def fz0_loss(r, var975_, es_, alpha=0.025):
    v = -var975_; e = -es_
    ind = (r <= v).astype(float)
    return -ind * (v - r) / (alpha * e) + v / e + np.log(-e) - 1.0

def dm_test(dloss, lag=10):
    d = dloss[np.isfinite(dloss)]
    Tn = len(d); dbar = d.mean(); d0 = d - dbar
    s = d0 @ d0 / Tn
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (d0[:-l] @ d0[l:]) / Tn
    stat = dbar / np.sqrt(s / Tn)
    return stat, 2 * (1 - norm.cdf(abs(stat)))

rng = np.random.default_rng(2026)

def as_null_sims(md, nsim=1000):
    """Симуляция нулевого распределения Z1/Z2 при верной модели."""
    z1s = np.full(nsim, np.nan); z2s = np.full(nsim, np.nan)
    if "mixture" in md:
        Sm = md["mixture"]; D = Sm.shape[1]
        for m in range(nsim):
            pick = rng.integers(0, D, size=T)
            r = Sm[np.arange(T), pick] * rng.standard_normal(T)
            z1s[m], z2s[m], _ = as_z(r, md["VaR975"], md["ES"], TAIL_ES)
    else:
        scale = np.sqrt((md["nu"] - 2.0) / md["nu"]) * md["sd"]
        for m in range(nsim):
            r = scale * tdist.rvs(df=md["nu"], random_state=rng)
            z1s[m], z2s[m], _ = as_z(r, md["VaR975"], md["ES"], TAIL_ES)
    return z1s, z2s

records = []
fz_store = {}
for name, md in models.items():
    viol = pr < -md["VaR"]
    pof_LR, pof_p = kupiec_pof(viol, ALPHA)
    cc_LR, cc_p, _, ind_p = christoffersen_cc(viol, ALPHA)
    last250 = pr[-250:] < -md["VaR"][-250:]
    tl = basel_tl(int(last250.sum()), 250)
    csum = np.concatenate([[0], np.cumsum(viol.astype(int))])
    wins = csum[250:] - csum[:-250]
    z1, z2, n_x = as_z(pr, md["VaR975"], md["ES"], TAIL_ES)
    z1s, z2s = as_null_sims(md)
    fz = fz0_loss(pr, md["VaR975"], md["ES"], TAIL_ES)
    fz_store[name] = fz
    records.append({
        "model": name, "n_obs": T, "n_viol": int(viol.sum()),
        "viol_%": 100 * viol.mean(),
        "POF_LR": pof_LR, "POF_p": pof_p, "CC_p": cc_p, "Ind_p": ind_p,
        "Basel_TL_250d": tl,
        "TL_share_G": float((wins <= 4).mean()),
        "TL_share_Y": float(((wins >= 5) & (wins <= 9)).mean()),
        "TL_share_R": float((wins >= 10).mean()),
        "TL_max_exc": int(wins.max()), "n_tail975": n_x,
        "Z1": z1, "Z1_p": float(np.nanmean(z1s >= z1)) if np.isfinite(z1) else np.nan,
        "Z2": z2, "Z2_p": float(np.nanmean(z2s >= z2)),
        "FZ0_mean": float(np.nanmean(fz))})

BENCH = "M3_cDCC_t"
for rec in records:
    if rec["model"] == BENCH:
        rec["DM_FZ0_vs_cDCC"] = np.nan; rec["DM_p"] = np.nan
    else:
        rec["DM_FZ0_vs_cDCC"], rec["DM_p"] = dm_test(fz_store[rec["model"]] - fz_store[BENCH])

bt = pd.DataFrame(records)
bt.to_csv(TAB / "table30_oos_backtest.csv", index=False)
print("\n=== OOS backtest (expanding window, 2015-2026) ===")
print(bt.round(4).to_string(index=False))

# ---------------------------------------------------------------- подпериоды
periods = [
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("COVID", "2020-02-20", "2020-06-30"),
    ("Inter", "2020-07-01", "2022-02-20"),
    ("2022 shock", "2022-02-21", "2022-12-30"),
    ("Recent", "2023-01-01", "2026-06-30"),
]
sub = []
for name, md in models.items():
    for pname, a, b in periods:
        sel = (oos_idx >= a) & (oos_idx <= b)
        if sel.sum() < 10:
            continue
        v = pr[sel] < -md["VaR"][sel]
        sub.append({"model": name, "period": pname, "n_obs": int(sel.sum()),
                    "n_viol": int(v.sum()), "viol_%": 100 * v.mean(),
                    "avg_VaR_%": 100 * md["VaR"][sel].mean()})
period_bt = pd.DataFrame(sub)
period_bt.to_csv(TAB / "table30b_oos_periods.csv", index=False)
print("\n=== OOS period-wise violations ===")
print(period_bt.round(3).to_string(index=False))

# ---------------------------------------------------------------- капитал
# Базельское правило IMA с надбавкой светофора («плюс-фактором»):
#   K_t = max(VaR_{t-1}, (m_c + plus_t) * avg60),
# где plus_t определяется числом пробоев VaR(99%) за скользящие 250 торговых
# дней до t-1 включительно: 0–4: +0,00; 5: +0,40; 6: +0,50; 7: +0,65;
# 8: +0,75; 9: +0,85; >=10: +1,00 (BCBS, Supervisory framework 1996).
# Прежняя (небазельская) форма mult*max(VaR, avg60) сохранена контрольными
# колонками *_legacy.
PLUS_FACTOR = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}

def capital_path_basel(var_arr, viol_arr, m_base=3.0, lookback=60, window=250):
    n = len(var_arr)
    K = np.zeros(n)
    cs = np.concatenate([[0], np.cumsum(viol_arr.astype(int))])
    for t in range(lookback, n):
        n_viol = int(cs[t] - cs[max(0, t - window)])
        plus = 0.0 if n_viol <= 4 else PLUS_FACTOR.get(n_viol, 1.00)
        K[t] = max(var_arr[t - 1],
                   (m_base + plus) * var_arr[max(0, t - lookback):t].mean())
    K[:lookback] = K[lookback]
    return K

def capital_path_legacy(var_arr, mult=3.0, lookback=60):
    """Контрольная (прежняя) форма: mult * max(VaR_{t-1}, avg60)."""
    n = len(var_arr)
    K = np.zeros(n)
    for t in range(lookback, n):
        K[t] = mult * max(var_arr[t - 1], var_arr[max(0, t - lookback):t].mean())
    K[:lookback] = K[lookback]
    return K

# Рекалибровка покрытия FMSV: множитель c* подбирается минимальным, при
# котором число пробоев VaR(99%) не превышает лучшую модель DCC-семейства
# (GDCC). Подбор ведётся на том же оценочном периоде, поэтому строка -
# верхняя граница возможностей модели, а не прогнозный результат
# (оговорка в § 3.7.4).
viol_target = int((pr < -models["M4_GDCC_t"]["VaR"]).sum())
lo_c, hi_c = 1.0, 1.5
for _ in range(60):
    mid = 0.5 * (lo_c + hi_c)
    if int((pr < -mid * models["M5_FMSV"]["VaR"]).sum()) > viol_target:
        lo_c = mid
    else:
        hi_c = mid
c_star = hi_c
n_recal = int((pr < -c_star * models["M5_FMSV"]["VaR"]).sum())
print(f"\nРекалибровка FMSV: c* = {c_star:.4f} -> {n_recal} пробоев (цель {viol_target})")

cap_rows = []
cap_paths = {}
entries = [(name, md["VaR"]) for name, md in models.items()]
entries.append(("M5_FMSV_recal", c_star * models["M5_FMSV"]["VaR"]))
for name, var_arr in entries:
    viol_arr = pr < -var_arr
    Kp = capital_path_basel(var_arr, viol_arr) * NOTIONAL
    Kl = capital_path_legacy(var_arr) * NOTIONAL
    cap_paths[name] = Kp
    cap_rows.append({"model": name,
                     "n_viol": int(viol_arr.sum()),
                     "avg_K_mln_RUB": Kp.mean() / 1e6,
                     "max_K_mln_RUB": Kp.max() / 1e6,
                     "annual_carry_cost_mln_RUB": Kp.mean() * WACC / 1e6,
                     "avg_K_legacy_mln_RUB": Kl.mean() / 1e6,
                     "max_K_legacy_mln_RUB": Kl.max() / 1e6,
                     "c_star": c_star if name == "M5_FMSV_recal" else np.nan})
cap = pd.DataFrame(cap_rows)
cap.to_csv(TAB / "table30c_oos_capital.csv", index=False)
print("\n=== OOS capital summary (Basel + plus-factor; *_legacy - контроль) ===")
print(cap.round(2).to_string(index=False))

# дневные пути капитала — вход панели (г) сводного рисунка (08b)
pd.DataFrame(cap_paths, index=oos_idx).to_csv(TAB / "table30d_oos_capital_paths.csv")

# ---------------------------------------------------------------- рисунок
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
colors = {"M0_EWMA": "#777", "M1_DCC_t": "#003049", "M2_ADCC_t": "#669bbc",
          "M3_cDCC_t": "#d62828", "M4_GDCC_t": "#f77f00", "M5_FMSV": "#003f88"}
for name, md in models.items():
    axes[0].plot(oos_idx, 100 * md["VaR"], color=colors.get(name, "k"), lw=0.7, label=name)
axes[0].set_ylabel("VaR(99%, 1d), %")
axes[0].set_title("Вневыборочный прогноз VaR(99%, 1d): расширяющееся окно, переоценка без будущих данных")
axes[0].legend(loc="upper right", ncol=3, frameon=False)
for name in models:
    axes[1].plot(oos_idx, cap_paths[name] / 1e9, color=colors.get(name, "k"), lw=0.7, label=name)
axes[1].set_ylabel("Капитал, млрд руб.")
axes[1].set_title("Требуемый капитал по вневыборочным прогнозам "
                  "(нотионал 10 млрд руб., базельское правило с плюс-фактором)")
axes[1].legend(loc="upper right", ncol=3, frameon=False)
for ax in axes:
    for vs, ve, c in [("2020-02-20", "2020-06-30", "purple"),
                      ("2022-02-21", "2022-06-30", "orange")]:
        ax.axvspan(pd.Timestamp(vs), pd.Timestamp(ve), alpha=0.1, color=c)
plt.tight_layout()
plt.savefig(FIG / "fig6_var_capital_full.png", dpi=200)
plt.close()
print("\nfig6_var_capital_full.png (OOS) saved. DONE-21")
