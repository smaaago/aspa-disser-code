"""Полновыборочная диагностика моделей риска: M0 EWMA, M1 DCC-t, M2 ADCC-t,
M3 cDCC-t, M4 GDCC-spectral-t, M5 FMSV-implied.

Портфельная конвенция и метрики — по протоколу § 2.3.5 диссертации.
ВАЖНО О СТАТУСЕ: параметры всех моделей (и сглаженные траектории FMSV)
оценены на полной выборке, поэтому упражнение является диагностикой
качества подгонки при фиксированных параметрах, а НЕ вневыборочным
бэктестом; честный out-of-sample протокол реализован в 19-21.

Аппарат валидации:
  - Купик (POF) и Кристофферсен (CC, Independence) для VaR(99%);
  - базельский светофор с ОФИЦИАЛЬНЫМИ границами зон (0-4/5-9/10+ на 250
    наблюдений, кумулятивные пороги 95%/99,99%) — на последнем 250-дневном
    окне и скользящей последовательностью всех 250-дневных окон;
  - Ачерби-Секели Z1/Z2 для ES(97,5%) с хвостовым событием по VaR(97,5%)
    (уровень согласован с уровнем ES) и p-значениями из симулированного
    нулевого распределения при верной модели;
  - совместно элиситируемая потеря Фисслера-Циглера (FZ0) для пары
    (VaR 97,5%, ES 97,5%) и тест Диболда-Мариано против эталона cDCC.

If Sigma_fmsv.rds (FMSV implied covariance tensor) is present, M5 is included;
otherwise M5 is skipped with a warning printed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

OUT = Path("output")
TAB = OUT / "tables"
FIG = OUT / "figures"

ret = pd.read_csv(OUT / "log_rets_clean.csv", parse_dates=["Datetime"]).set_index("Datetime")
vol = pd.read_csv(OUT / "cond_vol.csv", parse_dates=["Datetime"]).set_index("Datetime")
res = pd.read_csv(OUT / "std_residuals.csv", parse_dates=["Datetime"]).set_index("Datetime")
ret = ret.reindex(res.index)
cols = list(vol.columns)
T = len(res); N = len(cols)
print(f"Aligned: T={T}, N={N}")

# ===== load tensors =====
Rs_dcc = np.load(OUT / "R_dcc.npy")
Rs_adcc = np.load(OUT / "R_adcc.npy")
Rs_cdcc = np.load(OUT / "R_cdcc.npy")
Rs_gdcc = np.load(OUT / "R_gdcc.npy")
sig_assets = vol.values  # (T, N)

# ===== Веса модельного портфеля (протокол § 2.3.5) =====
# 70% RTSI + 30% зарубежных позиций; до 21.02.2022 зарубежная часть —
# 15% S&P 500 + 15% DAX (докризисная диверсификация), после — 15% SSEC +
# 15% NIFTY 50 (доступная траектория репликации после закрытия западных площадок)
W = pd.DataFrame(0.0, index=ret.index, columns=cols)
switch = pd.Timestamp("2022-02-21")
pre = ret.index < switch; post = ~pre
W.loc[pre, "rtsi"] = 0.70; W.loc[pre, "spx"] = 0.15; W.loc[pre, "dax"] = 0.15
W.loc[post, "rtsi"] = 0.70; W.loc[post, "ssec"] = 0.15; W.loc[post, "nifty"] = 0.15
W = W.values

port_ret = (W * ret.values).sum(axis=1)

# ===== Per-model portfolio SD =====
def sd_from_R(R_tensor):
    sd = np.zeros(T)
    for t in range(T):
        w = W[t]; s = sig_assets[t]
        cov = (s[:, None] * R_tensor[t] * s[None, :])
        sd[t] = float(np.sqrt(w @ cov @ w))
    return sd

# EWMA
LAM = 0.94
sig2 = np.zeros(T); sig2[0] = port_ret[0] ** 2
for t in range(1, T):
    sig2[t] = LAM * sig2[t - 1] + (1 - LAM) * port_ret[t - 1] ** 2
sd_ewma = np.sqrt(sig2)

sd_dcc = sd_from_R(Rs_dcc)
sd_adcc = sd_from_R(Rs_adcc)
sd_cdcc = sd_from_R(Rs_cdcc)
sd_gdcc = sd_from_R(Rs_gdcc)

# ===== Try to load FMSV covariance tensor =====
sd_fmsv = None
fmsv_rds = OUT / "Sigma_fmsv.rds"
if fmsv_rds.exists():
    print("Loading FMSV covariance via rpy2 fallback ...")
    try:
        import subprocess
        # Use R to dump Sigma_fmsv to .npy via a temp file
        rscript = """
        x <- readRDS('output/Sigma_fmsv.rds')
        Sig <- x$Sigma
        d <- dim(Sig)
        cat(sprintf('SIGMA_DIM:%d %d %d\\n', d[1], d[2], d[3]))
        # Save flat binary
        arr <- as.vector(aperm(Sig, c(1,2,3)))
        writeBin(as.double(arr), 'output/Sigma_fmsv_flat.bin', size=8)
        cat('DATE_LEN:', length(x$dates), '\\n')
        write(as.character(x$dates), 'output/Sigma_fmsv_dates.txt')
        """
        r = subprocess.run(["Rscript", "-e", rscript], capture_output=True, text=True)
        print(r.stdout)
        if r.returncode == 0:
            d_line = [l for l in r.stdout.splitlines() if l.startswith("SIGMA_DIM")][0]
            T_eff, N1, N2 = map(int, d_line.split(":")[1].strip().split())
            print(f"FMSV Sigma loaded shape ({T_eff}, {N1}, {N2})")
            flat = np.fromfile(OUT / "Sigma_fmsv_flat.bin", dtype=np.float64)
            Sig_fmsv = flat.reshape((T_eff, N1, N2), order="F")
            dates_fmsv = pd.read_csv(OUT / "Sigma_fmsv_dates.txt", header=None,
                                     names=["d"])["d"].values
            dates_fmsv = pd.to_datetime(dates_fmsv)
            # Align FMSV to full T via dataframe
            sd_fmsv_partial = np.zeros(T_eff)
            for t in range(T_eff):
                w = W[ret.index.get_loc(dates_fmsv[t])]
                sd_fmsv_partial[t] = float(np.sqrt(w @ Sig_fmsv[t] @ w))
            # Map back to full T
            sd_fmsv = np.full(T, np.nan)
            for t, d in enumerate(dates_fmsv):
                if d in ret.index:
                    sd_fmsv[ret.index.get_loc(d)] = sd_fmsv_partial[t]
            # Forward-fill missing rtsi-closure window with prior value
            for t in range(1, T):
                if np.isnan(sd_fmsv[t]):
                    sd_fmsv[t] = sd_fmsv[t - 1] if not np.isnan(sd_fmsv[t - 1]) else 0
    except Exception as e:
        print(f"FMSV load failed: {e}")
        sd_fmsv = None
else:
    print("FMSV Sigma rds not found yet — M5 will be skipped.")

# ===== VaR / ES =====
ALPHA = 0.01; TAIL_ES = 0.025
z_emp = port_ret / sd_ewma
nu_t = max(4.5, tdist.fit(z_emp[~np.isnan(z_emp)])[0])
print(f"port_ret t-df = {nu_t:.2f}")

def t_es(alpha, nu):
    z = tdist.rvs(df=nu, size=300000, random_state=2026) * np.sqrt((nu - 2) / nu)
    var_q = np.quantile(z, alpha)
    es_q = z[z <= var_q].mean()
    return var_q, es_q

qs_var, _ = t_es(ALPHA, nu_t)
qs_var975, es_var_es = t_es(TAIL_ES, nu_t)

models = {
    "M0_EWMA": sd_ewma, "M1_DCC_t": sd_dcc, "M2_ADCC_t": sd_adcc,
    "M3_cDCC_t": sd_cdcc, "M4_GDCC_t": sd_gdcc,
}
if sd_fmsv is not None and not np.all(np.isnan(sd_fmsv)):
    models["M5_FMSV"] = sd_fmsv

# Build VaR and ES dictionaries
VaR = {k: -qs_var * sd for k, sd in models.items()}
VaR975 = {k: -qs_var975 * sd for k, sd in models.items()}
ES = {k: -es_var_es * sd for k, sd in models.items()}

# Backtest helpers
def kupiec_pof(viol, alpha):
    n = len(viol); x = int(viol.sum())
    if x == 0 or x == n:
        return np.nan, np.nan
    pi_hat = x / n
    LR = -2 * (x * np.log(alpha) + (n - x) * np.log(1 - alpha)
               - x * np.log(pi_hat) - (n - x) * np.log(1 - pi_hat))
    return LR, 1 - chi2.cdf(LR, df=1)

def christoffersen_cc(viol, alpha):
    v = viol.astype(int)
    n00=n01=n10=n11=0
    for i in range(1, len(v)):
        if v[i-1]==0 and v[i]==0: n00+=1
        if v[i-1]==0 and v[i]==1: n01+=1
        if v[i-1]==1 and v[i]==0: n10+=1
        if v[i-1]==1 and v[i]==1: n11+=1
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1) if (n10 + n11) > 0 else 0
    pi_u = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    LR_ind = (-2 * ((n00 + n10) * np.log(1 - pi_u) + (n01 + n11) * np.log(pi_u))
              + 2 * (n00 * np.log(max(1 - pi01, 1e-10)) + n01 * np.log(max(pi01, 1e-10))
                     + n10 * np.log(max(1 - pi11, 1e-10)) + n11 * np.log(max(pi11, 1e-10))))
    LR_pof, _ = kupiec_pof(viol, alpha)
    return LR_pof + LR_ind, 1 - chi2.cdf(LR_pof + LR_ind, df=2), \
           LR_ind, 1 - chi2.cdf(max(LR_ind, 0), df=1)

def basel_tl(n_violations, n_obs, alpha=0.01):
    """Официальные зоны базельского светофора (MAR32.36): границы задаются
    кумулятивной биномиальной вероятностью 95% и 99,99%, что для 250
    наблюдений даёт 0-4 исключения — зелёная зона, 5-9 — жёлтая, 10+ —
    красная."""
    from scipy.stats import binom
    cum = binom.cdf(n_violations, n_obs, alpha)
    if cum < 0.95:
        return "Green"
    if cum < 0.9999:
        return "Yellow"
    return "Red"

def as_z(port_ret, var975_, es_, alpha_es=0.025):
    """Статистики Ачерби-Секели: Z1 — условная по хвостовым событиям, Z2 —
    безусловная. Хвостовое событие — пробой VaR того же уровня 97,5%, на
    котором задан ES (несоответствие уровней ломает нулевое ожидание Z2)."""
    excl = port_ret < -var975_
    n_x = int(excl.sum())
    z1 = (-(port_ret[excl] / es_[excl]).mean() - 1) if n_x else np.nan
    z2 = (-port_ret * excl / es_).sum() / (len(port_ret) * alpha_es) - 1
    return z1, z2, n_x

def as_null_sims(sd_arr, var975_, es_, nu, alpha_es=0.025, nsim=1000, seed=2026):
    """Нулевое распределение Z1/Z2 при верной модели: r*_t = sd_t * z*_t,
    z* ~ t_nu (единичная дисперсия). Возвращает массивы симулированных Z."""
    rng = np.random.default_rng(seed)
    Tn = len(sd_arr)
    scale = np.sqrt((nu - 2) / nu)
    z1s = np.full(nsim, np.nan)
    z2s = np.full(nsim, np.nan)
    for m in range(nsim):
        r = sd_arr * scale * rng.standard_t(nu, size=Tn)
        excl = r < -var975_
        if excl.any():
            z1s[m] = -(r[excl] / es_[excl]).mean() - 1
        z2s[m] = (-r * excl / es_).sum() / (Tn * alpha_es) - 1
    return z1s, z2s

def fz0_loss(ret_arr, var975_, es_, alpha=0.025):
    """Совместно элиситируемая потеря Фисслера-Циглера в нулевой
    параметризации Паттона-Циглера-Чена (FZ0) для пары (VaR, ES)."""
    v = -var975_
    e = -es_
    ind = (ret_arr <= v).astype(float)
    return -ind * (v - ret_arr) / (alpha * e) + v / e + np.log(-e) - 1.0

def dm_test(dloss, lag=10):
    """Тест Диболда-Мариано с HAC-дисперсией Ньюи-Уэста."""
    from scipy.stats import norm as _norm
    d = dloss[np.isfinite(dloss)]
    Tn = len(d)
    dbar = d.mean()
    d0 = d - dbar
    s = d0 @ d0 / Tn
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (d0[:-l] @ d0[l:]) / Tn
    stat = dbar / np.sqrt(s / Tn)
    return stat, 2 * (1 - _norm.cdf(abs(stat)))

# ===== Run backtest =====
start = 30
pr = port_ret[start:]
records = []
fz_store = {}
roll_zone = {}
for name, sd in models.items():
    var_a = VaR[name][start:]
    var975_a = VaR975[name][start:]
    es_a = ES[name][start:]
    viol = pr < -var_a
    pof_LR, pof_p = kupiec_pof(viol, ALPHA)
    cc_LR, cc_p, ind_LR, ind_p = christoffersen_cc(viol, ALPHA)
    last250 = pr[-250:] < -var_a[-250:]
    tl = basel_tl(int(last250.sum()), 250, ALPHA)
    # скользящая последовательность 250-дневных окон (официальные границы 4/9)
    csum = np.concatenate([[0], np.cumsum(viol.astype(int))])
    wins = csum[250:] - csum[:-250]
    share_g = float((wins <= 4).mean())
    share_y = float(((wins >= 5) & (wins <= 9)).mean())
    share_r = float((wins >= 10).mean())
    roll_zone[name] = wins
    # Ачерби-Секели на согласованном уровне 97,5% + симуляционные p-значения
    z1, z2, n_x975 = as_z(pr, var975_a, es_a, TAIL_ES)
    z1s, z2s = as_null_sims(sd[start:], var975_a, es_a, nu_t, TAIL_ES)
    p_z1 = float(np.nanmean(z1s >= z1)) if np.isfinite(z1) else np.nan
    p_z2 = float(np.nanmean(z2s >= z2))
    fz = fz0_loss(pr, var975_a, es_a, TAIL_ES)
    fz_store[name] = fz
    records.append({"model": name,
                    "n_viol": int(viol.sum()), "n_obs": len(pr),
                    "viol_%": 100 * viol.mean(),
                    "POF_LR": pof_LR, "POF_p": pof_p,
                    "CC_LR": cc_LR, "CC_p": cc_p,
                    "Ind_p": ind_p,
                    "Basel_TL_250d": tl,
                    "TL_share_G": share_g, "TL_share_Y": share_y,
                    "TL_share_R": share_r, "TL_max_exc": int(wins.max()),
                    "n_tail975": n_x975,
                    "Z1": z1, "Z1_p": p_z1, "Z2": z2, "Z2_p": p_z2,
                    "FZ0_mean": float(np.nanmean(fz))})

BENCH = "M3_cDCC_t"
for rec in records:
    if rec["model"] == BENCH:
        rec["DM_FZ0_vs_cDCC"], rec["DM_p"] = np.nan, np.nan
    else:
        rec["DM_FZ0_vs_cDCC"], rec["DM_p"] = dm_test(fz_store[rec["model"]] - fz_store[BENCH])

bt = pd.DataFrame(records)
bt.to_csv(TAB / "table14_var_backtest_full.csv", index=False)
print("\n=== In-sample diagnostic backtest (fixed full-sample parameters) ===")
print(bt.round(4).to_string(index=False))

# ===== Sub-period =====
periods = [
    ("GFC", "2008-09-01", "2009-06-30"),
    ("Calm", "2009-07-01", "2019-12-31"),
    ("COVID", "2020-02-20", "2020-06-30"),
    ("Inter", "2020-07-01", "2022-02-20"),
    ("2022 shock", "2022-02-21", "2022-12-30"),
    ("Recent", "2023-01-01", "2026-06-30"),
]
sub_rows = []
idx = res.index[start:]
for name, sd in models.items():
    var_a = VaR[name][start:]
    for pname, a, b in periods:
        sel = (idx >= a) & (idx <= b)
        if sel.sum() < 10: continue
        pr_sub = pr[sel]; viol = pr_sub < -var_a[sel]
        sub_rows.append({"model": name, "period": pname,
                         "n_obs": int(sel.sum()),
                         "n_viol": int(viol.sum()),
                         "viol_%": 100 * viol.mean(),
                         "avg_VaR_%": 100 * var_a[sel].mean()})
period_bt = pd.DataFrame(sub_rows)
period_bt.to_csv(TAB / "table15_var_periods_full.csv", index=False)
print("\n=== Period-wise violations ===")
print(period_bt.round(3).to_string(index=False))

# ===== Capital =====
NOTIONAL = 10e9; WACC = 0.13
def capital_path(var_arr, mult=3.0, lookback=60):
    n = len(var_arr)
    K = np.zeros(n)
    for t in range(lookback, n):
        rolling = var_arr[max(0, t - lookback):t].mean()
        K[t] = mult * max(var_arr[t - 1], rolling)
    K[:lookback] = K[lookback]
    return K

cap_rows = []
cap_paths = {}
for name, _ in models.items():
    Kpath = capital_path(VaR[name]) * NOTIONAL
    cap_paths[name] = Kpath
    cap_rows.append({"model": name,
                     "avg_K_mln_RUB": Kpath.mean() / 1e6,
                     "max_K_mln_RUB": Kpath.max() / 1e6,
                     "delta_vs_M0_mln_RUB": (Kpath - cap_paths["M0_EWMA"]).mean() / 1e6
                          if "M0_EWMA" in cap_paths else 0,
                     "annual_carry_cost_mln_RUB": Kpath.mean() * WACC / 1e6})
cap_summary = pd.DataFrame(cap_rows)
cap_summary.to_csv(TAB / "table16_capital_summary_full.csv", index=False)
print("\n=== Capital summary ===")
print(cap_summary.round(2).to_string(index=False))

cap_df = pd.DataFrame(cap_paths, index=res.index)
cap_df.to_csv(TAB / "table16_capital_path_full.csv")

# ===== Plots =====
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
colors = {"M0_EWMA": "#777", "M1_DCC_t": "#003049",
          "M2_ADCC_t": "#669bbc", "M3_cDCC_t": "#d62828",
          "M4_GDCC_t": "#f77f00", "M5_FMSV": "#003f88"}
for name in models:
    sl = slice(start, None)
    axes[0].plot(res.index[sl], 100 * VaR[name][sl], color=colors.get(name, "k"),
                 lw=0.7, label=name)
axes[0].set_ylabel("VaR(99%, 1d), %")
axes[0].set_title("Прогноз VaR(99%, 1d) для портфеля по конкурирующим моделям")
axes[0].legend(loc="upper right", ncol=3, frameon=False)
for vs, ve, c in [("2008-09-15", "2009-06-30", "red"),
                  ("2020-02-20", "2020-06-30", "purple"),
                  ("2022-02-21", "2022-06-30", "orange")]:
    axes[0].axvspan(pd.Timestamp(vs), pd.Timestamp(ve), alpha=0.1, color=c)

for name in models:
    axes[1].plot(res.index, cap_paths[name] / 1e9, color=colors.get(name, "k"),
                 lw=0.7, label=name)
axes[1].set_ylabel("Капитал, млрд руб.")
axes[1].set_title("Требуемый капитал (нотионал 10 млрд руб., m_c = 3,0)")
axes[1].legend(loc="upper right", ncol=3, frameon=False)
for vs, ve, c in [("2008-09-15", "2009-06-30", "red"),
                  ("2020-02-20", "2020-06-30", "purple"),
                  ("2022-02-21", "2022-06-30", "orange")]:
    axes[1].axvspan(pd.Timestamp(vs), pd.Timestamp(ve), alpha=0.1, color=c)
plt.tight_layout()
plt.savefig(FIG / "fig6_var_capital_full.png", dpi=200)
plt.close()
print("Plot saved.")
