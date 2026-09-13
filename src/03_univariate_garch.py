"""Этап 1 (глобальный уровень): одномерные модели условной волатильности (§ 3.2).

Для каждого индекса оцениваются GARCH(1,1), GJR-GARCH(1,1) и EGARCH(1,1)
со скошенным t-распределением; лучшая спецификация выбирается по BIC.
Выход: table3_best_garch.csv и table4_garch_params.csv (печатные Таблицы 6–7),
output/std_residuals.csv и output/cond_vol.csv — входы этапов 2–3 (сцепка § 1.3).
"""
from pathlib import Path
import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

DATA = Path("output/log_rets_clean.csv")
TAB = Path("output/tables")
TAB.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["Datetime"]).set_index("Datetime")

# Окно остановки торгов на Московской бирже (25.02.2022 — 24.03.2022) заполнено
# в панели нулями (правило протокола § 1.3). Контрольная серия rtsi_clean
# исключает это окно целиком: по ней оценивается альтернативная спецификация
# RTSI (примечание к печатной Таблице 7), подтверждающая устойчивость
# одномерных оценок к способу обработки эпизода остановки.
df["rtsi_clean"] = df["rtsi"].copy()
mask_freeze = (df.index >= "2022-02-25") & (df.index <= "2022-03-24")
df.loc[mask_freeze, "rtsi_clean"] = np.nan

records = []
sel_records = []
all_residuals = {}
all_sigma = {}

# Масштабирование доходностей на 100 (процентные пункты) — численная
# устойчивость оптимизатора правдоподобия (§ 3.2).
SCALE = 100.0

for c in df.columns:
    series = (df[c].dropna() * SCALE).rename(c)
    if series.empty:
        continue
    fits = {}
    for label, vol_kwargs in [
        ("GARCH",    dict(vol="GARCH", p=1, q=1, o=0)),
        ("GJR",      dict(vol="GARCH", p=1, q=1, o=1)),
        ("EGARCH",   dict(vol="EGARCH", p=1, q=1, o=1)),
    ]:
        am = arch_model(series, mean="Constant", dist="skewt", **vol_kwargs)
        try:
            res = am.fit(disp="off", show_warning=False)
            fits[label] = res
        except Exception as e:
            print(f"{c} {label} failed: {e}")
    best_label, best_bic = None, np.inf
    for lab, res in fits.items():
        if res.bic < best_bic:
            best_label, best_bic = lab, res.bic
    if best_label is None:
        continue
    sel_records.append({"series": c, "best_spec": best_label,
                        "best_BIC": best_bic,
                        **{f"BIC_{l}": fits[l].bic for l in fits}})
    res = fits[best_label]
    # Персистентность и хвостовые параметры лучшей спецификации
    params = res.params
    omega = float(params.get("omega", np.nan))
    a = float(params.get("alpha[1]", np.nan))
    b = float(params.get("beta[1]", np.nan))
    g = float(params.get("gamma[1]", np.nan))
    nu = float(params.get("nu", np.nan))
    xi = float(params.get("eta", params.get("lambda", np.nan)))  # скошенность skew-t
    pers = a + b + (0.5 * g if best_label == "GJR" else 0.0)
    records.append({"series": c, "best_spec": best_label,
                    "omega": omega, "alpha": a, "beta": b, "gamma": g,
                    "persistence": pers, "nu": nu, "xi": xi,
                    "loglik": res.loglikelihood, "bic": res.bic,
                    "aic": res.aic})
    all_residuals[c] = res.std_resid.values
    all_sigma[c] = res.conditional_volatility.values / SCALE

selection = pd.DataFrame(sel_records).set_index("series")
selection.to_csv(TAB / "table3_best_garch.csv")
params_tbl = pd.DataFrame(records).set_index("series")
params_tbl.to_csv(TAB / "table4_garch_params.csv")
print("=== Univariate GARCH selection by BIC ===")
print(selection)
print("\n=== Best-spec parameters ===")
print(params_tbl.round(4))

# Стандартизированные остатки и условные волатильности — входы этапов 2–3;
# контрольная rtsi_clean не сохраняется (другая длина ряда)
keep = [c for c in df.columns if c != "rtsi_clean"]
res_arr = {c: all_residuals[c] for c in keep if c in all_residuals}
sig_arr = {c: all_sigma[c] for c in keep if c in all_sigma}
n = len(next(iter(res_arr.values())))
res_df = pd.DataFrame(res_arr, index=df.index[-n:])
sig_df = pd.DataFrame(sig_arr, index=df.index[-n:])
res_df.to_csv("output/std_residuals.csv")
sig_df.to_csv("output/cond_vol.csv")
print("\nResiduals/vol files written.")
