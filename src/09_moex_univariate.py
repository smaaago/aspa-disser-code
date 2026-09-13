"""Этап 5 (локальный уровень), шаг 1: одномерные модели секторов (§ 3.6.1).

Панель: 7 отраслевых индексов Московской биржи (нефтегаз, финансы, металлы
и добыча, потребительский, электроэнергетика, химия, транспорт). Не включены:
MOEXTL (прекращён Мосбиржей 20.03.2026), MOEXIT и MOEXRE (история только
с 2020 г. — не покрывают кризисные эпизоды, критерий отбора § 1.3). Окно:
с первой общей торговой даты (январь 2008 г., задана появлением MOEXTN)
по 30.06.2026.

Зеркально повторяет 03_univariate_garch.py: GARCH/GJR/EGARCH(1,1)-skew-t
для каждого сектора, выбор по BIC; выходы — table17*/table17b* и
std_residuals_moex.csv / cond_vol_moex.csv (входы шагов 5.2–5.3).
"""
from pathlib import Path
import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data/moex_log_rets.csv")
TAB = Path("output/tables")
TAB.mkdir(parents=True, exist_ok=True)

SECTORS = ["moexog", "moexfn", "moexmm", "moexcn", "moexeu", "moexch", "moextn"]

df = pd.read_csv(DATA, parse_dates=["Datetime"]).set_index("Datetime")[SECTORS]
df = df.dropna()  # сбалансированная панель: от появления MOEXTN до 30.06.2026
print("Sectoral panel:", df.shape, "|", df.index.min().date(), "->", df.index.max().date())
print("Exact zeros per column:", (df == 0).sum().to_dict())

SCALE = 100.0
records, sel_records = [], []
all_residuals, all_sigma = {}, {}

for c in df.columns:
    series = (df[c] * SCALE).rename(c)
    fits = {}
    for label, vol_kwargs in [
        ("GARCH",  dict(vol="GARCH", p=1, q=1, o=0)),
        ("GJR",    dict(vol="GARCH", p=1, q=1, o=1)),
        ("EGARCH", dict(vol="EGARCH", p=1, q=1, o=1)),
    ]:
        am = arch_model(series, mean="Constant", dist="skewt", **vol_kwargs)
        try:
            fits[label] = am.fit(disp="off", show_warning=False)
        except Exception as e:
            print(f"{c} {label} failed: {e}")
    best_label = min(fits, key=lambda l: fits[l].bic)
    sel_records.append({"series": c, "best_spec": best_label,
                        "best_BIC": fits[best_label].bic,
                        **{f"BIC_{l}": fits[l].bic for l in fits}})
    res = fits[best_label]
    p = res.params
    a, b, g = (float(p.get(k, np.nan)) for k in ("alpha[1]", "beta[1]", "gamma[1]"))
    pers = a + b + (0.5 * g if best_label == "GJR" else 0.0)
    records.append({"series": c, "best_spec": best_label,
                    "omega": float(p.get("omega", np.nan)), "alpha": a, "beta": b,
                    "gamma": g, "persistence": pers,
                    "nu": float(p.get("nu", np.nan)),
                    "xi": float(p.get("eta", p.get("lambda", np.nan))),
                    "loglik": res.loglikelihood, "bic": res.bic, "aic": res.aic})
    all_residuals[c] = res.std_resid.values
    all_sigma[c] = res.conditional_volatility.values / SCALE

pd.DataFrame(sel_records).set_index("series").to_csv(TAB / "table17_moex_garch_sel.csv")
params_tbl = pd.DataFrame(records).set_index("series")
params_tbl.to_csv(TAB / "table17b_moex_garch_params.csv")
print("\n=== Sectoral GARCH selection (BIC) ===")
print(params_tbl[["best_spec", "alpha", "beta", "gamma", "persistence", "nu"]].round(4))

pd.DataFrame(all_residuals, index=df.index).to_csv("output/std_residuals_moex.csv")
pd.DataFrame(all_sigma, index=df.index).to_csv("output/cond_vol_moex.csv")
print("\nResiduals/vol files written (moex).")
