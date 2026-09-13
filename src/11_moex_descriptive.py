"""Зеркальная дескриптивная диагностика локальной (отраслевой) панели для Главы 2.

Та же выборка, что в 09_moex_univariate.py (7 секторов, сбалансированная панель
после dropna: 09.01.2008 – 30.06.2026, T = 4616), тот же набор статистик, что
в 02_descriptive.py для глобальной панели: описательные моменты + ADF, KPSS,
ARCH-LM(10), Ljung-Box(20) для квадратов. Выход: output/tables/table23_moex_desc.csv.
Запуск из корня: .venv/bin/python src/11_moex_descriptive.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore")

DATA = Path("data/moex_log_rets.csv")
TAB = Path("output/tables")
TAB.mkdir(parents=True, exist_ok=True)

SECTORS = ["moexog", "moexfn", "moexmm", "moexcn", "moexeu", "moexch", "moextn"]
NICE = {"moexog": "Нефть и газ", "moexfn": "Финансы", "moexmm": "Металлы и добыча",
        "moexcn": "Потребительский", "moexeu": "Электроэнергетика",
        "moexch": "Химия и нефтехимия", "moextn": "Транспорт"}

df = pd.read_csv(DATA, parse_dates=["Datetime"]).set_index("Datetime")[SECTORS].dropna()
print("Panel:", df.shape, "|", df.index.min().date(), "->", df.index.max().date())

rows = []
for c in SECTORS:
    x = df[c].values
    adf = adfuller(x, autolag="AIC")[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = kpss(x, regression="c", nlags="auto")[0]
    arch = het_arch(x, nlags=10)[0]
    lb2 = acorr_ljungbox(x ** 2, lags=[20], return_df=True)["lb_stat"].values[0]
    rows.append({
        "sector": NICE[c], "mean": np.mean(x), "std": np.std(x, ddof=1),
        "min": np.min(x), "skew": pd.Series(x).skew(), "kurtosis": pd.Series(x).kurtosis(),
        "ADF": adf, "KPSS": kp, "ARCH-LM": arch, "LB-squares": lb2,
    })

out = pd.DataFrame(rows).set_index("sector")
out.to_csv(TAB / "table23_moex_desc.csv")
print(out.round(3).to_string())
