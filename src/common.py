"""Общие компоненты вычислительного комплекса диссертации.

Модуль собирает то, что до рефакторинга дублировалось в скриптах 05, 09c, 12,
13, 14, 16: подпериодную сетку, оценивание VAR, обобщённую декомпозицию
дисперсии прогноза (GFEVD) по Песарану–Шину, её частотное разложение по
Барунику–Кржехлику и сборку таблицы направленных переливов Диболда–Йылмаза.

Все функции размерностно-агностичны (число узлов выводится из формы Sigma) и
применяются без изменений к обеим панелям работы — глобальной (15 индексов) и
локальной (7 отраслевых индексов Мосбиржи). Именно это и обеспечивает
зеркальность протокола двух уровней (§ 2.3.4 диссертации): один и тот же код,
разные данные.

Запуск скриптов — из корня проекта; пути относительные.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# --------------------------------------------------------------------- пути
ROOT = Path(".")
DATA = ROOT / "data"
OUT = ROOT / "output"
TAB = OUT / "tables"
FIG = OUT / "figures"

# ------------------------------------------------------- подпериодная сетка
# Внешняя (внемодельная) датировка системных эпизодов, § 1.3 диссертации.
# Границы едины для всех этапов анализа; отличается только стартовая дата
# докризисного окна: глобальная панель начинается 19.09.2007, локальная —
# 09.01.2008 (начало истории индекса транспорта).
PERIODS_GLOBAL = {
    "Pre-GFC (2007-09 to 2008-08)":     ("2007-09-19", "2008-08-31"),
    "GFC (2008-09 to 2009-06)":         ("2008-09-01", "2009-06-30"),
    "Calm (2009-07 to 2019-12)":        ("2009-07-01", "2019-12-31"),
    "COVID (2020-02 to 2020-06)":       ("2020-02-20", "2020-06-30"),
    "Inter (2020-07 to 2022-01)":       ("2020-07-01", "2022-01-31"),
    "2022 shock (2022-02 to 2022-12)":  ("2022-02-21", "2022-12-30"),
    "Recent (2023-01 to 2026-06)":      ("2023-01-01", "2026-06-30"),
}

PERIODS_LOCAL = {"Pre-GFC (2008-01 to 2008-08)": ("2008-01-01", "2008-08-31")}
PERIODS_LOCAL.update({k: v for k, v in PERIODS_GLOBAL.items()
                      if not k.startswith("Pre-GFC")})

# Краткие ярлыки — для скриптов робастности (12, 14, 16), где длинные имена
# столбцов таблиц неудобны.
PERIODS_SHORT = {
    "Pre-GFC": ("2007-09-19", "2008-08-31"),
    "GFC": ("2008-09-01", "2009-06-30"),
    "Calm": ("2009-07-01", "2019-12-31"),
    "COVID": ("2020-02-20", "2020-06-30"),
    "Inter": ("2020-07-01", "2022-01-31"),
    "2022 shock": ("2022-02-21", "2022-12-30"),
    "Recent": ("2023-01-01", "2026-06-30"),
}

# Кризисные окна для затенения на графиках.
SHADED = [("2008-09-15", "2009-06-30", "red"),
          ("2020-02-20", "2020-06-30", "purple"),
          ("2022-02-21", "2022-06-30", "orange")]

# Частотные полосы Баруника–Кржехлика (§ 2.3.3): короткая — циклы до 10 дней
# (преимущественно 1–5), длинная — свыше 10 дней.
BANDS = [(0.0, np.pi / 5), (np.pi / 5, np.pi)]
BAND_LABELS = {(0.0, np.pi / 5): "Long (>10d)", (np.pi / 5, np.pi): "Short (1-5d)"}


# ------------------------------------------------------------------- данные
def load_log_vol(path):
    """Условные волатильности -> логарифмы (стационарный вход VAR)."""
    vol = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
    return np.log(vol.clip(lower=1e-6))


# ------------------------------------------------------ VAR и связность DY
def fit_var(Y, p=2):
    """МНК-оценка VAR(p). Возвращает список матриц коэффициентов и Sigma_u."""
    res = VAR(np.asarray(Y)).fit(p)
    return list(np.asarray(res.coefs)), np.asarray(res.sigma_u)


def giir(B_list, H):
    """Матрицы скользящего среднего представления VAR (обобщённые IRF,
    Koop–Pesaran–Potter 1996, Pesaran–Shin 1998)."""
    n = B_list[0].shape[0]
    P = len(B_list)
    M = np.zeros((H, n, n))
    M[0] = np.eye(n)
    for h in range(1, H):
        s = np.zeros((n, n))
        for p in range(min(P, h)):
            s += B_list[p] @ M[h - 1 - p]
        M[h] = s
    return M


def fevd_generalised(B_list, Sigma, H=10):
    """Обобщённая декомпозиция дисперсии ошибки прогноза (Pesaran–Shin 1998;
    Diebold–Yilmaz 2012). Элемент (i, j) — доля H-шаговой дисперсии ошибки
    прогноза переменной i, атрибутируемая шокам переменной j; строки
    нормированы на единицу."""
    n = Sigma.shape[0]
    M = giir(B_list, H)
    sd = np.diag(Sigma).copy()
    sd[sd <= 0] = 1e-12
    num = np.zeros((n, n))
    den = np.zeros((n, n))
    for h in range(H):
        Mh = M[h]
        num += (Mh @ Sigma) ** 2 / sd[None, :]
        den += np.diag(Mh @ Sigma @ Mh.T)[:, None] * np.ones((1, n))
    theta = num / den
    return theta / theta.sum(axis=1, keepdims=True)


def fevd_bk(B_list, Sigma, bands=None, ngrid=200):
    """Частотное разложение обобщённой FEVD по Барунику–Кржехлику (2018).

    bands — список пар (lo, hi) в радианах. На каждой частоте вычисляется
    числитель обобщённого спектра каузальности
    num_{jk}(w) = sigma_kk^{-1} |(Psi(w) Sigma)_{jk}|^2; взвешивание частот
    по спектральной плотности Gamma_j(w) из статьи BK учтено точно — в
    произведении Gamma_j(w) f_{jk}(w) плотность (Psi Sigma Psi*)_{jj}
    сокращается, и вклад полосы сводится к интегралу num по полосе с
    построчной нормировкой на интеграл по всему спектру. Возвращает словарь
    полоса -> (N x N) матрица долей; сумма матриц по полосам даёт построчно
    нормированную безусловную (спектральную, H -> inf) таблицу GFEVD, а
    внедиагональная доля каждой полосы — её частотную связность C_d^F в
    терминах BK, аддитивную по полосам. Со спектральной таблицей сопоставима
    конечно-горизонтная таблица DY при H -> inf, поэтому сумма полос
    отличается от TCI при H = 10 на величину усечения горизонта."""
    bands = BANDS if bands is None else bands
    n = Sigma.shape[0]
    P = len(B_list)
    sd = np.diag(Sigma).copy()
    sd[sd <= 0] = 1e-12
    omegas = np.linspace(1e-6, np.pi - 1e-6, ngrid)
    acc = {b: np.zeros((n, n)) for b in bands}
    for w in omegas:
        A_inv = np.eye(n, dtype=complex)
        for p in range(P):
            A_inv -= B_list[p] * np.exp(-1j * (p + 1) * w)
        A = np.linalg.inv(A_inv)
        num = (np.abs(A @ Sigma) ** 2) / sd[None, :]
        for b in acc:
            lo, hi = b
            if lo <= w <= hi:
                acc[b] += num
    row_sum = sum(acc.values()).sum(axis=1, keepdims=True)
    return {b: acc[b] / row_sum for b in acc}


# ------------------------------------------------ агрегаты матрицы связности
def tci_of(theta):
    """Совокупный индекс связности системы, %."""
    n = theta.shape[0]
    return (theta.sum() - np.trace(theta)) / n * 100


def to_from_net(theta, i):
    """Передаваемая, принимаемая и чистая связность узла i, %."""
    to_ = (theta[:, i].sum() - theta[i, i]) * 100
    from_ = (theta[i, :].sum() - theta[i, i]) * 100
    return to_, from_, to_ - from_


def spillover_table(theta, labels):
    """Печатная матрица направленных переливов со строками TO и NET и
    столбцом FROM (проценты)."""
    df = pd.DataFrame(theta, index=labels, columns=labels) * 100
    df["FROM"] = df.sum(axis=1) - np.diag(df.values)
    to_others = df.iloc[:, :-1].sum(axis=0) - np.diag(df.iloc[:, :-1].values)
    df.loc["TO"] = list(to_others) + [to_others.sum()]
    df.loc["NET"] = list(to_others.values - df["FROM"].iloc[:-1].values) + [np.nan]
    return df


def rolling_connectedness(y, node=None, win=250, step=10, p=1, H=10):
    """Скользящая оценка TCI (и чистой позиции узла `node`, если задан).

    Конфигурация протокола § 2.3.3: окно 250 торговых дней, шаг 10, VAR(1)
    внутри окна, GFEVD на горизонте H = 10."""
    idx = list(y.columns).index(node) if node is not None else None
    rows, dates = [], []
    for end in range(win, len(y), step):
        sub = y.iloc[end - win:end]
        try:
            B, S = fit_var(sub, p=p)
            th = fevd_generalised(B, S, H=H)
        except Exception:
            continue
        rec = {"TCI": tci_of(th)}
        if idx is not None:
            rec["NET"] = to_from_net(th, idx)[2]
        rows.append(rec)
        dates.append(y.index[end - 1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))


def shade(ax):
    """Затенить кризисные эпизоды на оси."""
    for a, b, c in SHADED:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=0.15, color=c)
