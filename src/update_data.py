"""Обновление данных проекта.

Два блока:
  world — продлевает data/log_rets.csv (15 мировых индексов) до --till.
          Источники: Yahoo Finance (14 индексов) + MOEX ISS (RTSI).
          Конвенция прежняя: лог-доходности на собственных торговых днях,
          объединённый календарь, нули в неторговые дни. Старые строки не
          трогаются — новые доклеиваются после последней даты.
  moex  — собирает data/moex_log_rets.csv: 10 отраслевых индексов Мосбиржи
          + IMOEX + RTSI (MOEX ISS, вся доступная история). Та же конвенция;
          до первой котировки индекса — NaN (не нули). Уровни (закрытия)
          сохраняются в data/moex_levels.csv.

Запуск из корня проекта:
    .venv/bin/python src/update_data.py            # оба блока
    .venv/bin/python src/update_data.py world      # только мировые
    .venv/bin/python src/update_data.py moex       # только Мосбиржа
    ... [--till 2026-06-30]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# колонка в log_rets.csv -> тикер Yahoo (rtsi берём из MOEX ISS)
YAHOO = {
    "asx": "^AXJO", "bel": "^BFX", "bist": "XU100.IS", "bovespa": "^BVSP",
    "cac": "^FCHI", "dax": "^GDAXI", "ftse": "^FTSE", "hsi": "^HSI",
    "nifty": "^NSEI", "nikkei": "^N225", "smi": "^SSMI", "spx": "^GSPC",
    "ssec": "000001.SS", "tsx": "^GSPTSE",
}

MOEX_SECTORAL = [
    "MOEXOG",  # нефть и газ
    "MOEXFN",  # финансы
    "MOEXMM",  # металлы и добыча
    "MOEXCN",  # потребительский
    "MOEXEU",  # электроэнергетика
    "MOEXTL",  # телекоммуникации
    "MOEXCH",  # химия
    "MOEXTN",  # транспорт
    "MOEXIT",  # IT (с 2020)
    "MOEXRE",  # строительство/недвижимость (с 2020)
    "IMOEX",   # широкий рынок, рубли
    "RTSI",    # широкий рынок, доллары
]

ISS = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/{sec}.json"


def moex_close(secid: str, frm: str | None = None, till: str | None = None) -> pd.Series:
    """Цены закрытия индекса с MOEX ISS (с постраничной выгрузкой)."""
    rows, start = [], 0
    while True:
        params = {"iss.meta": "off", "iss.only": "history",
                  "history.columns": "TRADEDATE,CLOSE", "start": start}
        if frm:
            params["from"] = frm
        if till:
            params["till"] = till
        for attempt in range(4):
            try:
                data = requests.get(ISS.format(sec=secid), params=params, timeout=60).json()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
        page = data["history"]["data"]
        if not page:
            break
        rows.extend(page)
        start += len(page)
    s = pd.Series({d: c for d, c in rows if c is not None}, name=secid.lower(), dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def to_panel(prices: pd.DataFrame, zero_fill_from_inception: bool = True) -> pd.DataFrame:
    """Лог-доходности на собственных днях -> объединённый календарь будних дней.

    Неторговый день = 0. При zero_fill_from_inception нули ставятся только
    между первой и последней котировкой колонки (вне этого окна — NaN;
    важно для прекращённых индексов, напр. MOEXTL с 20.03.2026).
    """
    rets = pd.DataFrame({c: np.log(prices[c].dropna()).diff() for c in prices})
    rets = rets[rets.index.dayofweek < 5]
    out = rets.copy()
    for c in out:
        fv, lv = out[c].first_valid_index(), out[c].last_valid_index()
        if fv is None:
            continue
        lo = fv if zero_fill_from_inception else out.index[0]
        out.loc[lo:lv, c] = out.loc[lo:lv, c].fillna(0.0)
    return out


def update_world(till: str) -> None:
    path = DATA / "log_rets.csv"
    old = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
    last = old.index.max()
    print(f"[world] сейчас: {old.shape}, до {last.date()}; цель: до {till}")
    if last >= pd.Timestamp(till):
        print("[world] уже актуально, пропускаю")
        return

    import yfinance as yf
    dl_from = last - pd.Timedelta(days=45)  # запас для стыковки и сверки
    px = yf.download(list(YAHOO.values()), start=dl_from.date(),
                     end=(pd.Timestamp(till) + pd.Timedelta(days=1)).date(),
                     progress=False, auto_adjust=False)["Close"]
    px = px.rename(columns={v: k for k, v in YAHOO.items()})
    px["rtsi"] = moex_close("RTSI", frm=str(dl_from.date()), till=till)
    missing = [c for c in old.columns if c not in px or px[c].dropna().empty]
    if missing:
        sys.exit(f"[world] нет данных по {missing} — стоп, файл не тронут")

    new = to_panel(px[old.columns], zero_fill_from_inception=False)

    # сверка источников на перекрытии (информативно, не блокирует)
    overlap = new.index.intersection(old.index)
    if len(overlap):
        diff = (new.loc[overlap] - old.loc[overlap]).abs().max()
        print(f"[world] сверка на перекрытии ({len(overlap)} дн.): "
              f"max|Δ| = {diff.max():.2e} ({diff.idxmax()})")
        bad = diff[diff > 5e-3]
        if len(bad):
            print(f"[world] ВНИМАНИЕ, расхождение источников > 0.005: {dict(bad.round(4))}")

    add = new.loc[new.index > last]
    stale = add.columns[add.isna().any()]
    if len(stale):
        print(f"[world] ВНИМАНИЕ: NaN в хвосте у {list(stale)} — источник отстаёт, "
              f"эти дни останутся NaN (перезапусти позже)")
    backup = DATA / f"log_rets_until_{last.date()}.csv"
    if not backup.exists():
        old.to_csv(backup)
    combined = pd.concat([old, add])
    combined.index.name = "Datetime"
    combined.to_csv(path)
    # рабочая копия панели для конвейера: оценочные скрипты глобального уровня
    # читают output/log_rets_clean.csv (см. README, «Порядок прогона»)
    out = ROOT / "output"
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "log_rets_clean.csv")
    print(f"[world] +{len(add)} строк -> {combined.shape}, "
          f"до {combined.index.max().date()}; бэкап: {backup.name}")


def update_moex(till: str) -> None:
    print(f"[moex] выгрузка {len(MOEX_SECTORAL)} индексов с MOEX ISS...")
    levels = pd.DataFrame({s.name: s for s in
                           (moex_close(sec, till=till) for sec in MOEX_SECTORAL)})
    for c in levels:
        s = levels[c].dropna()
        print(f"  {c:8s} {s.index.min().date()} -> {s.index.max().date()}  ({len(s)} набл.)")
    levels.index.name = "Datetime"
    levels.to_csv(DATA / "moex_levels.csv")

    rets = to_panel(levels)
    rets.index.name = "Datetime"
    rets.to_csv(DATA / "moex_log_rets.csv")
    print(f"[moex] сохранено: moex_levels.csv, moex_log_rets.csv {rets.shape}, "
          f"{rets.index.min().date()} -> {rets.index.max().date()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("block", nargs="?", choices=["world", "moex"], default=None,
                   help="какой блок обновить (по умолчанию оба)")
    p.add_argument("--till", default="2026-06-30", help="последняя дата данных")
    a = p.parse_args()
    if a.block in (None, "world"):
        update_world(a.till)
    if a.block in (None, "moex"):
        update_moex(a.till)
