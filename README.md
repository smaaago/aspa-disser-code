# Вычислительный комплекс диссертации

Полный воспроизводимый код эмпирической части работы «Моделирование
кросс-биржевых эффектов перелива волатильности на примере российского фондового
рынка». Каждое число, таблица и рисунок диссертации получены прогоном этих
скриптов на данных каталога `data/`.

## Окружение

| Компонент | Версия | Пакеты |
|---|---|---|
| Python | 3.10 | `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `arch`, `matplotlib`, `yfinance`, `requests` (`requirements.txt`) |
| R | 4.5 | `factorstochvol`, `stochvol`, `rugarch`, `rmgarch`, `FinTS`, `frequencyConnectedness`, `vars` (сверка BK) |

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Все пути относительные, запуск — **из корня проекта**:
`.venv/bin/python src/<скрипт>.py`, `Rscript src/<скрипт>.R`.

## Порядок прогона

Скрипты исполняются в порядке нумерации имён; выход каждого этапа служит входом
следующего (сцепка этапов § 1.3 диссертации).

```bash
# рабочая копия глобальной панели для конвейера (update_data.py делает это
# автоматически при обновлении; для запуска на приложенных снимках — вручную):
mkdir -p output && cp data/log_rets.csv output/log_rets_clean.csv

.venv/bin/python src/update_data.py           # выгрузка/продление панелей (необязательно)

# глобальный уровень
.venv/bin/python src/02_descriptive.py
.venv/bin/python src/03_univariate_garch.py
.venv/bin/python src/04_dcc_adcc.py
.venv/bin/python src/04b_gdcc_clustered.py
.venv/bin/python src/05_connectedness.py
Rscript              src/22_bk_crosscheck.R   # сверка полос BK с frequencyConnectedness, ~2 мин
Rscript              src/06_fmsv.R            # MCMC, ~6–8 мин
Rscript              src/06d_fmsv_cov.R       # ковариационный тензор FMSV
.venv/bin/python src/06c_fmsv_plots.py
.venv/bin/python src/08c_fmsv_paths.py

# локальный уровень (зеркально)
.venv/bin/python src/11_moex_descriptive.py
.venv/bin/python src/09_moex_univariate.py
.venv/bin/python src/09b_moex_dcc_gdcc.py
.venv/bin/python src/09c_moex_connectedness.py

# единая система двух уровней (индекс RI, § 3.6.5)
.venv/bin/python src/18_ri_index.py

# прикладной блок: полновыборочная диагностика + вневыборочный бэктест
.venv/bin/python src/07b_var_es_full.py       # диагностика подгонки (Прил. Б.6)
.venv/bin/python src/19_oos_forecasts.py      # OOS-прогнозы GARCH/DCC, ~15–30 мин (Pool)
for i in 0 1 2 3; do Rscript src/20_fmsv_oos.R $i 4 & done; wait   # OOS-FMSV, ~20 мин
.venv/bin/python src/21_oos_backtest.py       # вневыборочный бэктест и капитал

# сводные рисунки, робастность
.venv/bin/python src/08_hero_figures.py
.venv/bin/python src/08b_hero_v2.py
.venv/bin/python src/10_ch1_figures.py
.venv/bin/python src/12_imputation_robustness.py
.venv/bin/python src/13_rtsi_imoex_fx.py
.venv/bin/python src/14_commodities.py        # ~3–5 мин
.venv/bin/python src/15_spread_break_tests.py # + поправка размерности спреда
.venv/bin/python src/16_tvp_var_connectedness.py
Rscript              src/17_fmsv_subsamples.R # FMSV до/после 2022 + K-чувствительность (Прил. Б.5, Б.7; ~15 мин)
```

## Карта «скрипт → результат»

Номера таблиц и рисунков — печатные, как в тексте диссертации.

| Скрипт | Этап методики | Печатные таблицы | Печатные рисунки |
|---|---|---|---|
| `common.py` | общие компоненты: подпериодная сетка, VAR, GFEVD, полосы Баруника–Кржехлика, агрегаты связности | — | — |
| `update_data.py` | формирование панелей (Yahoo Finance + MOEX ISS) | — | — |
| `02_descriptive.py` | дескриптивный анализ глобальной панели | 3, 4 | 4, 5 |
| `11_moex_descriptive.py` | зеркальная диагностика локальной панели | 5 | — |
| `03_univariate_garch.py` | этап 1: GARCH/GJR/EGARCH-skew-$t$ | 6, 7 | — |
| `04_dcc_adcc.py` | этап 2: DCC / ADCC / cDCC | 8, Б.1 | 7, 8 |
| `04b_gdcc_clustered.py` | этап 2: кластерно-регуляризованная GDCC | 8, Б.2, Б.3 | 6 |
| `05_connectedness.py` | этап 3: связность DY + полосы BK, сходимость по горизонту, граница полосы | 9, 10 | 11 |
| `22_bk_crosscheck.R` | этап 3: сверка полос BK с эталонной реализацией `frequencyConnectedness` | — | — |
| `06_fmsv.R`, `06d_fmsv_cov.R` | этап 4: байесовская FMSV, ковариационный тензор | 11 | — |
| `06c_fmsv_plots.py`, `08c_fmsv_paths.py` | визуализация FMSV | — | 12, 13 |
| `09_moex_univariate.py` | этап 5: одномерный этап локального уровня | — | — |
| `09b_moex_dcc_gdcc.py` | этап 5: корреляционные модели секторов | 12, Б.2, Б.3 | 14 |
| `09c_moex_connectedness.py` | этап 5: связность секторов и мост двух уровней | 13, 14, Б.4 | 15, 16 |
| `18_ri_index.py` | этап 5: единая система двух уровней, индексы RI и RI_avg, гранулярность внешнего блока (§ 3.6.5) | 15, Б.8 | 17 |
| `07b_var_es_full.py` | этап 6: полновыборочная диагностика моделей риска | Б.6 | 19 |
| `19_oos_forecasts.py`, `20_fmsv_oos.R` | этап 6: вневыборочные прогнозы (расширяющееся окно) | — | — |
| `21_oos_backtest.py` | этап 6: вневыборочный бэктест VaR/ES и капитал | 16, 17, 18 | 18 |
| `08_hero_figures.py`, `08b_hero_v2.py` | сводные рисунки | — | 9, 10, 19 |
| `10_ch1_figures.py` | рисунки Главы 1 | — | 1, 2, 3 |
| `12_imputation_robustness.py` | робастность § 3.8.1: EM-импутация пропусков | 19 | 20 |
| `13_rtsi_imoex_fx.py` | робастность § 3.8.2: пара (RTSI, IMOEX) | 20 | 21 |
| `14_commodities.py` | робастность § 3.8.3: товарные узлы Brent и золота | 21 | 22 |
| `15_spread_break_tests.py` | робастность § 3.8.4: тесты режимного сдвига + поправка размерности | 22 | 23 |
| `16_tvp_var_connectedness.py` | робастность § 3.8.5: TVP-VAR без скользящего окна | 23 | 24 |
| `17_fmsv_subsamples.R` | робастность § 3.5: FMSV до и после разрыва 2022 г. + чувствительность к числу факторов $K$ | Б.5, Б.7 | — |

## Данные

| Файл | Содержимое |
|---|---|
| `data/log_rets.csv` | глобальная панель: лог-доходности 15 индексов, 19.09.2007 — 30.06.2026, $T = 4893$ |
| `data/moex_log_rets.csv` | локальная панель: отраслевые индексы Мосбиржи + IMOEX + RTSI |
| `data/commodity_log_rets.csv` | Brent (BZ=F) и золото (GC=F) |
| `data/usdrub.csv` | курс USD/RUB (санитарная сверка валютного фактора) |
| `data/index_composition_snapshot.csv` | составы и веса индексных корзин MOEX ISS |

Промежуточные выходы (`output/cond_vol.csv`, `output/std_residuals.csv`,
`output/Sigma_fmsv.rds` и прочие) создаются скриптами и в репозиторий не
включаются; итоговые таблицы и рисунки создаются прогоном в `output/tables`
и `output/figures`.

## Замечания о воспроизводимости

- Оценивание FMSV байесовское; `set.seed(2026)` зафиксирован в R-скриптах, но
  точные постериорные средние могут отличаться в последних знаках при смене
  версии `factorstochvol`.
- Численная оптимизация правдоподобия GARCH-моделей (`arch`) даёт расхождения
  порядка $10^{-5}$ между прогонами; на печатную точность (1–2 знака) это не
  влияет.
- Вневыборочный блок FMSV (`20_fmsv_oos.R`) контролирует сходимость каждой
  цепи (доля розыгрышей с аномальным хвостом предиктивного распределения) и
  при необходимости перезапускает переоценку с новым зерном и удлинённым
  прогревом; зерно детерминировано датой переоценки, результат воспроизводим.
- Скрипт `update_data.py` обращается к внешним источникам; при недоступности
  Yahoo Finance или MOEX ISS расчёты воспроизводятся на приложенных снимках
  данных без обновления.
