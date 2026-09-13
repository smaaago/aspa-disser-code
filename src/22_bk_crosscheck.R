## Сверка частотной декомпозиции Баруника—Кржехлика с авторской эталонной
## реализацией — пакетом frequencyConnectedness (§ 3.4.4).
##
## На тех же логарифмах условных волатильностей глобальной панели
## сопоставляются:
##   (i)  совокупная связность DY на длинном горизонте (n.ahead = 200) —
##        контроль VAR/GFEVD-части конвейера;
##   (ii) частотные полосы «длинная (циклы > 10 дней) / короткая» с границей
##        pi/5 при спектральной аппроксимации n.ahead = 1000 — контроль
##        частотной части; наши значения берутся из table9_bk_*.csv,
##        посчитанных fevd_bk (common.py) на сетке ngrid = 5000.
##
## Выход: output/tables/table9d_bk_crosscheck.csv.
## Запуск из корня: Rscript src/22_bk_crosscheck.R  (~1-2 мин)
suppressMessages({
  library(frequencyConnectedness)
  library(vars)
})

vol <- read.csv("output/cond_vol.csv")
y <- log(pmax(as.matrix(vol[, -1]), 1e-6))
est <- VAR(y, p = 2, type = "const")

## (i) совокупная связность DY на длинном горизонте
dy200_pkg <- overall(spilloverDY12(est, n.ahead = 200, no.corr = FALSE))[[1]]

## (ii) частотные полосы: partition в убывающих границах, первая полоса —
## короткая (pi/5, pi], вторая — длинная [0, pi/5]; нижняя граница строго 0,
## иначе нулевая фурье-частота (основная масса спектра персистентных рядов)
## выпадает из длинной полосы
bk <- spilloverBK12(est, n.ahead = 1000, no.corr = FALSE,
                    partition = c(pi + 1e-6, pi / 5, 0))
ov <- overall(bk)
short_pkg <- ov[[1]]
long_pkg <- ov[[2]]

## Наши значения: связность полосы из сохранённых матриц (после ngrid = 5000)
band_tci <- function(path) {
  m <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
  m <- m[, colnames(m) != "FROM", drop = FALSE]
  (sum(m) - sum(diag(m))) / nrow(m)
}
long_own <- band_tci("output/tables/table9_bk_Long_(gt10d).csv")
short_own <- band_tci("output/tables/table9_bk_Short_(1-5d).csv")

hor <- read.csv("output/tables/table9b_tci_horizon.csv")
dy200_own <- as.numeric(hor$TCI[hor$H == "200"])

out <- data.frame(
  metric = c("DY_total_H200", "BK_long_band", "BK_short_band"),
  own = c(dy200_own, long_own, short_own),
  package = c(dy200_pkg, long_pkg, short_pkg))
out$abs_diff <- abs(out$own - out$package)
write.csv(out, "output/tables/table9d_bk_crosscheck.csv", row.names = FALSE)
print(out, digits = 6)
cat(sprintf("\nМаксимальное расхождение: %.4f п.п.\n", max(out$abs_diff)))
cat("DONE-22\n")
