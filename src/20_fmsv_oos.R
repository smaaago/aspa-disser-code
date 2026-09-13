## Вневыборочный блок FMSV (§ 2.3.5, § 3.7): последовательная байесовская
## переоценка на расширяющемся окне и постериорно-предиктивная ковариация.
##
## Протокол: начальное окно 19.09.2007 — 31.12.2014; далее модель
## переоценивается каждые 10 торговых дней ТОЛЬКО на данных до даты переоценки
## (окно остановки торгов Мосбиржи 25.02–24.03.2022 исключается из оценивания,
## как и в основном прогоне 06d), после чего для каждого из следующих 10 дней
## строится постериорно-предиктивное распределение ковариационной матрицы
## (predcov: h-шаговый прогноз латентных лог-волатильностей по розыгрышам
## параметров и последних состояний). Сохраняются розыгрыши портфельного
## условного СКО s_t^{(m)} = sqrt(w_t' Sigma_t^{(m)} w_t) — из них прикладной
## этап (21_oos_backtest.py) строит VaR/ES как квантили смеси, то есть
## полноценный posterior predictive VaR без plug-in подстановки средних.
##
## Отличие от 06d принципиально: там ковариация дня t восстанавливалась из
## СГЛАЖЕННЫХ траекторий (использующих всю выборку, включая будущее), здесь
## каждый прогноз опирается только на прошлое. Число MCMC-итераций сокращено
## до 1500+500 (полный прогон одной переоценки ~10-15 с; их ~290).
##
## Запуск из корня, четырьмя параллельными потоками:
##   for i in 0 1 2 3; do Rscript src/20_fmsv_oos.R $i 4 & done; wait
## Выход: output/oos_fmsv_sdraws_chunk<i>.bin (+ .dates, .meta)

suppressMessages(library(factorstochvol))

args <- commandArgs(trailingOnly = TRUE)
chunk <- if (length(args) >= 1) as.integer(args[1]) else 0
nchunks <- if (length(args) >= 2) as.integer(args[2]) else 1

df <- read.csv("output/log_rets_clean.csv")
df$Datetime <- as.Date(df$Datetime)
ret <- as.matrix(df[, -1]) * 100          # процентная шкала, как в 06d
nms <- colnames(ret)
Tn <- nrow(ret); N <- ncol(ret)

freeze <- df$Datetime >= as.Date("2022-02-25") & df$Datetime <= as.Date("2022-03-24")

## Портфельные веса протокола § 2.3.5 (переключение 21.02.2022)
w_pre <- setNames(numeric(N), nms); w_pre[c("rtsi", "spx", "dax")] <- c(0.70, 0.15, 0.15)
w_post <- setNames(numeric(N), nms); w_post[c("rtsi", "ssec", "nifty")] <- c(0.70, 0.15, 0.15)
switch_date <- as.Date("2022-02-21")

K <- 3; DRAWS <- 1500; BURN <- 500; STEP <- 10

t0 <- max(which(df$Datetime <= as.Date("2014-12-31")))
refits <- seq(t0, Tn - 1, by = STEP)
mine <- refits[(seq_along(refits) - 1) %% nchunks == chunk]
cat(sprintf("chunk %d/%d: %d refits (of %d total), first %s\n",
            chunk, nchunks, length(mine), length(refits),
            as.character(df$Datetime[mine[1]])))

all_dates <- character(0)
all_sd <- NULL   # строки — прогнозные дни, столбцы — розыгрыши

for (ri in seq_along(mine)) {
  t <- mine[ri]
  train <- ret[1:t, , drop = FALSE][!freeze[1:t], , drop = FALSE]
  h <- min(STEP, Tn - t)
  tt <- proc.time()
  ## Цепь изредка падает (Cholesky в full-conditional) или расходится, давая
  ## взорвавшиеся хвосты предиктивных дисперсий; обе ситуации лечатся
  ## перезапуском с другим зерном и удлинённым прогревом. Критерий приёмки
  ## цепи: доля розыгрышей с дневным портфельным СКО выше 50% — не более
  ## 0,1% (нормальные значения на этих данных на два порядка ниже 50%).
  sd_block <- NULL
  for (attempt in 0:7) {
    set.seed(2026 + t + 100000 * attempt)
    sd_try <- tryCatch({
      fit <- fsvsample(train, factors = K, draws = DRAWS,
                       burnin = BURN + 500 * attempt,
                       zeromean = TRUE, quiet = TRUE)
      pc <- predcov(fit, ahead = 1:h)       # N x N x draws x h
      m <- matrix(NA_real_, nrow = h, ncol = DRAWS)
      for (j in 1:h) {
        d_j <- df$Datetime[t + j]
        w <- if (d_j < switch_date) w_pre else w_post
        ## w' Sigma w по розыгрышам; обратно к долям из процентной шкалы
        m[j, ] <- apply(pc[, , , j], 3,
                        function(S) sqrt(max(c(w %*% S %*% w), 0))) / 100
      }
      frac <- mean(m > 0.5)
      if (!is.finite(frac) || frac > 0.001)
        stop(sprintf("diverged chain: frac(s > 50%%) = %.4f", frac))
      m
    }, error = function(e) { cat(sprintf("  retry %d (t=%d): %s\n", attempt + 1, t, conditionMessage(e))); NULL })
    if (!is.null(sd_try)) { sd_block <- sd_try; break }
  }
  if (is.null(sd_block)) stop(sprintf("fsvsample failed after retries at t=%d", t))
  all_dates <- c(all_dates, as.character(df$Datetime[(t + 1):(t + h)]))
  all_sd <- rbind(all_sd, sd_block)
  cat(sprintf("REFIT %d/%d t=%d date=%s T_train=%d h=%d %.1fs\n",
              ri, length(mine), t, as.character(df$Datetime[t]),
              nrow(train), h, (proc.time() - tt)[3]))
}

base <- sprintf("output/oos_fmsv_sdraws_chunk%d", chunk)
writeBin(as.double(as.vector(all_sd)), paste0(base, ".bin"), size = 8)
writeLines(all_dates, paste0(base, ".dates"))
writeLines(c(sprintf("rows %d", nrow(all_sd)), sprintf("cols %d", ncol(all_sd)),
             "layout column-major (R as.vector)"), paste0(base, ".meta"))
cat(sprintf("chunk %d done: %d prediction days saved\n", chunk, nrow(all_sd)))
