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
## Критерий приёмки цепи и перезапуски формализованы константами ниже;
## журнал всех попыток (включая принятые) пишется в
## output/oos_fmsv_retry_log_chunk<i>.csv — окно, дата, номер попытки, зерно,
## прогрев, исход, значение критерия.
##
## Запуск из корня, четырьмя параллельными потоками:
##   for i in 0 1 2 3; do Rscript src/20_fmsv_oos.R $i 4 & done; wait
## Выход: output/oos_fmsv_sdraws_chunk<i>.bin (+ .dates, .meta, retry-журнал)

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

## Критерий приёмки цепи: доля розыгрышей с дневным портфельным СКО выше
## SD_BLOWUP не превышает BLOWUP_SHARE_MAX. Порог — детектор численного
## взрыва цепи (нормальные значения СКО на этих данных на два порядка ниже
## 50%), цензурировать правдоподобные тяжёлые хвосты он не может. При
## нарушении — перезапуск с новым зерном и удлинённым прогревом, не более
## MAX_ATTEMPTS попыток; зерно детерминировано парой (окно, попытка).
SD_BLOWUP <- 0.5            # дневное портфельное СКО 50%
BLOWUP_SHARE_MAX <- 0.001   # допустимая доля таких розыгрышей
MAX_ATTEMPTS <- 8
BURN_STEP <- 500            # удлинение прогрева на каждую попытку

t0 <- max(which(df$Datetime <= as.Date("2014-12-31")))
refits <- seq(t0, Tn - 1, by = STEP)
mine <- refits[(seq_along(refits) - 1) %% nchunks == chunk]
cat(sprintf("chunk %d/%d: %d refits (of %d total), first %s\n",
            chunk, nchunks, length(mine), length(refits),
            as.character(df$Datetime[mine[1]])))

all_dates <- character(0)
all_sd <- NULL   # строки — прогнозные дни, столбцы — розыгрыши
retry_log <- list()

for (ri in seq_along(mine)) {
  t <- mine[ri]
  train <- ret[1:t, , drop = FALSE][!freeze[1:t], , drop = FALSE]
  h <- min(STEP, Tn - t)
  tt <- proc.time()
  ## Цепь изредка падает (Cholesky в full-conditional) или расходится, давая
  ## взорвавшиеся хвосты предиктивных дисперсий; обе ситуации лечатся
  ## перезапуском по критерию приёмки (константы выше).
  sd_block <- NULL
  for (attempt in 0:(MAX_ATTEMPTS - 1)) {
    seed <- 2026 + t + 100000 * attempt
    burn_a <- BURN + BURN_STEP * attempt
    set.seed(seed)
    frac_val <- NA_real_; fail_msg <- ""
    sd_try <- tryCatch({
      fit <- fsvsample(train, factors = K, draws = DRAWS,
                       burnin = burn_a,
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
      frac_val <<- mean(m > SD_BLOWUP)
      if (!is.finite(frac_val) || frac_val > BLOWUP_SHARE_MAX)
        stop(sprintf("diverged chain: frac(s > %.0f%%) = %.4f",
                     100 * SD_BLOWUP, frac_val))
      m
    }, error = function(e) {
      fail_msg <<- conditionMessage(e)
      cat(sprintf("  retry %d (t=%d): %s\n", attempt + 1, t, fail_msg))
      NULL
    })
    accepted <- !is.null(sd_try)
    retry_log[[length(retry_log) + 1]] <- data.frame(
      chunk = chunk, refit = ri - 1, t = t,
      refit_date = as.character(df$Datetime[t]),
      attempt = attempt + 1, seed = seed, burnin = burn_a,
      outcome = if (accepted) "accepted" else "retry",
      frac_blowup = frac_val,
      reason = if (accepted) "" else fail_msg)
    if (accepted) { sd_block <- sd_try; break }
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
write.csv(do.call(rbind, retry_log),
          sprintf("output/oos_fmsv_retry_log_chunk%d.csv", chunk),
          row.names = FALSE)
cat(sprintf("chunk %d done: %d prediction days saved, %d retry-log rows\n",
            chunk, nrow(all_sd), length(retry_log)))
