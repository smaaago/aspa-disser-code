## Устойчивость факторной структуры FMSV к разрыву 2022 г. (§ 3.5).
##
## Полновыборочные нагрузки — статические характеристики всей выборки
## 2007–2026 гг., и сами по себе они не доказывают сохранение экспозиции
## российского рынка на общие факторы ПОСЛЕ 2022 г.: высокая нагрузка может
## почти целиком определяться доразрывным периодом. Здесь модель оценивается
## раздельно на трёх окнах: (i) до 21.02.2022; (ii) после возобновления торгов
## 25.03.2022 — 30.06.2026; (iii) устоявшийся послешоковый режим 2023–2026 гг.
##
## Два выхода:
## 1) ротационно-инвариантная метрика — доля дисперсии RTSI, объясняемая
##    общими факторами (не зависит от перестановок/знаков факторов и потому
##    сопоставима между независимыми MCMC-прогонами);
## 2) постериорные нагрузки подвыборок, сопоставленные с полновыборочными
##    факторами по максимуму |корреляции| векторов нагрузок с выравниванием
##    знака (перестановочная неопределённость решается явно, а не ярлыками).
##
## Конфигурация MCMC — как в основном прогоне 06d (K = 3, 3000+1000, thin 2).
## Выход: output/tables/table29_fmsv_subsample_loadings.csv,
##        output/tables/table29b_fmsv_subsample_shares.csv
## Запуск из корня: Rscript src/17_fmsv_subsamples.R  (~3-5 мин)

suppressMessages(library(factorstochvol))

df <- read.csv("output/log_rets_clean.csv")
df$Datetime <- as.Date(df$Datetime)
ret <- as.matrix(df[, -1]) * 100
nms <- colnames(ret)
K <- 3
freeze <- df$Datetime >= as.Date("2022-02-25") & df$Datetime <= as.Date("2022-03-24")

## Полновыборочные референсы (из основного прогона 06d)
lam_full <- as.matrix(read.csv("output/tables/table10_fmsv_loadings_mean.csv", row.names = 1))
asset_h_full <- read.csv("output/tables/table12_fmsv_asset_logvol.csv")
fac_h_full <- read.csv("output/tables/table13_fmsv_factor_logvol.csv")

rtsi_share <- function(lam, fac_h, asset_h_rtsi) {
  ## Доля дисперсии RTSI, объясняемая общими факторами, по дням; вход —
  ## постериорные средние нагрузок и лог-волатильностей (plug-in описание).
  lr <- lam["rtsi", ]
  common <- as.matrix(exp(fac_h)) %*% (lr^2)
  common / (common + exp(asset_h_rtsi))
}

share_full <- rtsi_share(lam_full, fac_h_full[, -1], asset_h_full$rtsi)
dates_full <- as.Date(fac_h_full$date)

subsamples <- list(
  pre2022    = df$Datetime <= as.Date("2022-02-21"),
  post2022   = df$Datetime >= as.Date("2022-03-25"),
  recent2023 = df$Datetime >= as.Date("2023-01-01")
)

match_factors <- function(lam_sub, lam_ref) {
  ## Перестановка и знак: максимум суммы |corr| по 6 перестановкам K = 3.
  perms <- rbind(c(1,2,3), c(1,3,2), c(2,1,3), c(2,3,1), c(3,1,2), c(3,2,1))
  cc <- cor(lam_ref, lam_sub)             # K x K
  best <- perms[which.max(apply(perms, 1, function(p) sum(abs(cc[cbind(1:3, p)])))), ]
  out <- lam_sub[, best]
  for (k in 1:3) if (cc[k, best[k]] < 0) out[, k] <- -out[, k]
  colnames(out) <- colnames(lam_ref)
  out
}

load_rows <- list(); share_rows <- list()
share_rows[["full"]] <- data.frame(
  subsample = "full (2007-2026)", T_eff = length(share_full),
  rtsi_factor_share = mean(share_full),
  rtsi_factor_share_2023plus = mean(share_full[dates_full >= as.Date("2023-01-01")]))

for (nm in names(subsamples)) {
  sel <- subsamples[[nm]] & !freeze
  sub <- ret[sel, , drop = FALSE]
  cat(sprintf("--- %s: T=%d (%s .. %s)\n", nm, nrow(sub),
              min(df$Datetime[sel]), max(df$Datetime[sel])))
  fit <- NULL
  for (attempt in 0:7) {
    set.seed(2026 + 31 * attempt)
    fit <- tryCatch(
      fsvsample(sub, factors = K, draws = 3000, burnin = 1000, thin = 2,
                zeromean = TRUE, runningstore = 6, runningstoremoments = 2,
                quiet = TRUE),
      error = function(e) { cat("  retry:", conditionMessage(e), "\n"); NULL })
    if (!is.null(fit)) break
  }
  if (is.null(fit)) stop("fsvsample failed: ", nm)

  lam_mean <- apply(fit$facload, c(1, 2), mean)
  lam_sd <- apply(fit$facload, c(1, 2), sd)
  rownames(lam_mean) <- rownames(lam_sd) <- nms

  lv <- fit$runningstore$logvar[, , 1]
  asset_h <- lv[, 1:length(nms)]; fac_h <- lv[, (length(nms) + 1):(length(nms) + K)]
  colnames(asset_h) <- nms

  ## Инвариантная метрика — до сопоставления факторов
  sh <- rtsi_share(lam_mean, fac_h, asset_h[, "rtsi"])
  share_rows[[nm]] <- data.frame(subsample = nm, T_eff = nrow(sub),
                                 rtsi_factor_share = mean(sh),
                                 rtsi_factor_share_2023plus = NA)

  ## Сопоставленные нагрузки (та же перестановка и знак — для SD)
  perms <- rbind(c(1,2,3), c(1,3,2), c(2,1,3), c(2,3,1), c(3,1,2), c(3,2,1))
  cc <- cor(lam_full, lam_mean)
  best <- perms[which.max(apply(perms, 1, function(p) sum(abs(cc[cbind(1:3, p)])))), ]
  lam_m <- lam_mean[, best]; lam_s <- lam_sd[, best]
  for (k in 1:3) if (cc[k, best[k]] < 0) lam_m[, k] <- -lam_m[, k]
  colnames(lam_m) <- colnames(lam_s) <- colnames(lam_full)
  cat(sprintf("  matched perm: %s; match |corr|: %s\n", paste(best, collapse = ""),
              paste(sprintf("%.2f", abs(cc[cbind(1:3, best)])), collapse = " ")))

  d <- data.frame(subsample = nm, market = nms, lam_m, check.names = FALSE)
  d[paste0(colnames(lam_full), "_sd")] <- lam_s
  d$match_abscorr <- paste(sprintf("%.3f", abs(cc[cbind(1:3, best)])), collapse = "; ")
  load_rows[[nm]] <- d
}

loadings_out <- do.call(rbind, load_rows)
write.csv(loadings_out, "output/tables/table29_fmsv_subsample_loadings.csv", row.names = FALSE)
shares_out <- do.call(rbind, share_rows)
write.csv(shares_out, "output/tables/table29b_fmsv_subsample_shares.csv", row.names = FALSE)
print(shares_out, digits = 3)
cat("\nRTSI loadings, matched (mean):\n")
rt <- loadings_out[loadings_out$market == "rtsi", ]
for (i in seq_len(nrow(rt)))
  cat(sprintf("  %-11s F1=%6.3f F2=%6.3f F3=%6.3f\n", rt$subsample[i],
              rt[i, 3], rt[i, 4], rt[i, 5]))
## ---- Чувствительность к числу факторов K (Прил. Б.5) ------------------------
## Модель переоценивается при K = 2, 3, 4, 5 раздельно до и после разрыва
## 2022 г. Доля общих факторов в дисперсии RTSI сопровождается приближённым
## 95%-м постериорным интервалом: доля пересчитывается по каждому розыгрышу
## нагрузок при постериорно-средних траекториях лог-волатильностей (полный
## пер-розыгрышный расчёт с траекториями оставлен направлением развития, § 3.10).
share_ci <- function(fit, Kk) {
  lv <- fit$runningstore$logvar[, , 1]
  asset_h <- lv[, 1:length(nms)]
  fac_h <- lv[, (length(nms) + 1):(length(nms) + Kk), drop = FALSE]
  efac <- as.matrix(exp(fac_h))
  eres <- exp(asset_h[, which(nms == "rtsi")])
  ir <- which(nms == "rtsi")
  draws <- dim(fit$facload)[3]
  means <- numeric(draws)
  for (d in seq_len(draws)) {
    lr <- fit$facload[ir, , d]
    common <- efac %*% (lr^2)
    means[d] <- mean(common / (common + eres))
  }
  c(mean = mean(means),
    lo = unname(quantile(means, 0.025)),
    hi = unname(quantile(means, 0.975)))
}

k_rows <- list()
for (Kk in c(2, 3, 4, 5)) {
  for (nm in c("pre2022", "post2022")) {
    sel <- subsamples[[nm]] & !freeze
    sub <- ret[sel, , drop = FALSE]
    fit <- NULL
    for (attempt in 0:7) {
      set.seed(2026 + 1000 * Kk + 31 * attempt)   # детерминированное зерно на (K, попытка)
      fit <- tryCatch(
        fsvsample(sub, factors = Kk, draws = 3000, burnin = 1000, thin = 2,
                  zeromean = TRUE, runningstore = 6, runningstoremoments = 2,
                  quiet = TRUE),
        error = function(e) { cat("  retry:", conditionMessage(e), "\n"); NULL })
      if (!is.null(fit)) break
    }
    if (is.null(fit)) stop("fsvsample failed: K=", Kk, " ", nm)
    ci <- share_ci(fit, Kk)
    cat(sprintf("K=%d %-9s share=%7.3f%%  CI95=[%.3f; %.3f]%%\n", Kk, nm,
                100 * ci["mean"], 100 * ci["lo"], 100 * ci["hi"]))
    k_rows[[paste(Kk, nm)]] <- data.frame(
      K = Kk, subsample = nm, T_eff = nrow(sub),
      share_mean = ci["mean"], ci_lo = ci["lo"], ci_hi = ci["hi"])
  }
}
write.csv(do.call(rbind, k_rows),
          "output/tables/table29c_fmsv_k_sensitivity.csv", row.names = FALSE)

## ---- Модельно-свободный бенчмарк: R² RTSI по 3 главным компонентам ----------
## Регрессия доходности RTSI на первые три главные компоненты остальных
## 14 рынков (корреляционная PCA) — верхняя граница объяснимой общей динамики,
## не зависящая от спецификации факторной модели.
pca_rows <- list()
for (nm in names(subsamples)) {
  sel <- subsamples[[nm]] & !freeze
  sub <- ret[sel, , drop = FALSE]
  x <- sub[, setdiff(nms, "rtsi")]
  pc <- prcomp(x, scale. = TRUE)
  r2 <- summary(lm(sub[, "rtsi"] ~ pc$x[, 1:3]))$r.squared
  pca_rows[[nm]] <- data.frame(subsample = nm, T_eff = nrow(sub), R2_3pc = r2)
  cat(sprintf("PCA-бенчмарк %-11s R2 = %.3f\n", nm, r2))
}
write.csv(do.call(rbind, pca_rows),
          "output/tables/table29d_pca_benchmark.csv", row.names = FALSE)

cat("DONE-17\n")
