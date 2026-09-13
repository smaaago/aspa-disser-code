## Этап 4 (продолжение): условная ковариационная матрица FMSV для прикладного
## блока VaR/ES (§ 2.3.2, § 3.7). Модель переоценивается с сохранением
## постериорных траекторий (runningstore); по постериорным средним нагрузок
## и лог-волатильностей строится тензор
##   Sigma_t = Lambda diag(exp(h_f,t)) Lambda' + diag(exp(h_eps,t)).
## Выход: output/Sigma_fmsv.rds (вход 07b), траектории table12/table13
## (входы 08c и 17), нагрузки table10* (печатная Таблица 11), СКО RTSI table13b.

suppressMessages({
  library(factorstochvol)
})

set.seed(2026)
df <- read.csv("output/log_rets_clean.csv")
df$Datetime <- as.Date(df$Datetime)
ret <- as.matrix(df[, -1]) * 100  # процентная шкала, как на всех этапах
nms <- colnames(ret)
T_full <- nrow(ret); N <- ncol(ret)

# Окно остановки Мосбиржи (25.02.2022 — 24.03.2022) исключается из оценочной
# выборки; вектор дат сохраняется для согласования временного индекса
# с остальным конвейером
freeze <- df$Datetime >= as.Date("2022-02-25") & df$Datetime <= as.Date("2022-03-24")

K <- 3

cat("Fitting FMSV K=", K, "with runningstore (cov + logvar) ...\n")
fit <- fsvsample(ret[!freeze, ],
                 factors = K, draws = 3000, burnin = 1000, thin = 2,
                 zeromean = TRUE, runningstore = 6,
                 runningstoremoments = 2)
cat("Done.\n")

# Диагностический вывод состава runningstore
cat("runningstore names: "); print(names(fit$runningstore))
for (n in names(fit$runningstore)) {
  obj <- fit$runningstore[[n]]
  if (is.array(obj)) cat(n, ":", paste(dim(obj), collapse = "x"), "\n")
}

# factorstochvol хранит подразумеваемые ковариации в runningstore$cov
# (размерность T_eff x N x N x 2: среднее и SD по розыгрышам)
if (!is.null(fit$runningstore$cov)) {
  cov_obj <- fit$runningstore$cov
  cat("cov dim:", paste(dim(cov_obj), collapse = "x"), "\n")
  cov_mean <- cov_obj
}

# Постериорные средние латентных лог-дисперсий (активы + факторы)
if (!is.null(fit$runningstore$logvar)) {
  lv <- fit$runningstore$logvar
  cat("logvar dim:", paste(dim(lv), collapse = "x"), "\n")
}

# Нагрузки: постериорные среднее и SD (печатная Таблица 11)
load_mean <- apply(fit$facload, c(1, 2), mean)
load_sd <- apply(fit$facload, c(1, 2), sd)
rownames(load_mean) <- nms; colnames(load_mean) <- paste0("F", 1:K)
rownames(load_sd) <- nms; colnames(load_sd) <- paste0("F", 1:K)
write.csv(load_mean, "output/tables/table10_fmsv_loadings_mean.csv")
write.csv(load_sd, "output/tables/table10b_fmsv_loadings_sd.csv")

# ---- Построение траектории условных ковариаций ----
# Sigma_t = Lambda diag(exp(h_f,t)) Lambda' + diag(exp(h_eps,t));
# runningstore$logvar имеет форму T x (N+K) x статистики
lv <- fit$runningstore$logvar
if (is.array(lv) && length(dim(lv)) == 3) {
  # первый срез измерения статистик — постериорное среднее по розыгрышам
  lv_mean <- lv[, , 1]
  T_eff <- nrow(lv_mean)
  # лог-дисперсии: первые N столбцов — идиосинкратические (активы), следующие K — факторные
  asset_h <- lv_mean[, 1:N]
  fac_h <- lv_mean[, (N + 1):(N + K)]
  colnames(asset_h) <- nms
  colnames(fac_h) <- paste0("F", 1:K)
  dt <- df$Datetime[!freeze]
  write.csv(data.frame(date = dt, asset_h),
            "output/tables/table12_fmsv_asset_logvol.csv",
            row.names = FALSE)
  write.csv(data.frame(date = dt, fac_h),
            "output/tables/table13_fmsv_factor_logvol.csv",
            row.names = FALSE)

  # Тензор Sigma_t на каждую дату; сохраняется в бинарном виде для Python (07b)
  Sigma_arr <- array(0, dim = c(T_eff, N, N))
  Lam <- load_mean  # постериорное среднее нагрузок
  for (t in 1:T_eff) {
    Df <- diag(exp(fac_h[t, ]))
    De <- diag(exp(asset_h[t, ]))
    Sigma_arr[t, , ] <- Lam %*% Df %*% t(Lam) + De
  }
  # Возврат к исходной шкале доходностей (данные оценивались в шкале ×100)
  Sigma_arr <- Sigma_arr / 10000

  save_path <- "output/Sigma_fmsv.rds"
  saveRDS(list(Sigma = Sigma_arr, dates = dt, names = nms), save_path)
  cat("Sigma_fmsv tensor saved to", save_path, "shape", T_eff, N, N, "\n")

  # Подразумеваемое условное СКО RTSI — вход графиков 08c
  rtsi_idx <- which(nms == "rtsi")
  sd_rtsi <- sqrt(Sigma_arr[, rtsi_idx, rtsi_idx])
  write.csv(data.frame(date = dt, sd_rtsi = sd_rtsi),
            "output/tables/table13b_fmsv_rtsi_sd.csv",
            row.names = FALSE)
} else {
  cat("logvar runningstore not a 3D array, skipping covariance construction\n")
}

cat("\nDone.\n")
