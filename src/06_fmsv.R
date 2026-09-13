## Этап 4: байесовская факторная модель стохастической волатильности (§ 3.5).
## Оценивание в реализации пакета factorstochvol (Кастнер—Хоссейни, ASIS-
## интерливинг); конфигурация протокола § 2.3.2: K = 3 фактора, 4000 розыгрышей
## после 1500 прогрева, прореживание 2.
## Выход: постериорные нагрузки (table10*, печатная Таблица 11), траектории
## факторов (table11*) и лог-волатильностей (table12/table13) — входы 06c/08c.

suppressMessages({
  library(factorstochvol)
})

DATA <- "output/log_rets_clean.csv"
OUT_TAB <- "output/tables"
OUT_FIG <- "output/figures"

set.seed(2026)

df <- read.csv(DATA)
df$Datetime <- as.Date(df$Datetime)
ret <- as.matrix(df[, -1]) * 100  # процентная шкала — численная устойчивость (как в 03)
T_obs <- nrow(ret); N <- ncol(ret)
nms <- colnames(ret)
cat("Returns matrix:", T_obs, "x", N, "\n")

# Окно остановки торгов на Московской бирже (25.02.2022 — 24.03.2022)
# исключается из оценочной выборки (§ 3.1, § 3.5)
freeze <- df$Datetime >= as.Date("2022-02-25") & df$Datetime <= as.Date("2022-03-24")
ret <- ret[!freeze, ]
T_obs <- nrow(ret)
cat("After removing 2022 freeze block:", T_obs, "x", N, "\n")

# Рабочая конфигурация протокола § 2.3.2
K <- 3
draws <- 4000
burn <- 1500

cat("Fitting FMSV with K=", K, " factors, draws=", draws, "...\n")
fit <- fsvsample(
  ret,
  factors = K,
  draws = draws,
  burnin = burn,
  thin = 2,
  zeromean = TRUE,
  runningstore = 6
)
cat("Done.\n")

# Постериорные среднее и SD факторных нагрузок (печатная Таблица 11)
load_arr <- fit$facload  # массив N x K x число розыгрышей
load_mean <- apply(load_arr, c(1, 2), mean)
load_sd   <- apply(load_arr, c(1, 2), sd)
rownames(load_mean) <- nms
colnames(load_mean) <- paste0("F", 1:K)
rownames(load_sd)   <- nms
colnames(load_sd)   <- paste0("F", 1:K)
cat("\n== Posterior mean of factor loadings ==\n")
print(round(load_mean, 3))

write.csv(load_mean, file.path(OUT_TAB, "table10_fmsv_loadings_mean.csv"))
write.csv(load_sd, file.path(OUT_TAB, "table10b_fmsv_loadings_sd.csv"))

# Латентные траектории факторов: постериорное среднее по розыгрышам
# (fit$fac — массив T x K x число розыгрышей)
if (!is.null(fit$fac)) {
  fac_mean <- apply(fit$fac, c(1, 2), mean)
  colnames(fac_mean) <- paste0("F", 1:K)
  fac_df <- data.frame(date = df$Datetime[!freeze], fac_mean)
  write.csv(fac_df, file.path(OUT_TAB, "table11_fmsv_factors_mean.csv"), row.names = FALSE)
}

# Латентные лог-волатильности: первые N столбцов — идиосинкратические
# (активы), следующие K — факторные
if (!is.null(fit$logvar)) {
  logvar_mean <- apply(fit$logvar, c(1, 2), mean)  # T x (N+K)
  fcoll <- ncol(logvar_mean)
  factor_logvar <- logvar_mean[, (N + 1):fcoll, drop = FALSE]
  asset_logvar <- logvar_mean[, 1:N, drop = FALSE]
  colnames(factor_logvar) <- paste0("F", 1:K)
  colnames(asset_logvar) <- nms
  write.csv(data.frame(date = df$Datetime[!freeze], factor_logvar),
            file.path(OUT_TAB, "table12_fmsv_factor_logvol.csv"), row.names = FALSE)
  write.csv(data.frame(date = df$Datetime[!freeze], asset_logvar),
            file.path(OUT_TAB, "table13_fmsv_asset_logvol.csv"), row.names = FALSE)
}

cat("Files written.\n")
