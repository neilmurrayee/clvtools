#!/usr/bin/env Rscript
# Oracle fixtures for the Pareto/NBD with time-varying covariates (S3.3, S6.4.2).
#
# Kept separate from generate_fixtures.R because fitting this model takes over a
# minute; the rest of the suite should not pay for that on every re-baseline.
#
#     ./tools/setup_oracle.sh
#     R_LIBS=.Rlib Rscript tools/oracle/generate_dyncov_fixtures.R
#
# What is dumped, and why:
#
#   * the *walk* structures CLVTools builds from the transaction log and the
#     covariate time series -- flat covariate matrices plus per-customer index
#     tables. These are the hard part of the model, and having them lets the
#     Python walk construction be tested directly rather than only through the
#     likelihood it feeds.
#   * every intermediate quantity of the per-customer likelihood: A1T, AkT,
#     A1sum, B1, BT, Bjsum, Bksum, C1T, CkT, D1, DT, DkT, the F components and
#     the a/b arguments of the hypergeometrics. `pnbd_dyncov_LL_ind` returns all
#     thirty when asked, so each block of the likelihood can be held to the
#     oracle on its own instead of only the total.
#   * the same at a second, off-optimum parameter vector, so the likelihood is
#     testable before any optimiser runs.

suppressMessages({
  library(CLVTools)
  library(data.table)
})

OUT <- "tests/fixtures"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

ns <- asNamespace("CLVTools")
cpp <- function(name) get(name, envir = ns)

write_csv <- function(dt, name) {
  fwrite(dt, file.path(OUT, paste0(name, ".csv")), dateTimeAs = "ISO")
  cat(sprintf("  %-38s %7d rows\n", paste0(name, ".csv"), nrow(dt)))
}
write_json <- function(x, name) {
  enc <- function(v) {
    if (is.list(v)) {
      paste0("{", paste(sprintf('"%s": %s', names(v), vapply(v, enc, "")),
                        collapse = ", "), "}")
    } else if (is.character(v)) {
      if (length(v) == 1) sprintf('"%s"', v)
      else paste0("[", paste(sprintf('"%s"', v), collapse = ", "), "]")
    } else if (is.logical(v)) {
      if (length(v) == 1) tolower(as.character(v))
      else paste0("[", paste(tolower(as.character(v)), collapse = ", "), "]")
    } else {
      f <- format(v, digits = 17, scientific = FALSE, trim = TRUE)
      if (length(v) == 1) f else paste0("[", paste(f, collapse = ", "), "]")
    }
  }
  writeLines(enc(x), file.path(OUT, paste0(name, ".json")))
  cat(sprintf("  %-38s\n", paste0(name, ".json")))
}

NAMES_COV <- c("High.Season", "Gender", "Channel")

data("apparelTrans")
data("apparelDynCov")

clv <- clvdata(apparelTrans, date.format = "ymd", time.unit = "week",
               estimation.split = 104,
               name.id = "Id", name.date = "Date", name.price = "Price")
dyn <- SetDynamicCovariates(
  clv,
  data.cov.life = apparelDynCov, data.cov.trans = apparelDynCov,
  names.cov.life = NAMES_COV, names.cov.trans = NAMES_COV,
  name.id = "Id", name.date = "Cov.Date"
)

cat("\n== fitting (this takes a minute) ==\n")
fit <- latentAttrition(
  ~ High.Season + Gender + Channel | High.Season + Gender + Channel,
  family = pnbd, data = dyn, verbose = FALSE,
  optimx.args = list(hessian = FALSE)
)
cat("  LL =", as.numeric(logLik(fit)), "\n")

write_json(list(
  names.cov = NAMES_COV,
  coefficients = as.list(coef(fit)),
  logLik = as.numeric(logLik(fit)),
  nobs = nobs(fit)
), "dyncov_fit")

# -- the walk structures ------------------------------------------------------

cat("\n== walks ==\n")

args <- cpp("pnbd_dyncov_getLLcallargs_ind")(fit)

cbs <- copy(fit@cbs)
write_csv(cbs[, .(Id, x, t.x, T.cal, d_omega)], "dyncov_cbs")

# Per-customer index into the flat covariate vectors. 1-based, inclusive, as
# the C++ walk constructor reads them.
write_csv(data.table(
  Id = cbs$Id,
  aux_life_from  = args$walkinfo_aux_life[, 1],
  aux_life_to    = args$walkinfo_aux_life[, 2],
  real_life_from = args$walkinfo_real_life[, 1],
  real_life_to   = args$walkinfo_real_life[, 2],
  aux_trans_from = args$walkinfo_aux_trans[, 1],
  aux_trans_to   = args$walkinfo_aux_trans[, 2],
  aux_trans_d1   = args$walkinfo_aux_trans[, 3],
  aux_trans_tjk  = args$walkinfo_aux_trans[, 4],
  # Zero-repeaters have no real transaction walks and carry NA here.
  real_trans_from = args$walkinfo_trans_real_from,
  real_trans_to   = args$walkinfo_trans_real_to
), "dyncov_walkinfo")

dt.wrt <- as.data.table(args$walkinfo_real_trans)
setnames(dt.wrt, c("walk_from", "walk_to", "d1", "tjk"))
write_csv(dt.wrt, "dyncov_walkinfo_real_trans")

# The raw covariate matrices. exp(covdata %*% gamma) happens inside the LL, so
# these are the untransformed dummies and stay small.
for (nm in c("aux_life", "real_life", "aux_trans", "real_trans")) {
  m <- args[[paste0("covdata_", nm)]]
  dt <- as.data.table(m)
  setnames(dt, NAMES_COV)
  write_csv(dt, paste0("dyncov_covdata_", nm))
}

# -- the likelihood, decomposed ----------------------------------------------

cat("\n== likelihood ==\n")

fitted_params <- coef(fit)
grid <- list(
  mle = c(log(fitted_params[1:4]), fitted_params[5:10]),
  offset = c(log(c(1.0, 50.0, 1.0, 60.0)),
             c(0.2, -0.3, 0.4, 0.1, 0.25, -0.15))
)

for (nm in names(grid)) {
  ll.args <- args
  ll.args[["params"]] <- unname(grid[[nm]])
  ll.args[["return_intermediate_results"]] <- TRUE
  res <- do.call(cpp("pnbd_dyncov_LL_ind"), ll.args)
  dt <- data.table(Id = cbs$Id, as.data.table(res))
  write_csv(dt, paste0("dyncov_ll_", nm))
}

write_json(list(
  params = lapply(grid, function(p) as.numeric(unname(p))),
  param.names = c("log.r", "log.alpha", "log.s", "log.beta",
                  paste0("life.", NAMES_COV), paste0("trans.", NAMES_COV)),
  LL.sum = lapply(grid, function(p) {
    ll.args <- args
    ll.args[["params"]] <- unname(p)
    ll.args[["vN"]] <- rep(1, nrow(cbs))
    -as.numeric(do.call(cpp("pnbd_dyncov_LL_negsum"), ll.args))
  })
), "dyncov_ll_grid")

# The summed individual likelihood at the optimum must be logLik() of the fit.
ll.args <- args
ll.args[["params"]] <- unname(grid$mle)
ll.args[["return_intermediate_results"]] <- FALSE
total <- sum(do.call(cpp("pnbd_dyncov_LL_ind"), ll.args)[, 1])
if (!isTRUE(all.equal(total, as.numeric(logLik(fit)), tolerance = 1e-6))) {
  stop(sprintf("self-check failed: summed LL %.10g vs logLik %.10g",
               total, as.numeric(logLik(fit))))
}
cat(sprintf("  ok  %-34s %.6f\n", "dyncov LL at MLE", total))

cat("\nfixtures written to ", OUT, "\n", sep = "")

# -- prediction with time-varying covariates, S6.4.2 ---------------------------
#
# Three layers, so a failure localises:
#
#   * the ABCD table -- the per-period A_i, B_i, C_i, D_i built from the
#     covariates a customer is alive for, which CET and DECT are both sums over.
#     Dumped for a handful of customers only; it is one row per customer per
#     period and would otherwise be a hundred thousand rows.
#   * PAlive, CET and DECT for every customer, from the public predict().
#   * the paper's own three printed rows, which need a second fit: on the full
#     data, with the covariate series extended into the prediction window.

cat("\n== prediction (S6.4.2) ==\n")

SAMPLE_IDS <- c("1", "10", "100", "1000", "1001")

dt.palive <- cpp("pnbd_dyncov_palive")(fit)
write_csv(dt.palive, "dyncov_palive")

date.holdout.end <- clv@clv.time@timepoint.holdout.end
dt.abcd <- cpp("pnbd_dyncov_ABCD")(
  clv.fitted = fit, prediction.end.date = date.holdout.end)
write_csv(dt.abcd[Id %in% SAMPLE_IDS], "dyncov_abcd_sample")

dt.pred <- as.data.table(predict(fit, verbose = FALSE))
check.palive <- merge(dt.pred[, .(Id, PAlive)], dt.palive, by = "Id")
if (!isTRUE(all.equal(check.palive$PAlive, check.palive$palive, tolerance = 1e-10))) {
  stop("predict()'s PAlive disagrees with pnbd_dyncov_palive()")
}
cat("  ok  predict() PAlive matches pnbd_dyncov_palive()\n")
write_csv(dt.pred, "dyncov_predict_holdout")
write_json(list(
  prediction.end = as.character(date.holdout.end),
  period.length = dt.pred$period.length[1],
  continuous.discount.factor = log(1 + 0.1),
  sample.ids = SAMPLE_IDS
), "dyncov_predict_holdout_settings")

# -- the paper's table, S6.4.2 ------------------------------------------------

cat("\n== the paper's prediction table (this takes another minute) ==\n")

data("apparelDynCovFuture")
clv.full <- clvdata(apparelTrans, date.format = "ymd", time.unit = "week",
                    estimation.split = NULL,
                    name.id = "Id", name.date = "Date", name.price = "Price")
dyn.full <- SetDynamicCovariates(
  clv.full,
  data.cov.life = rbind(apparelDynCov, apparelDynCovFuture),
  data.cov.trans = rbind(apparelDynCov, apparelDynCovFuture),
  names.cov.life = NAMES_COV, names.cov.trans = NAMES_COV,
  name.id = "Id", name.date = "Cov.Date"
)
fit.full <- latentAttrition(
  ~ High.Season + Gender + Channel | High.Season + Gender + Channel,
  family = pnbd, data = dyn.full, verbose = FALSE,
  optimx.args = list(hessian = FALSE)
)
est.gg.full <- spending(family = gg, data = clv.full, verbose = FALSE)
cat("  LL =", as.numeric(logLik(fit.full)), "\n")

dt.pred.future <- as.data.table(predict(
  fit.full, predict.spending = est.gg.full, prediction.end = 95,
  continuous.discount.factor = log(1 + 0.075) / 52, verbose = FALSE))
write_csv(dt.pred.future, "dyncov_predict_future")
write_json(list(
  coefficients = as.list(coef(fit.full)),
  logLik = as.numeric(logLik(fit.full)),
  spending.coefficients = as.list(coef(est.gg.full)),
  prediction.periods = 95,
  continuous.discount.factor = log(1 + 0.075) / 52
), "dyncov_fit_full")

# The covariate series the prediction window needs, as a plain table: the
# Python side reads it from data/apparelDynCovFuture.csv.
write_json(list(
  n.rows.past = nrow(apparelDynCov),
  n.rows.future = nrow(apparelDynCovFuture),
  first.future.date = as.character(min(apparelDynCovFuture$Cov.Date)),
  last.future.date = as.character(max(apparelDynCovFuture$Cov.Date))
), "dyncov_future_covariates")

# -- a prospective customer with time-varying covariates, S6.3.4 --------------

cat("\n== newcustomer.dynamic() ==\n")

dt.nc.cov <- data.table(
  Cov.Date = seq(from = as.Date("2010-12-19"), by = "week", length.out = 15),
  High.Season = 1, Gender = 0, Channel = 1
)
nc.dyn <- as.numeric(predict(fit.full, newdata = newcustomer.dynamic(
  num.periods = 10,
  data.cov.life = dt.nc.cov, data.cov.trans = dt.nc.cov,
  first.transaction = as.Date("2010-12-21")
)))
write_json(list(
  num.periods = 10,
  first.transaction = "2010-12-21",
  cov.dates = as.character(dt.nc.cov$Cov.Date),
  High.Season = 1, Gender = 0, Channel = 1,
  expected.num.transactions = nc.dyn
), "dyncov_newcustomer")

# A horizon inside the customer's first covariate period, which takes the other
# branch of the expectation: there is no earlier period for the sum to
# telescope through.
nc.dyn.single <- as.numeric(predict(fit.full, newdata = newcustomer.dynamic(
  num.periods = 0.5,
  data.cov.life = dt.nc.cov, data.cov.trans = dt.nc.cov,
  first.transaction = as.Date("2010-12-21")
)))
write_json(list(
  num.periods = 0.5,
  first.transaction = "2010-12-21",
  expected.num.transactions = nc.dyn.single
), "dyncov_newcustomer_single_period")

cat("\ndone\n")
