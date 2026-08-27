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
