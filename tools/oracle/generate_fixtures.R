#!/usr/bin/env Rscript
# Generate oracle fixtures from the R package CLVTools.
#
# The Python test suite is held to these files, not to a live R session: they
# are committed under tests/fixtures/ so that `uv run pytest` needs no R.
# Regenerate them only when deliberately re-baselining:
#
#     ./tools/setup_oracle.sh
#     R_LIBS=.Rlib Rscript tools/oracle/generate_fixtures.R
#
# The point of dumping per-customer vectors at *arbitrary* parameter vectors --
# not just at the MLE -- is that it makes each expression testable on its own,
# before the optimiser works. A likelihood can be verified against a fixture
# long before `fit()` exists to find its maximum.

suppressMessages({
  library(CLVTools)
  library(data.table)
})

OUT <- "tests/fixtures"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# CLVTools exposes its per-customer expressions as internal Rcpp entry points.
# Reaching for them with ::: is deliberate: the public interface only ever
# reports quantities at the fitted optimum, which is far too coarse to drive
# equation-level tests.
ns <- asNamespace("CLVTools")
cpp <- function(name) get(name, envir = ns)

# Conventions of those entry points, confirmed against src/pnbd.cpp and
# src/gg_LL.cpp, and each one re-checked below against a public generic:
#
#   * the four latent-attrition model parameters are passed on the LOG scale;
#     covariate parameters are passed on the natural scale
#   * `*_LL_sum` and `gg_LL` return the NEGATIVE sum -- they are objectives
#     handed to a minimiser, not log-likelihoods
#   * `vN` weights each row, because CLVTools collapses duplicate CBS rows
#   * static-covariate arguments are ordered (life, trans), and the parameter
#     vector is c(log model params, life params, trans params)
#
# Getting any of these wrong produces fixtures that are plausible and wrong,
# so `check` below asserts every fixture family against a public generic
# before anything is written.

check <- function(label, got, want, tol = 1e-6) {
  if (!isTRUE(all.equal(as.numeric(got), as.numeric(want), tolerance = tol))) {
    stop(sprintf("oracle self-check failed for %s: got %.10g, want %.10g",
                 label, as.numeric(got), as.numeric(want)), call. = FALSE)
  }
  cat(sprintf("  ok  %-34s %.6f\n", label, as.numeric(want)))
}

write_csv  <- function(dt, name) {
  fwrite(dt, file.path(OUT, paste0(name, ".csv")), dateTimeAs = "ISO")
  cat(sprintf("  %-38s %6d rows\n", paste0(name, ".csv"), nrow(dt)))
}
write_json <- function(x, name) {
  # Hand-rolled so the fixtures carry no dependency beyond base R + data.table.
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

# -- Provenance ---------------------------------------------------------------

write_json(list(
  clvtools.version = as.character(packageVersion("CLVTools")),
  r.version        = R.version.string,
  generated.by     = "tools/oracle/generate_fixtures.R"
), "_manifest")

# -- The case study data objects, S6.1 ----------------------------------------

data("apparelTrans")
data("apparelStaticCov")

clv.apparel <- clvdata(
  apparelTrans, date.format = "ymd", time.unit = "week",
  estimation.split = 104,
  name.id = "Id", name.date = "Date", name.price = "Price"
)
clv.full <- clvdata(
  apparelTrans, date.format = "ymd", time.unit = "week",
  estimation.split = NULL,
  name.id = "Id", name.date = "Date", name.price = "Price"
)

cat("\n== data layer (clvdata) ==\n")

est.pnbd  <- latentAttrition(family = pnbd, data = clv.apparel, verbose = FALSE)
est.full  <- latentAttrition(family = pnbd, data = clv.full,    verbose = FALSE)

# (x, t_x, T) per customer -- the sufficient statistics every model consumes.
write_csv(est.pnbd@cbs, "cbs_estimation")
write_csv(est.full@cbs, "cbs_full")

# -- Pareto/NBD without covariates, S6.2.1 ------------------------------------

cat("\n== pnbd, no covariates ==\n")

cbs <- est.pnbd@cbs
x <- cbs$x; t.x <- cbs$t.x; T.cal <- cbs$T.cal

# Parameter vectors at which to evaluate every expression: the MLE first, then
# points deliberately off it -- including alpha < beta, which selects the other
# branch of A_1 / A_2 in the appendix likelihood.
grid <- list(
  mle        = c(r = 1.4490, alpha = 48.6361, s = 0.5613, beta = 46.8844),
  alpha.gt.beta = c(r = 1.0, alpha = 10.0, s = 1.0, beta = 5.0),
  alpha.lt.beta = c(r = 1.0, alpha = 5.0,  s = 1.0, beta = 10.0),
  alpha.eq.beta = c(r = 0.5, alpha = 7.0,  s = 2.0, beta = 7.0),
  small.shapes  = c(r = 0.25, alpha = 4.0, s = 0.3, beta = 15.0),
  large.shapes  = c(r = 5.0,  alpha = 80.0, s = 4.0, beta = 60.0)
)

vN <- rep(1, nrow(cbs))

for (nm in names(grid)) {
  p <- grid[[nm]]; vp <- unname(p)
  dt <- data.table(
    Id     = cbs$Id,
    LL.ind = as.numeric(cpp("pnbd_nocov_LL_ind")(log(vp), x, t.x, T.cal)),
    PAlive = as.numeric(cpp("pnbd_nocov_PAlive")(vp[1], vp[2], vp[3], vp[4], x, t.x, T.cal)),
    CET    = as.numeric(cpp("pnbd_nocov_CET")(vp[1], vp[2], vp[3], vp[4], 52, x, t.x, T.cal)),
    DERT   = as.numeric(cpp("pnbd_nocov_DERT")(vp[1], vp[2], vp[3], vp[4],
                                               log(1 + 0.075) / 52, x, t.x, T.cal))
  )
  write_csv(dt, paste0("pnbd_nocov_", nm))
}

# LL.sum is stored as a true log-likelihood: the C++ objective is negated here
# once, so the Python side never has to remember the minimiser's sign.
write_json(list(
  params = lapply(grid, function(p) as.list(p)),
  LL.sum = lapply(grid, function(p)
    -as.numeric(cpp("pnbd_nocov_LL_sum")(log(unname(p)), x, t.x, T.cal, vN))),
  CET.horizon.weeks = 52,
  DERT.continuous.discount.factor = log(1 + 0.075) / 52
), "pnbd_nocov_grid")

# At the MLE the summed individual likelihood must be logLik() of the fit.
check("pnbd nocov LL at MLE",
      sum(cpp("pnbd_nocov_LL_ind")(log(unname(grid$mle)), x, t.x, T.cal)),
      logLik(est.pnbd), tol = 1e-4)
check("pnbd nocov LL_sum sign",
      -cpp("pnbd_nocov_LL_sum")(log(unname(grid$mle)), x, t.x, T.cal, vN),
      logLik(est.pnbd), tol = 1e-4)

# PMF: expected number of customers making exactly k repeat transactions.
pmf.dt <- data.table(Id = cbs$Id)
for (k in 0:10) {
  pmf.dt[[paste0("pmf.", k)]] <-
    as.numeric(cpp("pnbd_nocov_PMF")(1.4490, 48.6361, 0.5613, 46.8844, k, T.cal))
}
write_csv(pmf.dt, "pnbd_nocov_pmf_mle")

# Unconditional expectation E[X(t)] -- drives the tracking plot and the
# prospective-customer prediction of S6.3.4.
t.grid <- seq(0, 200, by = 0.5)
write_csv(data.table(
  t = t.grid,
  expectation = as.numeric(cpp("pnbd_nocov_expectation")(
    1.4490, 48.6361, 0.5613, 46.8844, t.grid))
), "pnbd_nocov_expectation_mle")

# The fit itself, with the standard errors the paper reports for the
# covariate model and that summary() reports here.
write_json(list(
  coefficients = as.list(coef(est.pnbd)),
  logLik       = as.numeric(logLik(est.pnbd)),
  AIC          = AIC(est.pnbd),
  BIC          = BIC(est.pnbd),
  nobs         = nobs(est.pnbd),
  vcov         = as.numeric(vcov(est.pnbd)),
  vcov.names   = colnames(vcov(est.pnbd))
), "pnbd_nocov_fit")

write_json(list(
  coefficients = as.list(coef(est.full)),
  logLik       = as.numeric(logLik(est.full)),
  nobs         = nobs(est.full)
), "pnbd_nocov_fit_full")

# -- The Gaussian hypergeometric, Appendix A ----------------------------------
#
# The A_1 / A_2 terms of the marginalised likelihood are 2F1 evaluations.
# CLVTools calls GSL for these; dumping a grid lets the Python 2F1 be tested
# in isolation, which matters because scipy's hyp2f1 is not reliable across
# the whole argument range this likelihood visits.

cat("\n== hypergeometric ==\n")

# The GSL wrappers return list(value, status); status 0 means GSL converged,
# and rows where it did not are dropped rather than baked in as expectations.
hyp.grid <- CJ(a = c(0.5, 1.5, 3.0, 12.0), b = c(0.3, 1.0, 4.0),
               c = c(2.0, 5.5, 13.0), z = c(-5, -0.9, -0.1, 0, 0.1, 0.5, 0.9))
hyp.grid <- hyp.grid[c > b & c > a - 1]
res <- cpp("vec_gsl_hyp2f1_e")(hyp.grid$a, hyp.grid$b, hyp.grid$c, hyp.grid$z)
hyp.grid[, `:=`(value = as.numeric(res$value), status = as.integer(res$status))]
write_csv(hyp.grid[status == 0L, .(a, b, c, z, value)], "hyp2f1")

hyp0 <- CJ(a = c(0.5, 2.0, 6.0), b = c(1.0, 3.0), z = c(-2, -0.5, -0.05))
res0 <- cpp("vec_gsl_hyp2f0_e")(hyp0$a, hyp0$b, hyp0$z)
hyp0[, `:=`(value = as.numeric(res0$value), status = as.integer(res0$status))]
write_csv(hyp0[status == 0L, .(a, b, z, value)], "hyp2f0")

# -- Gamma-Gamma spending model, S6.2.3 ---------------------------------------

cat("\n== gamma-gamma ==\n")

est.gg       <- spending(family = gg, data = clv.apparel, verbose = FALSE)
est.gg.full  <- spending(family = gg, data = clv.full,    verbose = FALSE)
est.gg.first <- spending(family = gg, data = clv.full, verbose = FALSE,
                         remove.first.transaction = FALSE)

# The spending CBS differs from the transaction CBS: by default the first
# transaction is dropped, so single-purchase customers fall out entirely.
write_csv(est.gg@cbs,       "cbs_spending_estimation")
write_csv(est.gg.first@cbs, "cbs_spending_full_with_first")

gg.cbs <- est.gg@cbs
gg.grid <- list(
  mle    = c(p = 3.099, q = 5.654, gamma = 56.504),
  offset = c(p = 2.0,   q = 4.0,   gamma = 40.0),
  small  = c(p = 0.8,   q = 1.5,   gamma = 10.0)
)
write_json(list(
  params = lapply(gg.grid, as.list),
  LL     = lapply(gg.grid, function(p)
    -as.numeric(cpp("gg_LL")(log(unname(p)), gg.cbs$x, gg.cbs$Spending,
                             rep(1, nrow(gg.cbs)))))
), "gg_grid")

check("gg LL at MLE",
      -cpp("gg_LL")(log(unname(gg.grid$mle)), gg.cbs$x, gg.cbs$Spending,
                    rep(1, nrow(gg.cbs))),
      logLik(est.gg), tol = 1e-4)

write_json(list(
  coefficients = as.list(coef(est.gg)),
  logLik       = as.numeric(logLik(est.gg)),
  nobs         = nobs(est.gg)
), "gg_fit")
write_json(list(
  coefficients = as.list(coef(est.gg.full)),
  logLik       = as.numeric(logLik(est.gg.full))
), "gg_fit_full")
write_json(list(
  coefficients = as.list(coef(est.gg.first)),
  logLik       = as.numeric(logLik(est.gg.first))
), "gg_fit_full_with_first")

# -- Combined prediction, S6.3 ------------------------------------------------

cat("\n== predictions ==\n")

write_csv(predict(est.pnbd, predict.spending = est.gg, verbose = FALSE),
          "predict_holdout")

write_csv(predict(est.full, predict.spending = est.gg.full,
                  prediction.end = 95,
                  continuous.discount.factor = log(1 + 0.075) / 52,
                  verbose = FALSE),
          "predict_full")

write_json(list(
  num.periods                = 52,
  expected.num.transactions  = as.numeric(
    predict(est.full, newdata = newcustomer(num.periods = 52), verbose = FALSE)),
  expected.spending          = as.numeric(
    predict(est.gg.first, newdata = newcustomer.spending(), verbose = FALSE))
), "newcustomer")

# -- Pareto/NBD with time-invariant covariates, S6.4.1 ------------------------

cat("\n== pnbd, static covariates ==\n")

clv.static <- SetStaticCovariates(
  clv.data = clv.apparel,
  data.cov.life  = apparelStaticCov, data.cov.trans = apparelStaticCov,
  names.cov.life = c("Gender", "Channel"),
  names.cov.trans = c("Gender", "Channel"),
  name.id = "Id"
)
est.static <- latentAttrition(~ Gender + Channel | Gender + Channel,
                              family = pnbd, data = clv.static, verbose = FALSE)

write_json(list(
  coefficients = as.list(coef(est.static)),
  logLik       = as.numeric(logLik(est.static)),
  AIC          = AIC(est.static),
  BIC          = BIC(est.static),
  nobs         = nobs(est.static),
  se           = as.list(sqrt(diag(vcov(est.static)))),
  vcov         = as.numeric(vcov(est.static)),
  vcov.names   = colnames(vcov(est.static))
), "pnbd_staticcov_fit")

# The design matrices, so the Python side can be checked to build the same
# k-1 dummy coding from the raw covariate frame.
write_csv(as.data.table(clv.static@data.cov.life,  keep.rownames = FALSE),
          "staticcov_design_life")
write_csv(as.data.table(clv.static@data.cov.trans, keep.rownames = FALSE),
          "staticcov_design_trans")

# alpha_i / beta_i per customer, and the per-customer expressions built on
# them, at the fitted parameters and at an off-optimum point.
m.life  <- as.matrix(clv.static@data.cov.life[,  c("Gender", "Channel"), with = FALSE])
m.trans <- as.matrix(clv.static@data.cov.trans[, c("Gender", "Channel"), with = FALSE])
sc <- est.static@cbs

sc.grid <- list(
  mle = list(model = c(r = 1.8378, alpha = 92.9123, s = 0.5920, beta = 49.6227),
             life  = c(Gender = -0.6430, Channel = 0.7907),
             trans = c(Gender = 0.2859,  Channel = 0.6241)),
  offset = list(model = c(r = 1.0, alpha = 50.0, s = 1.0, beta = 30.0),
                life  = c(Gender = 0.2, Channel = -0.3),
                trans = c(Gender = -0.1, Channel = 0.4))
)

# Parameter vector layout, per src/pnbd.cpp: the four model parameters on the
# log scale, then the life covariate parameters, then the transaction ones.
sc.vec <- function(g) c(log(unname(g$model)), unname(g$life), unname(g$trans))
sc.N   <- rep(1, nrow(sc))

# Called with named arguments throughout: CET/PAlive take
# (vCovParams_trans, vCovParams_life, mCov_trans, mCov_life) while DERT takes
# (mCov_life, mCov_trans, vCovParams_life, vCovParams_trans), and positional
# calls would silently swap the two processes.
for (nm in names(sc.grid)) {
  g <- sc.grid[[nm]]
  a.i <- as.numeric(cpp("pnbd_staticcov_alpha_i")(
    alpha_0 = g$model[["alpha"]], vCovParams_trans = unname(g$trans), mCov_trans = m.trans))
  b.i <- as.numeric(cpp("pnbd_staticcov_beta_i")(
    beta_0 = g$model[["beta"]], vCovParams_life = unname(g$life), mCov_life = m.life))
  write_csv(data.table(
    Id = sc$Id, alpha.i = a.i, beta.i = b.i,
    LL.ind = as.numeric(cpp("pnbd_staticcov_LL_ind")(
      sc.vec(g), sc$x, sc$t.x, sc$T.cal, m.life, m.trans)),
    PAlive = as.numeric(cpp("pnbd_staticcov_PAlive")(
      r = g$model[["r"]], alpha_0 = g$model[["alpha"]],
      s = g$model[["s"]], beta_0 = g$model[["beta"]],
      vX = sc$x, vT_x = sc$t.x, vT_cal = sc$T.cal,
      vCovParams_trans = unname(g$trans), vCovParams_life = unname(g$life),
      mCov_trans = m.trans, mCov_life = m.life)),
    CET = as.numeric(cpp("pnbd_staticcov_CET")(
      r = g$model[["r"]], alpha_0 = g$model[["alpha"]],
      s = g$model[["s"]], beta_0 = g$model[["beta"]], dPeriods = 52,
      vX = sc$x, vT_x = sc$t.x, vT_cal = sc$T.cal,
      vCovParams_trans = unname(g$trans), vCovParams_life = unname(g$life),
      mCov_trans = m.trans, mCov_life = m.life)),
    DERT = as.numeric(cpp("pnbd_staticcov_DERT")(
      r = g$model[["r"]], alpha_0 = g$model[["alpha"]],
      s = g$model[["s"]], beta_0 = g$model[["beta"]],
      continuous_discount_factor = log(1 + 0.075) / 52,
      vX = sc$x, vT_x = sc$t.x, vT_cal = sc$T.cal,
      mCov_life = m.life, mCov_trans = m.trans,
      vCovParams_life = unname(g$life), vCovParams_trans = unname(g$trans)))
  ), paste0("pnbd_staticcov_", nm))
}

write_json(list(
  params = lapply(sc.grid, function(g)
    list(model = as.list(g$model), life = as.list(g$life), trans = as.list(g$trans))),
  LL.sum = lapply(sc.grid, function(g)
    -as.numeric(cpp("pnbd_staticcov_LL_sum")(
      sc.vec(g), sc$x, sc$t.x, sc$T.cal, sc.N, m.life, m.trans)))
), "pnbd_staticcov_grid")

check("pnbd staticcov LL at MLE",
      sum(cpp("pnbd_staticcov_LL_ind")(sc.vec(sc.grid$mle), sc$x, sc$t.x, sc$T.cal,
                                       m.life, m.trans)),
      logLik(est.static), tol = 1e-3)

cat("\nfixtures written to ", OUT, "\n", sep = "")
