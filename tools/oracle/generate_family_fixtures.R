#!/usr/bin/env Rscript
# Oracle fixtures for the two alternative latent attrition models, Table 3:
# the BG/NBD of Fader et al. and the GGom/NBD of Bemmaor & Glady.
#
#     R_LIBS=.Rlib Rscript tools/oracle/generate_family_fixtures.R
#
# S6.2.1: "As an alternative to the Pareto/NBD model, CLVTools features the
# Beta-Geometric/NBD model and the Gamma-Gompertz/NBD model. To use these
# models, set the parameter family to either bgnbd or to ggomnbd."
#
# As with the Pareto/NBD, every expression is dumped at several parameter
# vectors rather than only at the optimum, so each can be tested before any
# optimiser runs.

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
  cat(sprintf("  %-40s %6d rows\n", paste0(name, ".csv"), nrow(dt)))
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
  cat(sprintf("  %-40s\n", paste0(name, ".json")))
}
check <- function(label, got, want, tol = 1e-6) {
  if (!isTRUE(all.equal(as.numeric(got), as.numeric(want), tolerance = tol))) {
    stop(sprintf("self-check failed for %s: got %.10g, want %.10g",
                 label, as.numeric(got), as.numeric(want)), call. = FALSE)
  }
  cat(sprintf("  ok  %-36s %.6f\n", label, as.numeric(want)))
}

data("apparelTrans")
data("apparelStaticCov")

clv <- clvdata(apparelTrans, date.format = "ymd", time.unit = "week",
               estimation.split = 104,
               name.id = "Id", name.date = "Date", name.price = "Price")
clv.static <- SetStaticCovariates(
  clv, data.cov.life = apparelStaticCov, data.cov.trans = apparelStaticCov,
  names.cov.life = c("Gender", "Channel"),
  names.cov.trans = c("Gender", "Channel"), name.id = "Id"
)

T_HORIZON <- 52
vN <- NULL

# -- BG/NBD, Fader et al. -----------------------------------------------------

cat("\n== bgnbd ==\n")

fit.bgnbd <- latentAttrition(family = bgnbd, data = clv, verbose = FALSE)
cbs <- fit.bgnbd@cbs
x <- cbs$x; t.x <- cbs$t.x; T.cal <- cbs$T.cal
vN <- rep(1, nrow(cbs))

write_csv(cbs[, .(Id, x, t.x, T.cal)], "bgnbd_cbs")
write_json(list(
  coefficients = as.list(coef(fit.bgnbd)),
  logLik = as.numeric(logLik(fit.bgnbd)),
  AIC = AIC(fit.bgnbd), BIC = BIC(fit.bgnbd), nobs = nobs(fit.bgnbd)
), "bgnbd_fit")

# a and b are the beta parameters of the dropout process; r and alpha the gamma
# parameters of the transaction process (Table 3).
bg.grid <- list(
  mle = as.list(coef(fit.bgnbd)),
  offset = list(r = 0.6, alpha = 12.0, a = 1.4, b = 3.2),
  small = list(r = 0.25, alpha = 4.0, a = 0.8, b = 1.5)
)

for (nm in names(bg.grid)) {
  p <- bg.grid[[nm]]
  write_csv(data.table(
    Id = cbs$Id,
    LL.ind = as.numeric(cpp("bgnbd_nocov_LL_ind")(
      vLogparams = log(unlist(p[c("r", "alpha", "a", "b")])),
      vX = x, vT_x = t.x, vT_cal = T.cal)),
    PAlive = as.numeric(cpp("bgnbd_nocov_PAlive")(
      r = p$r, alpha = p$alpha, a = p$a, b = p$b,
      vX = x, vT_x = t.x, vT_cal = T.cal)),
    CET = as.numeric(cpp("bgnbd_nocov_CET")(
      r = p$r, alpha = p$alpha, a = p$a, b = p$b, dPeriods = T_HORIZON,
      vX = x, vT_x = t.x, vT_cal = T.cal))
  ), paste0("bgnbd_nocov_", nm))
}

write_json(list(
  params = bg.grid,
  CET.horizon = T_HORIZON,
  LL.sum = lapply(bg.grid, function(p)
    -as.numeric(cpp("bgnbd_nocov_LL_sum")(
      vLogparams = log(unlist(p[c("r", "alpha", "a", "b")])),
      vX = x, vT_x = t.x, vT_cal = T.cal, vN = vN)))
), "bgnbd_nocov_grid")

check("bgnbd LL at MLE",
      sum(cpp("bgnbd_nocov_LL_ind")(
        log(unlist(bg.grid$mle[c("r", "alpha", "a", "b")])), x, t.x, T.cal)),
      logLik(fit.bgnbd), tol = 1e-4)

# PMF and the unconditional expectation, at the fitted parameters.
mle <- bg.grid$mle
pmf.dt <- data.table(Id = cbs$Id)
for (k in 0:10) {
  pmf.dt[[paste0("pmf.", k)]] <- as.numeric(cpp("bgnbd_nocov_PMF")(
    r = mle$r, alpha = mle$alpha, a = mle$a, b = mle$b, x = k, vT_i = T.cal))
}
write_csv(pmf.dt, "bgnbd_nocov_pmf_mle")

t.grid <- seq(0, 200, by = 0.5)
write_csv(data.table(t = t.grid, expectation = as.numeric(
  cpp("bgnbd_nocov_expectation")(r = mle$r, alpha = mle$alpha,
                                 a = mle$a, b = mle$b, vT_i = t.grid))),
  "bgnbd_nocov_expectation_mle")

# With time-invariant covariates. Note the sign convention differs from the
# Pareto/NBD: alpha_i uses exp(-gamma'x) but a_i and b_i use exp(+gamma'x).
fit.bgnbd.cov <- latentAttrition(~ Gender + Channel | Gender + Channel,
                                 family = bgnbd, data = clv.static, verbose = FALSE)
write_json(list(
  coefficients = as.list(coef(fit.bgnbd.cov)),
  logLik = as.numeric(logLik(fit.bgnbd.cov)),
  nobs = nobs(fit.bgnbd.cov)
), "bgnbd_staticcov_fit")

m.life <- as.matrix(clv.static@data.cov.life[, c("Gender", "Channel"), with = FALSE])
m.trans <- as.matrix(clv.static@data.cov.trans[, c("Gender", "Channel"), with = FALSE])
sc <- fit.bgnbd.cov@cbs
sc.p <- as.list(coef(fit.bgnbd.cov))
g.life <- c(sc.p[["life.Gender"]], sc.p[["life.Channel"]])
g.trans <- c(sc.p[["trans.Gender"]], sc.p[["trans.Channel"]])

write_csv(data.table(
  Id = sc$Id,
  alpha.i = as.numeric(cpp("bgnbd_staticcov_alpha_i")(sc.p$alpha, g.trans, m.trans)),
  a.i = as.numeric(cpp("bgnbd_staticcov_a_i")(sc.p$a, g.life, m.life)),
  b.i = as.numeric(cpp("bgnbd_staticcov_b_i")(sc.p$b, g.life, m.life)),
  LL.ind = as.numeric(cpp("bgnbd_staticcov_LL_ind")(
    vParams = c(log(c(sc.p$r, sc.p$alpha, sc.p$a, sc.p$b)), g.life, g.trans),
    vX = sc$x, vT_x = sc$t.x, vT_cal = sc$T.cal,
    mCov_life = m.life, mCov_trans = m.trans))
), "bgnbd_staticcov_mle")

check("bgnbd staticcov LL at MLE",
      sum(cpp("bgnbd_staticcov_LL_ind")(
        c(log(c(sc.p$r, sc.p$alpha, sc.p$a, sc.p$b)), g.life, g.trans),
        sc$x, sc$t.x, sc$T.cal, m.life, m.trans)),
      logLik(fit.bgnbd.cov), tol = 1e-4)

# -- GGom/NBD, Bemmaor & Glady ------------------------------------------------

cat("\n== ggomnbd ==\n")

fit.ggom <- latentAttrition(family = ggomnbd, data = clv, verbose = FALSE)
write_json(list(
  coefficients = as.list(coef(fit.ggom)),
  logLik = as.numeric(logLik(fit.ggom)),
  AIC = AIC(fit.ggom), BIC = BIC(fit.ggom), nobs = nobs(fit.ggom)
), "ggomnbd_fit")

# Table 3: the GGom/NBD carries five parameters, r, alpha, beta, b and s.
gg.grid <- list(
  mle = as.list(coef(fit.ggom)),
  offset = list(r = 0.8, alpha = 10.0, b = 0.02, s = 1.5, beta = 4.0),
  small = list(r = 0.3, alpha = 5.0, b = 0.005, s = 0.6, beta = 1.2)
)
order.names <- c("r", "alpha", "b", "s", "beta")

for (nm in names(gg.grid)) {
  p <- gg.grid[[nm]]
  write_csv(data.table(
    Id = cbs$Id,
    LL.ind = as.numeric(cpp("ggomnbd_nocov_LL_ind")(
      vLogparams = log(unlist(p[order.names])),
      vX = x, vT_x = t.x, vT_cal = T.cal)),
    PAlive = as.numeric(cpp("ggomnbd_nocov_PAlive")(
      r = p$r, alpha_0 = p$alpha, b = p$b, s = p$s, beta_0 = p$beta,
      vX = x, vT_x = t.x, vT_cal = T.cal)),
    CET = as.numeric(cpp("ggomnbd_nocov_CET")(
      r = p$r, alpha_0 = p$alpha, b = p$b, s = p$s, beta_0 = p$beta,
      dPeriods = T_HORIZON, vX = x, vT_x = t.x, vT_cal = T.cal))
  ), paste0("ggomnbd_nocov_", nm))
}

write_json(list(
  params = gg.grid,
  CET.horizon = T_HORIZON,
  LL.sum = lapply(gg.grid, function(p)
    -as.numeric(cpp("ggomnbd_nocov_LL_sum")(
      vLogparams = log(unlist(p[order.names])),
      vX = x, vT_x = t.x, vT_cal = T.cal, vN = vN)))
), "ggomnbd_nocov_grid")

check("ggomnbd LL at MLE",
      sum(cpp("ggomnbd_nocov_LL_ind")(
        log(unlist(gg.grid$mle[order.names])), x, t.x, T.cal)),
      logLik(fit.ggom), tol = 1e-4)

mle <- gg.grid$mle
pmf.dt <- data.table(Id = cbs$Id)
for (k in 0:6) {
  pmf.dt[[paste0("pmf.", k)]] <- as.numeric(cpp("ggomnbd_nocov_PMF")(
    r = mle$r, alpha_0 = mle$alpha, b = mle$b, s = mle$s, beta_0 = mle$beta,
    x = k, vT_i = T.cal))
}
write_csv(pmf.dt, "ggomnbd_nocov_pmf_mle")

write_csv(data.table(t = t.grid, expectation = as.numeric(
  cpp("ggomnbd_nocov_expectation")(r = mle$r, alpha_0 = mle$alpha, b = mle$b,
                                   s = mle$s, beta_0 = mle$beta, vT_i = t.grid))),
  "ggomnbd_nocov_expectation_mle")

cat("\nfixtures written to ", OUT, "\n", sep = "")
