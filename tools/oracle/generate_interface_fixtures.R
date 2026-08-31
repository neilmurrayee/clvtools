#!/usr/bin/env Rscript
# Oracle fixtures for the interface layer: the descriptive statistics and plots
# of S6.1.2, the inference generics of Table 2, and the prediction tables of
# the families the Pareto/NBD's own generator does not cover.
#
#     ./tools/setup_oracle.sh
#     R_LIBS=.Rlib Rscript tools/oracle/generate_interface_fixtures.R
#
# Same discipline as generate_fixtures.R: every fixture family is checked
# against the public generic that produces it before anything is written, so a
# convention slip cannot ship as a plausible-looking expectation.

suppressMessages({
  library(CLVTools)
  library(data.table)
})

OUT <- "tests/fixtures"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

write_csv <- function(dt, name) {
  fwrite(dt, file.path(OUT, paste0(name, ".csv")), dateTimeAs = "ISO")
  cat(sprintf("  %-42s %6d rows\n", paste0(name, ".csv"), nrow(dt)))
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
  cat(sprintf("  %-42s\n", paste0(name, ".json")))
}
check <- function(label, got, want, tol = 1e-6) {
  if (!isTRUE(all.equal(as.character(got), as.character(want)))) {
    if (!isTRUE(all.equal(as.numeric(got), as.numeric(want), tolerance = tol))) {
      stop(sprintf("oracle self-check failed for %s", label), call. = FALSE)
    }
  }
  cat(sprintf("  ok  %s\n", label))
}

data("apparelTrans")

clv.apparel <- clvdata(
  apparelTrans, date.format = "ymd", time.unit = "week",
  estimation.split = 104,
  name.id = "Id", name.date = "Date", name.price = "Price"
)

# -- S6.1.2, summary() of the transaction data --------------------------------
#
# summary() formats every value to three decimals before printing. The values
# themselves are recomputed here at full precision from the same expressions
# CLVTools uses (clv.data.make.descriptives), and the formatted result is
# checked against summary()'s own table cell by cell below.

cat("\n== descriptive statistics (S6.1.2) ==\n")

dt.all <- clv.apparel@data.transactions
tp.est.start <- clv.apparel@clv.time@timepoint.estimation.start
tp.est.end <- clv.apparel@clv.time@timepoint.estimation.end
tp.hold.start <- clv.apparel@clv.time@timepoint.holdout.start
tp.hold.end <- clv.apparel@clv.time@timepoint.holdout.end

descriptives <- function(dt.data, sample.name) {
  dt.interp <- CLVTools:::clv.data.mean.interpurchase.times(
    clv.data = clv.apparel, dt.transactions = dt.data)
  by.cust <- dt.data[, .N, by = "Id"]
  is.est <- sample.name == "Estimation"
  list(
    `Period Start` = as.character(switch(
      sample.name, Estimation = tp.est.start, Holdout = tp.hold.start,
      Total = tp.est.start)),
    `Period End` = as.character(switch(
      sample.name, Estimation = tp.est.end, Holdout = tp.hold.end,
      Total = tp.hold.end)),
    `Number of customers` = if (sample.name == "Total") nrow(by.cust) else NA_real_,
    `First Transaction in period` = as.character(dt.data[, min(Date)]),
    `Last Transaction in period` = as.character(dt.data[, max(Date)]),
    `Total # Transactions` = nrow(dt.data),
    `Mean # Transactions per cust` = by.cust[, mean(N)],
    `(SD)` = by.cust[, sd(N)],
    `Mean Spending per Transaction` = dt.data[, mean(Price)],
    `(SD) ` = dt.data[, sd(Price)],
    `Total Spending` = dt.data[, sum(Price)],
    `Total # zero repeaters` = if (is.est) by.cust[, sum(N == 1)] else NA_real_,
    `Percentage of zero repeaters` = if (is.est) by.cust[, mean(N == 1) * 100] else NA_real_,
    `Mean Interpurchase time` = dt.interp[, mean(interp.time, na.rm = TRUE)],
    `(SD)   ` = dt.interp[, sd(interp.time, na.rm = TRUE)]
  )
}

l.est <- descriptives(dt.all[Date <= tp.est.end], "Estimation")
l.hold <- descriptives(dt.all[Date >= tp.hold.start], "Holdout")
l.total <- descriptives(dt.all, "Total")

dt.desc <- data.table(
  Name = names(l.est),
  Estimation = vapply(l.est, function(v) as.character(format(v, digits = 17, trim = TRUE)), ""),
  Holdout = vapply(l.hold, function(v) as.character(format(v, digits = 17, trim = TRUE)), ""),
  Total = vapply(l.total, function(v) as.character(format(v, digits = 17, trim = TRUE)), "")
)

# Every cell must agree with what summary() prints, once rounded the way it
# rounds. This is what makes the full-precision recomputation above legitimate.
dt.printed <- as.data.table(summary(clv.apparel)$descriptives.transactions)
for (j in c("Estimation", "Holdout", "Total")) {
  for (i in seq_len(nrow(dt.desc))) {
    printed <- trimws(dt.printed[[j]][i])
    mine <- trimws(dt.desc[[j]][i])
    if (printed == "-") {
      if (mine != "NA") stop(sprintf("expected '-' at %s/%s", j, dt.desc$Name[i]))
      next
    }
    num <- suppressWarnings(as.numeric(printed))
    if (is.na(num)) {
      if (printed != mine) stop(sprintf("date mismatch at %s/%s", j, dt.desc$Name[i]))
    } else {
      if (!isTRUE(all.equal(num, as.numeric(mine), tolerance = 1e-3))) {
        stop(sprintf("value mismatch at %s/%s: %s vs %s", j, dt.desc$Name[i],
                     printed, mine))
      }
    }
  }
}
cat("  ok  every cell agrees with summary()\n")
write_csv(dt.desc, "descriptives_summary")

# as.data.frame() and subset() default to the FULL sample, where the
# descriptive plots below default to the estimation period. Both defaults are
# pinned here so the difference cannot be lost in translation.
write_json(list(
  nobs = nobs(clv.apparel),
  n.transactions.total = nrow(dt.all),
  n.transactions.estimation = nrow(dt.all[Date <= tp.est.end]),
  n.transactions.holdout = nrow(dt.all[Date >= tp.hold.start]),
  n.rows.raw = nrow(apparelTrans),
  n.default.as.data.frame = nrow(as.data.frame(clv.apparel)),
  n.ids.1 = nrow(as.data.frame(clv.apparel, ids = "1")),
  n.price.50.to.100 = nrow(subset(clv.apparel, Price >= 50 & Price <= 100)),
  n.price.50.to.100.estimation = nrow(
    subset(clv.apparel, Price >= 50 & Price <= 100, sample = "estimation")),
  n.ids.7.9.holdout = nrow(
    subset(clv.apparel, Id == "7" | Id == "9", sample = "holdout"))
), "data_samples")

# -- S6.1.2, the five descriptive plots (Table 3) -----------------------------
#
# Each is taken through the public plot(..., plot = FALSE), which is the
# interface the paper itself uses in S6.3.3 to get at plot data.

cat("\n== descriptive plots (Table 3) ==\n")

plot_data <- function(...) as.data.table(plot(clv.apparel, ..., plot = FALSE, verbose = FALSE))

write_csv(plot_data(which = "tracking"), "plot_data_tracking")
write_csv(plot_data(which = "tracking", cumulative = TRUE), "plot_data_tracking_cumulative")

dt.freq <- plot_data(which = "frequency")
check("frequency counts sum to the customers",
      dt.freq[, sum(num.customers)], nobs(clv.apparel))
write_csv(dt.freq, "plot_data_frequency")
write_csv(plot_data(which = "frequency", count.remaining = FALSE),
          "plot_data_frequency_no_remaining")
write_csv(plot_data(which = "frequency", trans.bins = 0:4, label.remaining = "5+"),
          "plot_data_frequency_five_bins")

write_csv(plot_data(which = "interpurchasetime"), "plot_data_interpurchasetime")
write_csv(plot_data(which = "interpurchasetime", sample = "holdout"),
          "plot_data_interpurchasetime_holdout")

write_csv(plot_data(which = "spending"), "plot_data_spending_mean")
write_csv(plot_data(which = "spending", sample = "holdout"),
          "plot_data_spending_mean_holdout")
write_csv(plot_data(which = "spending", mean.spending = FALSE),
          "plot_data_spending_transactions")

write_csv(plot_data(which = "timings", ids = c("1", "2", "3")), "plot_data_timings")

cat("\ndone\n")

# -- Table 2's model details: vcov(), confint(), fitted() ----------------------
#
# All three are evaluated at CLVTools' own optimum, and the coefficients are
# written alongside so the Python side can be held to them at the same point
# rather than at its own -- these likelihoods are flat enough near the optimum
# that comparing curvature at two different points would prove nothing.

cat("\n== model details (Table 2) ==\n")

data("apparelStaticCov")
clv.static <- SetStaticCovariates(
  clv.data = clv.apparel,
  data.cov.life = apparelStaticCov, data.cov.trans = apparelStaticCov,
  names.cov.life = c("Gender", "Channel"),
  names.cov.trans = c("Gender", "Channel"), name.id = "Id"
)

inference <- function(fitted, name) {
  cf <- coef(fitted)
  vc <- vcov(fitted)
  ci <- confint(fitted)
  # confint is the Wald interval: estimate +- 1.96 standard errors. Asserting
  # it here is what makes the fixture safe to hold Python to.
  se <- sqrt(diag(vc))
  check(paste0(name, ": confint is Wald"),
        as.numeric(ci[, 1]), as.numeric(cf - qnorm(0.975) * se), tol = 1e-8)
  write_json(list(
    coefficients = as.list(cf),
    names = names(cf),
    se = as.numeric(se),
    vcov = as.numeric(vc),
    confint.lower = as.numeric(ci[, 1]),
    confint.upper = as.numeric(ci[, 2]),
    logLik = as.numeric(logLik(fitted)),
    nobs = nobs(fitted),
    AIC = AIC(fitted),
    BIC = BIC(fitted)
  ), paste0("inference_", name))
}

est.pnbd <- latentAttrition(family = pnbd, data = clv.apparel, verbose = FALSE)
est.bgnbd <- latentAttrition(family = bgnbd, data = clv.apparel, verbose = FALSE)
est.ggomnbd <- latentAttrition(family = ggomnbd, data = clv.apparel, verbose = FALSE)
est.gg <- spending(family = gg, data = clv.apparel, verbose = FALSE)
est.static <- latentAttrition(
  formula = ~ Gender + Channel | Gender + Channel,
  family = pnbd, data = clv.static, verbose = FALSE)

inference(est.pnbd, "pnbd")
inference(est.bgnbd, "bgnbd")
inference(est.ggomnbd, "ggomnbd")
inference(est.gg, "gg")
inference(est.static, "pnbd_staticcov")

# The constrained fit reports one coefficient where the unconstrained one
# reports two, and its standard errors are over that shorter vector -- the
# curvature of the likelihood in the parameters actually estimated.
est.static.constr <- latentAttrition(
  formula = ~ Gender + Channel | Gender + Channel,
  names.cov.constr = "Gender",
  family = pnbd, data = clv.static, verbose = FALSE)
inference(est.static.constr, "pnbd_staticcov_constrained")

# fitted(): the model's unconditional expectation over the estimation period.
dt.fitted <- as.data.table(fitted(est.pnbd))
check("fitted() opens at zero", dt.fitted[1, expectation], 0)
write_csv(dt.fitted, "fitted_pnbd")

# -- Prediction for the other two families (Table 4) ---------------------------
#
# These two report CET and PAlive but no DERT: there is no closed form for the
# discounted expected residual transactions of either.

cat("\n== predict() for the BG/NBD and GGom/NBD ==\n")

for (nm in c("bgnbd", "ggomnbd")) {
  fitted <- if (nm == "bgnbd") est.bgnbd else est.ggomnbd
  dt.pred <- as.data.table(predict(fitted, predict.spending = est.gg, verbose = FALSE))
  check(paste0(nm, ": no DERT column"), "DERT" %in% names(dt.pred), FALSE)
  write_csv(dt.pred, paste0("predict_", nm))
  write_json(as.list(coef(fitted)), paste0("predict_", nm, "_coefficients"))
}

# -- Covariate prediction for the other two families --------------------------
#
# Table 4 gives time-invariant covariates to all three families, so all three
# can predict from them. Only the Pareto/NBD's covariate prediction was covered
# by the Pareto/NBD generator.

cat("\n== predict() with covariates, BG/NBD and GGom/NBD ==\n")

for (nm in c("bgnbd", "ggomnbd")) {
  fitted <- latentAttrition(
    formula = ~ Gender + Channel | Gender + Channel,
    family = if (nm == "bgnbd") bgnbd else ggomnbd,
    data = clv.static, verbose = FALSE)
  dt.pred <- as.data.table(predict(fitted, verbose = FALSE))
  write_csv(dt.pred, paste0("predict_", nm, "_staticcov"))
  write_json(as.list(coef(fitted)), paste0("predict_", nm, "_staticcov_coefficients"))
}

# -- Prospective customers, S6.3.4 --------------------------------------------

cat("\n== newcustomer() ==\n")

clv.full <- clvdata(
  apparelTrans, date.format = "ymd", time.unit = "week", estimation.split = NULL,
  name.id = "Id", name.date = "Date", name.price = "Price"
)
clv.static.full <- SetStaticCovariates(
  clv.data = clv.full,
  data.cov.life = apparelStaticCov, data.cov.trans = apparelStaticCov,
  names.cov.life = c("Gender", "Channel"),
  names.cov.trans = c("Gender", "Channel"), name.id = "Id"
)
est.pnbd.full <- latentAttrition(family = pnbd, data = clv.full, verbose = FALSE)
est.static.full <- latentAttrition(
  formula = ~ Gender + Channel | Gender + Channel,
  family = pnbd, data = clv.static.full, verbose = FALSE)

# Covariate scenarios: the two levels of Gender at each level of Channel, which
# is the "region A versus region B" comparison S6.3.4 describes.
scenarios <- list(
  gender0.channel0 = data.frame(Gender = 0, Channel = 0),
  gender1.channel0 = data.frame(Gender = 1, Channel = 0),
  gender0.channel1 = data.frame(Gender = 0, Channel = 1),
  gender1.channel1 = data.frame(Gender = 1, Channel = 1)
)
l.static <- lapply(scenarios, function(cov) {
  as.numeric(predict(est.static.full, newdata = newcustomer.static(
    num.periods = 52, data.cov.life = cov, data.cov.trans = cov)))
})
# The same scenario question, asked of the other two families.
for (nm in c("bgnbd", "ggomnbd")) {
  fitted.other <- latentAttrition(
    formula = ~ Gender + Channel | Gender + Channel,
    family = if (nm == "bgnbd") bgnbd else ggomnbd,
    data = clv.static.full, verbose = FALSE)
  scenario <- function(g, c) as.numeric(predict(
    fitted.other, newdata = newcustomer.static(
      num.periods = 52,
      data.cov.life = data.frame(Gender = g, Channel = c),
      data.cov.trans = data.frame(Gender = g, Channel = c))))
  write_json(list(
    num.periods = 52,
    coefficients = as.list(coef(fitted.other)),
    gender0.channel0 = scenario(0, 0),
    gender1.channel1 = scenario(1, 1)
  ), paste0("newcustomer_static_", nm))
}

write_json(c(
  list(num.periods = 52,
       coefficients = as.list(coef(est.static.full)),
       nocov.transactions = as.numeric(predict(
         est.pnbd.full, newdata = newcustomer(num.periods = 52)))),
  l.static
), "newcustomer_static")

# -- Likelihood ratio test, S6.5.3 --------------------------------------------

cat("\n== lrtest() ==\n")

lr <- lmtest::lrtest(est.static.constr, est.static)
# lmtest's "#Df" is the parameter count of each model; "Df" is the difference,
# which is NA on the first row.
check("lrtest: the constraint costs one parameter",
      lr[["#Df"]][2] - lr[["#Df"]][1], 1)
write_json(list(
  n.parameters.restricted = lr[["#Df"]][1],
  n.parameters.unrestricted = lr[["#Df"]][2],
  loglik.restricted = lr$LogLik[1], loglik.unrestricted = lr$LogLik[2],
  df = lr$Df[2], chisq = lr$Chisq[2], p.value = lr[["Pr(>Chisq)"]][2],
  coefficients.restricted = as.list(coef(est.static.constr))
), "lrtest_pnbd_staticcov")

# -- Correlated Pareto/NBD: prediction ignores the correlation ----------------
#
# The Sarmanov correlation enters estimation only. predict() uses the plain
# uncorrelated PAlive and CET at the fitted (r, alpha, s, beta), which this
# fixture pins: the difference below is exactly zero, not merely small.

cat("\n== correlated Pareto/NBD ==\n")

est.cor <- latentAttrition(family = pnbd, data = clv.apparel, use.cor = TRUE,
                           verbose = FALSE)
cf <- coef(est.cor)
dt.pred.cor <- as.data.table(predict(est.cor, verbose = FALSE))
cbs <- est.cor@cbs
plain.palive <- as.numeric(CLVTools:::pnbd_nocov_PAlive(
  r = cf[["r"]], alpha_0 = cf[["alpha"]], s = cf[["s"]], beta_0 = cf[["beta"]],
  vX = cbs$x, vT_x = cbs$t.x, vT_cal = cbs$T.cal))
check("correlated PAlive is the uncorrelated one",
      max(abs(dt.pred.cor$PAlive - plain.palive)), 0, tol = 1e-15)
write_json(list(
  coefficients = as.list(cf),
  correlation = as.numeric(cf[["Cor(life,trans)"]]),
  logLik = as.numeric(logLik(est.cor)),
  palive.head = head(dt.pred.cor$PAlive, 5),
  cet.head = head(dt.pred.cor$CET, 5),
  period.length = dt.pred.cor$period.length[1]
), "correlated_predict")

cat("\ndone\n")
