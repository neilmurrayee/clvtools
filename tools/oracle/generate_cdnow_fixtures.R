#!/usr/bin/env Rscript
# Oracle fixtures for the CDNOW data.
#
#     R_LIBS=.Rlib Rscript tools/oracle/generate_cdnow_fixtures.R
#
# `cdnow` is the dataset CLVTools' own documentation reaches for by default --
# `?clvdata`, `?pmf`, `?subset.clv.data`, `?as.data.frame.clv.data`,
# `?spending` and `?clv.bootstrapped.apply` all use it -- while the paper uses
# `apparelTrans` throughout. It is a second, independent dataset for machinery
# otherwise exercised on one: 2,357 customers rather than 600, no covariates,
# and a split given in weeks from the first transaction.
#
# `?pmf` prints a PMF table to six decimals with the empirical frequencies
# beside it. This dumps the same quantities at full precision, so the published
# table can be checked to the digits it prints and the fit behind it to more.

suppressMessages({
  library(CLVTools)
  library(data.table)
})

OUT <- "tests/fixtures"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

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

data("cdnow")

# `?pmf` and `?clvdata`: 37 weeks from the first transaction.
clv <- clvdata(cdnow, date.format = "ymd", time.unit = "w",
               estimation.split = 37)

cat("cdnow, estimation.split = 37\n")
fit <- pnbd(clv, verbose = FALSE)

cf <- coef(fit)
se <- sqrt(diag(vcov(fit)))
write_json(list(
  coefficients = as.list(cf),
  standard.errors = as.list(se),
  logLik = as.numeric(logLik(fit)),
  AIC = AIC(fit),
  BIC = BIC(fit),
  nobs = nobs(fit),
  vcov = as.numeric(vcov(fit)),
  vcov.names = names(cf)
), "cdnow_pnbd_fit")

# The self-check CLAUDE.md asks for: the dumped coefficients are the ones the
# public generic reports, so a name or ordering slip cannot ship.
check("coef r", cf[["r"]], coef(fit)[["r"]])
check("logLik", as.numeric(logLik(fit)), sum(fit@optimx.estimation.output$value) * -1,
      tol = 1e-4)

# `?pmf`: the model PMF for x = 0..10, per customer. The man page prints the
# column means; both are dumped so the aggregation can be checked separately
# from the expression.
X_MAX <- 10
dt.pmf <- pmf(fit, x = 0:X_MAX)
setnames(dt.pmf, c("Id", paste0("pmf.x.", 0:X_MAX)))
write_csv(dt.pmf, "cdnow_pmf")

means <- vapply(paste0("pmf.x.", 0:X_MAX),
                function(col) mean(dt.pmf[[col]]), numeric(1))
write_json(list(x = 0:X_MAX, mean.pmf = unname(means)), "cdnow_pmf_means")

# The published table's own check: the man page prints mean(pmf) at x = 0 as
# 0.616514.
check("mean pmf at x=0", means[[1]], 0.616514, tol = 1e-4)

# `?pmf` prints the empirical frequencies beside the model ones. Taken from
# CLVTools' own CBS so the comparison is like for like.
cbs <- fit@cbs
freq <- vapply(0:X_MAX, function(k) sum(cbs$x == k), numeric(1))
write_json(list(
  x = 0:X_MAX,
  count = freq,
  n.customers = nrow(cbs)
), "cdnow_frequencies")

check("customers", nrow(cbs), 2357)
check("zero-repeaters", freq[[1]], 1432)

cat("done\n")
