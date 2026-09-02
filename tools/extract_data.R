#!/usr/bin/env Rscript
# Export the datasets bundled with the R package CLVTools to CSV, into the
# package itself at src/clvtools/data/ -- they ship in the wheel.
#
# These are the datasets used throughout Meierer et al. (JSS 5634). Exporting
# them to plain CSV means the Python test suite reads exactly the same bytes
# the R oracle does, with no R dependency at test time.
#
# Run from the project root:  R_LIBS=.Rlib Rscript tools/extract_data.R

suppressMessages({library(CLVTools); library(data.table)})

dir.create(file.path("src", "clvtools", "data"), showWarnings = FALSE, recursive = TRUE)

for (nm in c("apparelTrans", "apparelStaticCov", "apparelDynCov",
             "apparelDynCovFuture", "cdnow")) {
  data(list = nm, package = "CLVTools", envir = environment())
  dt <- get(nm, envir = environment())
  # Dates are written ISO-8601 so pandas parses them unambiguously.
  fwrite(dt, file.path("src", "clvtools", "data", paste0(nm, ".csv")),
         dateTimeAs = "ISO")
  cat(sprintf("%-20s %7d rows x %d cols  ->  src/clvtools/data/%s.csv\n",
              nm, nrow(dt), ncol(dt), nm))
}
