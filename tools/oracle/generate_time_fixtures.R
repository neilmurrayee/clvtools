# S5's time arithmetic, from CLVTools' own `clv.time` classes.
#
#     R_LIBS=.Rlib Rscript tools/oracle/generate_time_fixtures.R
#
# Writes `time_elapsed.csv` (840 spans) and `time_add_periods.csv` (280
# additions), which `tests/test_timeunit.py` reads.
#
# Backlog item 33. Both files were committed in `e385a16` by a script that was
# never committed with them, so for a while they were the one thing in
# `tests/fixtures/` that could not be re-baselined -- and CLAUDE.md's claim that
# the fixtures come from `tools/oracle/*.R` was not true of them. This is that
# script, reconstructed from the grid the committed files describe.
#
# The grid is chosen for the calendar's awkward cases rather than for coverage:
# two leap days, the day after one, a 31st, a year boundary, and a start in
# mid-month. `clv.time` is a *calendar* abstraction for months and years -- it
# counts whole anniversaries and expresses the remainder as a fraction of the
# period that would have followed -- so those are where it can be wrong.

suppressMessages({
  library(CLVTools)
  library(data.table)
  library(lubridate)
})

OUT <- "tests/fixtures"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

ns <- asNamespace("CLVTools")

write_csv <- function(dt, name) {
  # `write.csv`, not the `ISO` its sibling generators use: the committed files
  # carry `2005-01-02 00:00:00` and ISO would rewrite every row as
  # `2005-01-02T00:00:00Z`. Same instants, 560 lines of noise, and
  # `tests/test_timeunit.py` parses the space-separated form.
  fwrite(dt, file.path(OUT, paste0(name, ".csv")), dateTimeAs = "write.csv")
  cat(sprintf("  %-38s %6d rows\n", paste0(name, ".csv"), nrow(dt)))
}

check <- function(label, got, want, tol = 1e-9) {
  if (!isTRUE(all.equal(as.numeric(got), as.numeric(want), tolerance = tol))) {
    stop(sprintf("oracle self-check failed for %s: got %.10g, want %.10g",
                 label, as.numeric(got), as.numeric(want)), call. = FALSE)
  }
  cat(sprintf("  ok  %-34s %.6f\n", label, as.numeric(want)))
}

# `months` is deliberately absent: CLVTools has no month unit at all -- it
# rejects both "month" and "months" -- while S5 names one, so this package
# implements calendar months and holds them to internal consistency instead.
# See the README's findings and `tests/test_timeunit.py`.
UNITS <- c("hours", "days", "weeks", "years")

# Order matters only because it is the order the committed files are in, and a
# regenerated fixture that differs from them by 616 reordered lines is a diff
# nobody can read.
STARTS <- as.Date(c(
  "2005-01-02",  # the apparel cohort's own first day
  "2004-02-29",  # a leap day: lubridate's period(n, "years") returns NA here
  "2005-01-31",  # a 31st, so month-ends have no counterpart in short months
  "2000-01-01",  # a century leap year
  "2003-12-31",  # a year boundary from the last day of a year
  "2004-03-01",  # the day after a leap day, where the anniversary is not
  "2005-06-15",  # an unremarkable mid-month day, as a control
  "1999-02-28",  # 28 February in a non-leap year
  "2008-02-29",  # a second leap day, four years on
  "2005-12-01"
))

# Offsets in days, chosen to straddle every boundary the units have: a day
# either side of 4, 8, 12 and 52 weeks, of one, two, three and four years, and
# of the leap day in each.
OFFSETS <- c(0, 1, 27, 28, 29, 30, 59, 60, 89, 180, 181,
             364, 365, 366, 367, 730, 731, 1095, 1461, 2178, 3650)

# Period counts spanning one period to a year of them.
COUNTS <- c(1, 2, 3, 12, 52, 104, 365)

clv_time <- function(unit) get(paste0("clv.time.", unit), ns)(time.format = "ymd")
elapsed  <- get("clv.time.interval.in.number.tu", ns)
as_period <- get("clv.time.number.timeunits.to.timeperiod", ns)

cat("== elapsed ==\n")
rows <- rbindlist(lapply(UNITS, function(unit) {
  ct <- clv_time(unit)
  rbindlist(lapply(STARTS, function(start) {
    ends <- start + OFFSETS
    data.table(
      unit = unit, start = start, end = ends, offset_days = OFFSETS,
      elapsed = vapply(
        ends,
        function(end) as.numeric(elapsed(ct, interv = interval(start, end))),
        numeric(1)
      )
    )
  }))
}))
write_csv(rows, "time_elapsed")

cat("== add periods ==\n")
added <- rbindlist(lapply(UNITS, function(unit) {
  ct <- clv_time(unit)
  rbindlist(lapply(STARTS, function(start) {
    # `as.POSIXct` because the hour unit's additions are not whole days, and a
    # `Date` column would silently truncate them.
    origin <- as.POSIXct(paste(format(start), "00:00:00"), tz = "UTC")
    ends <- do.call(c, lapply(
      COUNTS, function(n) origin + as_period(ct, user.number.periods = n)
    ))
    # Formatted here rather than left to `fwrite`, which renders a POSIXct at
    # exactly midnight as a bare date and would write `2005-01-02` where the
    # committed file has `2005-01-02 00:00:00`. The hour unit's own additions
    # are not whole days, so the column has to keep its time either way.
    data.table(
      unit = unit,
      start = format(origin, "%Y-%m-%d %H:%M:%S"),
      n = COUNTS,
      end = format(ends, "%Y-%m-%d %H:%M:%S")
    )
  }))
}))
write_csv(added, "time_add_periods")

# The self-checks CLAUDE.md asks for. Each is a value the *public* calendar
# agrees on, so a unit or ordering slip in the grid above cannot ship.
cat("== self-checks ==\n")
check("days elapsed over a week",
      rows[unit == "days" & offset_days == 365, elapsed][[1]], 365)
check("weeks elapsed over 364 days",
      rows[unit == "weeks" & offset_days == 364, elapsed][[1]], 52)
check("hours elapsed over one day",
      rows[unit == "hours" & offset_days == 1, elapsed][[1]], 24)

# A whole anniversary is exactly 1, which is the property that separates a
# calendar unit from a division by a fixed length.
anniversary <- rows[unit == "years" & start == as.Date("2005-01-02") &
                    offset_days == 365, elapsed][[1]]
check("years elapsed to the anniversary", anniversary, 1)

# And `add` inverts `elapsed`, which is the invariant the Python side asserts
# in both directions.
ct <- clv_time("weeks")
start <- as.POSIXct("2005-01-02 00:00:00", tz = "UTC")
check("add(52 weeks) then elapsed",
      elapsed(ct, interv = interval(start,
              start + as_period(ct, user.number.periods = 52))), 52)

cat("\ndone\n")
