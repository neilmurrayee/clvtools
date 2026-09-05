r"""S5 - the unit of time, and the arithmetic that depends on it.

S5: "the business context also determines the choice of the relevant *unit of
time* (e.g., hour, day, week, month, year). To define the input data for a
latent attrition model, the chosen time unit is used to convert relevant time
spans into meaningfully scaled numbers. […] The choice of time unit is
arbitrary, but it can be a convenient tool to facilitate interpretation and
improve numerical stability."

Two kinds of unit, and they are not interchangeable.

**Fixed** units -- hours, days, weeks -- are a constant number of days, so a
span is a division and adding :math:`n` periods is an addition. That is all the
weekly analysis of S6 ever needs.

**Calendar** units -- months, years -- are not. A year is 365 or 366 days
depending on where it starts, so "1.5 years" has to mean something in terms of
actual anniversaries. The convention here is the one CLVTools inherits from
lubridate: count whole calendar anniversaries, then express the remainder as a
fraction of the year that *would have followed*. So 2005-01-02 to 2006-12-31 is
1.9945 years, not 728/365 = 1.9945… by coincidence, but because it is one whole
year plus 363/365 of the next.

Leap days are the awkward case, and the convention is worth stating: the
anniversary of 29 February rolls **forward** to 1 March in non-leap years. So
2004-02-29 to 2005-03-01 is exactly 1.0 years, and the year it spans is 366 days
long. :class:`Years` reproduces that.

.. note::
   CLVTools supports hours, days, weeks and years. Months are implemented here
   as well, by the same calendar rule, because S5 names them and S5 recommends
   reasoning in them -- "it often makes sense to at least look at the last 12
   months […] extended to 24 months when strong seasonal patterns are observed."
   They have no counterpart in the reference implementation, so they are checked
   for internal consistency rather than against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["TIME_UNITS", "Days", "Hours", "Months", "TimeUnit", "Weeks", "Years", "get"]


class TimeUnit:
    """A unit of time, and the four operations the package needs of one."""

    name: str
    #: How many of this unit make a year. Used to scale discount rates, S6.3.2.
    periods_per_year: float

    def elapsed(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """The span ``[start, end]``, expressed in this unit."""
        raise NotImplementedError

    def add(self, when: pd.Timestamp, periods: float) -> pd.Timestamp:
        """``when`` advanced by ``periods`` of this unit."""
        raise NotImplementedError

    def floor(self, when: pd.Timestamp) -> pd.Timestamp:
        """The start of the period ``when`` falls in."""
        raise NotImplementedError

    def ceiling(self, when: pd.Timestamp) -> pd.Timestamp:
        """The start of the *next* period.

        A timepoint already sitting on a boundary advances a whole period, which
        is CLVTools' ``change_on_boundary = TRUE``. The dynamic-covariate walks
        depend on it: without it a transaction on a period boundary would
        contribute nothing to its own period.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, repr=False)
class _Fixed(TimeUnit):
    """A unit of constant length. Spans are division; addition is addition."""

    name: str
    days: float
    freq: str
    #: Declared rather than derived as ``365 / days``. S6.3.2 fixes the
    #: convention: "k is the number of time units per year (e.g. k = 52 for
    #: weekly, k = 365 for daily units)" -- so a year holds 52 weeks, not
    #: 365/7 = 52.14.
    periods_per_year: float

    def elapsed(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        return (end - start).total_seconds() / (86400.0 * self.days)

    def add(self, when: pd.Timestamp, periods: float) -> pd.Timestamp:
        return when + pd.Timedelta(days=float(periods) * self.days)

    def floor(self, when: pd.Timestamp) -> pd.Timestamp:
        return when.floor(self.freq)

    def ceiling(self, when: pd.Timestamp) -> pd.Timestamp:
        return self.floor(when) + pd.Timedelta(days=self.days)


class _Calendar(TimeUnit):
    """A unit of variable length, measured in whole anniversaries plus a fraction."""

    #: How many calendar months make one period of this unit.
    months: int

    def _anniversary(self, when: pd.Timestamp, periods: int) -> pd.Timestamp:
        r"""``when`` advanced by whole periods, keeping the day of the month.

        29 February rolls **forward** to 1 March where the target year has no
        such date, matching lubridate and so CLVTools. Note that pandas'
        ``DateOffset`` clamps *backward* to 28 February instead, which is why
        this is written out rather than delegated.
        """
        total = (when.year * 12 + (when.month - 1)) + periods * self.months
        year, month = divmod(total, 12)
        month += 1
        try:
            return pd.Timestamp(
                year=year, month=month, day=when.day,
                hour=when.hour, minute=when.minute, second=when.second,
            )
        except ValueError:
            # The day does not exist in the target month: 31st of a short
            # month, or 29 February of a common year. Roll into the next month.
            first_of_next = pd.Timestamp(
                year=year + (month == 12), month=(month % 12) + 1, day=1,
                hour=when.hour, minute=when.minute, second=when.second,
            )
            overflow = when.day - pd.Timestamp(year=year, month=month, day=1).days_in_month
            return first_of_next + pd.Timedelta(days=overflow - 1)

    def elapsed(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        if end == start:
            return 0.0
        sign = 1 if end > start else -1
        if sign < 0:
            start, end = end, start

        # Whole anniversaries first, estimated from the month difference and
        # then corrected downward. The correction is only ever downward:
        # `_anniversary(start, k)` never falls *earlier* than `start` plus `k`
        # calendar months, because an impossible day rolls forward rather than
        # clamping back. So the estimate can overshoot but never undershoot.
        # `test_the_anniversary_estimate_only_ever_overshoots` checks that over
        # 18,000 date pairs.
        whole = (end.year * 12 + end.month - start.year * 12 - start.month) // self.months
        while self._anniversary(start, whole) > end:
            whole -= 1

        anchor = self._anniversary(start, whole)
        if anchor == end:
            return float(sign * whole)

        following = self._anniversary(start, whole + 1)
        fraction = (end - anchor).total_seconds() / (following - anchor).total_seconds()
        return float(sign * (whole + fraction))

    def add(self, when: pd.Timestamp, periods: float) -> pd.Timestamp:
        r"""Advance by ``periods``, which need not be whole.

        A fractional part is taken as that fraction of the period it lands in,
        which is the inverse of :meth:`elapsed`.
        """
        whole = int(periods // 1)
        remainder = periods - whole
        anchor = self._anniversary(when, whole)
        if remainder == 0.0:
            return anchor
        following = self._anniversary(when, whole + 1)
        return anchor + (following - anchor) * remainder

    def floor(self, when: pd.Timestamp) -> pd.Timestamp:
        if self.months == 12:
            return pd.Timestamp(year=when.year, month=1, day=1)
        return pd.Timestamp(year=when.year, month=when.month, day=1)

    def ceiling(self, when: pd.Timestamp) -> pd.Timestamp:
        return self._anniversary(self.floor(when), 1)


class Hours(_Fixed):
    """One hour. S6.1 aggregates to the second at this resolution."""

    def __init__(self) -> None:
        super().__init__(name="hour", days=1.0 / 24.0, freq="h",
                         periods_per_year=8760.0)


class Days(_Fixed):
    """One day."""

    def __init__(self) -> None:
        super().__init__(name="day", days=1.0, freq="D",
                         periods_per_year=365.0)


class Weeks(_Fixed):
    r"""One week -- S5's recommendation.

    "In practice, a time unit may be chosen such that the typical interpurchase
    time is a few single-digit time units long. In most settings, this means
    choosing ``"weeks"`` as time unit."
    """

    def __init__(self) -> None:
        super().__init__(name="week", days=7.0, freq="D",
                         periods_per_year=52.0)

    def ceiling(self, when: pd.Timestamp) -> pd.Timestamp:
        # Weeks are not anchored to a calendar week here: a "week" is seven days
        # from wherever the data starts, so flooring is to the day and the
        # ceiling is the next day boundary a week later. The covariate grid in
        # the dynamic-covariate model supplies its own boundaries anyway.
        return self.floor(when) + pd.Timedelta(days=7.0)


class Months(_Calendar):
    """One calendar month. No counterpart in CLVTools; see the module docstring."""

    name = "month"
    months = 1
    periods_per_year = 12.0


class Years(_Calendar):
    """One calendar year, counted by anniversary."""

    name = "year"
    months = 12
    periods_per_year = 1.0


#: Every unit the package accepts, by the name :class:`~clvtools.data.ClvData`
#: takes.
TIME_UNITS: dict[str, TimeUnit] = {
    "hour": Hours(),
    "day": Days(),
    "week": Weeks(),
    "month": Months(),
    "year": Years(),
}


def _resolve(name: str) -> str | None:
    """The unit ``name`` refers to, in CLVTools' spelling or any of R's.

    ``clvdata()`` matches its ``time.unit`` the way R's ``match.arg`` does:
    case-insensitively, and on any unambiguous prefix. Asked, it accepts
    ``"w"``, ``"week"``, ``"weeks"``, ``"Weeks"`` and ``"WEEK"`` alike, and
    this package took only the exact lowercase singular -- so code that works
    against CLVTools failed here on a spelling. Spec ``T-07``.

    An ambiguous prefix is refused rather than guessed:

    >>> _resolve("Weeks"), _resolve("d"), _resolve("YEAR")
    ('week', 'day', 'year')
    >>> _resolve("fortnight") is None
    True

    Note the one place the two differ in the *other* direction: CLVTools has no
    month unit at all -- it rejects ``"month"`` and ``"months"`` -- while this
    package implements calendar months, which S5 describes. That is a
    deliberate extension and is recorded in the README's findings.
    """
    lowered = name.strip().lower()
    if lowered in TIME_UNITS:
        return lowered
    singular = lowered.removesuffix("s")
    if singular in TIME_UNITS:
        return singular
    matches = [unit for unit in TIME_UNITS if unit.startswith(singular)]
    return matches[0] if len(matches) == 1 and singular else None


def get(name: str) -> TimeUnit:
    """Look up a unit by name.

    Examples
    --------
    >>> import pandas as pd
    >>> week = get("week")
    >>> week.elapsed(pd.Timestamp("2005-01-02"), pd.Timestamp("2006-12-31"))
    104.0

    Calendar units count anniversaries, not days. The same span is a little
    under two years, because the leftover 363 days are measured against the
    365-day year that would have followed:

    >>> year = get("year")
    >>> round(year.elapsed(pd.Timestamp("2005-01-02"), pd.Timestamp("2006-12-31")), 6)
    1.994521

    A leap-day anniversary falls on 1 March, so the year it spans is 366 days:

    >>> round(year.elapsed(pd.Timestamp("2004-02-29"), pd.Timestamp("2005-02-28")), 6)
    0.997268
    >>> year.elapsed(pd.Timestamp("2004-02-29"), pd.Timestamp("2005-03-01"))
    1.0
    """
    resolved = _resolve(name)
    if resolved is None:
        raise ValueError(
            f"time_unit must be one of {sorted(TIME_UNITS)}, got {name!r}"
        )
    name = resolved
    return TIME_UNITS[name]
