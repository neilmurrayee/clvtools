r"""S5 - the unit of time.

The fixed units are arithmetic and hold no surprises. The calendar units do,
and the fixtures here pin them: 840 spans and 280 additions across ten start
dates -- including two leap days, a 31st, and a year boundary -- generated from
CLVTools' own ``clv.time`` classes.

Months have no counterpart in the reference, so they are checked for internal
consistency instead: against the year unit where the two must agree, against
their own inverse, and against the identities any sane definition has to
satisfy.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_csv

from clvtools import timeunit

#: The reference's names for the units it implements.
R_NAMES = {"hours": "hour", "days": "day", "weeks": "week", "years": "year"}


@pytest.fixture(scope="module")
def elapsed_grid():
    return fixture_csv("time_elapsed", parse_dates=["start", "end"])


@pytest.fixture(scope="module")
def add_grid():
    return fixture_csv("time_add_periods", parse_dates=["start"])


@pytest.mark.oracle
class TestAgainstOracle:
    @pytest.mark.parametrize("r_name,name", list(R_NAMES.items()))
    def test_elapsed_matches(self, elapsed_grid, r_name, name):
        rows = elapsed_grid[elapsed_grid["unit"] == r_name]
        unit = timeunit.get(name)
        got = np.array(
            [unit.elapsed(a, b) for a, b in zip(rows["start"], rows["end"], strict=True)]
        )
        np.testing.assert_allclose(got, rows["elapsed"], rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("r_name,name", list(R_NAMES.items()))
    def test_add_matches(self, add_grid, r_name, name):
        rows = add_grid[add_grid["unit"] == r_name]
        unit = timeunit.get(name)
        compared = 0
        for _, row in rows.iterrows():
            if pd.isna(row["end"]):
                continue  # see test_leap_day_addition_is_defined_here
            assert unit.add(row["start"], row["n"]) == pd.Timestamp(row["end"]), (
                f"{name}: {row['start']} + {row['n']}"
            )
            compared += 1
        assert compared > 0

    def test_leap_day_addition_is_defined_here(self, add_grid):
        r"""Eight rows where the reference returns ``NA`` and this does not.

        lubridate's ``period(n, "years")`` added to 29 February yields ``NA``,
        so CLVTools cannot express an estimation split of *n* years from a
        leap-day start. Its own ``time_length`` is more forgiving: it treats
        the anniversary as 1 March, which is why 2004-02-29 to 2005-03-01 comes
        back as exactly 1.0. This package takes that second convention for both
        directions, so ``add`` and ``elapsed`` stay inverse to each other.
        """
        rows = add_grid[(add_grid["unit"] == "years") & add_grid["end"].isna()]
        assert len(rows) == 8
        assert set(rows["start"].dt.strftime("%m-%d")) == {"02-29"}

        year = timeunit.get("year")
        got = year.add(pd.Timestamp("2004-02-29"), 1)
        assert got == pd.Timestamp("2005-03-01")
        assert year.elapsed(pd.Timestamp("2004-02-29"), got) == 1.0


class TestFixedUnits:
    @pytest.mark.parametrize("name,days", [("hour", 1 / 24), ("day", 1.0), ("week", 7.0)])
    def test_a_span_is_a_division(self, name, days):
        unit = timeunit.get(name)
        start = pd.Timestamp("2005-01-02")
        assert unit.elapsed(start, start + pd.Timedelta(days=days * 3)) == 3.0

    @pytest.mark.parametrize("name", ["hour", "day", "week"])
    def test_add_inverts_elapsed(self, name):
        unit = timeunit.get(name)
        start = pd.Timestamp("2005-06-15 13:45:00")
        for n in (0.0, 1.0, 2.5, 104.0):
            assert unit.elapsed(start, unit.add(start, n)) == pytest.approx(n)

    def test_periods_per_year_follows_the_paper(self):
        r"""S6.3.2: "k = 52 for weekly, k = 365 for daily units".

        Declared rather than derived: 365/7 is 52.14, and the paper says 52.
        """
        assert timeunit.get("week").periods_per_year == 52.0
        assert timeunit.get("day").periods_per_year == 365.0
        assert timeunit.get("hour").periods_per_year == 8760.0
        assert timeunit.get("year").periods_per_year == 1.0
        assert timeunit.get("month").periods_per_year == 12.0


class TestCalendarUnits:
    r"""Years and months count anniversaries, not days."""

    def test_a_whole_anniversary_is_an_integer(self):
        year = timeunit.get("year")
        for start in ("2005-01-02", "2004-01-01", "2003-12-31", "2005-06-15"):
            for n in (1, 2, 5):
                start_ts = pd.Timestamp(start)
                assert year.elapsed(start_ts, year.add(start_ts, n)) == float(n)

    def test_the_same_number_of_days_can_be_different_fractions(self):
        r"""Which is the whole reason a calendar unit is not a division.

        365 days from a common-year start is exactly 1; the same 365 days from
        a leap-year start is a little less, because the year it is measured
        against is 366 days long.
        """
        year = timeunit.get("year")
        common = year.elapsed(pd.Timestamp("2005-01-02"), pd.Timestamp("2006-01-02"))
        leap = year.elapsed(pd.Timestamp("2004-01-01"), pd.Timestamp("2004-12-31"))
        assert common == 1.0
        assert leap < 1.0

    def test_add_inverts_elapsed_for_fractional_periods(self):
        for name in ("year", "month"):
            unit = timeunit.get(name)
            start = pd.Timestamp("2005-03-15")
            for n in (0.0, 0.25, 1.0, 2.5, 13.75):
                assert unit.elapsed(start, unit.add(start, n)) == pytest.approx(
                    n, abs=1e-9
                ), f"{name} at {n}"

    def test_twelve_months_is_a_year(self):
        """Where the two calendar units overlap, they must agree exactly."""
        month, year = timeunit.get("month"), timeunit.get("year")
        for start in ("2005-01-02", "2004-02-29", "2005-01-31", "2003-12-31"):
            start_ts = pd.Timestamp(start)
            for n in (1, 2, 5):
                assert month.add(start_ts, 12 * n) == year.add(start_ts, n)
                anniversary = year.add(start_ts, n)
                assert month.elapsed(start_ts, anniversary) == float(12 * n)

    def test_a_partial_span_is_not_twelve_times_the_yearly_one(self):
        r"""And should not be expected to be.

        Both count whole anniversaries and then take a fraction, but of
        different things: months divide by the length of the next *month*,
        years by the length of the next *year*. 2005-01-02 to 2007-08-09 is
        31.2258 months and 2.6 years -- 31.2 months if scaled. The two agree
        only on whole periods, which is what the test above pins.
        """
        month, year = timeunit.get("month"), timeunit.get("year")
        a, b = pd.Timestamp("2005-01-02"), pd.Timestamp("2007-08-09")
        assert month.elapsed(a, b) == pytest.approx(31.225806, abs=1e-6)
        assert 12 * year.elapsed(a, b) == pytest.approx(31.2, abs=1e-6)

    def test_a_month_end_start_rolls_forward(self):
        r"""31 January plus one month has no 31st to land on.

        The same rule as 29 February: roll into the next month rather than
        clamping back, so that ``add`` stays monotone in the number of periods.
        """
        month = timeunit.get("month")
        assert month.add(pd.Timestamp("2005-01-31"), 1) == pd.Timestamp("2005-03-03")
        assert month.add(pd.Timestamp("2005-01-31"), 2) == pd.Timestamp("2005-03-31")

    def test_add_is_monotone(self):
        for name in ("year", "month"):
            unit = timeunit.get(name)
            for start in ("2005-01-31", "2004-02-29", "2005-06-15"):
                start_ts = pd.Timestamp(start)
                points = [unit.add(start_ts, n) for n in range(30)]
                assert all(a < b for a, b in pairwise(points)), name

    def test_the_anniversary_estimate_only_ever_overshoots(self):
        r"""Which is why :meth:`elapsed` corrects in one direction only.

        ``_anniversary(start, k)`` never lands earlier than ``start`` plus
        ``k`` calendar months, because a day that does not exist in the target
        month rolls *forward*. So the estimate taken from the month difference
        can be too large but never too small, and an upward correction would be
        unreachable code. Checked here across every start day and a range of
        spans, since the argument is easy to state and easy to get wrong.
        """
        for name in ("month", "year"):
            unit = timeunit.get(name)
            for year in range(1999, 2009):
                for month_of in range(1, 13):
                    for day in (1, 15, 28, 29, 30, 31):
                        try:
                            start = pd.Timestamp(year=year, month=month_of, day=day)
                        except ValueError:
                            continue
                        for offset in (1, 28, 29, 31, 59, 180, 365, 366, 730, 1461):
                            end = start + pd.Timedelta(days=offset)
                            whole = (
                                end.year * 12 + end.month
                                - start.year * 12 - start.month
                            ) // unit.months
                            while unit._anniversary(start, whole) > end:
                                whole -= 1
                            assert unit._anniversary(start, whole + 1) > end

    def test_whole_periods_are_counted_by_correction_not_by_estimate(self):
        r"""The month-difference estimate can be one short, and is fixed up.

        From 31 January to 3 March the month difference is 2, but only one whole
        month has passed -- 31 January plus one month is 3 March. Both the
        upward and downward correction have to work.
        """
        month = timeunit.get("month")
        assert month.elapsed(
            pd.Timestamp("2005-01-31"), pd.Timestamp("2005-03-03")
        ) == 1.0
        assert month.elapsed(
            pd.Timestamp("2005-01-31"), pd.Timestamp("2005-03-02")
        ) < 1.0
        # And from a month start, where the estimate is already right.
        assert month.elapsed(
            pd.Timestamp("2005-01-01"), pd.Timestamp("2005-03-01")
        ) == 2.0

    def test_elapsed_is_signed(self):
        year = timeunit.get("year")
        a, b = pd.Timestamp("2005-01-02"), pd.Timestamp("2007-01-02")
        assert year.elapsed(a, b) == 2.0
        assert year.elapsed(b, a) == -2.0

    def test_elapsed_is_zero_over_no_time(self):
        for name in ("year", "month"):
            when = pd.Timestamp("2005-06-15")
            assert timeunit.get(name).elapsed(when, when) == 0.0


class TestFloorAndCeiling:
    def test_year_boundaries(self):
        year = timeunit.get("year")
        assert year.floor(pd.Timestamp("2005-06-15")) == pd.Timestamp("2005-01-01")
        assert year.ceiling(pd.Timestamp("2005-06-15")) == pd.Timestamp("2006-01-01")

    def test_ceiling_advances_a_whole_period_on_a_boundary(self):
        r"""CLVTools' ``change_on_boundary = TRUE``.

        The dynamic-covariate walks depend on it -- see ``d_omega``, which is 1
        for every apparel customer precisely because their first purchase lands
        on a covariate boundary.
        """
        year = timeunit.get("year")
        assert year.ceiling(pd.Timestamp("2005-01-01")) == pd.Timestamp("2006-01-01")
        month = timeunit.get("month")
        assert month.ceiling(pd.Timestamp("2005-06-01")) == pd.Timestamp("2005-07-01")

    def test_month_boundaries(self):
        month = timeunit.get("month")
        assert month.floor(pd.Timestamp("2005-06-15")) == pd.Timestamp("2005-06-01")
        assert month.ceiling(pd.Timestamp("2005-06-15")) == pd.Timestamp("2005-07-01")

    @pytest.mark.parametrize(
        "name,floored,ceiled",
        [
            ("hour", "2005-06-15 13:00:00", "2005-06-15 14:00:00"),
            ("day", "2005-06-15 00:00:00", "2005-06-16 00:00:00"),
            ("week", "2005-06-15 00:00:00", "2005-06-22 00:00:00"),
        ],
    )
    def test_fixed_unit_boundaries(self, name, floored, ceiled):
        r"""Weeks floor to the day, not to a calendar week.

        A "week" here is seven days from wherever the data starts, so anchoring
        to Monday or Sunday would be meaningless. The dynamic-covariate model
        takes its boundaries from the covariate grid it is given rather than
        from this.
        """
        unit = timeunit.get(name)
        when = pd.Timestamp("2005-06-15 13:45:30")
        assert unit.floor(when) == pd.Timestamp(floored)
        assert unit.ceiling(when) == pd.Timestamp(ceiled)


class TestLookup:
    def test_every_unit_the_paper_names_is_available(self):
        r"""S5: "hour, day, week, month, year"."""
        assert set(timeunit.TIME_UNITS) == {"hour", "day", "week", "month", "year"}

    def test_rejects_an_unknown_unit(self):
        with pytest.raises(ValueError, match="time_unit must be one of"):
            timeunit.get("fortnight")

    def test_repr_names_the_class(self):
        assert repr(timeunit.get("year")) == "Years()"

    def test_the_base_class_defines_the_interface(self):
        """Subclasses must supply all four operations."""
        base = timeunit.TimeUnit()
        when = pd.Timestamp("2005-01-02")
        for call in (
            lambda: base.elapsed(when, when),
            lambda: base.add(when, 1),
            lambda: base.floor(when),
            lambda: base.ceiling(when),
        ):
            with pytest.raises(NotImplementedError):
                call()


class TestClvDataWithCalendarUnits:
    """The unit reaching through to the model inputs."""

    @staticmethod
    @pytest.fixture(scope="class")
    def transactions():
        from clvtools import load_apparel_trans

        return load_apparel_trans()

    def test_a_two_year_split_matches_the_104_week_one(self, transactions):
        r"""2005-01-02 plus 104 weeks is 2006-12-31; plus 2 years is 2007-01-02.

        Not the same date, so not the same ``x`` -- but the two should be close,
        and this is the check that the year path reaches the model inputs at
        all.
        """
        from clvtools import ClvData

        weekly = ClvData(transactions, time_unit="week", estimation_split=104)
        yearly = ClvData(transactions, time_unit="year", estimation_split=2)

        assert weekly.estimation_end == pd.Timestamp("2006-12-31")
        assert yearly.estimation_end == pd.Timestamp("2007-01-02")

        summary = yearly.customer_summary()
        assert len(summary) == 600
        # Every customer's window is exactly two years, by construction.
        np.testing.assert_allclose(summary["T"], 2.0)

    def test_recency_is_expressed_in_the_chosen_unit(self, transactions):
        from clvtools import ClvData

        weekly = ClvData(transactions, time_unit="week", estimation_split=104)
        yearly = ClvData(transactions, time_unit="year", estimation_split=2)
        w = weekly.customer_summary().set_index("Id")
        y = yearly.customer_summary().set_index("Id")

        # A customer whose last purchase is inside both windows: the same
        # instant, expressed in weeks and in years.
        both = w.index[(w["x"] > 0) & (w["t_x"] < 100)]
        assert len(both) > 0
        for customer in both[:20]:
            assert y.loc[customer, "t_x"] * 52 == pytest.approx(
                w.loc[customer, "t_x"], rel=0.02
            )

    def test_monthly_units_work_end_to_end(self, transactions):
        """Months have no oracle, so this asserts the shape rather than digits."""
        from clvtools import ClvData
        from clvtools.pnbd import fit_pnbd

        monthly = ClvData(transactions, time_unit="month", estimation_split=24)
        assert monthly.estimation_end == pd.Timestamp("2007-01-02")

        summary = monthly.customer_summary()
        np.testing.assert_allclose(summary["T"], 24.0)

        fitted = fit_pnbd(
            summary["x"], summary["t_x"], summary["T"], hessian=False
        )
        assert fitted.converged
        assert np.isfinite(fitted.log_likelihood)

    def test_rescaling_the_unit_only_shifts_the_likelihood_by_a_jacobian(
        self, transactions
    ):
        r"""S5: "The choice of time unit is arbitrary".

        Rescaling time by :math:`c` divides both latent rates by :math:`c`,
        which multiplies :math:`\alpha` and :math:`\beta` by :math:`c`. That
        leaves the *model* unchanged, but not the log-likelihood: it carries a
        density in :math:`t_x`, so each of the :math:`\sum x` repeat
        transactions contributes a Jacobian factor. Measuring in days instead of
        weeks therefore shifts it by exactly :math:`-\sum x \log 7` and by
        nothing else.

        This is the sharp form of "arbitrary" -- and a check that the two units
        produce genuinely consistent inputs, not merely similar ones.
        """
        from clvtools import ClvData
        from clvtools.pnbd import log_likelihood

        weekly = ClvData(transactions, time_unit="week", estimation_split=104)
        daily = ClvData(transactions, time_unit="day", estimation_split=728)
        assert weekly.estimation_end == daily.estimation_end

        w = weekly.customer_summary()
        d = daily.customer_summary()
        np.testing.assert_array_equal(w["x"], d["x"])

        in_weeks = log_likelihood(
            w["x"], w["t_x"], w["T"], 1.4490, 48.6361, 0.5613, 46.8844
        )
        in_days = log_likelihood(
            d["x"], d["t_x"], d["T"], 1.4490, 48.6361 * 7, 0.5613, 46.8844 * 7
        )
        jacobian = float(w["x"].sum()) * np.log(7.0)
        assert in_days == pytest.approx(in_weeks - jacobian, abs=1e-9)

    def test_a_rescaled_unit_gives_the_same_fitted_model(self, transactions):
        """And the fits agree once the rates are put back on one scale."""
        from clvtools import ClvData
        from clvtools.pnbd import fit_pnbd

        w = ClvData(transactions, time_unit="week", estimation_split=104
                    ).customer_summary()
        d = ClvData(transactions, time_unit="day", estimation_split=728
                    ).customer_summary()
        in_weeks = fit_pnbd(w["x"], w["t_x"], w["T"], hessian=False)
        in_days = fit_pnbd(d["x"], d["t_x"], d["T"], hessian=False)

        assert in_days.r == pytest.approx(in_weeks.r, rel=1e-3)
        assert in_days.s == pytest.approx(in_weeks.s, rel=1e-3)
        assert in_days.alpha == pytest.approx(in_weeks.alpha * 7, rel=1e-3)
        assert in_days.beta == pytest.approx(in_weeks.beta * 7, rel=1e-3)
