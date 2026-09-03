r"""Building the time-varying covariate walks from raw data.

Split out of ``test_pnbd_dyncov.py``, which had outgrown the 700-code-line
module limit. What lives here is everything about *constructing* the walks --
the oracle's own walk tables, the calendar the split falls on, and the input
validation -- as against evaluating the likelihood over them, which stays
there.

Two of these classes exist because the apparel data cannot exercise what they
check. Every dyncov test in this repository runs at ``estimation_split=104``,
which lands exactly on the weekly covariate grid, and every one of the 600
apparel customers was born on a Sunday; so ``d1`` and ``d_omega`` are 1
throughout and six of the seven possible alignments are never seen. Findings
B5 and D4 of ``docs/spec-audit.md``.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from conftest import DYNCOV_CASES, DYNCOV_GRID, dyncov_grid_case, fixture_csv

from clvtools.pnbd.dyncov import log_likelihood


@pytest.mark.oracle
class TestWalkConstruction:
    r"""Building the walks from raw data, against the oracle's own tables.

    This is checked separately from the likelihood, and before it: the
    :func:`dyncov_walks` fixture loads CLVTools' walks directly, so a fault in
    construction cannot hide behind a correct likelihood or the reverse.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def built():
        from clvtools import ClvData, load_apparel_dyn_cov, load_apparel_trans
        from clvtools.pnbd.dyncov import build_walks

        return build_walks(
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
            load_apparel_dyn_cov(),
            names_cov_life=["High.Season", "Gender", "Channel"],
            names_cov_trans=["High.Season", "Gender", "Channel"],
        )

    def test_customer_summary_matches(self, built):
        want = fixture_csv("dyncov_cbs").set_index("Id").loc[built.ids]
        np.testing.assert_array_equal(built.x, want["x"])
        np.testing.assert_array_equal(built.T_cal, want["T.cal"])
        np.testing.assert_array_equal(built.d_omega, want["d_omega"])
        # t_x is computed from whole days here and is the exact value; the
        # fixture went through CSV at 15 significant digits.
        np.testing.assert_allclose(built.t_x, want["t.x"], atol=1e-12)

    def test_d_omega_is_one_for_this_cohort(self, built):
        r"""Every first purchase falls on a covariate boundary.

        The apparel cohort's covariate grid starts on the cohort's own first
        purchase date, so ``d_omega`` takes the boundary value of 1 throughout.
        This is what pins the "``d`` is 1 on the lower boundary" rule: with the
        naive definition it would be 0 for all 600 customers.
        """
        np.testing.assert_array_equal(built.d_omega, np.ones(600))

    def test_auxiliary_walk_indices_match(self, built):
        want = fixture_csv("dyncov_walkinfo").set_index("Id").loc[built.ids]
        np.testing.assert_array_equal(
            built.walkinfo_aux_life, want[["aux_life_from", "aux_life_to"]]
        )
        np.testing.assert_array_equal(
            built.walkinfo_aux_trans[:, :2],
            want[["aux_trans_from", "aux_trans_to"]],
        )

    def test_auxiliary_transaction_walk_d1_and_tjk_match(self, built):
        want = fixture_csv("dyncov_walkinfo").set_index("Id").loc[built.ids]
        np.testing.assert_allclose(
            built.walkinfo_aux_trans[:, 2:],
            want[["aux_trans_d1", "aux_trans_tjk"]],
            atol=1e-12,
        )

    def test_real_lifetime_walk_indices_match(self, built):
        want = fixture_csv("dyncov_walkinfo").set_index("Id").loc[built.ids]
        want_values = want[["real_life_from", "real_life_to"]].to_numpy(float)
        np.testing.assert_array_equal(
            np.isfinite(built.walkinfo_real_life), np.isfinite(want_values)
        )
        present = np.isfinite(want_values)
        np.testing.assert_array_equal(
            built.walkinfo_real_life[present], want_values[present]
        )

    def test_real_transaction_walks_match(self, built):
        want = fixture_csv("dyncov_walkinfo_real_trans")[
            ["walk_from", "walk_to", "d1", "tjk"]
        ].to_numpy(float)
        assert built.walkinfo_real_trans.shape == want.shape
        np.testing.assert_allclose(built.walkinfo_real_trans, want, atol=1e-12)

    def test_covariate_matrices_match(self, built):
        for kind in ("aux_life", "real_life", "aux_trans", "real_trans"):
            want = fixture_csv(f"dyncov_covdata_{kind}").to_numpy(dtype=float)
            got = getattr(built, f"covdata_{kind}")
            np.testing.assert_array_equal(got, want, err_msg=kind)

    def test_the_likelihood_agrees_with_the_oracles_walks(self, built):
        """End to end: raw data in, CLVTools' log-likelihood out."""
        for case in DYNCOV_CASES:
            r, alpha, s, beta, g_life, g_trans = dyncov_grid_case(case)
            total = log_likelihood(built, r, alpha, s, beta, g_life, g_trans)
            assert total == pytest.approx(DYNCOV_GRID["LL.sum"][case], abs=1e-8), case

    def test_day_aggregation_is_applied(self, built):
        """S6.1's collapse to the day, which the walks inherit.

        Passing a raw log instead would give a customer who bought twice in one
        day an extra transaction and an extra zero-length walk.
        """
        from clvtools import ClvData, load_apparel_trans

        data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        summary = data.customer_summary().set_index("Id").loc[built.ids]
        np.testing.assert_array_equal(built.x, summary["x"])

    def test_every_customer_has_an_auxiliary_walk(self, built):
        """It runs from the last transaction to the window end, always non-empty."""
        lengths = built.walkinfo_aux_trans[:, 1] - built.walkinfo_aux_trans[:, 0] + 1
        assert (lengths >= 1).all()

    def test_repeat_purchases_get_one_transaction_walk_each(self, built):
        counts = np.where(
            np.isfinite(built.real_trans_from),
            built.real_trans_to - built.real_trans_from + 1,
            0,
        )
        np.testing.assert_array_equal(counts, built.x)


class TestEveryWeekdaySplit:
    r"""DY-22: "All walks are basically correct for an ``estimation.split`` on
    every day of the week" -- seven cases in CLVTools' suite.

    Finding D4 of ``docs/spec-audit.md``. Every dyncov test in both files runs
    at ``estimation_split=104``, which lands exactly on the weekly covariate
    grid, so all 600-customer oracle comparisons here have been made at one
    alignment out of seven. The other six put the split part-way through a
    covariate interval, which is where the walk arithmetic has something to do.

    No oracle is needed for it. S3.3's nesting holds at any split -- with every
    coefficient zero this model *is* the plain Pareto/NBD -- and the plain
    likelihood is a closed form of the split's own ``(x, t_x, T)``. So each of
    the seven alignments has an independent answer to be checked against, and
    the seven answers differ: the customers are observed for a little longer
    each day, so the likelihood falls monotonically from -5848.1 to -5879.7.
    """

    #: 2006-12-31 is the Sunday the 104-week split falls on, and the covariate
    #: grid's own weekday. The following six days walk through the alignments.
    SPLITS: ClassVar[list[str]] = [
        (pd.Timestamp("2006-12-31") + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
        for n in range(7)
    ]

    @staticmethod
    @pytest.fixture(scope="class")
    def built():
        from clvtools import ClvData, load_apparel_dyn_cov, load_apparel_trans
        from clvtools.data import ClvDataDynCov

        names = ["High.Season", "Gender", "Channel"]
        transactions, covariates = load_apparel_trans(), load_apparel_dyn_cov()
        out = {}
        for split in TestEveryWeekdaySplit.SPLITS:
            base = ClvData(transactions, time_unit="week", estimation_split=split)
            data = ClvDataDynCov(
                base, covariates, names_cov_life=names, names_cov_trans=names
            )
            out[split] = (base, data.walks())
        return out

    @pytest.mark.parametrize("split", SPLITS)
    def test_the_walks_carry_the_splits_own_summary(self, built, split):
        base, walks = built[split]
        cbs = base.customer_summary().set_index("Id").loc[walks.ids]
        np.testing.assert_array_equal(walks.x, cbs["x"].to_numpy(dtype=float))
        np.testing.assert_allclose(walks.t_x, cbs["t_x"], rtol=1e-14)
        np.testing.assert_allclose(walks.T_cal, cbs["T"], rtol=1e-14)

    @pytest.mark.parametrize("split", SPLITS)
    def test_one_real_transaction_walk_per_repeat_purchase(self, built, split):
        _, walks = built[split]
        customers = walks.customers(np.zeros(3), np.zeros(3))
        assert len(customers) == 600
        for customer in customers:
            assert len(customer.real_walks_trans) == customer.x

    @pytest.mark.parametrize("split", SPLITS)
    def test_zero_coefficients_give_the_plain_likelihood(self, built, split):
        """The check with content: the plain Pareto/NBD is a closed form, so
        each alignment is compared against an answer the walk machinery had no
        part in."""
        from clvtools.pnbd import log_likelihood as plain_log_likelihood

        _, walks = built[split]
        r, alpha, s, beta = 1.4490, 48.6361, 0.5613, 46.8844
        dynamic = log_likelihood(
            walks, r, alpha, s, beta, np.zeros(3), np.zeros(3)
        )
        plain = plain_log_likelihood(
            walks.x, walks.t_x, walks.T_cal, r, alpha, s, beta
        )
        assert dynamic == pytest.approx(plain, rel=1e-12)

    def test_the_split_moves_the_window_but_not_the_births(self, built):
        r""":math:`d_\omega` is the fraction of their first covariate interval a
        customer is alive for, so it is fixed by their birth and cannot depend
        on where the estimation period is cut. ``T_cal`` must move, by a day a
        time."""
        d_omegas = [built[split][1].d_omega for split in self.SPLITS]
        for later in d_omegas[1:]:
            np.testing.assert_array_equal(later, d_omegas[0])

        spans = [float(built[split][1].T_cal.max()) for split in self.SPLITS]
        np.testing.assert_allclose(
            spans, [104.0 + n / 7.0 for n in range(7)], rtol=1e-12
        )

    def test_the_seven_answers_are_seven_different_answers(self, built):
        """What makes the parametrisation above worth running seven times.

        Each day observes the customers a little longer, so the likelihood of
        what was observed falls: had the split not reached the walks at all,
        these would be one number repeated.
        """
        from clvtools.pnbd import log_likelihood as plain_log_likelihood

        values = [
            plain_log_likelihood(
                built[split][1].x, built[split][1].t_x, built[split][1].T_cal,
                1.4490, 48.6361, 0.5613, 46.8844,
            )
            for split in self.SPLITS
        ]
        assert values == sorted(values, reverse=True)
        assert float(values[0]) == pytest.approx(-5848.0978, abs=1e-4)
        assert float(values[-1]) == pytest.approx(-5879.6527, abs=1e-4)


class TestDOmegaOffTheBoundary:
    r"""``d_omega`` where the apparel data cannot take it.

    Finding B5 of ``docs/spec-audit.md``: every one of the 600 apparel
    customers made their first purchase on a **Sunday**, and the covariate grid
    starts on a Sunday too, so ``d_omega`` is 1 for all of them. Comparing that
    column against the oracle discriminates nothing, and the second branch of
    :func:`~clvtools.pnbd.dyncov_walks._distance_to_interval_end` -- the one
    that measures a real distance -- is never reached through it.

    ``d_omega`` is the fraction of their first covariate interval a customer is
    alive for, so it is determined by the calendar alone. Four synthetic
    customers born on four different weekdays fix all four answers without an
    oracle, and CLVTools' own rule for the boundary -- "d shall be 1 if it is
    exactly on the time unit lower boundary" -- is the first of them.
    """

    #: Sunday, Wednesday, Friday, Saturday. The covariate grid below is the
    #: weeks beginning Sunday, so the days left in the birth week are 7, 4, 2
    #: and 1 -- and the Sunday customer gets a whole period, not zero.
    BIRTHS: ClassVar[dict[str, str]] = {
        "sunday": "2020-01-05",
        "wednesday": "2020-01-08",
        "friday": "2020-01-10",
        "saturday": "2020-01-11",
    }

    @staticmethod
    @pytest.fixture(scope="class")
    def walks():
        from clvtools import ClvData
        from clvtools.data import ClvDataDynCov

        grid = pd.date_range("2020-01-05", periods=30, freq="7D")
        rows = [
            (customer, date, price)
            for customer, birth in TestDOmegaOffTheBoundary.BIRTHS.items()
            for date, price in (
                (birth, 10.0), ("2020-03-01", 20.0), ("2020-06-07", 30.0)
            )
        ]
        transactions = pd.DataFrame(rows, columns=["Id", "Date", "Price"])
        transactions["Date"] = pd.to_datetime(transactions["Date"])
        covariates = pd.DataFrame(
            [(c, d, 1.0) for c in TestDOmegaOffTheBoundary.BIRTHS for d in grid],
            columns=["Id", "Cov.Date", "Marketing"],
        )
        data = ClvDataDynCov(
            ClvData(transactions, time_unit="week", estimation_split="2020-04-05"),
            covariates, names_cov_life=["Marketing"], names_cov_trans=["Marketing"],
        )
        return data.walks()

    def test_the_days_left_in_the_birth_week_are_the_multiplier(self, walks):
        # The ids come back sorted, as everywhere else, so the answers are
        # looked up by name rather than by position.
        days = dict(zip(walks.ids, walks.d_omega * 7.0, strict=True))
        assert days == pytest.approx(
            {"sunday": 7.0, "wednesday": 4.0, "friday": 2.0, "saturday": 1.0},
            rel=1e-12,
        )

    def test_the_apparel_data_could_not_have_said_this(self):
        """The reason this class exists, asserted rather than described."""
        from clvtools import load_apparel_trans

        first = load_apparel_trans().groupby("Id")["Date"].min()
        assert set(first.dt.day_name()) == {"Sunday"}


class TestWalkConstructionValidation:
    @staticmethod
    @pytest.fixture(scope="class")
    def data():
        from clvtools import ClvData, load_apparel_trans

        return ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)

    def test_rejects_covariates_without_a_date_column(self, data):
        from clvtools.pnbd.dyncov import build_walks

        with pytest.raises(ValueError, match=r"no 'Cov\.Date' column"):
            build_walks(data, pd.DataFrame({"Id": ["1"], "Gender": [0]}))

    def test_rejects_an_unknown_covariate_name(self, data):
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        with pytest.raises(ValueError, match="missing columns"):
            build_walks(data, load_apparel_dyn_cov(), names_cov_life=["Nope"])

    def test_rejects_customers_without_covariate_data(self, data):
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        partial = load_apparel_dyn_cov()
        partial = partial[partial["Id"] != "1"]
        with pytest.raises(ValueError, match="have no covariate data"):
            build_walks(
                data, partial,
                names_cov_life=["High.Season"], names_cov_trans=["High.Season"],
            )

    def test_rejects_two_processes_on_different_date_grids(self, data):
        """The walk indices are derived once and used to slice both matrices.

        Nothing downstream would notice a transactional grid shifted against
        the lifetime one: every walk would simply read the wrong intervals and
        the likelihood would come back wrong but finite.
        """
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        covariates = load_apparel_dyn_cov()
        shifted = covariates.copy()
        shifted["Cov.Date"] = shifted["Cov.Date"] + pd.Timedelta(days=7)
        with pytest.raises(ValueError, match="share one date grid"):
            build_walks(
                data, covariates, shifted,
                names_cov_life=["High.Season"], names_cov_trans=["High.Season"],
            )

    def test_rejects_a_series_too_short_for_the_walks_it_must_cover(self, data):
        """A short series would slice to fewer rows than the walk spans.

        The two grids still agree everywhere they overlap, so the grid check
        passes; it is the stacking that would silently shift every later walk.
        """
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        covariates = load_apparel_dyn_cov()
        short = covariates[covariates["Cov.Date"] <= pd.Timestamp("2006-06-01")]
        with pytest.raises(ValueError, match="periods its walk spans"):
            build_walks(
                data, covariates, short,
                names_cov_life=["High.Season"], names_cov_trans=["High.Season"],
            )

    def test_covariate_names_default_to_every_column(self, data):
        from clvtools import load_apparel_dyn_cov
        from clvtools.pnbd.dyncov import build_walks

        walks = build_walks(data, load_apparel_dyn_cov())
        assert walks.n_cov_life == walks.n_cov_trans == 3



class TestTheLifeWalksReconstructTheCovariateSeries:
    r"""Spec DY-20, which `docs/spec-audit.md` called an exact round-trip
    invariant and marked `weak`: "matrices match; the round-trip is never
    asserted".

    S3.3 cuts a customer's covariate path into a **real** life walk, from birth
    to the end of the estimation period, and an **auxiliary** one covering what
    follows. Laid end to end those are the path itself, so

    .. math::
        [\,\text{real} \mathbin\Vert \text{aux}\,]_i
        = \exp(\boldsymbol{\gamma}'\mathbf{x}_i)

    over the customer's own covariate intervals, taken from their birth.
    Nothing said so. The two halves were each compared against oracle tables,
    which pins their *values* and not the claim that they partition one series
    with nothing lost or repeated at the join -- exactly the seam an off-by-one
    would open.

    Asserted **bit for bit** (``atol=0, rtol=0``): both sides are the same
    products of the same floats, so anything looser would be hiding a real
    difference rather than tolerating a rounding one. Backlog item 34, round 5.
    """

    #: Deliberately not zero. With :math:`\gamma = 0` every multiplier is 1 and
    #: the reconstruction holds for a reason that has nothing to do with the
    #: walks -- which is how DY-15's ``d_omega`` oracle came to be degenerate.
    GAMMA: ClassVar[np.ndarray] = np.array([0.7, -0.3, 0.45])

    #: The three covariates, in the order `dyncov_walks` was built with.
    COVARIATES: ClassVar[list[str]] = ["High.Season", "Gender", "Channel"]

    @pytest.fixture(scope="class")
    def multipliers(self):
        """``exp(gamma'x)`` per customer, straight from the covariate frame."""
        from clvtools import load_apparel_dyn_cov

        frame = load_apparel_dyn_cov().sort_values(["Id", "Cov.Date"])
        return {
            customer: np.exp(group[self.COVARIATES].to_numpy() @ self.GAMMA)
            for customer, group in frame.groupby("Id", sort=False)
        }

    def test_every_customer_reconstructs_exactly(self, dyncov_walks, multipliers):
        mismatched = []
        for customer_id, customer in zip(
            dyncov_walks.ids,
            dyncov_walks.customers(self.GAMMA, self.GAMMA),
            strict=True,
        ):
            joined = np.concatenate([
                customer.real_walk_life.values,  # noqa: PD011 - a Walk field
                customer.aux_walk_life.values,  # noqa: PD011 - not a DataFrame
            ])
            want = multipliers[customer_id][: len(joined)]
            if not np.array_equal(joined, want):
                mismatched.append(customer_id)
        assert not mismatched, (
            f"{len(mismatched)} customers whose life walks do not lay end to "
            f"end into their covariate series: {mismatched[:5]}"
        )

    def test_the_auxiliary_walk_is_never_empty_and_the_real_one_can_be(
        self, dyncov_walks, multipliers
    ):
        """The seam: no interval is dropped or counted twice at the join.

        Writing this asserted ``real > 0`` first, which is false for **214 of
        the 600** -- and the exception turns out to be exactly the structure
        worth stating. The real life walk spans birth to the last repeat
        purchase in *a later covariate interval*, so it is empty when there is
        no such purchase: the 213 customers with ``x = 0``, plus customer 129,
        who buys again at ``t_x = 0.43`` weeks and so never leaves the interval
        they were born in. ``test_pnbd_dyncov.py`` already names 129 for the
        same reason, from finding B2.
        """
        empty_real, empty_aux = [], []
        for customer_id, customer in zip(
            dyncov_walks.ids,
            dyncov_walks.customers(self.GAMMA, self.GAMMA),
            strict=True,
        ):
            real = len(customer.real_walk_life.values)
            aux = len(customer.aux_walk_life.values)
            # The two together cannot exceed the series they came from, which
            # is what rules out an interval being counted twice at the join.
            assert real + aux <= len(multipliers[customer_id])
            if real == 0:
                empty_real.append(customer_id)
            if aux == 0:
                empty_aux.append(customer_id)

        # Every customer is alive for some part of the auxiliary window, so
        # this half is never empty -- unlike the real one.
        assert not empty_aux
        assert len(empty_real) == 214
        assert "129" in empty_real


class TestTheCovariateGridMustReachTheEstimationEnd:
    """Spec T-18, `weak`: "only the on/on combination".

    T-18 asks that covariate date ranges be right for all four combinations of
    {start on, start off} x {end on, end off} a period boundary. All four build
    correctly -- but chasing the fourth found that a grid which stops *short*
    of the estimation end raised ``IndexError: index 4 is out of bounds for
    axis 0 with size 4`` from inside ``_distance_to_interval_end``, rather than
    saying the covariates do not cover the data.

    That is the shape items 21, 27 and 29 spent their time on, and the model
    already had the right words for it at the other end:
    ``dyncov_predict._require_coverage`` refuses a *prediction horizon* the
    covariates cannot reach. Construction had no equivalent. Backlog item 34,
    round 5.
    """

    START: ClassVar[pd.Timestamp] = pd.Timestamp("2005-01-03")

    def _data(self):
        from clvtools import ClvData

        trans = pd.DataFrame([
            {"Id": customer, "Date": self.START + pd.Timedelta(weeks=week)}
            for customer in ("a", "b")
            for week in range(6)
        ])
        return ClvData(trans, time_unit="week", estimation_split=4)

    def _grid(self, weeks: int, offset_days: int = 0):
        first = self.START - pd.Timedelta(days=offset_days)
        return pd.DataFrame([
            {"Id": customer, "Cov.Date": first + pd.Timedelta(weeks=week),
             "S": week % 2}
            for customer in ("a", "b")
            for week in range(weeks)
        ])

    def _build(self, weeks: int, offset_days: int = 0):
        from clvtools import ClvDataDynCov

        return ClvDataDynCov(
            self._data(), self._grid(weeks, offset_days),
            names_cov_life=["S"], names_cov_trans=["S"],
        ).walks()

    @pytest.mark.parametrize("start_offset", [0, 3], ids=["start-on", "start-off"])
    @pytest.mark.parametrize("weeks", [9, 10], ids=["end-on", "end-off"])
    def test_all_four_boundary_combinations_build(self, start_offset, weeks):
        """The claim as written: four combinations, all correct."""
        walks = self._build(weeks, start_offset)
        assert walks.n_customers == 2
        assert walks.n_cov_life == 1

    def test_a_grid_ending_exactly_at_the_estimation_end_is_enough(self):
        """A covariate date describes the period *starting* there.

        So the final interval is covered by a grid whose last date is the
        estimation end, and refusing it would be one period too strict.
        """
        assert self._build(5).n_customers == 2

    @pytest.mark.parametrize("weeks", [4, 3, 1])
    def test_but_a_shorter_one_says_so_rather_than_indexing_past_it(self, weeks):
        with pytest.raises(ValueError, match="stops before the estimation period"):
            self._build(weeks)

    def test_the_message_names_a_customer_and_both_dates(self):
        """What the `IndexError` could not: which grid, whose, and how short."""
        with pytest.raises(ValueError, match="covariate series") as excinfo:
            self._build(4)
        message = str(excinfo.value)
        assert "lifetime" in message
        assert "customer a" in message
        assert "2005-01-24" in message   # the grid's last date
        assert "2005-01-31" in message   # the estimation end

    def test_the_apparel_data_is_unaffected(self, dyncov_walks):
        """The regime every other dyncov test runs in, still built."""
        assert dyncov_walks.n_customers == 600

    def test_a_short_transaction_grid_keeps_its_own_more_specific_error(self):
        """The two checks divide the work, and the division is deliberate.

        Every walk's interval indices come from the *lifetime* grid and then
        slice both matrices, so a short transaction series is refused later, by
        `_stack`, with "periods its walk spans". The first draft of
        `_check_covariate_span` checked both grids and preempted that message --
        `test_rejects_a_series_too_short_for_the_walks_it_must_cover` above
        caught the substitution. This pins the boundary from the other side.
        """
        from clvtools import ClvData, load_apparel_dyn_cov, load_apparel_trans
        from clvtools.pnbd.dyncov import build_walks

        data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        full = load_apparel_dyn_cov()
        short = full[full["Cov.Date"] <= pd.Timestamp("2006-06-01")]
        with pytest.raises(ValueError, match="periods its walk spans"):
            build_walks(
                data, full, short,
                names_cov_life=["High.Season"], names_cov_trans=["High.Season"],
            )


class TestTheTwoPeriodAuxiliaryWalk:
    """Spec DY-17, `weak`: "the named 2-period edge case never constructed".

    DY-17 has two claims. The general splitting is covered; the specific one is
    that an auxiliary walk is **2 periods** when ``T`` sits on a week start and
    the customer comes alive shortly before it with *no real life walk*. The
    apparel cohort cannot show it -- every one of its 600 customers has an
    auxiliary life walk of 12 -- so it needed building.

    Two periods is the interesting length because it is the shortest walk with
    an interior: a one-period walk has only its own ends, and the ``d_omega``
    and ``d1`` corrections that scale the first and last intervals coincide.
    Backlog item 34, round 5.
    """

    GRID: ClassVar[pd.Timestamp] = pd.Timestamp("2005-01-03")   # a Monday

    def _walks(self, birth_offset_days: int, split_weeks: int = 4):
        from clvtools import ClvData, ClvDataDynCov

        rows = [
            # The customer under test: one purchase, so no real life walk.
            {"Id": "a", "Date": self.GRID + pd.Timedelta(days=birth_offset_days)},
            # A second customer, so the estimation window spans the grid.
            {"Id": "b", "Date": self.GRID},
            {"Id": "b", "Date": self.GRID + pd.Timedelta(weeks=split_weeks)},
        ]
        data = ClvData(
            pd.DataFrame(rows), time_unit="week", estimation_split=split_weeks
        )
        grid = pd.DataFrame([
            {"Id": customer, "Cov.Date": self.GRID + pd.Timedelta(weeks=week),
             "S": week % 2}
            for customer in ("a", "b")
            for week in range(12)
        ])
        built = ClvDataDynCov(
            data, grid, names_cov_life=["S"], names_cov_trans=["S"]
        ).walks()
        lengths = {}
        for customer_id, customer in zip(
            built.ids, built.customers([0.3], [0.3]), strict=True
        ):
            lengths[customer_id] = (
                len(customer.real_walk_life.values),
                len(customer.aux_walk_life.values),
            )
        return data.estimation_end, lengths

    def test_the_named_case_gives_exactly_two_periods(self):
        """`T` on a week start, alive a week before it, no real life walk."""
        end, lengths = self._walks(birth_offset_days=21)
        assert end == pd.Timestamp("2005-01-31")          # a Monday, on the grid
        assert lengths["a"] == (0, 2)

    @pytest.mark.parametrize("offset,expected_aux", [
        (0, 5), (13, 4), (20, 3), (21, 2), (27, 2),
    ])
    def test_the_auxiliary_walk_shortens_as_the_customer_arrives_later(
        self, offset, expected_aux
    ):
        """The surrounding behaviour, so the 2 above is not a coincidence.

        A customer born later has fewer covariate periods left before ``T``, and
        the walk shortens one period at a time -- monotonically, and never below
        the one period every customer alive at ``T`` must have.
        """
        _end, lengths = self._walks(birth_offset_days=offset)
        assert lengths["a"] == (0, expected_aux)

    def test_and_a_repeat_buyer_beside_them_keeps_a_real_walk(self):
        """Customer `b` buys twice, so the real/auxiliary split is exercised."""
        _end, lengths = self._walks(birth_offset_days=21)
        real, aux = lengths["b"]
        assert real == 4
        assert aux == 1


class TestTransactionsAnEpsilonApartCannotLoseAWalk:
    """Spec DY-19's third claim, which the audit left **undecided**.

    DY-19 asks that no walk be lost when transactions are "only one epsilon
    apart". The audit's note reads "the epsilon-apart claim is unreachable (day
    aggregation), undecided" -- and it is right on both counts, so this decides
    it rather than testing the unreachable thing.

    S6.1 collapses the log to at most one record per customer-day *before*
    anything else looks at it: "For any customer-day combination, multiple
    purchases are combined into a single record". Two purchases an epsilon apart
    are therefore **one** transaction by the time walks are built, so there is
    no second walk to lose. A test asserting "the walk survives" would be
    asserting something the data layer has already made vacuous.

    What is asserted instead is the step that makes it vacuous, which is the
    thing that could actually regress. Same shape as `T-01`. Backlog item 34,
    round 5.
    """

    def test_two_purchases_an_epsilon_apart_are_one_transaction(self):
        from clvtools import ClvData

        rows = [
            {"Id": "a", "Date": pd.Timestamp("2005-01-03 00:00:00")},
            {"Id": "a", "Date": pd.Timestamp("2005-01-03 00:00:01")},
        ]
        data = ClvData(pd.DataFrame(rows), time_unit="week")
        assert len(data.as_data_frame()) == 1
        assert int(data.customer_summary()["x"].iloc[0]) == 0

    def test_so_a_repeat_needs_a_later_day_not_a_later_second(self):
        """The contrast: a genuine repeat does produce one."""
        from clvtools import ClvData

        rows = [
            {"Id": "a", "Date": pd.Timestamp("2005-01-03")},
            {"Id": "a", "Date": pd.Timestamp("2005-01-04")},
        ]
        data = ClvData(pd.DataFrame(rows), time_unit="week")
        assert len(data.as_data_frame()) == 2
        assert int(data.customer_summary()["x"].iloc[0]) == 1
