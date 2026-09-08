r"""S6.3.3 - bootstrapped standard errors and confidence intervals.

Split out of ``test_diagnostics.py``, which had grown to 677 code lines against
the 700-line limit by carrying two subjects: the diagnostic frames of S6.2.2
and S6.2.4, and this. They share a data fixture and nothing else -- the frames
are checked against CLVTools' ``plot(..., plot = FALSE)`` output row for row,
and the bootstrap cannot be, because it is random and the paper's own example
seeds R's generator, which has no Python equivalent.

So the bootstrap is checked against the properties S6.3.3 states instead: the
periods are preserved, a customer drawn twice counts twice, the intervals
bracket the point estimate, and a draw that fails is reported rather than
silently dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clvtools import ClvData, bootstrap, load_apparel_trans


class TestBootstrapData:
    r"""Rebuilding the data from a resample, per S6.3.3."""

    def test_a_customer_drawn_twice_becomes_two_customers(self, data):
        rebuilt = bootstrap.bootstrap_data(data, ["1", "1", "10"])
        summary = rebuilt.customer_summary()
        assert len(summary) == 3
        assert set(summary["Id"]) == {"1", f"1{bootstrap.BOOTSTRAP_SUFFIX}1", "10"}

    def test_the_duplicate_has_the_originals_history(self, data):
        r"""S6.3.3: "Customers, together with their entire purchasing history,
        are sampled with replacement"."""
        rebuilt = bootstrap.bootstrap_data(data, ["1", "1"])
        summary = rebuilt.customer_summary().set_index("Id")
        original = summary.loc["1"]
        copy = summary.loc[f"1{bootstrap.BOOTSTRAP_SUFFIX}1"]
        assert original["x"] == copy["x"]
        assert original["t_x"] == copy["t_x"]
        assert original["T"] == copy["T"]

    def test_the_periods_are_preserved(self, data):
        r"""The detail S6.3.3 calls out.

        "simply sampling customers with their orders and creating a data object
        may yield different estimation and holdout periods because the end of
        the data is determined by the last order. This method makes sure that
        the estimation and holdout periods are preserved."

        Resampling to a single customer whose last purchase is years before the
        data ends would otherwise move both boundaries.
        """
        summary = data.customer_summary().set_index("Id")
        early = summary["t_x"].idxmin()
        rebuilt = bootstrap.bootstrap_data(data, [early])

        assert rebuilt.estimation_end == data.estimation_end
        assert rebuilt.data_end == data.data_end
        assert rebuilt.has_holdout == data.has_holdout

    def test_the_model_inputs_are_unchanged_by_resampling(self, data):
        r"""S6.3.3: "the model inputs (i.e, (x, t_x, T)) in each iteration
        remain for each customer the same as in the original data"."""
        chosen = ["1", "10", "100", "1"]
        original = data.customer_summary().set_index("Id")
        rebuilt = bootstrap.bootstrap_data(data, chosen).customer_summary()
        rebuilt["Id"] = rebuilt["Id"].str.replace(
            rf"{bootstrap.BOOTSTRAP_SUFFIX}\d+$", "", regex=True
        )
        for _, row in rebuilt.iterrows():
            source = original.loc[row["Id"]]
            assert row["x"] == source["x"]
            assert row["t_x"] == pytest.approx(source["t_x"])
            assert row["T"] == pytest.approx(source["T"])

    def test_spending_survives_the_rebuild(self, data):
        rebuilt = bootstrap.bootstrap_data(data, ["1", "10"])
        assert rebuilt.has_spending
        spend = rebuilt.spending_summary().set_index("Id")
        original = data.spending_summary().set_index("Id")
        assert spend.loc["1", "Spending"] == pytest.approx(
            original.loc["1", "Spending"]
        )

    def test_rejects_ids_that_are_not_in_the_data(self, data):
        with pytest.raises(ValueError, match="not in the data"):
            bootstrap.bootstrap_data(data, ["1", "nobody"])


class TestBootstrapApply:
    def test_it_runs_the_requested_number_of_iterations(self, data):
        seen = bootstrap.bootstrap_apply(
            data, lambda d: len(d.customer_summary()), num_boots=5, seed=1
        )
        assert seen == [600] * 5

    def test_it_is_reproducible_from_a_seed(self, data):
        r"""S6.3.3's example opens with ``set.seed(1)``."""
        summarise = lambda d: float(d.customer_summary()["x"].sum())  # noqa: E731
        first = bootstrap.bootstrap_apply(data, summarise, num_boots=4, seed=7)
        again = bootstrap.bootstrap_apply(data, summarise, num_boots=4, seed=7)
        different = bootstrap.bootstrap_apply(data, summarise, num_boots=4, seed=8)
        assert first == again
        assert first != different

    def test_resampling_varies_the_totals(self, data):
        """Otherwise there would be no uncertainty to measure."""
        totals = bootstrap.bootstrap_apply(
            data, lambda d: float(d.customer_summary()["x"].sum()),
            num_boots=8, seed=1,
        )
        assert len(set(totals)) > 1
        observed = float(data.customer_summary()["x"].sum())
        assert 0.8 * observed < np.mean(totals) < 1.2 * observed

    def test_a_custom_sampler_is_honoured(self, data):
        """S6.3.3 exposes ``fn.sample`` for exactly this."""
        drawn = bootstrap.bootstrap_apply(
            data, lambda d: sorted(d.customer_summary()["Id"]),
            num_boots=2, sample=lambda pool: ["1", "10"],
        )
        assert drawn == [["1", "10"], ["1", "10"]]

    def test_a_sampler_may_draw_fewer_customers(self, data):
        """``?clv.bootstrapped.apply``'s own example.

        Its sampler takes half the customers *without* replacement::

            fn.sample = function(x) sample(x, size = as.integer(0.5*length(x)),
                                           replace = FALSE)

        which is a different shape from the default: the resampled data is
        smaller than the original and holds no duplicates, so nothing is
        suffixed. Worth its own test because every other sampler here returns
        as many customers as it was given.
        """
        rng = np.random.default_rng(11)

        def half(pool):
            return rng.choice(pool, size=len(pool) // 2, replace=False)

        drawn = bootstrap.bootstrap_apply(
            data, lambda d: sorted(d.customer_summary()["Id"]),
            num_boots=3, sample=half,
        )
        assert [len(ids) for ids in drawn] == [300, 300, 300]
        for ids in drawn:
            assert len(set(ids)) == len(ids)
            assert not any(bootstrap.BOOTSTRAP_SUFFIX in i for i in ids)

    def test_the_documented_use_is_bootstrapping_coefficients(self, data):
        """``fn.boot.apply = coef`` -- what the man page reaches for first."""
        from clvtools.pnbd import fit_pnbd

        fits = bootstrap.bootstrap_apply(
            data,
            lambda d: fit_pnbd(
                *(d.customer_summary()[c] for c in ("x", "t_x", "T")),
                hessian=False, maxiter=200,
            ).coefficients,
            num_boots=2, seed=3,
        )
        assert len(fits) == 2
        for coefficients in fits:
            assert set(coefficients) == {"r", "alpha", "s", "beta"}
            assert all(v > 0 for v in coefficients.values())

    def test_rejects_a_meaningless_iteration_count(self, data):
        with pytest.raises(ValueError, match="at least 1"):
            bootstrap.bootstrap_apply(data, lambda d: None, num_boots=0)

    @pytest.mark.slow
    def test_covariates_are_carried_across(self):
        from clvtools import ClvDataStaticCov, load_apparel_static_cov

        static = ClvDataStaticCov(
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
            load_apparel_static_cov(),
            names_cov_life=["Gender", "Channel"],
            names_cov_trans=["Gender", "Channel"],
        )
        shapes = bootstrap.bootstrap_apply(
            static, lambda d: d.design_life().shape, num_boots=2, seed=1
        )
        assert shapes == [(600, 2), (600, 2)]


class TestConfidenceIntervals:
    def test_bounds_follow_the_requested_level(self):
        draws = []
        for i in range(100):
            frame = pd.DataFrame({"CET": [float(i)]}, index=["a"])
            frame.index.name = "Id"
            draws.append(frame)

        ninety = bootstrap.confidence_intervals(draws, level=0.9)
        fifty = bootstrap.confidence_intervals(draws, level=0.5)
        assert list(ninety.columns) == ["CET.CI.5", "CET.CI.95"]
        assert list(fifty.columns) == ["CET.CI.25", "CET.CI.75"]
        # A narrower level gives a narrower interval.
        assert (
            fifty.loc["a", "CET.CI.75"] - fifty.loc["a", "CET.CI.25"]
            < ninety.loc["a", "CET.CI.95"] - ninety.loc["a", "CET.CI.5"]
        )

    def test_duplicate_draws_of_one_customer_are_pooled(self):
        r"""The suffix that kept them apart while resampling is stripped here.

        A customer drawn three times in one iteration contributes three values
        to their own interval, which is where the spread comes from.
        """
        frame = pd.DataFrame(
            {"CET": [1.0, 5.0, 9.0]},
            index=["a", f"a{bootstrap.BOOTSTRAP_SUFFIX}1",
                   f"a{bootstrap.BOOTSTRAP_SUFFIX}2"],
        )
        frame.index.name = "Id"
        intervals = bootstrap.confidence_intervals([frame], level=0.9)
        assert list(intervals.index) == ["a"]
        assert intervals.loc["a", "CET.CI.5"] < intervals.loc["a", "CET.CI.95"]

    def test_only_numeric_columns_are_summarised(self):
        frame = pd.DataFrame(
            {"CET": [1.0], "label": ["x"]}, index=pd.Index(["a"], name="Id")
        )
        intervals = bootstrap.confidence_intervals([frame])
        assert list(intervals.columns) == ["CET.CI.5", "CET.CI.95"]

    def test_columns_can_be_chosen(self):
        frame = pd.DataFrame(
            {"CET": [1.0], "DERT": [2.0]}, index=pd.Index(["a"], name="Id")
        )
        intervals = bootstrap.confidence_intervals([frame], columns=["DERT"])
        assert list(intervals.columns) == ["DERT.CI.5", "DERT.CI.95"]

    def test_an_index_column_is_accepted_in_place_of_an_index(self):
        frame = pd.DataFrame({"Id": ["a", "b"], "CET": [1.0, 2.0]})
        intervals = bootstrap.confidence_intervals([frame])
        assert list(intervals.index) == ["a", "b"]

    def test_rejects_a_meaningless_level(self):
        frame = pd.DataFrame({"CET": [1.0]}, index=pd.Index(["a"], name="Id"))
        for level in (0.0, 1.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="strictly between 0 and 1"):
                bootstrap.confidence_intervals([frame], level=level)

    def test_rejects_an_empty_set_of_draws(self):
        with pytest.raises(ValueError, match="no bootstrap draws"):
            bootstrap.confidence_intervals([])


@pytest.mark.slow
class TestPredictIntervals:
    r"""The whole routine, as S6.3.3 describes it."""

    @staticmethod
    @pytest.fixture(scope="class")
    def refit():
        from clvtools.gg import fit_gg as fit_spending
        from clvtools.pnbd import fit_pnbd
        from clvtools.predict import predict

        def run(sample):
            cbs, spend = sample.customer_summary(), sample.spending_summary()
            return predict(
                sample,
                fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False),
                fit_spending(spend["x"], spend["Spending"]),
            )

        return run

    @staticmethod
    @pytest.fixture(scope="class")
    def intervals(data, refit):
        return bootstrap.predict_intervals(
            data, refit, num_boots=10, level=0.9,
            columns=["PAlive", "CET", "predicted.CLV"], seed=1,
        )

    def test_every_customer_gets_an_interval(self, data, intervals):
        assert len(intervals) == 600
        assert list(intervals.columns) == [
            "PAlive.CI.5", "PAlive.CI.95",
            "CET.CI.5", "CET.CI.95",
            "predicted.CLV.CI.5", "predicted.CLV.CI.95",
        ]

    def test_the_bounds_are_ordered(self, intervals):
        for quantity in ("PAlive", "CET", "predicted.CLV"):
            assert (
                intervals[f"{quantity}.CI.5"] <= intervals[f"{quantity}.CI.95"]
            ).all()

    def test_palive_intervals_stay_in_the_unit_interval(self, intervals):
        assert (intervals["PAlive.CI.5"] >= 0).all()
        assert (intervals["PAlive.CI.95"] <= 1).all()

    def test_the_intervals_bracket_the_point_estimate(self, data, refit):
        r"""Not guaranteed for every customer, but it should hold for most.

        A percentile bootstrap interval is not built to contain the point
        estimate, so a handful of customers falling outside is expected; a large
        fraction would mean the resampling was not reproducing the fit.
        """
        intervals = bootstrap.predict_intervals(
            data, refit, num_boots=10, columns=["CET"], seed=1
        )
        point = refit(data)
        inside = (
            (point["CET"] >= intervals["CET.CI.5"])
            & (point["CET"] <= intervals["CET.CI.95"])
        )
        assert inside.mean() > 0.9

    def test_it_measures_parameter_uncertainty_only(self, data, refit):
        r"""S6.3.3: "bootstrapping only accounts for uncertainty in model
        parameters (epistemic uncertainty), and not sampling variability in the
        actual outcomes (aleatoric uncertainty)."

        So the intervals are narrow relative to the spread of actual outcomes:
        they describe where the *expectation* sits, not where a customer's next
        year of purchases will land.
        """
        intervals = bootstrap.predict_intervals(
            data, refit, num_boots=10, columns=["CET"], seed=1
        )
        width = (intervals["CET.CI.95"] - intervals["CET.CI.5"]).median()
        outcomes = refit(data)["actual.x"].std()
        assert width < outcomes


class TestTheBootstrapReportsWhatHappened:
    """Finding 11: three silences and a rebuild that dominated the run.

    The rebuild is measured rather than gated -- 0.965 s a draw on CDNOW
    against 0.134 s for the summary and fit together, now 0.016 s -- because
    this repo gates operations rather than clocks. What is asserted here is the
    behaviour: a failed draw is reported instead of discarding the run, and a
    caller's own sampler receives the seeded generator.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def small():
        from clvtools import ClvData, load_apparel_trans

        trans = load_apparel_trans()
        ids = sorted(trans["Id"].unique())[:40]
        return ClvData(
            trans[trans["Id"].isin(ids)], time_unit="week", estimation_split=104
        )

    def test_one_failed_draw_does_not_lose_the_others(self, small):
        """It used to propagate out of the loop, discarding every draw before
        it -- a resample is a random object and some are degenerate, so losing
        five minutes of refits to the third one is the wrong trade."""
        from clvtools._validate import ConvergenceWarning
        from clvtools.bootstrap import bootstrap_apply

        calls = {"n": 0}

        def flaky(data):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("degenerate resample")
            return data.nobs()

        with pytest.warns(ConvergenceWarning, match="1 of 3 bootstrap draws failed"):
            got = bootstrap_apply(small, flaky, num_boots=3, seed=1)
        assert len(got) == 2

    def test_every_draw_failing_is_an_error(self, small):
        from clvtools.bootstrap import bootstrap_apply

        def always(data):
            raise RuntimeError("no")

        with pytest.raises(ValueError, match="all 2 bootstrap draws failed"):
            bootstrap_apply(small, always, num_boots=2, seed=1)

    def test_a_custom_sampler_is_given_the_seeded_generator(self, small):
        """``seed`` used to be silently ignored whenever ``sample`` was passed,
        so runs that looked reproducible were not."""
        from clvtools.bootstrap import bootstrap_apply

        def half(pool, rng):
            return rng.choice(pool, size=len(pool) // 2, replace=False)

        first = bootstrap_apply(small, lambda d: d.nobs(), num_boots=2,
                                sample=half, seed=7)
        again = bootstrap_apply(small, lambda d: d.nobs(), num_boots=2,
                                sample=half, seed=7)
        assert first == again
        assert all(n == small.nobs() // 2 for n in first)

    def test_a_one_argument_sampler_still_works(self, small):
        """``?clv.bootstrapped.apply``'s own example has that shape."""
        from clvtools.bootstrap import bootstrap_apply

        def half(pool):
            return pool[: len(pool) // 2]

        got = bootstrap_apply(small, lambda d: d.nobs(), num_boots=1, sample=half)
        assert got == [small.nobs() // 2]


class TestBootstrappingDynamicCovariatesRefuses:
    """Finding A2: the audit's only wrong *answer*, rather than a silence.

    ``ClvDataDynCov`` subclasses ``ClvData`` and not ``ClvDataStaticCov``, so
    the covariate-resampling branch never fired for it. Reproduced before the
    guard existed: a ``ClvDataDynCov`` went in and ``apply`` received a plain
    ``ClvData`` -- every covariate gone -- which then refitted a model that is
    *defined* by those covariates without them, and returned an interval from
    it.
    """

    def test_it_raises_rather_than_dropping_the_covariates(self):
        from clvtools import ClvData, load_apparel_dyn_cov, load_apparel_trans
        from clvtools.bootstrap import bootstrap_apply
        from clvtools.data import ClvDataDynCov

        names = ["High.Season", "Gender", "Channel"]
        trans = load_apparel_trans()
        ids = sorted(trans["Id"].unique())[:20]
        covariates = load_apparel_dyn_cov()
        data = ClvDataDynCov(
            ClvData(trans[trans["Id"].isin(ids)], time_unit="week",
                    estimation_split=104),
            covariates[covariates["Id"].isin(ids)],
            names_cov_life=names, names_cov_trans=names,
        )
        with pytest.raises(NotImplementedError, match="time-varying covariate"):
            bootstrap_apply(data, lambda resampled: resampled.nobs(), num_boots=1)


class TestBootstrapArgumentsAndCovariateAlignment:
    """Spec B-01, B-08 and B-15, all `weak`.

    `B-01` -- the default `num_boots` is 100 -- was never pinned, and a default
    is exactly the kind of thing that drifts unnoticed.

    `B-08` is the one worth the care. Resampling *with replacement* draws the
    same customer more than once, and each copy gets a synthetic id
    (``101_BOOTSTRAP_ID_1``) so the cbs can hold both. The claim is that the
    static covariates follow: same ids, sorted the same way as the cbs. The
    suite asserted a ``(600, 2)`` shape, which a design matrix built in the
    wrong order would also satisfy -- and that is a fit whose every coefficient
    is estimated against the wrong customer's covariates.

    `B-15`'s six argument checks were two-of-six, and one landed badly.
    """

    def test_the_default_number_of_draws_is_a_hundred(self):
        import inspect

        from clvtools import bootstrap

        default = inspect.signature(
            bootstrap.bootstrap_apply
        ).parameters["num_boots"].default
        assert default == 100

    @pytest.mark.slow
    def test_every_design_row_is_its_own_customers_covariates(self, static_data):
        """B-08, through `bootstrap_apply` rather than `bootstrap_data`.

        The latter is the low-level helper and returns plain data on purpose;
        `bootstrap_apply` rebuilds the covariates around it. Checking the helper
        would have reported a defect that is not there.
        """
        from clvtools import bootstrap, load_apparel_static_cov

        original = load_apparel_static_cov().set_index("Id")
        seen = []

        def check(resampled):
            cbs = resampled.customer_summary()
            design = np.asarray(resampled.design_life())
            assert isinstance(resampled, type(static_data))
            assert design.shape[0] == len(cbs)
            for row, customer in enumerate(cbs["Id"]):
                # A duplicated customer carries a synthetic id; the covariates
                # it should carry are the original customer's.
                source = str(customer).split("_BOOTSTRAP_ID_")[0]
                assert float(design[row, 0]) == float(
                    original.loc[source, "Gender"]
                ), f"row {row} carries the wrong customer's covariates"
            seen.append(len(cbs))
            return 1.0

        bootstrap.bootstrap_apply(static_data, apply=check, num_boots=3, seed=7)
        assert seen == [600, 600, 600]

    def test_a_non_callable_apply_is_refused_before_any_draw_runs(self):
        """B-15. It used to run all 100 draws and report that all 100 failed.

        A hundred symptoms of one mistake, and a hundred resamples spent to say
        so -- the count is what makes this worth fixing rather than rewording.
        """
        from clvtools import ClvData, bootstrap, load_apparel_trans

        data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        with pytest.raises(TypeError, match="apply must be callable, not int"):
            bootstrap.bootstrap_apply(data, apply=5, num_boots=100)

    @pytest.mark.parametrize("num_boots", [0, -1])
    def test_a_non_positive_draw_count_is_refused(self, num_boots):
        from clvtools import ClvData, bootstrap, load_apparel_trans

        data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        with pytest.raises(ValueError, match="num_boots must be at least 1"):
            bootstrap.bootstrap_apply(data, apply=lambda d: 1.0, num_boots=num_boots)

    @pytest.mark.parametrize("level", [1.5, -0.1, 0.0, 1.0])
    def test_a_confidence_level_outside_the_unit_interval_is_refused(self, level):
        from clvtools import bootstrap

        draws = pd.DataFrame({"Id": ["a", "a", "b"], "value": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="level must lie strictly between"):
            bootstrap.confidence_intervals(
                draws, columns=["value"], by="Id", level=level
            )


class TestBootstrapDrawsKeepTheOriginalWindow:
    """Spec B-12 and B-14.

    `B-12`: a resampled draw contains a *different* set of transactions, so its
    own first and last purchase differ from the cohort's -- and the estimation
    end must not follow them, or every draw would predict over a slightly
    different window and the interval would be built from answers to different
    questions. It holds; nothing said so.

    `B-14` asks that ``predict(uncertainty = "boots")`` work across the model
    families. **There is no such argument here**, and that is the deliberate
    shape the README already records for the bootstrap: `apply` receives the
    resampled data and does its own fitting, so intervals are composed by the
    caller from `bootstrap_apply` and `confidence_intervals` rather than
    requested inside `predict`. Pinned as a divergence, not a gap.
    """

    def test_every_draw_keeps_the_cohorts_estimation_end(self, apparel_trans):
        from clvtools import ClvData, bootstrap

        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        seen = []

        def note(resampled):
            seen.append((resampled.estimation_end, resampled.data_end))
            return 1.0

        bootstrap.bootstrap_apply(data, apply=note, num_boots=4, seed=3)
        assert len(seen) == 4
        assert all(end == data.estimation_end for end, _ in seen)
        assert all(end == data.data_end for _, end in seen)

    def test_predict_takes_no_uncertainty_argument(self):
        """B-14's divergence, pinned so it reads as a decision.

        R asks `predict` for bootstrap intervals; here the caller composes them,
        which is the same choice the README records under *Deliberately not
        ported* for `clv.bootstrapped.apply`.
        """
        import inspect

        from clvtools import predict

        assert "uncertainty" not in inspect.signature(predict).parameters

    def test_and_the_pieces_that_replace_it_are_both_present(self):
        from clvtools import bootstrap

        assert callable(bootstrap.bootstrap_apply)
        assert callable(bootstrap.confidence_intervals)
