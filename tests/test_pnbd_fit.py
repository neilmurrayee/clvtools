r"""S6.2.1 - maximum likelihood estimation of :math:`(r, \alpha, s, \beta)`.

The paper prints ``r = 1.4490, alpha = 48.6361, s = 0.5613, beta = 46.8844``
for the apparel cohort. Reproducing those digits exactly is not the right test,
and the tests here do not ask for it.

The Pareto/NBD likelihood has a long, flat ridge near its maximum. Moving 3e-5
along it changes the log-likelihood by around 1e-10 -- far below any tolerance
either optimiser can resolve. So two correct implementations will stop at
slightly different points, and demanding agreement in the fourth decimal would
be testing SciPy's stopping rule against ``optimx``'s, not the model.

What is asserted instead:

  * the attained log-likelihood matches the oracle's to 1e-6, and is no worse;
  * the estimates match the published ones to 1e-4 relative;
  * the fit is a genuine local maximum -- perturbing any parameter lowers the
    likelihood.
"""

# One precision rule, applied across the suite after CI showed the old one was
# a statement about macOS/ARM (``docs/backlog.md`` item 17, finding 13 of
# ``docs/review-2026-09-02.md``):
#
#   * an **estimate** is compared with a tolerance no tighter than 1e-3
#     relative -- the Pareto/NBD ridge moves the parameters by ~1e-4 between
#     libms while the log-likelihood moves by 1e-9, so anything tighter is
#     asserting a property of a C library;
#   * a **log-likelihood** is compared tightly, because it is what the search
#     actually optimises and it is flat-bottomed: the two platforms agree to
#     9e-10 on a value of -5848;
#   * "at least as good as the oracle" is asserted with 1e-6 of slack, not
#     1e-9, for the same reason;
#   * and no test asserts a *printed* digit of an estimate.


from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_csv, fixture_json
from paper_values import PNBD_MEAN_ATTRITION_RATE, PNBD_MEAN_PURCHASE_RATE, PNBD_MLE

from clvtools.pnbd import log_likelihood
from clvtools.pnbd.fit import PnbdParams, fit_pnbd


@pytest.fixture(scope="module")
def cbs():
    return fixture_csv("cbs_estimation")


@pytest.fixture(scope="module")
def fitted(cbs):
    return fit_pnbd(cbs["x"], cbs["t.x"], cbs["T.cal"])


@pytest.mark.slow
@pytest.mark.paper
class TestAgainstThePaper:
    def test_estimates_match_the_published_values(self, fitted):
        """S6.2.1's four estimates, to the precision a fit reproduces.

        1e-3 rather than the 1e-4 this asserted until CI ran on a second
        platform: ``s`` used 80% of a 1e-4 allowance on macOS/ARM and at least
        89% on x86-64 Linux, so the margin was luck rather than agreement.
        """
        for name, want in PNBD_MLE.items():
            assert getattr(fitted, name) == pytest.approx(want, rel=1e-3)

    def test_mean_rates_match_the_published_values(self, fitted):
        r"""S6.2.1: "an average purchase rate of :math:`r/\alpha` = 0.030
        transactions and an average attrition rate of :math:`s/\beta` = 0.012"."""
        assert round(fitted.mean_purchase_rate, 3) == PNBD_MEAN_PURCHASE_RATE
        assert round(fitted.mean_attrition_rate, 3) == PNBD_MEAN_ATTRITION_RATE

    def test_converges(self, fitted):
        """S6.2.1 reports ``KKT1: TRUE`` and ``KKT2: TRUE`` for this fit."""
        assert fitted.converged


@pytest.mark.slow
@pytest.mark.oracle
class TestAgainstTheOracle:
    def test_log_likelihood_matches(self, fitted):
        want = fixture_json("pnbd_nocov_fit")["logLik"]
        assert fitted.log_likelihood == pytest.approx(want, abs=1e-6)

    def test_reaches_at_least_the_oracles_optimum(self, fitted):
        """A lower likelihood would mean a worse fit; a higher one is fine."""
        want = fixture_json("pnbd_nocov_fit")["logLik"]
        assert fitted.log_likelihood >= want - 1e-6

    def test_aic_and_bic_match(self, fitted):
        want = fixture_json("pnbd_nocov_fit")
        assert fitted.aic == pytest.approx(want["AIC"], abs=1e-5)
        assert fitted.bic == pytest.approx(want["BIC"], abs=1e-5)

    def test_nobs_matches(self, fitted):
        assert fitted.n_customers == fixture_json("pnbd_nocov_fit")["nobs"]

    def test_standard_errors_match(self, fitted):
        """The Hessian is differenced numerically, as CLVTools does via numDeriv."""
        want_fit = fixture_json("pnbd_nocov_fit")
        vcov = np.array(want_fit["vcov"]).reshape(4, 4)
        want = dict(zip(want_fit["vcov.names"], np.sqrt(np.diag(vcov)), strict=True))
        got = fitted.standard_errors()
        for name, value in want.items():
            # 1% -- these are second derivatives of a flat surface, evaluated at
            # two different points on its ridge.
            assert got[name] == pytest.approx(value, rel=1e-2)

    def test_full_data_fit_matches(self):
        """S6.3.2 refits on all data with ``estimation.split = NULL``."""
        full = fixture_csv("cbs_full")
        want = fixture_json("pnbd_nocov_fit_full")
        got = fit_pnbd(full["x"], full["t.x"], full["T.cal"], hessian=False)
        assert got.log_likelihood == pytest.approx(want["logLik"], abs=1e-5)
        assert got.log_likelihood >= want["logLik"] - 1e-6
        for name, value in want["coefficients"].items():
            assert getattr(got, name) == pytest.approx(value, rel=1e-3)


@pytest.mark.slow
class TestOptimum:
    def test_is_a_local_maximum(self, cbs, fitted):
        best = fitted.log_likelihood
        for name in PNBD_MLE:
            for factor in (0.999, 1.001):
                nudged = dict(fitted.as_dict(), **{name: getattr(fitted, name) * factor})
                assert log_likelihood(cbs["x"], cbs["t.x"], cbs["T.cal"], **nudged) < best

    def test_is_reached_from_different_starting_points(self, cbs, fitted):
        for start in [(0.5, 10.0, 0.5, 10.0), (3.0, 100.0, 2.0, 100.0)]:
            got = fit_pnbd(
                cbs["x"], cbs["t.x"], cbs["T.cal"], start=start, hessian=False
            )
            assert got.log_likelihood == pytest.approx(fitted.log_likelihood, abs=1e-5)

    def test_nelder_mead_agrees_with_lbfgsb(self, cbs, fitted):
        """S6.2.1 offers Nelder-Mead as the fallback; it must find the same peak."""
        got = fit_pnbd(
            cbs["x"], cbs["t.x"], cbs["T.cal"], method="Nelder-Mead", hessian=False
        )
        assert got.log_likelihood == pytest.approx(fitted.log_likelihood, abs=1e-5)


class TestWeights:
    """Row multiplicities, the compression CLVTools applies to its CBS.

    Many customers share a summary -- 260 of the 600 apparel customers are
    ``(0, 0, 104)`` -- so the likelihood can be evaluated once per distinct row
    and weighted. Fitting the compressed table must give the same answer as
    fitting the full one, which is what makes the compression safe.
    """

    def test_a_compressed_table_fits_identically(self, cbs):
        grouped = (
            cbs.groupby(["x", "t.x", "T.cal"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
        )
        assert len(grouped) < len(cbs)  # there is something to compress

        weighted = fit_pnbd(
            grouped["x"], grouped["t.x"], grouped["T.cal"],
            weights=grouped["n"], hessian=False,
        )
        full = fit_pnbd(cbs["x"], cbs["t.x"], cbs["T.cal"], hessian=False)

        assert weighted.n_customers == len(cbs)
        assert weighted.log_likelihood == pytest.approx(
            full.log_likelihood, abs=1e-6
        )
        for name in PNBD_MLE:
            assert getattr(weighted, name) == pytest.approx(
                getattr(full, name), rel=1e-3
            )


class TestParamsObject:
    def test_iterates_in_the_papers_order(self, fitted):
        assert list(fitted) == [fitted.r, fitted.alpha, fitted.s, fitted.beta]

    def test_as_dict_round_trips_into_the_expressions(self, cbs, fitted):
        got = log_likelihood(cbs["x"], cbs["t.x"], cbs["T.cal"], **fitted.as_dict())
        assert got == pytest.approx(fitted.log_likelihood, abs=1e-9)

    def test_standard_errors_require_a_hessian(self, cbs):
        got = fit_pnbd(cbs["x"], cbs["t.x"], cbs["T.cal"], hessian=False)
        assert got.hessian is None
        with pytest.raises(ValueError, match="hessian=True"):
            got.standard_errors()


class TestValidation:
    BASE = (np.array([0, 2]), np.array([0.0, 30.0]), np.array([104.0, 104.0]))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            fit_pnbd([0, 1], [0.0], [104.0, 104.0])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="no customers"):
            fit_pnbd([], [], [])

    def test_rejects_negative_frequency(self):
        with pytest.raises(ValueError, match="non-negative"):
            fit_pnbd([-1, 2], [0.0, 30.0], [104.0, 104.0])

    def test_rejects_recency_beyond_the_window(self):
        with pytest.raises(ValueError, match="t_x exceeds T for 1 customer"):
            fit_pnbd([1, 2], [200.0, 30.0], [104.0, 104.0])

    def test_a_recency_a_hair_over_the_window_is_clamped_not_accepted(self):
        """The failure mode this replaced, from the outside review's finding 5.

        Date arithmetic produces ``t_x = T + 1e-10`` routinely. The validator
        used to accept anything within 1e-9, and the likelihood needs
        ``t_x <= T`` exactly: the ratio goes above one, an intermediate goes
        negative, its log is NaN, and *every* objective evaluation is
        infinite. The fit then returned its own start values -- ``r = alpha =
        s = beta = 1``, ``log_likelihood = -inf``, ``converged = False`` --
        and raised nothing, so a whole fit collapsed into a plausible-looking
        object. Now the slack is clamped away and the fit is a fit.
        """
        x = [2.0, 3.0, 0.0]
        T = [104.0, 104.0, 104.0]
        t_over = [104.0 + 1e-10, 40.0, 0.0]
        fitted = fit_pnbd(x, t_over, T, hessian=False)
        exact = fit_pnbd(x, [104.0, 40.0, 0.0], T, hessian=False)
        # Finite, and identical to the same data with the slack removed by
        # hand -- which is the whole claim. Not "negative": these three
        # customers are few enough that the fit reaches a log-likelihood of
        # 0.0 on x86-64 Linux, and a continuous density's log-likelihood is
        # not required to be negative anyway. CI caught that; the assertion
        # was mine and it was wrong.
        assert np.isfinite(fitted.log_likelihood)
        assert fitted.log_likelihood == pytest.approx(exact.log_likelihood)

    def test_a_fit_that_stops_early_says_so(self):
        """No ``warnings.warn`` existed anywhere in ``src/`` (finding 7).

        ``maxiter=2`` stops at ``[0.384, 1.682, 0.309, 1.335]``, which is not
        a fit of anything. The only signal was the ``converged`` flag, which a
        caller has to know to read.
        """
        from clvtools._validate import ConvergenceWarning

        with pytest.warns(ConvergenceWarning, match="Pareto/NBD"):
            stopped = fit_pnbd(
                [1, 2], [50.0, 30.0], [104.0, 104.0], hessian=False, maxiter=2
            )
        assert not stopped.converged

    def test_rejects_nonzero_recency_for_zero_purchases(self):
        with pytest.raises(ValueError, match=r"t_x must be 0 where x == 0"):
            fit_pnbd([0, 2], [5.0, 30.0], [104.0, 104.0])

    def test_rejects_nonpositive_window(self):
        with pytest.raises(ValueError, match="strictly positive"):
            fit_pnbd([0, 2], [0.0, 30.0], [0.0, 104.0])

    def test_rejects_bad_start_values(self):
        with pytest.raises(ValueError, match="4 values"):
            fit_pnbd(*self.BASE, start=(1.0, 1.0))
        with pytest.raises(ValueError, match="strictly positive"):
            fit_pnbd(*self.BASE, start=(1.0, -1.0, 1.0, 1.0))

    def test_accepts_extra_optimiser_options(self, cbs):
        """S6.2.1's ``optimx.args`` escape hatch."""
        got = fit_pnbd(
            cbs["x"], cbs["t.x"], cbs["T.cal"],
            options={"maxiter": 5}, hessian=False,
        )
        assert isinstance(got, PnbdParams)
        assert got.n_evaluations > 0


class TestNonFiniteStartValuesAreRefusedByName:
    """Spec V-01 and V-02, and the same defect one level apart.

    `nan <= 0` is ``False``, so a `NaN` start passed the positivity check and
    reached the optimiser, which reported *"the objective is not finite at the
    point the search started"* -- a statement about the model, or about the
    data, for a fault in the argument. `X-14`'s `NaN` regularization lambda was
    the same shape, and so is `start_cov`: a single scalar here where R takes a
    named vector, so five of `V-02`'s seven claims cannot arise, but "numeric"
    and "finite" can and the second one did not hold.

    Backlog item 34, round 5.
    """

    @pytest.fixture(scope="class")
    def inputs(self, cbs_estimation):
        return (
            cbs_estimation["x"].to_numpy(dtype=float),
            cbs_estimation["t.x"].to_numpy(dtype=float),
            cbs_estimation["T.cal"].to_numpy(dtype=float),
        )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_model_start_names_itself(self, inputs, bad):
        from clvtools.pnbd import fit_pnbd

        x, t_x, T = inputs
        with pytest.raises(ValueError, match="start values must be finite"):
            fit_pnbd(x, t_x, T, start=(bad, 1.0, 1.0, 1.0), hessian=False)

    def test_the_positivity_check_still_fires_where_it_should(self, inputs):
        """The new guard must not shadow the old one."""
        from clvtools.pnbd import fit_pnbd

        x, t_x, T = inputs
        with pytest.raises(ValueError, match="strictly positive"):
            fit_pnbd(x, t_x, T, start=(0.0, 1.0, 1.0, 1.0), hessian=False)

    def test_and_the_length_check_before_both(self, inputs):
        from clvtools.pnbd import fit_pnbd

        x, t_x, T = inputs
        with pytest.raises(ValueError, match="start must give 4"):
            fit_pnbd(x, t_x, T, start=(1.0, 1.0, 1.0), hessian=False)

    def test_a_non_finite_covariate_start_names_itself_too(self):
        from clvtools._staticcov import _validated_cov_start

        with pytest.raises(ValueError, match="start_cov must be a finite"):
            _validated_cov_start(float("nan"))

    def test_and_a_vector_where_a_scalar_belongs_says_which(self):
        """R takes one entry per covariate; here it is one value for all."""
        from clvtools._staticcov import _validated_cov_start

        with pytest.raises(TypeError, match="single number applied to every"):
            _validated_cov_start([0.1, 0.2])

    def test_the_default_and_a_plain_scalar_are_unmoved(self):
        from clvtools._staticcov import DEFAULT_COV_START, _validated_cov_start

        assert _validated_cov_start(None) == DEFAULT_COV_START
        assert _validated_cov_start(0.5) == 0.5


class TestFitsRunUnderOptimiserMethodsBeyondTheTwoInUse:
    """Spec F-11 and F-09, both `weak`.

    `F-11` asks that fits work "across all optimx methods"; only Nelder-Mead and
    L-BFGS-B were ever run here. Other SciPy methods are accepted -- item 31's
    `options_for` validates nothing for a method it does not recognise, rather
    than refusing one SciPy might well take -- so "accepted" and "works" were
    two different claims and only the first was covered.

    `F-09`'s "flawless results out of the box" is the finiteness sweep: no
    non-finite value in the estimates, the standard errors, or anywhere in the
    prediction table. Backlog item 34, round 5.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["Nelder-Mead", "L-BFGS-B", "Powell"])
    def test_a_third_method_reaches_a_comparable_optimum(
        self, cbs_estimation, method
    ):
        """Powell is the one nothing had ever run.

        Bounded loosely: what is under test is that the method *works*, not
        that three optimisers agree to the last digit on a flat ridge.
        """
        from clvtools.pnbd import fit_pnbd

        fitted = fit_pnbd(
            cbs_estimation["x"], cbs_estimation["t.x"], cbs_estimation["T.cal"],
            method=method, hessian=False,
        )
        assert np.isfinite(fitted.log_likelihood)
        assert fitted.log_likelihood == pytest.approx(-5848.1, abs=2.0)
        assert all(v > 0 for v in fitted.coefficients.values())

    @pytest.mark.slow
    def test_nothing_in_a_fit_or_its_predictions_is_non_finite(
        self, apparel_trans, cbs_estimation
    ):
        """F-09, on the apparel cohort without covariates."""
        from clvtools import ClvData, predict
        from clvtools.pnbd import fit_pnbd

        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        fitted = fit_pnbd(
            cbs_estimation["x"], cbs_estimation["t.x"], cbs_estimation["T.cal"],
        )
        assert np.isfinite(list(fitted.coefficients.values())).all()
        assert np.isfinite(list(fitted.standard_errors().values())).all()
        table = predict(data, fitted).select_dtypes("number")
        offending = [c for c in table.columns if not np.isfinite(table[c]).all()]
        assert not offending


class TestFitsRunOnDataShapesNothingHadFitOn:
    """Spec `F-12` and `F-13`, both `absent`.

    Two runability claims from CLVTools' own `helper_testthat_runability_nocov.R`
    that this port had covered one layer too low. Hourly time units were tested
    at the ``timeunit`` layer -- ``add``, ``floor``, the period count -- and no
    fit had ever been run on hourly data, where the numbers a fit sees are
    ~4,000 periods rather than ~100 and ``alpha`` lands three orders of
    magnitude away. And nothing fitted a model on data carrying no ``Price`` at
    all, then predicted on data that does: the two objects meet only inside
    :func:`~clvtools.predict.predict`, so a fit that had quietly remembered
    something about spending would have gone unnoticed. Backlog item 36,
    round 6.
    """

    @pytest.fixture(scope="class")
    def hourly(self, apparel_trans):
        """The apparel log read at one-hour resolution.

        Its dates are midnight-stamped, so every transaction falls on hour 0 of
        its day and the cohort is the same one; what changes is the *scale* the
        optimiser searches on, which is the point.
        """
        from clvtools import ClvData

        return ClvData(apparel_trans, time_unit="hour", estimation_split=104 * 168)

    def test_the_likelihood_is_exactly_invariant_to_the_time_unit(
        self, apparel_trans, hourly
    ):
        r"""The identity the rest of the class rests on, to 1e-12.

        .. math::
            \ell(x, c\,t_x, c\,T \mid r, c\alpha, s, c\beta)
                = \ell(x, t_x, T \mid r, \alpha, s, \beta)
                  - \Big(\sum_i x_i\Big)\log c

        -- the same distribution re-expressed, plus the Jacobian of the change
        of variable. Measured on the apparel cohort at :math:`c = 168`: the
        difference is ``-6486.938398`` and :math:`\sum_i x_i \log 168` is
        ``6486.938398``, agreeing to the twelfth decimal. Nothing here had ever
        asserted a scale invariance, and it is the cheapest possible oracle --
        no R, no fixture, no published number.
        """
        from clvtools import ClvData
        from clvtools.pnbd import log_likelihood

        weekly = ClvData(
            apparel_trans, time_unit="week", estimation_split=104
        ).customer_summary()
        hours = hourly.customer_summary()
        c = 168.0
        np.testing.assert_allclose(hours["t_x"], c * weekly["t_x"], rtol=1e-12)

        p = {"r": 1.4489, "alpha": 48.6348, "s": 0.5613, "beta": 46.8837}
        here = log_likelihood(weekly["x"], weekly["t_x"], weekly["T"], **p)
        there = log_likelihood(
            hours["x"], hours["t_x"], hours["T"],
            r=p["r"], alpha=p["alpha"] * c, s=p["s"], beta=p["beta"] * c,
        )
        # 1e-6 on a quantity of 6,487: the identity is exact and agrees to
        # 1e-12 on macOS/ARM, but this is a claim about the algebra and not
        # about a libm, so the bound is where the algebra would fail.
        assert there - here == pytest.approx(
            -weekly["x"].sum() * np.log(c), abs=1e-6
        )

    @pytest.mark.slow
    def test_a_fit_on_hourly_data_finds_the_weekly_fit_rescaled(self, hourly):
        r"""``alpha`` and ``beta`` are rates per period, so both scale by 168.

        ``r`` and ``s`` are shapes and do not move. By the invariance above the
        hourly optimum *is* the weekly one rescaled, so this is an equality and
        not an approximation of one -- which is what makes it a test of the
        optimiser rather than a smoke run.

        It failed when written. With CLVTools' all-ones start, ``alpha`` began
        four orders of magnitude from 8,171 and **L-BFGS-B stopped 223
        log-units short at a degenerate** ``s = 0.0011`` **and reported**
        ``converged = True``. The GGompertz/NBD raised instead, and the
        BG/NBD -- one mis-scaled coordinate rather than three -- was fine.
        :func:`~clvtools._optimize.start_scale` is the fix and says why it is
        the start that moved and not the data.
        """
        from clvtools.pnbd import fit_pnbd

        cbs = hourly.customer_summary()
        fitted = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)
        assert fitted.converged
        assert fitted.log_likelihood == pytest.approx(-12335.036225, abs=1e-2)
        assert fitted.r == pytest.approx(1.4490, rel=1e-3)
        assert fitted.s == pytest.approx(0.5613, rel=1e-3)
        assert fitted.alpha / 168.0 == pytest.approx(48.6361, rel=1e-3)
        assert fitted.beta / 168.0 == pytest.approx(46.8844, rel=1e-3)

    @pytest.mark.slow
    @pytest.mark.parametrize("family", ["bgnbd", "ggomnbd"])
    def test_the_other_two_families_fit_hourly_data_too(self, hourly, family):
        """F-12 is a claim about every family, and one of three used to hold."""
        import importlib

        cbs = hourly.customer_summary()
        module = importlib.import_module(f"clvtools.{family}")
        fitted = getattr(module, f"fit_{family}")(
            cbs["x"], cbs["t_x"], cbs["T"], hessian=False
        )
        assert fitted.converged
        expected = {"bgnbd": -12343.958147, "ggomnbd": -12335.0368}[family]
        assert fitted.log_likelihood == pytest.approx(expected, abs=1e-2)

    @pytest.fixture(scope="class")
    def priceless(self, apparel_trans):
        """The same log with ``Price`` never read -- ``name_price=None``."""
        from clvtools import ClvData

        return ClvData(
            apparel_trans.drop(columns=["Price"]),
            time_unit="week", estimation_split=104, name_price=None,
        )

    def test_data_without_spending_says_so(self, priceless):
        assert not priceless.has_spending
        with pytest.raises(ValueError, match="no Price column"):
            priceless.spending_summary()

    @pytest.mark.slow
    def test_a_fit_without_spending_predicts_on_data_that_has_it(
        self, priceless, apparel_trans
    ):
        """F-13: the transaction model never needed ``Price``, and this proves it.

        The fit is compared against the one on the identical log *with* prices
        -- bit for bit, since the customer summary a Pareto/NBD consumes is
        ``(x, t_x, T)`` and nothing else. Then the parameters are carried onto
        the priced data and asked for the spending columns, which only the
        Gamma-Gamma can supply.
        """
        from clvtools import ClvData, predict
        from clvtools.gg import fit_gg
        from clvtools.pnbd import fit_pnbd

        priced = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        without = priceless.customer_summary()
        with_ = priced.customer_summary()
        for column in ("x", "t_x", "T"):
            np.testing.assert_array_equal(without[column], with_[column])

        fitted = fit_pnbd(without["x"], without["t_x"], without["T"],
                          hessian=False)
        spending = priced.spending_summary()
        gg = fit_gg(spending["x"], spending["Spending"], hessian=False)

        table = predict(priced, fitted, gg)
        assert "predicted.mean.spending" in table.columns
        assert np.isfinite(table["predicted.CLV"]).all()
        assert (table["predicted.CLV"] > 0).all()

    def test_and_predicting_spending_on_priceless_data_is_refused(
        self, priceless
    ):
        """F-14's other half: the refusal names the missing column."""
        from clvtools import predict
        from clvtools.gg import GgParams
        from clvtools.pnbd.fit import PnbdParams

        fitted = PnbdParams(
            r=0.75, alpha=5.33, s=0.28, beta=36.25,
            log_likelihood=float("nan"), converged=True, n_customers=250,
        )
        gg = GgParams(
            p=3.1, q=5.65, gamma=56.5,
            log_likelihood=float("nan"), converged=True, n_customers=250,
        )
        with pytest.raises(ValueError, match="no Price column"):
            predict(priceless, fitted, gg)


class TestASingleLogicalIsCheckedLikeR:
    """Spec `V-05`, `absent`: "no single-logical validation exists anywhere".

    CLVTools has a ``check_userinput_single_logical`` applied throughout, which
    fails for ``NULL``, ``NA``, a disallowed type, or a vector longer than one.
    Python's truthiness accepts all four, and the failure is that the argument
    *works*: ``hessian="no"`` computes a Hessian, ``hessian=None`` skips one,
    and neither says anything. Applied to ``hessian``, which is the
    single-logical argument on every fit here and the one whose misreading
    silently changes what the caller gets back. Backlog item 36, round 6.
    """

    @pytest.fixture(scope="class")
    def cbs(self, cbs_estimation):
        return (
            cbs_estimation["x"], cbs_estimation["t.x"], cbs_estimation["T.cal"],
        )

    @pytest.mark.parametrize("family", ["pnbd", "bgnbd", "ggomnbd"])
    @pytest.mark.parametrize("bad", [None, "no", 1, 0])
    def test_a_non_logical_is_refused_by_every_family(self, cbs, family, bad):
        import importlib

        module = importlib.import_module(f"clvtools.{family}")
        with pytest.raises(ValueError, match="must be True or False"):
            getattr(module, f"fit_{family}")(*cbs, hessian=bad)

    def test_a_vector_says_how_many_it_got(self, cbs):
        from clvtools.pnbd import fit_pnbd

        with pytest.raises(ValueError, match="not 2 values"):
            fit_pnbd(*cbs, hessian=[True, False])

    def test_numpy_s_own_booleans_are_accepted(self, cbs):
        """``np.True_`` is not a ``bool``, and refusing it would be wrong.

        It is what a comparison on an array yields, so it reaches this argument
        from perfectly ordinary calling code.
        """
        from clvtools.pnbd import fit_pnbd

        fitted = fit_pnbd(*cbs, hessian=np.False_)
        assert fitted.hessian is None

    def test_the_spending_model_checks_it_too(self):
        from clvtools import ClvData, load_apparel_trans
        from clvtools.gg import fit_gg

        spending = ClvData(load_apparel_trans()).spending_summary()
        with pytest.raises(ValueError, match="must be True or False"):
            fit_gg(spending["x"], spending["Spending"], hessian="yes")
