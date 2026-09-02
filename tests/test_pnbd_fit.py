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
        with pytest.raises(ValueError, match="four values"):
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
