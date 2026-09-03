r"""Invariants that determine their own answer, and so need no oracle.

Findings D5 and D6 of ``docs/spec-audit.md``: claims whose truth is fixed by
the inputs rather than by anything CLVTools printed. They are the cheapest
tests in that audit and they cover the mechanisms most likely to break
silently, because every one of them joins two code paths that agreement at a
single fitted optimum would never separate.

Three groups.

**The nesting, past the likelihood.** S3.3: "With covariate effects set to
zero, we arrive at the standard model." ``tests/test_pnbd_staticcov.py``
already asserts that of the individual log-likelihood; spec items X-01, X-04
and X-05 assert it of the *fit*, of :func:`~clvtools.predict.predict` and of
the two diagnostics. Those run through
:func:`~clvtools.predict._model_rates`, a different path from the likelihood's,
and a sign slip there would leave the likelihood tests green.

**Two cross-model agreements.** PR-08, that the spending column inside
``predict()`` is the standalone Gamma-Gamma's own answer, and FI-12, that a
spending model's cbs ``x`` is the Pareto/NBD's. Both sides of FI-12 are
separately oracle-pinned and the agreement between them is not; they are
computed by different methods on :class:`~clvtools.data.ClvData`.

**The bootstrap identity**, B-02 and B-11. Draw every customer exactly once
and the resample must be the original: same cbs, same covariates, same
estimates. It is the strongest available test of the whole resampling path,
and it is the one that would have caught finding A2 -- the dyncov bootstrap
that silently refitted without covariates.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from clvtools import (
    ClvData,
    ClvDataStaticCov,
    bootstrap,
    diagnostics,
    load_apparel_static_cov,
    load_apparel_trans,
    predict,
)
from clvtools.gg import expected_mean_spending, fit_gg
from clvtools.pnbd import PnbdParams, expectation, fit_pnbd, pmf
from clvtools.pnbd.staticcov import (
    PnbdStaticCovParams,
    alpha_i,
    beta_i,
    fit_pnbd_staticcov,
)

#: The paper's own Pareto/NBD estimates on the apparel data, S6.2.2. Used
#: rather than a fit wherever the claim is about what happens *given*
#: parameters, so that the invariant is not entangled with the optimiser.
PNBD = (1.4490, 48.6361, 0.5613, 46.8844)

NAMES = ["Gender", "Channel"]


@pytest.fixture(scope="module")
def data() -> ClvData:
    return ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)


@pytest.fixture(scope="module")
def zero_cov(data) -> ClvDataStaticCov:
    """The same customers, carrying one covariate that is identically zero.

    X-01's data: a design matrix of zeros cannot move any rate, whatever
    coefficient is put on it.
    """
    ids = sorted(data.transactions["Id"].unique())
    frame = pd.DataFrame({"Id": ids, "Zero": np.zeros(len(ids))})
    return ClvDataStaticCov(
        data, frame, names_cov_life=["Zero"], names_cov_trans=["Zero"]
    )


@pytest.fixture(scope="module")
def static_cov(data) -> ClvDataStaticCov:
    """Gender and Channel, the real covariates of S6.4."""
    return ClvDataStaticCov(
        data, load_apparel_static_cov(),
        names_cov_life=NAMES, names_cov_trans=NAMES,
    )


def nested_params(gamma_life, gamma_trans, names) -> PnbdStaticCovParams:
    """A covariate fit at :data:`PNBD` with the given coefficients."""
    r, alpha, s, beta = PNBD
    return PnbdStaticCovParams(
        r=r, alpha=alpha, s=s, beta=beta,
        gamma_life=np.asarray(gamma_life, dtype=float),
        gamma_trans=np.asarray(gamma_trans, dtype=float),
        names_cov_life=list(names), names_cov_trans=list(names),
        log_likelihood=float("nan"), converged=True, n_customers=600,
    )


def plain_params() -> PnbdParams:
    """The same four numbers, with no covariates. The other side of X-04."""
    r, alpha, s, beta = PNBD
    return PnbdParams(
        r=r, alpha=alpha, s=s, beta=beta,
        log_likelihood=float("nan"), converged=True, n_customers=600,
    )


def nested_rates(cov_data: ClvDataStaticCov) -> dict:
    r"""``(r, alpha_i, s, beta_i)`` at :data:`PNBD` with every coefficient zero.

    What the diagnostics of X-05 are handed. :math:`\exp(\mathbf{0}'\gamma)`
    is one for every customer, so these arrays are the plain scalars repeated
    -- which is the claim, not an assumption: they come out of the rate
    builders rather than out of ``np.full``.
    """
    r, alpha, s, beta = PNBD
    return {
        "r": r,
        "alpha": alpha_i(alpha, np.zeros(2), cov_data.design_trans(NAMES)),
        "s": s,
        "beta": beta_i(beta, np.zeros(2), cov_data.design_life(NAMES)),
    }


# -- the nesting, past the likelihood: X-01, X-04, X-05 -----------------------


class TestZeroCovariatesRecoverThePlainModel:
    r"""S3.3: "With covariate effects set to zero, we arrive at the standard
    model."

    ``tests/test_pnbd_staticcov.py`` asserts that of the individual
    log-likelihood. These assert it of everything downstream of it, which runs
    through :func:`~clvtools.predict._model_rates` rather than through
    :func:`~clvtools.pnbd.staticcov.log_likelihood_staticcov_ind`.
    """

    def test_an_all_zero_design_fits_the_plain_estimates(self, data, zero_cov):
        """X-01, at CLVTools' own tolerance of 0.001.

        A design matrix of zeros cannot move any rate, so the four model
        parameters must land where the no-covariate fit lands. The two
        coefficients are then unidentified -- the optimiser leaves them
        wherever the search happened to stop, and that is correct rather than a
        defect, so only the model parameters are compared.
        """
        cbs = data.customer_summary()
        plain = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)
        nested = fit_pnbd_staticcov(zero_cov, hessian=False)

        np.testing.assert_allclose(
            [nested.r, nested.alpha, nested.s, nested.beta], list(plain),
            rtol=1e-3,
        )
        assert nested.log_likelihood == pytest.approx(plain.log_likelihood, abs=1e-6)

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"prediction_end": 6}, {"continuous_discount_factor": 0.25}],
        ids=["as-is", "prediction-end", "discount-factor"],
    )
    def test_zero_coefficients_predict_the_plain_table(
        self, data, static_cov, kwargs
    ):
        """X-04, asserted the three ways CLVTools' own suite asserts it.

        The covariates here are the real Gender and Channel, so the design is
        not degenerate; it is the coefficients that are zero. Nothing is
        approximate about this one -- :math:`\\exp(0) = 1` exactly -- so it is
        compared without a tolerance.
        """
        plain = predict(data, plain_params(), **kwargs)
        nested = predict(
            static_cov, nested_params(np.zeros(2), np.zeros(2), NAMES), **kwargs
        )
        pd.testing.assert_frame_equal(plain, nested, check_exact=True)

    def test_zero_coefficients_give_the_plain_pmf_plot(self, data, static_cov):
        """X-05's PMF half. ``pmf_data`` sums a per-customer PMF, so the
        covariate model reaches it as an array of :math:`\\alpha_i`."""
        r, alpha, s, beta = PNBD
        plain = diagnostics.pmf_data(data, lambda k, T: pmf(k, T, r, alpha, s, beta))
        rates = nested_rates(static_cov)
        nested = diagnostics.pmf_data(static_cov, lambda k, T: pmf(k, T, **rates))
        pd.testing.assert_frame_equal(plain, nested, check_exact=True)

    def test_zero_coefficients_give_the_plain_tracking_plot(self, data, static_cov):
        """X-05's tracking half. The unconditional expectation does not depend
        on a customer's history, so the plain series is ``n`` times one
        customer's and the covariate series is the sum over customers."""
        r, alpha, s, beta = PNBD
        n = data.nobs()
        plain = diagnostics.tracking_data(
            data, lambda t: n * expectation(t, r, alpha, s, beta)
        )
        rates = nested_rates(static_cov)
        nested = diagnostics.tracking_data(
            static_cov, lambda t: expectation(t, **rates).sum()
        )
        # The only comparison here with a tolerance, and the reason is
        # arithmetic rather than the model: the plain series multiplies one
        # customer's expectation by 600 while the covariate series adds 600
        # copies of it, so they part company in the last two bits. Exact
        # equality is asserted everywhere else in this class.
        pd.testing.assert_frame_equal(plain, nested, rtol=1e-13)


# -- two cross-model agreements: PR-08, FI-12 --------------------------------


class TestCrossModelAgreements:
    """Two claims that join modules whose sides are each oracle-pinned alone."""

    def test_the_predicted_spending_is_the_gamma_gammas_own_answer(self, data):
        """PR-08. ``predict()`` takes a fitted spending model and reports
        ``predicted.mean.spending``; that column must be nothing more than
        :func:`~clvtools.gg.expected_mean_spending` on the spending summary,
        with no rescaling picked up on the way through."""
        cbs, spend = data.customer_summary(), data.spending_summary()
        pnbd = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)
        gg = fit_gg(spend["x"], spend["Spending"], hessian=False)

        table = predict(data, pnbd, gg)
        standalone = expected_mean_spending(
            spend["x"], spend["Spending"], gg.p, gg.q, gg.gamma
        )
        np.testing.assert_array_equal(
            table["predicted.mean.spending"].to_numpy(), np.asarray(standalone)
        )

    @pytest.mark.parametrize("split", [104, None], ids=["holdout", "no-holdout"])
    def test_the_spending_cbs_counts_the_same_transactions(self, split):
        """FI-12, "asserted with and without a holdout".

        ``customer_summary`` counts repeat transactions by subtracting each
        customer's first from their total; ``spending_summary`` drops the first
        row per customer and counts what is left. Different methods on the same
        object (``data.py``), separately pinned against the oracle, and their
        agreement stated nowhere.
        """
        data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=split)
        cbs, spend = data.customer_summary(), data.spending_summary()

        np.testing.assert_array_equal(cbs["Id"].to_numpy(), spend["Id"].to_numpy())
        np.testing.assert_array_equal(cbs["x"].to_numpy(), spend["x"].to_numpy())


class TestRepeatTransactionsInTheEstimationPeriod:
    r"""D-17: dropping first transactions and cutting at the split commute.

    One of the two spec items the audit never reached. It is a set equality
    with no oracle behind it, and it holds because the estimation period always
    contains every customer's first transaction -- the estimation *start* is
    the earliest of them. Were that not so, a customer whose first purchase
    fell outside the window would have a different "first" on each side and the
    two sets would part company.
    """

    def test_dropping_firsts_and_cutting_at_the_split_commute(self, data):
        transactions = data.transactions
        cut = transactions["Date"] <= data.estimation_end

        first_overall = transactions.groupby("Id")["Date"].transform("min")
        repeats_then_cut = transactions[(transactions["Date"] > first_overall) & cut]

        within = transactions[cut]
        first_within = within.groupby("Id")["Date"].transform("min")
        cut_then_repeats = within[within["Date"] > first_within]

        pd.testing.assert_frame_equal(
            repeats_then_cut.reset_index(drop=True),
            cut_then_repeats.reset_index(drop=True),
        )

    def test_the_three_counts_of_them_agree(self, data):
        """The same 1,266, reached three ways: by construction above, by
        ``customer_summary``'s ``x``, and by the descriptive tracking series
        restricted to the estimation period. Three code paths, one answer."""
        transactions = data.transactions
        first_overall = transactions.groupby("Id")["Date"].transform("min")
        constructed = int(
            (
                (transactions["Date"] > first_overall)
                & (transactions["Date"] <= data.estimation_end)
            ).sum()
        )

        summarised = int(data.customer_summary()["x"].sum())
        tracked = diagnostics.tracking_data(data)
        plotted = float(
            tracked.loc[
                tracked["period.until"] <= data.estimation_end, "value"
            ].sum()
        )

        assert constructed == summarised == 1266
        assert plotted == 1266.0


# -- the bootstrap identity: B-02, B-11 --------------------------------------


def everyone(pool: np.ndarray) -> np.ndarray:
    """The degenerate resample: every customer, exactly once."""
    return pool


@pytest.fixture(scope="module")
def resampled(data) -> ClvData:
    """The whole pool, drawn once, through the real ``bootstrap_apply`` path."""
    [same] = bootstrap.bootstrap_apply(data, lambda d: d, num_boots=1, sample=everyone)
    return same


class TestDrawingEveryCustomerOnce:
    r"""B-02 and B-11: the resample of the whole pool must be the original.

    Nothing here is approximate. Each customer is drawn once, so no id is
    suffixed, no history is duplicated, and every quantity the model sees
    should come back bit for bit -- including the estimation and holdout
    boundaries, which S6.3.3 says are pinned rather than re-derived.

    This is the test finding A2 needed. A dyncov object used to fall through
    the covariate branch and arrive as a plain :class:`ClvData` with every
    covariate gone; drawing everyone would have shown it immediately, because
    the covariates would have been missing from a resample that changed
    nothing else.
    """

    def test_the_customer_summary_comes_back_unchanged(self, data, resampled):
        pd.testing.assert_frame_equal(
            data.customer_summary(), resampled.customer_summary()
        )

    def test_the_spending_summary_comes_back_unchanged(self, data, resampled):
        pd.testing.assert_frame_equal(
            data.spending_summary(), resampled.spending_summary()
        )

    def test_the_periods_come_back_unchanged(self, data, resampled):
        assert resampled.estimation_end == data.estimation_end
        assert resampled.estimation_start == data.estimation_start
        assert resampled.data_end == data.data_end
        assert resampled.has_holdout == data.has_holdout
        assert resampled.time_unit == data.time_unit

    def test_the_covariates_come_back_unchanged(self, static_cov):
        """B-02's static-covariate arm, and B-08's "sorted the same as the
        cbs" -- the design matrices are compared row for row, so a resample
        that reordered them against the customer summary would fail here."""
        [same] = bootstrap.bootstrap_apply(
            static_cov, lambda d: d, num_boots=1, sample=everyone
        )
        assert isinstance(same, ClvDataStaticCov)
        assert same.names_cov_life == static_cov.names_cov_life
        assert same.names_cov_trans == static_cov.names_cov_trans
        np.testing.assert_array_equal(
            same.design_life(NAMES), static_cov.design_life(NAMES)
        )
        np.testing.assert_array_equal(
            same.design_trans(NAMES), static_cov.design_trans(NAMES)
        )
        pd.testing.assert_frame_equal(
            same.customer_summary(), static_cov.customer_summary()
        )

    def test_the_estimate_comes_back_unchanged(self, data, resampled):
        """B-11. Same data, same optimiser, same start: the same optimum, to
        the last bit rather than to a tolerance."""
        original = data.customer_summary()
        drawn = resampled.customer_summary()
        first = fit_pnbd(original["x"], original["t_x"], original["T"], hessian=False)
        again = fit_pnbd(drawn["x"], drawn["t_x"], drawn["T"], hessian=False)
        assert list(again) == list(first)
        assert again.log_likelihood == first.log_likelihood


class TestZeroCovariatesMatchTheNoCovariateModelForAnyGamma:
    """Spec X-02, `weak`: the existing test used a *fixed* gamma.

    With covariate data identically zero, ``exp(gamma'x) = exp(0) = 1`` for
    **every** gamma, so a covariate model is the plain one. The audit's note is
    the whole point: "the randomness is the claim: it must hold for any gamma".
    A fixed gamma leaves open the one way this could fail -- a gamma reaching
    the likelihood by some path other than the multiplier.

    Drawn from a seeded generator across a wide range, so a coefficient of 40 is
    tried beside one of 1e-9, and for all three families rather than the
    Pareto/NBD alone. Element-wise where the family exposes a per-customer
    entry point, summed where it does not. Backlog item 34, round 5.
    """

    N_DRAWS = 8
    SEED = 20260903
    MODEL: ClassVar[dict] = {
        "pnbd": (1.4490, 48.6361, 0.5613, 46.8844),
        "bgnbd": (0.2426, 4.4136, 0.7929, 2.4259),
        "ggomnbd": (0.6, 10.0, 1e-6, 0.6, 12.0),
    }

    @pytest.fixture(scope="class")
    def inputs(self, cbs_estimation):
        x = cbs_estimation["x"].to_numpy(dtype=float)
        t_x = cbs_estimation["t.x"].to_numpy(dtype=float)
        T = cbs_estimation["T.cal"].to_numpy(dtype=float)
        return x, t_x, T, np.zeros((x.size, 2))

    def _gammas(self):
        rng = np.random.default_rng(self.SEED)
        for _ in range(self.N_DRAWS):
            # Wide on purpose: exp(40 * 0) is 1 exactly, and so is exp(1e-9 * 0).
            yield rng.uniform(-40.0, 40.0, size=2)

    def test_the_pareto_nbd_agrees_element_wise_for_every_drawn_gamma(
        self, inputs
    ):
        from clvtools.pnbd import log_likelihood_ind
        from clvtools.pnbd.staticcov import log_likelihood_staticcov_ind

        x, t_x, T, zeros = inputs
        plain = log_likelihood_ind(x, t_x, T, *self.MODEL["pnbd"])
        for gamma in self._gammas():
            covaried = log_likelihood_staticcov_ind(
                x, t_x, T, *self.MODEL["pnbd"],
                gamma_life=gamma, gamma_trans=gamma,
                cov_life=zeros, cov_trans=zeros,
            )
            # Exact, not close: `exp(0)` is 1 to the bit, and a tolerance here
            # would hide a multiplier that was 1 + 1e-13 for a reason.
            assert np.array_equal(covaried, plain), f"moved for gamma={gamma}"

    @pytest.mark.parametrize("family", ["bgnbd", "ggomnbd"])
    def test_and_the_other_families_agree_in_the_sum(self, inputs, family):
        """Summed because these two expose no per-customer covariate entry."""
        import clvtools

        x, t_x, T, zeros = inputs
        module = getattr(clvtools, family)
        plain = float(np.sum(
            module.log_likelihood_ind(x, t_x, T, *self.MODEL[family])
        ))
        for gamma in self._gammas():
            covaried = module.log_likelihood_staticcov(
                x, t_x, T, *self.MODEL[family],
                gamma_life=gamma, gamma_trans=gamma,
                cov_life=zeros, cov_trans=zeros,
            )
            assert float(covaried) == pytest.approx(plain, rel=1e-15), (
                f"{family} moved for gamma={gamma}"
            )
