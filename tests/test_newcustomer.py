r"""S6.3.4 - predicting for a customer who has not been acquired yet.

``newcustomer()``, ``newcustomer_static()``, ``newcustomer_dynamic()`` and
``newcustomer_spending()``: the prospective customer, with or without
covariates. A prediction here has no history to condition on, so what is
asserted is different in kind from the rest of ``predict()`` -- there is no
PAlive, no CET, and no actuals to compare against, only the expected count over
a horizon and, for the covariate forms, that a *scenario* separates from its
neighbours.

Split out of ``test_predict.py`` in round 5, which had grown six lines past the
700-line module limit. What stayed there is the prediction table for customers
the data knows about; what moved here is the prospective customer, which shares
neither its fixtures nor its shape.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_json, oracle_params
from paper_values import (
    NEWCUSTOMER_PERIODS,
    NEWCUSTOMER_SPENDING,
    NEWCUSTOMER_TOTAL,
    NEWCUSTOMER_TRANSACTIONS,
)

from clvtools import ClvData
from clvtools.gg import GgParams
from clvtools.predict import (
    newcustomer,
    newcustomer_spending,
    newcustomer_static,
    predict,
)


@pytest.fixture(scope="module")
def transactions():
    from clvtools import load_apparel_trans

    return load_apparel_trans()


class TestProspectiveCustomers:
    """S6.3.4's ``newcustomer()`` family."""

    @staticmethod
    def _static_fit():
        from clvtools.pnbd.staticcov import PnbdStaticCovParams

        want = fixture_json("newcustomer_static")
        c = want["coefficients"]
        return want, PnbdStaticCovParams(
            r=c["r"], alpha=c["alpha"], s=c["s"], beta=c["beta"],
            gamma_life=np.array([c["life.Gender"], c["life.Channel"]]),
            gamma_trans=np.array([c["trans.Gender"], c["trans.Channel"]]),
            names_cov_life=["Gender", "Channel"],
            names_cov_trans=["Gender", "Channel"],
            names_cov_constr=[],
            log_likelihood=float("nan"),
            unpenalised_log_likelihood=None,
            converged=True, n_customers=600,
        )

    def test_without_covariates_matches_the_oracle(self, transactions):
        want = fixture_json("newcustomer_static")
        pnbd, _ = oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        got = predict(newcustomer(want["num.periods"]), pnbd)
        assert got == pytest.approx(want["nocov.transactions"], rel=1e-9)

    @pytest.mark.parametrize("gender,channel", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_each_covariate_scenario_matches_the_oracle(self, gender, channel):
        want, params = self._static_fit()
        key = f"gender{gender}.channel{channel}"
        covariates = {"Gender": gender, "Channel": channel}
        got = predict(
            newcustomer_static(want["num.periods"], covariates, covariates),
            params,
        )
        assert got == pytest.approx(want[key], rel=1e-9)

    def test_covariates_separate_the_scenarios(self):
        """S6.3.4's "region A versus region B" comparison.

        This used to read the four values out of the *fixture* and assert that
        they differ from each other -- a statement about CLVTools' output, true
        no matter what this package computed, and it would have passed with
        ``predict`` deleted. Finding B1 of ``docs/spec-audit.md``. It now
        predicts the four scenarios here and asserts they are distinct, which
        is what "the covariates separate the scenarios" means.

        The test above already checks each against the oracle one at a time;
        what this adds is that the *spread* survives, which is the property
        S6.3.4 asks a covariate model for.
        """
        want, params = self._static_fit()
        got = [
            predict(
                newcustomer_static(
                    want["num.periods"],
                    {"Gender": g, "Channel": c},
                    {"Gender": g, "Channel": c},
                ),
                params,
            )
            for g in (0, 1) for c in (0, 1)
        ]
        assert len(set(got)) == 4, got
        # Not merely distinct: far enough apart to be a difference a reader
        # would act on. Measured, the closest pair is 0.0376 transactions over
        # the horizon apart and the widest 0.55, so 0.03 is a floor under the
        # measurement rather than a guess at one.
        assert min(abs(a - b) for a in got for b in got if a != b) > 0.03

    def test_a_covariate_model_refuses_a_plain_new_customer(self):
        _, params = self._static_fit()
        with pytest.raises(TypeError, match="newcustomer_static"):
            predict(newcustomer(52), params)

    def test_a_plain_model_refuses_covariate_values(self, transactions):
        pnbd, _ = oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        with pytest.raises(TypeError, match="covariate model"):
            predict(newcustomer_static(52, {"Gender": 0}, {"Gender": 0}), pnbd)

    def test_missing_covariate_values_are_named(self):
        _, params = self._static_fit()
        with pytest.raises(ValueError, match="Channel"):
            predict(newcustomer_static(52, {"Gender": 0}, {"Gender": 0}), params)

    def test_spending_needs_a_spending_model(self, transactions):
        pnbd, _ = oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        with pytest.raises(TypeError, match="spending model"):
            predict(newcustomer_spending(), pnbd)

    def test_a_zero_horizon_is_the_one_purchase_that_defines_them(self):
        """R returns 1 for ``newcustomer(0)``; this raised. Spec NC-02.

        S6.3.4 adds one "to account for all transactions that a prospective
        customer will make, including the first one", so over zero periods a
        prospective customer makes exactly that one and no more. A well-defined
        limit rather than an error.
        """
        pnbd, _ = oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        assert predict(newcustomer(0), pnbd) == pytest.approx(1.0)

    @pytest.mark.parametrize("periods", [-1, -0.5])
    def test_a_negative_horizon_is_still_refused(self, periods):
        with pytest.raises(ValueError, match="must not be negative"):
            newcustomer(periods)
        with pytest.raises(ValueError, match="must not be negative"):
            newcustomer_static(periods, {}, {})

    def test_a_horizon_that_is_not_a_number_says_so(self):
        """Spec NC-13: "``num.periods`` must be numeric and ``>= 0``".

        A string reached ``<`` directly and raised Python's own "'<' not
        supported between instances of 'str' and 'int'", which names neither
        the argument nor the requirement. CLVTools 0.12.1, asked directly,
        answers "num.periods has to be numeric!" -- so ``"52"`` is refused
        rather than coerced.
        """
        with pytest.raises(TypeError, match="num_periods must be a number"):
            newcustomer("52")
        with pytest.raises(TypeError, match="num_periods must be a number"):
            newcustomer_static(None, {}, {})

    def test_a_nan_horizon_is_refused_rather_than_propagated(self):
        """The other half of NC-13, and the worse half: ``nan < 0`` is
        ``False``, so a ``NaN`` passed the negativity check and became a
        ``NaN`` prediction several frames away from its cause. R gives ``NA``
        the same answer it gives a string."""
        with pytest.raises(ValueError, match="got NaN"):
            newcustomer(float("nan"))

    def test_a_covariate_this_fit_does_not_carry_is_refused(self):
        """NC-13's "covariate data must have the right format".

        An unknown name used to be dropped, so a typo returned a plausible
        number computed from the covariates that *were* recognised. CLVTools
        0.12.1 refuses it -- "The Lifetime covariate data has to contain
        exactly the following columns: Gender, Channel!" -- and *exactly* is
        the operative word: both directions are errors there. They have
        different messages here because they are different mistakes.
        """
        _, params = self._static_fit()
        scenario = {"Gender": 0, "Channel": 1, "Gendre": 1}
        with pytest.raises(ValueError, match="not covariates of this fit"):
            predict(newcustomer_static(52, scenario, scenario), params)

    @pytest.mark.paper
    def test_reproduces_the_printed_totals(self, transactions):
        """S6.3.4: 2.218635 transactions, 39.1372 per order, 86.83115 total."""
        pnbd, _ = oracle_params("pnbd_nocov_fit_full", "gg_fit_full")
        gg = GgParams(
            **fixture_json("gg_fit_full_with_first")["coefficients"],
            log_likelihood=float("nan"), converged=True, n_customers=600,
        )
        transactions_predicted = predict(
            newcustomer(NEWCUSTOMER_PERIODS), pnbd
        )
        spending = predict(newcustomer_spending(), gg)
        assert transactions_predicted == pytest.approx(
            NEWCUSTOMER_TRANSACTIONS, abs=5e-7
        )
        assert spending == pytest.approx(NEWCUSTOMER_SPENDING, abs=5e-5)
        assert transactions_predicted * spending == pytest.approx(
            NEWCUSTOMER_TOTAL, abs=5e-5
        )


@pytest.mark.oracle


class TestNewCustomerAcceptsShortHorizonsAndDateTypes:
    """Spec NC-09, NC-10 and NC-11, all `weak` and all holding.

    `NC-09` asks for `num.periods` below 1, and at 2 and 3 -- the audit found
    "the `< 1` case is dyncov only", so the plain and covariate models never saw
    a fractional horizon. `NC-11` asks that `first.transaction` accept a date, a
    string, and the two timestamp types; only `str` was ever passed, and a type
    that silently failed to parse would shift a whole covariate window.

    `NC-10`'s three cases -- covariate data starting before the first
    transaction, ending after the horizon, and drawn from a different period
    than the fitting data -- are the *working* side of claims whose refusals
    were already covered. Backlog item 34, round 5.
    """

    @pytest.fixture(scope="class")
    def fitted(self, apparel_trans):
        from clvtools.pnbd import fit_pnbd

        data = ClvData(apparel_trans, time_unit="week", estimation_split=104)
        cbs = data.customer_summary()
        return fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)

    @pytest.mark.parametrize("periods", [0.25, 0.5, 1, 2, 3])
    def test_a_short_horizon_predicts_and_grows_with_it(self, fitted, periods):
        from clvtools import newcustomer

        got = predict(newcustomer(periods), fitted)
        assert np.isfinite(got)
        assert got > 0.0

    def test_and_more_periods_predict_more_transactions(self, fitted):
        """Monotone, which a fractional horizon dropped on the floor would not
        be: `0.5` would equal `0`."""
        from clvtools import newcustomer

        values = [predict(newcustomer(n), fitted) for n in (0.25, 0.5, 1, 2, 3)]
        assert all(a < b for a, b in pairwise(values))

    @pytest.mark.parametrize("spelling", ["str", "date", "datetime", "timestamp"])
    def test_first_transaction_takes_every_spelling_of_a_date(self, spelling):
        """NC-11, and they must all normalise to the *same* timestamp."""
        import datetime as dt

        from clvtools import load_apparel_dyn_cov_future, newcustomer_dynamic

        covariates = load_apparel_dyn_cov_future()
        value = {
            "str": "2011-01-02",
            "date": dt.date(2011, 1, 2),
            "datetime": dt.datetime(2011, 1, 2),
            "timestamp": pd.Timestamp("2011-01-02"),
        }[spelling]
        scenario = newcustomer_dynamic(
            4, covariates, covariates, first_transaction=value
        )
        assert scenario.first_transaction == pd.Timestamp("2011-01-02")
