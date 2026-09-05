r"""S3.3 and S6.4.1 - the Pareto/NBD with time-invariant covariates.

The extension is checked three ways:

  * against the oracle's per-customer :math:`\alpha_i`, :math:`\beta_i`,
    likelihood, PAlive, CET and DERT at two parameter vectors;
  * against S6.4.1's printed coefficient table, standard errors and z-values;
  * against the model *without* covariates, which S3.3 says is nested inside
    it: "With covariate effects set to zero, we arrive at the standard model."

The nesting test is the one that would catch a sign error in
:func:`~clvtools.pnbd.staticcov.alpha_i`, which no amount of agreement at the
fitted optimum would reveal.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_csv, fixture_json
from paper_values import (
    PNBD_STATIC_AIC,
    PNBD_STATIC_BIC,
    PNBD_STATIC_LL,
    PNBD_STATIC_MLE,
    PNBD_STATIC_SE,
    PNBD_STATIC_Z,
)

from clvtools import (
    ClvData,
    ClvDataStaticCov,
    load_apparel_static_cov,
    load_apparel_trans,
)
from clvtools.pnbd import log_likelihood_ind
from clvtools.pnbd.aggregate import (
    conditional_expected_transactions,
    discounted_expected_residual_transactions,
    probability_alive,
)
from clvtools.pnbd.staticcov import (
    alpha_i,
    beta_i,
    fit_pnbd_staticcov,
    log_likelihood,
    log_likelihood_staticcov_ind,
)

NAMES = ["Gender", "Channel"]
GRID = fixture_json("pnbd_staticcov_grid")
CASES = list(GRID["params"])
DELTA = float(np.log(1.075) / 52)


@pytest.fixture(scope="module")
def data():
    return ClvDataStaticCov(
        ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
        load_apparel_static_cov(),
        names_cov_life=NAMES,
        names_cov_trans=NAMES,
    )


@pytest.fixture(scope="module")
def inputs(data):
    cbs = data.customer_summary().set_index("Id")
    return (
        cbs.index,
        cbs["x"].to_numpy(),
        cbs["t_x"].to_numpy(),
        cbs["T"].to_numpy(),
        data.design_life(),
        data.design_trans(),
    )


@pytest.fixture(scope="module")
def fitted(data):
    return fit_pnbd_staticcov(data)


def _params(case):
    p = GRID["params"][case]
    return (
        p["model"],
        np.array(list(p["life"].values())),
        np.array(list(p["trans"].values())),
    )


class TestDesignMatrices:
    """S6.4: the covariate frame lined up one row per customer."""

    @pytest.mark.oracle
    @pytest.mark.parametrize("which", ["life", "trans"])
    def test_matches_the_oracles_design(self, data, inputs, which):
        ids = inputs[0]
        want = (
            fixture_csv(f"staticcov_design_{which}")
            .set_index("Id").loc[ids, NAMES].to_numpy(dtype=float)
        )
        got = getattr(data, f"design_{which}")()
        np.testing.assert_array_equal(got, want)

    def test_rows_follow_the_customer_summary_order(self, data, inputs):
        assert data.design_life().shape == (len(inputs[0]), 2)
        assert list(data.customers) == list(inputs[0])

    def test_categorical_columns_become_k_minus_one_dummies(self):
        """S6.4: "Categorical data is turned into k-1 dummy variables"."""
        trans = load_apparel_trans()
        ids = sorted(trans["Id"].unique())
        covariates = pd.DataFrame({
            "Id": ids,
            "Region": ["north", "south", "east"] * (len(ids) // 3),
        })
        data = ClvDataStaticCov(
            ClvData(trans, time_unit="week", estimation_split=104), covariates
        )
        # Three levels give two columns, not three.
        assert sorted(data.names_cov_life) == ["Region_north", "Region_south"]
        assert data.design_life().shape[1] == 2

    def test_rejects_covariates_missing_a_customer(self):
        trans = load_apparel_trans()
        partial = load_apparel_static_cov().iloc[:100]
        with pytest.raises(ValueError, match="missing 500 customers"):
            ClvDataStaticCov(
                ClvData(trans, time_unit="week", estimation_split=104), partial
            )

    def test_rejects_an_unknown_covariate_name(self, ):
        trans = load_apparel_trans()
        with pytest.raises(ValueError, match="life covariates not in the data"):
            ClvDataStaticCov(
                ClvData(trans, time_unit="week", estimation_split=104),
                load_apparel_static_cov(),
                names_cov_life=["Nonexistent"],
            )

    def test_rejects_covariate_data_without_an_id_column(self):
        trans = load_apparel_trans()
        with pytest.raises(ValueError, match="no 'Id' column"):
            ClvDataStaticCov(
                ClvData(trans, time_unit="week", estimation_split=104),
                pd.DataFrame({"Gender": [0, 1]}),
            )

    def test_the_two_processes_can_take_different_covariates(self, ):
        """S6.4: "The covariates for the transaction process and attrition
        process may differ"."""
        data = ClvDataStaticCov(
            ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
            load_apparel_static_cov(),
            names_cov_life=["Gender"],
            names_cov_trans=["Gender", "Channel"],
        )
        assert data.design_life().shape[1] == 1
        assert data.design_trans().shape[1] == 2


@pytest.mark.oracle
class TestAgainstOracle:
    @pytest.mark.parametrize("case", CASES)
    def test_per_customer_rates(self, inputs, case):
        ids, _, _, _, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        want = fixture_csv(f"pnbd_staticcov_{case}").set_index("Id").loc[ids]

        np.testing.assert_allclose(
            alpha_i(model["alpha"], g_trans, cov_trans), want["alpha.i"], rtol=1e-12
        )
        np.testing.assert_allclose(
            beta_i(model["beta"], g_life, cov_life), want["beta.i"], rtol=1e-12
        )

    @pytest.mark.parametrize("case", CASES)
    def test_individual_log_likelihood(self, inputs, case):
        ids, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        want = fixture_csv(f"pnbd_staticcov_{case}").set_index("Id").loc[ids]

        got = log_likelihood_staticcov_ind(
            x, t_x, T, model["r"], model["alpha"], model["s"], model["beta"],
            g_life, g_trans, cov_life, cov_trans,
        )
        np.testing.assert_allclose(got, want["LL.ind"], rtol=1e-11)

    @pytest.mark.parametrize("case", CASES)
    def test_sample_log_likelihood(self, inputs, case):
        _, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        got = log_likelihood(
            x, t_x, T, model["r"], model["alpha"], model["s"], model["beta"],
            g_life, g_trans, cov_life, cov_trans,
        )
        assert got == pytest.approx(GRID["LL.sum"][case], rel=1e-11)

    def test_weights_repeat_rows(self, inputs):
        """Row multiplicities, as the CBS compression uses."""
        _, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params("mle")
        args = (model["r"], model["alpha"], model["s"], model["beta"],
                g_life, g_trans, cov_life[:2], cov_trans[:2])
        each = log_likelihood_staticcov_ind(x[:2], t_x[:2], T[:2], *args)
        weighted = log_likelihood(
            x[:2], t_x[:2], T[:2], *args, weights=[2.0, 3.0]
        )
        assert weighted == pytest.approx(2 * each[0] + 3 * each[1])

    @pytest.mark.parametrize("case", CASES)
    def test_probability_alive(self, inputs, case):
        ids, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        want = fixture_csv(f"pnbd_staticcov_{case}").set_index("Id").loc[ids]
        got = probability_alive(
            x, t_x, T,
            r=model["r"], alpha=alpha_i(model["alpha"], g_trans, cov_trans),
            s=model["s"], beta=beta_i(model["beta"], g_life, cov_life),
        )
        np.testing.assert_allclose(got, want["PAlive"], rtol=1e-10)

    @pytest.mark.parametrize(
        "case", [c for c in CASES if GRID["params"][c]["model"]["s"] != 1.0]
    )
    def test_cet_and_dert(self, inputs, case):
        ids, x, t_x, T, cov_life, cov_trans = inputs
        model, g_life, g_trans = _params(case)
        want = fixture_csv(f"pnbd_staticcov_{case}").set_index("Id").loc[ids]
        rates = {
            "r": model["r"], "alpha": alpha_i(model["alpha"], g_trans, cov_trans),
            "s": model["s"], "beta": beta_i(model["beta"], g_life, cov_life),
        }
        np.testing.assert_allclose(
            conditional_expected_transactions(x, t_x, T, 52.0, **rates),
            want["CET"], rtol=1e-10,
        )
        np.testing.assert_allclose(
            discounted_expected_residual_transactions(x, t_x, T, DELTA, **rates),
            want["DERT"], rtol=1e-10,
        )

    def test_fit_reaches_the_oracles_optimum(self, fitted):
        want = fixture_json("pnbd_staticcov_fit")
        assert fitted.log_likelihood == pytest.approx(want["logLik"], abs=1e-4)
        assert fitted.log_likelihood >= want["logLik"] - 1e-6


class TestNesting:
    r"""S3.3: "With covariate effects set to zero, we arrive at the standard
    model." A sign error in the rate builders would break this and nothing
    else."""

    def test_zero_coefficients_reproduce_the_plain_likelihood(self, inputs):
        _, x, t_x, T, cov_life, cov_trans = inputs
        plain = log_likelihood_ind(x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844)
        nested = log_likelihood_staticcov_ind(
            x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844,
            gamma_life=np.zeros(2), gamma_trans=np.zeros(2),
            cov_life=cov_life, cov_trans=cov_trans,
        )
        np.testing.assert_allclose(nested, plain, rtol=1e-15)

    def test_all_zero_covariates_reproduce_the_plain_likelihood(self, inputs):
        """Non-zero coefficients on an all-zero design must also collapse."""
        _, x, t_x, T, _, _ = inputs
        zeros = np.zeros((x.size, 2))
        plain = log_likelihood_ind(x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844)
        nested = log_likelihood_staticcov_ind(
            x, t_x, T, 1.4490, 48.6361, 0.5613, 46.8844,
            gamma_life=[0.5, -0.3], gamma_trans=[-0.2, 0.9],
            cov_life=zeros, cov_trans=zeros,
        )
        np.testing.assert_allclose(nested, plain, rtol=1e-15)

    def test_the_covariate_fit_beats_the_plain_one(self, fitted):
        """More parameters, nested model: the likelihood cannot be lower."""
        plain = fixture_json("pnbd_nocov_fit")["logLik"]
        assert fitted.log_likelihood > plain

    def test_a_positive_transaction_coefficient_lowers_alpha(self):
        r"""S6.4.1 reads ``trans.Gender = 0.2859`` as a higher purchase rate;
        since :math:`\alpha` is a rate parameter that means a lower
        :math:`\alpha_i`."""
        covariates = np.array([[0.0], [1.0]])
        got = alpha_i(50.0, [0.2859], covariates)
        assert got[1] < got[0]

    def test_a_positive_lifetime_coefficient_lowers_beta(self):
        """``life.Channel = 0.7907``: offline customers "drop out more
        quickly", a higher attrition rate, so a lower beta."""
        covariates = np.array([[0.0], [1.0]])
        got = beta_i(50.0, [0.7907], covariates)
        assert got[1] < got[0]


@pytest.mark.slow
@pytest.mark.paper
class TestAgainstThePaper:
    def test_log_likelihood_matches(self, fitted):
        assert fitted.log_likelihood == pytest.approx(PNBD_STATIC_LL, abs=1e-3)

    def test_aic_and_bic_match(self, fitted):
        assert fitted.aic == pytest.approx(PNBD_STATIC_AIC, abs=1e-2)
        assert fitted.bic == pytest.approx(PNBD_STATIC_BIC, abs=1e-2)

    def test_covariate_coefficients_match(self, fitted):
        """The four the paper interprets, to the precision it prints them."""
        got = fitted.coefficients
        for name in ("life.Gender", "life.Channel", "trans.Gender", "trans.Channel"):
            assert got[name] == pytest.approx(PNBD_STATIC_MLE[name], abs=2e-3), name

    def test_base_parameters_match_approximately(self, fitted):
        """Looser, because the eight-parameter likelihood ridge is broader.

        The attained log-likelihood is checked separately and exceeds the
        published one, so this bound is about where the optimiser stopped, not
        about the model.
        """
        got = fitted.coefficients
        for name in ("r", "alpha", "s", "beta"):
            assert got[name] == pytest.approx(PNBD_STATIC_MLE[name], rel=5e-3), name

    def test_standard_errors_match(self, fitted):
        """The paper's printed values, at the precision printing leaves."""
        got = fitted.standard_errors()
        for name in ("life.Gender", "life.Channel", "trans.Gender", "trans.Channel"):
            assert got[name] == pytest.approx(PNBD_STATIC_SE[name], rel=5e-2), name

    @pytest.mark.oracle
    def test_standard_errors_match_the_oracle_exactly(self, fitted):
        """And the oracle's own, which are not rounded to four decimals.

        ``inference_pnbd_staticcov.json`` has carried CLVTools' full-precision
        standard errors all along while the check above compared against the
        paper's printed four, at 5% -- so the fixture was written, committed
        and never read (finding 16 of the 2026-09 review). 2e-3 is
        what two optimisers stopping at different points on this ridge can
        agree to, and it is 25 times tighter than the printed comparison.
        """
        want = fixture_json("inference_pnbd_staticcov")
        oracle = dict(zip(want["names"], want["se"], strict=True))
        got = fitted.standard_errors()
        for name in ("life.Gender", "life.Channel", "trans.Gender", "trans.Channel"):
            assert got[name] == pytest.approx(oracle[name], rel=2e-3), name

    def test_z_values_match(self, fitted):
        """S6.4.1's ``z-val`` column, and the significance it reports."""
        table = fitted.summary()
        for name, want in PNBD_STATIC_Z.items():
            assert table.loc[name, "z-val"] == pytest.approx(want, rel=5e-2), name
        # All four are flagged significant at 5% in the printed table.
        for name in PNBD_STATIC_Z:
            assert table.loc[name, "Pr(>|z|)"] < 0.05

    def test_base_parameters_have_no_z_value(self, fitted):
        r"""S6.4.1: "a null hypothesis of :math:`\theta = 0` lies outside the
        admissible parameter space", so those cells are blank."""
        table = fitted.summary()
        for name in ("r", "alpha", "s", "beta"):
            assert np.isnan(table.loc[name, "z-val"])
            assert np.isnan(table.loc[name, "Pr(>|z|)"])


@pytest.mark.slow
class TestFitting:
    def test_converges(self, fitted):
        """S6.4.1 reports ``KKT 1 TRUE`` and ``KKT 2 TRUE``."""
        assert fitted.converged

    def test_nelder_mead_agrees(self, data, fitted):
        got = fit_pnbd_staticcov(data, method="Nelder-Mead", hessian=False)
        assert got.log_likelihood == pytest.approx(fitted.log_likelihood, abs=1e-4)

    def test_is_reached_from_different_starting_points(self, data, fitted):
        got = fit_pnbd_staticcov(
            data, start=(0.5, 30.0, 1.5, 80.0), start_cov=-0.2, hessian=False
        )
        assert got.log_likelihood == pytest.approx(fitted.log_likelihood, abs=1e-4)

    def test_parameter_count_includes_the_covariates(self, fitted):
        assert fitted.n_parameters == 8

    def test_information_criteria_use_the_plain_likelihood_when_unpenalised(
        self, fitted
    ):
        """Without a penalty the two likelihoods coincide."""
        assert fitted.unpenalised_log_likelihood == pytest.approx(
            fitted.log_likelihood
        )
        assert fitted.aic == pytest.approx(16 - 2 * fitted.log_likelihood)

    def test_a_params_object_without_an_unpenalised_value_falls_back(self):
        """Hand-built parameters, as when replaying a published fit."""
        from clvtools.pnbd.staticcov import PnbdStaticCovParams

        params = PnbdStaticCovParams(
            r=1.8378, alpha=92.9123, s=0.5920, beta=49.6227,
            gamma_life=np.array([-0.6430, 0.7907]),
            gamma_trans=np.array([0.2859, 0.6241]),
            names_cov_life=NAMES, names_cov_trans=NAMES,
            log_likelihood=-5821.0627, converged=True, n_customers=600,
        )
        assert params.unpenalised_log_likelihood is None
        assert params.aic == pytest.approx(16 - 2 * -5821.0627)
        assert params.bic == pytest.approx(
            8 * np.log(600) - 2 * -5821.0627
        )

    def test_names_follow_clvtools_convention(self, fitted):
        assert fitted.names == [
            "r", "alpha", "s", "beta",
            "life.Gender", "life.Channel", "trans.Gender", "trans.Channel",
        ]

    def test_rejects_bad_start_values(self, data):
        with pytest.raises(ValueError, match="4 model parameters"):
            fit_pnbd_staticcov(data, start=(1.0, 1.0))
        with pytest.raises(ValueError, match="strictly positive"):
            fit_pnbd_staticcov(data, start=(1.0, -1.0, 1.0, 1.0))

    def test_standard_errors_require_a_hessian(self, data):
        got = fit_pnbd_staticcov(data, hessian=False)
        with pytest.raises(ValueError, match="hessian=True"):
            got.standard_errors()


class TestRateBuilderValidation:
    def test_rejects_a_covariate_parameter_mismatch(self):
        with pytest.raises(ValueError, match="2 transaction covariates but 1"):
            alpha_i(50.0, [0.1], np.zeros((3, 2)))
        with pytest.raises(ValueError, match="2 attrition covariates but 3"):
            beta_i(50.0, [0.1, 0.2, 0.3], np.zeros((3, 2)))


@pytest.mark.slow
class TestPrediction:
    def test_predict_uses_per_customer_rates(self, data, fitted):
        from clvtools.predict import predict

        table = predict(data, fitted, prediction_end=52)
        assert len(table) == 600
        assert ((table["PAlive"] >= 0) & (table["PAlive"] <= 1)).all()
        # Customers differ in covariates, so their predictions must differ even
        # at identical (x, t_x, T).
        cbs = data.customer_summary().set_index("Id")
        zero_purchase = cbs.index[cbs["x"] == 0]
        assert table.loc[zero_purchase, "PAlive"].nunique() > 1  # noqa: PD101

    def test_predict_rejects_data_without_covariates(self, fitted):
        from clvtools.predict import predict

        plain = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        with pytest.raises(TypeError, match="covariate model needs covariate data"):
            predict(plain, fitted, prediction_end=52)


class TestTheCovariateJoinDoesNotDependOnOrder:
    """X-07 and X-08: nothing in the suite ever shuffled or reversed them.

    Every oracle frame arrives in the order the implementation sorts into, so
    a design matrix mis-joined to the customer summary, or ``names_cov_*``
    drifting from column *position*, would be invisible — and
    ``get_dummies`` moving generated dummies to the end is exactly that
    mechanism. Finding D2 of the 2026-09 spec audit.

    Neither test needs an oracle: the fit is compared with itself under a
    permutation that must not matter.
    """

    @staticmethod
    def _fit(covariates, names):
        data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
        static = ClvDataStaticCov(
            data, covariates, names_cov_life=names, names_cov_trans=names
        )
        return fit_pnbd_staticcov(static, hessian=False)

    def test_shuffling_the_covariate_rows_changes_nothing(self):
        """A row is joined to a customer by id, not by position."""
        covariates = load_apparel_static_cov()
        shuffled = covariates.sample(frac=1.0, random_state=20260902)
        assert not shuffled.index.equals(covariates.index)

        names = ["Gender", "Channel"]
        canonical = self._fit(covariates, names)
        permuted = self._fit(shuffled, names)

        assert permuted.coefficients.keys() == canonical.coefficients.keys()
        for name, value in canonical.coefficients.items():
            assert permuted.coefficients[name] == pytest.approx(value, rel=1e-9), name

    def test_reversing_the_covariate_columns_changes_nothing(self):
        """A coefficient follows its name, not the column it arrived in.

        ``names_cov_life=["Channel", "Gender"]`` on a frame whose columns are
        in the other order must give the same ``life.Gender`` as the canonical
        fit -- if the design matrix were built by position, the two
        coefficients would swap and this would fail.
        """
        covariates = load_apparel_static_cov()
        reversed_columns = covariates[["Id", "Channel", "Gender"]]

        canonical = self._fit(covariates, ["Gender", "Channel"])
        reversed_fit = self._fit(reversed_columns, ["Channel", "Gender"])

        # 1e-5 rather than 1e-9: reversing the columns reverses the order of
        # the sums inside the design-matrix product, so the search follows a
        # slightly different path and stops 1e-7 away on this flat ridge. What
        # the test is for survives that by four orders of magnitude -- if the
        # coefficients followed *position* instead of name, `life.Gender` and
        # `life.Channel` would swap, and they differ by 1.43.
        for name, value in canonical.coefficients.items():
            assert reversed_fit.coefficients[name] == pytest.approx(
                value, rel=1e-5
            ), name

        assert abs(
            canonical.coefficients["life.Gender"]
            - canonical.coefficients["life.Channel"]
        ) > 1.0


class TestCategoricalCovariatesBecomeDummies:
    """Spec C-01 to C-04, all `weak` and all holding -- one arm each was tested.

    Four claims about how covariate columns reach the design matrix: character
    and factor give the same dummies with and without a holdout (`C-01`); a
    2-category variable gives 1 dummy and a 3-category one gives 2 (`C-02`);
    categories convert whether or not numeric covariates are present (`C-03`);
    and numeric covariates stay numeric either way (`C-04`).

    The mixed case is the one worth reaching, and the audit said why: it "is
    where ``get_dummies`` reorders". It does -- numerics come out first
    regardless of the input order -- so what matters is not the order itself but
    that :attr:`names_cov_life` still describes the matrix column for column.
    A silent transposition there would give every coefficient the wrong name.
    """

    @staticmethod
    def _built(covariates: pd.DataFrame, *, holdout: bool = True):
        from clvtools import ClvData, ClvDataStaticCov

        transactions = pd.DataFrame([
            {"Id": customer, "Date": pd.Timestamp("2005-01-03")
             + pd.Timedelta(weeks=week)}
            for customer in covariates["Id"]
            for week in (0, 2, 6)
        ])
        data = ClvData(
            transactions, time_unit="week",
            estimation_split=3 if holdout else None,
        )
        return ClvDataStaticCov(data, covariates)

    IDS: ClassVar[list[str]] = ["a", "b", "c", "d"]

    def test_two_categories_give_one_dummy(self):
        frame = pd.DataFrame({"Id": self.IDS, "G": ["m", "f", "m", "f"]})
        assert self._built(frame).names_cov_life == ["G_m"]

    def test_three_categories_give_two(self):
        frame = pd.DataFrame({"Id": self.IDS, "G": ["m", "f", "x", "m"]})
        assert self._built(frame).names_cov_life == ["G_m", "G_x"]

    @pytest.mark.parametrize("holdout", [True, False])
    def test_character_and_categorical_agree_either_side_of_a_split(
        self, holdout
    ):
        """C-01's two arms: the dtype must not change the encoding."""
        character = pd.DataFrame({"Id": self.IDS, "G": ["m", "f", "x", "m"]})
        categorical = character.copy()
        categorical["G"] = categorical["G"].astype("category")
        assert (
            self._built(character, holdout=holdout).names_cov_life
            == self._built(categorical, holdout=holdout).names_cov_life
            == ["G_m", "G_x"]
        )

    def test_numeric_covariates_stay_numeric_beside_categories(self):
        """C-03 and C-04's mixed arm, and `get_dummies` reordering it."""
        frame = pd.DataFrame({
            "Id": self.IDS, "G": ["m", "f", "x", "m"],
            "Num": [10.0, 20.0, 30.0, 40.0],
        })
        built = self._built(frame)
        # Numerics first, whatever order the input frame had them in.
        assert built.names_cov_life == ["Num", "G_m", "G_x"]

    def test_and_the_names_still_describe_the_matrix_column_for_column(self):
        """The claim the reordering actually threatens.

        `Num` is the identity column, `G_m` marks a and d, `G_x` marks c. If the
        names and the matrix disagreed, every coefficient would be labelled
        with a neighbour's name and nothing else would notice.
        """
        frame = pd.DataFrame({
            "Id": self.IDS, "G": ["m", "f", "x", "m"],
            "Num": [10.0, 20.0, 30.0, 40.0],
        })
        built = self._built(frame)
        matrix = np.asarray(built.design_life())
        assert built.names_cov_life == ["Num", "G_m", "G_x"]
        np.testing.assert_array_equal(matrix[:, 0], [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(matrix[:, 1], [1.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(matrix[:, 2], [0.0, 0.0, 1.0, 0.0])

    def test_numeric_only_data_produces_no_dummies(self):
        frame = pd.DataFrame({"Id": self.IDS, "Num": [1.0, 2.0, 3.0, 4.0]})
        assert self._built(frame).names_cov_life == ["Num"]


class TestACategoricalCovariateCanBeNamedByItsOwnName:
    """Spec C-09, `absent` — and a bigger gap than the row describes.

    S6.4 turns a categorical covariate into k-1 dummies, so a column ``Region``
    with three levels becomes ``Region_b`` and ``Region_c``. The requested names
    were compared against the **encoded** frame, so naming a categorical
    covariate by its own name reported ``covariates not in the data:
    ['Region']`` -- of a column plainly in the data the caller passed. Selecting
    one was impossible; you had to know the dummy spelling.

    Nothing caught it because the apparel cohort's covariates are 0/1 **numeric**
    and keep their names through the encoding. Every test in this repository
    uses them.

    `C-09` itself is the single-category case, which now earns its own message:
    k-1 is zero, so the column contributes no dummies and carries no
    information -- "not in the data" would send the reader hunting a typo.
    """

    @pytest.fixture(scope="class")
    def base(self, apparel_trans):
        from clvtools import ClvData

        return ClvData(apparel_trans, time_unit="week", estimation_split=104)

    @staticmethod
    def _built(base, frame, names):
        from clvtools import ClvDataStaticCov

        return ClvDataStaticCov(
            base, frame, names_cov_life=names, names_cov_trans=names
        )

    def test_two_levels_resolve_to_the_one_dummy(self, base, apparel_static_cov):
        frame = apparel_static_cov.assign(
            Two=["a", "b"] * (len(apparel_static_cov) // 2)
        )
        built = self._built(base, frame, ["Two"])
        assert built.names_cov_life == ["Two_b"]
        assert built.design_life().shape == (600, 1)

    def test_three_levels_resolve_to_both(self, base, apparel_static_cov):
        frame = apparel_static_cov.assign(
            R=["a", "b", "c"] * (len(apparel_static_cov) // 3)
        )
        built = self._built(base, frame, ["R"])
        assert built.names_cov_life == ["R_b", "R_c"]
        assert built.design_life().shape == (600, 2)

    def test_a_numeric_covariate_keeps_its_own_name(self, base, apparel_static_cov):
        """Which is why nothing here ever exercised the branch above."""
        built = self._built(base, apparel_static_cov, ["Gender"])
        assert built.names_cov_life == ["Gender"]

    def test_a_single_level_column_says_it_carries_no_information(
        self, base, apparel_static_cov
    ):
        """C-09 proper, and not "not in the data"."""
        frame = apparel_static_cov.assign(Const="same")
        with pytest.raises(ValueError, match="carries no information"):
            self._built(base, frame, ["Const"])

    def test_and_a_genuinely_absent_one_still_says_that(
        self, base, apparel_static_cov
    ):
        """The three failure modes have to stay distinguishable."""
        with pytest.raises(ValueError, match="not in the data"):
            self._built(base, apparel_static_cov, ["Nope"])

    def test_the_dummy_name_may_still_be_given_directly(
        self, base, apparel_static_cov
    ):
        """Selecting one level of a categorical, which the expansion must allow."""
        frame = apparel_static_cov.assign(
            R=["a", "b", "c"] * (len(apparel_static_cov) // 3)
        )
        built = self._built(base, frame, ["R_c"])
        assert built.names_cov_life == ["R_c"]


class TestCovariateNamesThatWereNeverExercised:
    """Spec `FI-05`, `X-11` and `C-14`, all `absent`.

    Three claims about the *names* covariates arrive under, which the apparel
    cohort could never have raised: its two covariates are called ``Gender``
    and ``Channel``, are numeric 0/1, and are each named once. Backlog item 36,
    round 6.
    """

    @pytest.fixture(scope="class")
    def frames(self, apparel_trans, apparel_static_cov):
        from clvtools import ClvData

        return (
            ClvData(apparel_trans, time_unit="week", estimation_split=104),
            apparel_static_cov,
        )

    def test_a_name_given_twice_is_refused(self, frames):
        """`FI-05`: it used to build a rank-deficient design in silence.

        ``names_cov_life=["Gender", "Gender"]`` produced a ``(600, 2)`` matrix
        of two identical columns. The fit then reports two coefficients for one
        covariate, whose *sum* rather than either one is what the data
        identifies, each with a standard error the Hessian cannot support. The
        audit called this "a defect, untested"; it is now refused instead.
        """
        from clvtools import ClvDataStaticCov

        data, cov = frames
        with pytest.raises(ValueError, match="more than once"):
            ClvDataStaticCov(
                data, cov,
                names_cov_life=["Gender", "Gender"],
                names_cov_trans=["Gender"],
            )

    def test_the_message_names_the_process_that_repeated(self, frames):
        """Life and transaction covariates are separate lists and separate bugs."""
        from clvtools import ClvDataStaticCov

        data, cov = frames
        with pytest.raises(ValueError, match=r"^trans covariates name"):
            ClvDataStaticCov(
                data, cov,
                names_cov_life=["Gender"],
                names_cov_trans=["Channel", "Channel"],
            )

    def test_a_syntactically_illegal_name_works(self, frames):
        """`X-11`: R needs backticks for these; here they are just strings.

        A column called ``my gender!`` is not a legal R name and CLVTools' own
        suite tests that it survives ``make.names``. Nothing in this port has a
        name-mangling step at all, so the claim is that none crept in -- which
        is worth an assertion precisely because it would be invisible until
        someone used such a column.
        """
        from clvtools import ClvDataStaticCov

        data, cov = frames
        odd = cov.rename(columns={"Gender": "my gender!", "Channel": "2nd/chan"})
        built = ClvDataStaticCov(
            data, odd,
            names_cov_life=["my gender!", "2nd/chan"],
            names_cov_trans=["my gender!"],
        )
        assert built.names_cov_life == ["my gender!", "2nd/chan"]
        assert built.names_cov_trans == ["my gender!"]
        np.testing.assert_array_equal(
            built._cov_life["my gender!"].to_numpy(),
            cov.set_index("Id")
            .loc[built._cov_life.index, "Gender"]
            .to_numpy(),
        )

    def test_the_covariate_id_column_can_be_called_something_else(self, frames):
        """`C-14`: ``name_id`` on the covariate frame, which had no test."""
        from clvtools import ClvDataStaticCov

        data, cov = frames
        renamed = cov.rename(columns={"Id": "customer"})
        built = ClvDataStaticCov(
            data, renamed, names_cov_life=["Gender"], names_cov_trans=["Gender"],
            name_id="customer",
        )
        plain = ClvDataStaticCov(
            data, cov, names_cov_life=["Gender"], names_cov_trans=["Gender"],
        )
        pd.testing.assert_frame_equal(built._cov_life, plain._cov_life)
        assert built.names_cov_life == plain.names_cov_life
