"""Numbers printed in the R package's own documentation, in one place.

The paper is not the only place CLVTools 0.12.1 prints results. Its vignettes
and man pages print several tables the paper never does, on the same data --
and they exercise features (equality constraints, regularization, ``lrtest``,
``pmf``) for which the paper prints nothing at all.

Sources, all under ``.Rlib/CLVTools/``:

``doc/CLVTools_advanced_techniques.pdf``
    Sections 2 and 4: full ``summary()`` output for a regularized and for a
    constrained static-covariate Pareto/NBD, and the ``lrtest()`` comparing the
    latter against the unconstrained fit.
``help/`` (``?pmf``)
    A Pareto/NBD PMF table on ``cdnow``, with the empirical frequencies beside
    it.

Extract them again with::

    pdftotext -layout .Rlib/CLVTools/doc/CLVTools_advanced_techniques.pdf -
    R_LIBS=.Rlib Rscript -e 'tools::Rd2ex(tools::Rd_db("CLVTools",
      lib.loc=".Rlib")[["pmf.Rd"]])'

Kept out of ``conftest.py`` so that test modules can import them normally,
exactly as :mod:`paper_values` is.
"""

from __future__ import annotations

# -- Pareto/NBD with static covariates and an equality constraint -------------
#
# CLVTools_advanced_techniques.pdf S4:
#
#   latentAttrition(formula = ~ . | ., names.cov.constr = "Gender",
#                   family = pnbd, data = clv.apparel.static, verbose = FALSE)
#
# on apparelTrans with estimation.split = 104 -- the paper's own case study
# data. The unconstrained fit alongside it is the paper's table, already in
# `paper_values.PNBD_STATIC_*`.

CONSTRAINED_MLE = {
    "r": 1.7939,
    "alpha": 94.7223,
    "s": 0.4287,
    "beta": 59.0743,
    "life.Channel": 1.0228,
    "trans.Channel": 0.6384,
    "constr.Gender": 0.3283,
}

CONSTRAINED_SE = {
    "r": 0.3318,
    "alpha": 17.2216,
    "s": 0.1418,
    "beta": 34.5098,
    "life.Channel": 0.3542,
    "trans.Channel": 0.1064,
    "constr.Gender": 0.1074,
}

#: Only the covariate rows. The vignette also prints z-values for `r`, `alpha`,
#: `s` and `beta` (5.406, 5.500, 3.025, 1.712), which contradicts both the
#: paper -- S6.4.1: a null of zero "lies outside the admissible parameter
#: space" -- and CLVTools' own `?pnbd`, which says the indicators "are set to
#: NA on purpose". This package follows the paper and reports NaN there;
#: `test_model_parameters_carry_no_z_value` pins that.
CONSTRAINED_Z = {
    "life.Channel": 2.888,
    "trans.Channel": 5.998,
    "constr.Gender": 3.056,
}

CONSTRAINED_LL = -5826.5342
CONSTRAINED_AIC = 11667.0684
CONSTRAINED_BIC = 11697.8469
CONSTRAINED_N_PARAMETERS = 7

# -- lrtest(constrained, unconstrained), same section -------------------------
#
#   Model 1: Constrained Model
#   Model 2: Unconstrained Model
#     #Df LogLik Df Chisq Pr(>Chisq)
#   1   7 -5826.5
#   2   8 -5821.1 1 10.943 0.0009396 ***

LRTEST = {
    "df_restricted": 7,
    "df_unrestricted": 8,
    "df": 1,
    "chisq": 10.943,
    "p_value": 0.0009396,
}

# -- Regularized Pareto/NBD with static covariates ----------------------------
#
# CLVTools_advanced_techniques.pdf S2, with *asymmetric* weights -- the paper
# and the oracle fixtures only ever use equal ones:
#
#   reg.lambdas = c(trans = 0.1, life = 0.2)

#: ``(life, trans)``, the order this package takes them in.
REGULARIZED_LAMBDAS = (0.2, 0.1)

REGULARIZED_MLE = {
    "r": 1.73887,
    "alpha": 69.85288,
    "s": 0.53350,
    "beta": 39.68346,
    "life.Gender": -0.04437,
    "life.Channel": 0.02465,
    "trans.Gender": 0.17178,
    "trans.Channel": 0.23676,
}

#: The penalised *mean* objective, not a log-likelihood. Independent
#: confirmation of the trap: the same model unpenalised is at -5821.06.
REGULARIZED_LL = -9.7313

#: What CLVTools prints, computed from `REGULARIZED_LL` rather than from a
#: log-likelihood. This package deliberately differs -- see the README's
#: findings and `TestRegularizationAgainstTheVignette`.
REGULARIZED_AIC_CLVTOOLS = 35.4626
REGULARIZED_BIC_CLVTOOLS = 70.6380

# -- ?pmf, on the CDNOW data --------------------------------------------------
#
#   pnbd.cdnow <- pnbd(clvdata(cdnow, time.unit="w", estimation.split=37,
#                              date.format="ymd"))
#   pmf(pnbd.cdnow, x=0:10)

CDNOW_ESTIMATION_WEEKS = 37
CDNOW_N_CUSTOMERS = 2357
CDNOW_N_TRANSACTIONS = 6696

#: ``mean(pmf)`` for x = 0..10, to the six decimals the man page prints.
CDNOW_PMF = [
    0.616514, 0.168309, 0.080971, 0.046190, 0.028566, 0.018506,
    0.012351, 0.008415, 0.005822, 0.004074, 0.002877,
]

#: "actual percentage of x", printed as counts over `CDNOW_N_CUSTOMERS`.
CDNOW_FREQUENCIES = [1432, 436, 208, 100, 60, 36, 27, 21, 5, 4, 7]
