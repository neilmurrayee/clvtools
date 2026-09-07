"""Time-invariant covariates, for all three latent-attrition families.

This module exists because of a gap the paired oracle made countable. Of the 76
per-customer entry points CLVTools exposes, 38 are called by no generator in
``tools/oracle/``, and the largest block of those is the static-covariate
machinery: every GGompertz/NBD covariate expression, and all of the BG/NBD's
beyond the three scale transforms. They were reachable only through
``predict()`` and ``plot()`` output, which is a fit away from the equation and
cannot be evaluated off the optimum at all.

Fifteen of them are paired here. What each pair asserts is worth naming, because
it is a claim the paper makes and this package's implementation depends on:
CLVTools computes a covariate quantity in dedicated C++, while this package
computes the *no-covariate* quantity at per-customer scale parameters. Nothing
guarantees those coincide except the algebra in S3.3 -- covariates enter only by
scaling `alpha` and `beta` (or `a` and `b`) -- and these pairs are that algebra,
checked against an independent implementation of it.

The parameter blocks are named rather than flat, and that is not cosmetic.
CLVTools' own entry points disagree about the order of the two processes:
``pnbd_staticcov_PAlive`` takes ``vCovParams_trans`` before ``vCovParams_life``
and ``mCov_trans`` before ``mCov_life``, while ``pnbd_staticcov_DERT`` takes
both the other way round, and ``bgnbd_staticcov_PAlive`` mixes the two orders
again. A flat vector would let a transposition through silently. With
``p$life`` and ``p$trans`` named at the call site, the snippet says which
process it means and the reader can check it against the signature.

Sign conventions differ by family too, and the pairs pin that: the Pareto/NBD
and the GGompertz/NBD scale with ``exp(-gamma'x)`` on both processes, while the
BG/NBD uses ``exp(-gamma'x)`` for ``alpha`` and ``exp(+gamma'x)`` for ``a`` and
``b``.
"""

from __future__ import annotations

import numpy as np
from pairs import pair

from clvtools import bgnbd, ggomnbd
from clvtools.pnbd import aggregate, staticcov

HORIZON_WEEKS = 52.0

#: The count the PMF pairs evaluate at. One value, not a sweep: what is under
#: test is the expression, and a second `k` would re-test the same code path.
PMF_K = 3


def cbs(d):
    """``(x, t_x, T)`` from the prelude's data."""
    return d["x"], d["t.x"], d["T.cal"]


def design(d):
    """The two design matrices, column-stacked in the order CLVTools names them.

    Gender then Channel, on both processes, which is S6.4's specification. The
    prelude records them one column at a time precisely so that this stacking
    is written here, in the open, rather than recovered from a flattened
    matrix by a reshape that could silently transpose.
    """
    return (
        np.column_stack([d["life.Gender"], d["life.Channel"]]),
        np.column_stack([d["trans.Gender"], d["trans.Channel"]]),
    )


def blocks(p):
    """``(model, gamma_life, gamma_trans)`` from a named input."""
    return (
        np.asarray(p["model"], float),
        np.asarray(p["life"], float),
        np.asarray(p["trans"], float),
    )


# -- Pareto/NBD ---------------------------------------------------------------

#: S6.4's fitted covariate model, and a point deliberately off it. Identical to
#: the grid in ``tools/oracle/generate_fixtures.R``.
PNBD_GRID = {
    # From `pnbd_staticcov_fit.json`, to full precision rather than the four
    # decimals the paper prints.
    "mle": {
        "model": (1.8378432988515905, 92.91235159438746, 0.5919984547642078,
                  49.62271824538944),
        "life": (-0.642993394016565, 0.7906789609698832),
        "trans": (0.2858739773063453, 0.624124096048895),
    },
    # `s = 1` exactly, which is where S4's expectation and CET divide by zero.
    # Kept deliberately: it is the branch, and the pair below declares what
    # each implementation does there rather than avoiding the question.
    "offset": {
        "model": (1.0, 50.0, 1.0, 30.0),
        "life": (0.2, -0.3),
        "trans": (-0.1, 0.4),
    },
    # And a third point off the optimum with `s != 1`, so that the quantities
    # undefined at `s = 1` are still compared numerically somewhere.
    "s.above.one": {
        "model": (0.8, 40.0, 1.4, 25.0),
        "life": (-0.35, 0.5),
        "trans": (0.15, -0.2),
    },
}

PNBD = {"family": "pnbd_staticcov", "prelude": "apparel_staticcov", "inputs": PNBD_GRID}


@pair(
    id="pnbd.staticcov.alpha_i",
    spec="M-03",
    tol=1e-13,
    r='cpp("pnbd_staticcov_alpha_i")(alpha_0 = p$model[2],'
      " vCovParams_trans = p$trans, mCov_trans = mTrans)",
    **PNBD,
)
def pnbd_alpha_i(p, d):
    """`alpha` carries the transaction process. S3.3."""
    model, _life, trans = blocks(p)
    return staticcov.alpha_i(model[1], trans, design(d)[1])


@pair(
    id="pnbd.staticcov.beta_i",
    spec="M-03",
    tol=1e-13,
    r='cpp("pnbd_staticcov_beta_i")(beta_0 = p$model[4],'
      " vCovParams_life = p$life, mCov_life = mLife)",
    **PNBD,
)
def pnbd_beta_i(p, d):
    """`beta` carries the lifetime process.

    Getting these two the wrong way round reproduces the note the generator
    beside this one carries in capitals.
    """
    model, life, _trans = blocks(p)
    return staticcov.beta_i(model[3], life, design(d)[0])


@pair(
    id="pnbd.staticcov.LL_ind",
    spec="M-01",
    tol=1e-12,
    r='cpp("pnbd_staticcov_LL_ind")(c(log(p$model), p$life, p$trans),'
      " x, tx, Tc, mLife, mTrans)",
    **PNBD,
)
def pnbd_ll_ind(p, d):
    """The covariate likelihood, per customer."""
    model, life, trans = blocks(p)
    m_life, m_trans = design(d)
    return staticcov.log_likelihood_staticcov_ind(
        *cbs(d), *model, life, trans, m_life, m_trans
    )


@pair(
    id="pnbd.staticcov.expectation",
    spec="M-03",
    tol=1e-12,
    # S4.4 divides by (s - 1) just as S4.2's CET does, so the `offset` point at
    # `s = 1` has no value: CLVTools returns NaN, this package raises. Declared
    # rather than dropped, which asserts both halves.
    undefined_at=("offset",),
    # No generator has ever called this one.
    r='cpp("pnbd_staticcov_expectation")(r = p$model[1], s = p$model[3],'
      ' vAlpha_i = cpp("pnbd_staticcov_alpha_i")(p$model[2], p$trans, mTrans),'
      ' vBeta_i = cpp("pnbd_staticcov_beta_i")(p$model[4], p$life, mLife),'
      " vT_i = rep(52, length(x)))",
    **PNBD,
)
def pnbd_expectation(p, d):
    """S4.4's expectation at per-customer scale parameters."""
    model, life, trans = blocks(p)
    m_life, m_trans = design(d)
    return aggregate.expectation(
        np.full(d["x"].shape, HORIZON_WEEKS),
        model[0],
        staticcov.alpha_i(model[1], trans, m_trans),
        model[2],
        staticcov.beta_i(model[3], life, m_life),
    )


@pair(
    id="pnbd.staticcov.PMF",
    spec="PMF-01",
    tol=1e-12,
    r='cpp("pnbd_staticcov_PMF")(r = p$model[1], s = p$model[3], x = 3,'
      ' vAlpha_i = cpp("pnbd_staticcov_alpha_i")(p$model[2], p$trans, mTrans),'
      ' vBeta_i = cpp("pnbd_staticcov_beta_i")(p$model[4], p$life, mLife),'
      " vT_i = Tc)",
    **PNBD,
)
def pnbd_pmf(p, d):
    """P(X = 3) over each customer's own observation window."""
    model, life, trans = blocks(p)
    m_life, m_trans = design(d)
    return aggregate.pmf(
        PMF_K,
        d["T.cal"],
        model[0],
        staticcov.alpha_i(model[1], trans, m_trans),
        model[2],
        staticcov.beta_i(model[3], life, m_life),
    )


# -- BG/NBD -------------------------------------------------------------------
#
# Five of the six below are called by no generator. Note the sign convention:
# `alpha_i` scales with exp(-gamma'x) as in the Pareto/NBD, but `a_i` and `b_i`
# scale with exp(+gamma'x). One family, two directions.

BGNBD_GRID = {
    # The fitted covariate model, from `bgnbd_staticcov_fit.json`. `a` and `b`
    # in the thousands is not a typo: the BG/NBD's beta parameters are barely
    # identified under covariates on this data, which the README records. An
    # oracle check at a point the optimiser actually reaches is worth more than
    # one at a comfortable invented point.
    "mle": {
        "model": (0.6662431039141171, 38.19086463420248, 4603.050621369979,
                  33985.17084170346),
        "life": (-1.390294175239601, -7.778505511184465),
        "trans": (0.4612080504904093, 0.5678289853332902),
    },
    "offset": {
        "model": (1.0, 5.0, 0.8, 2.0),
        "life": (0.25, -0.4),
        "trans": (-0.15, 0.3),
    },
}

BGNBD = {"family": "bgnbd_staticcov", "prelude": "apparel_staticcov", "inputs": BGNBD_GRID}


def bgnbd_scaled(p, d):
    """``(r, alpha_i, a_i, b_i)`` -- the model at per-customer scale."""
    model, life, trans = blocks(p)
    m_life, m_trans = design(d)
    return (
        model[0],
        bgnbd.alpha_i(model[1], trans, m_trans),
        bgnbd.a_i(model[2], life, m_life),
        bgnbd.b_i(model[3], life, m_life),
    )


@pair(
    id="bgnbd.staticcov.LL_ind",
    spec="M-05",
    # 1e-9, and the reason is the `mle` point below rather than the expression.
    # At `a = 4.6e3`, `b = 3.4e4` the likelihood's log-beta ratio differences
    # two nearly equal quantities, and how much of that cancellation survives
    # depends on the platform's `lgamma`: this pair agrees to better than 1e-11
    # on macOS and to 2.6e-11 on a Linux runner, against the same R. A bound
    # tight enough to be a real check on the expression and loose enough not to
    # be a check on libm.
    tol=1e-9,
    r='cpp("bgnbd_staticcov_LL_ind")(vParams = c(log(p$model), p$life, p$trans),'
      " vX = x, vT_x = tx, vT_cal = Tc, mCov_life = mLife, mCov_trans = mTrans)",
    **BGNBD,
)
def bgnbd_ll_ind(p, d):
    """The covariate likelihood, per customer.

    CLVTools computes this in dedicated C++; here it is the *no-covariate*
    likelihood evaluated at per-customer `(alpha_i, a_i, b_i)`. That the two
    agree is S3.3's claim rather than an implementation detail, and this is
    where it is checked.
    """
    return bgnbd.log_likelihood_ind(*cbs(d), *bgnbd_scaled(p, d))


@pair(
    id="bgnbd.staticcov.LL_sum",
    spec="M-05",
    tol=1e-10,
    # Negated at the call: the C++ returns a minimiser's objective. This one
    # goes through the covariate likelihood itself rather than the composition
    # above, so the two pairs together check the sum and its parts.
    r='-cpp("bgnbd_staticcov_LL_sum")(c(log(p$model), p$life, p$trans),'
      " x, tx, Tc, vN, mLife, mTrans)",
    **BGNBD,
)
def bgnbd_ll_sum(p, d):
    """The sample log-likelihood the covariate fit maximises."""
    model, life, trans = blocks(p)
    m_life, m_trans = design(d)
    return bgnbd.log_likelihood_staticcov(
        *cbs(d), *model, life, trans, m_life, m_trans
    )


@pair(
    id="bgnbd.staticcov.PAlive",
    spec="M-05",
    tol=1e-11,
    # `mCov_trans` before `mCov_life` here, the opposite of the Pareto/NBD's
    # DERT. Named arguments on both sides are what make that safe to read.
    r='cpp("bgnbd_staticcov_PAlive")(r = p$model[1], alpha = p$model[2],'
      " a = p$model[3], b = p$model[4], vX = x, vT_x = tx, vT_cal = Tc,"
      " vCovParams_trans = p$trans, vCovParams_life = p$life,"
      " mCov_trans = mTrans, mCov_life = mLife)",
    **BGNBD,
)
def bgnbd_palive(p, d):
    """P(alive) at per-customer scale parameters."""
    return bgnbd.probability_alive(*cbs(d), *bgnbd_scaled(p, d))


@pair(
    id="bgnbd.staticcov.CET",
    spec="M-05",
    tol=1e-10,
    r='cpp("bgnbd_staticcov_CET")(r = p$model[1], alpha = p$model[2],'
      " a = p$model[3], b = p$model[4], dPeriods = 52,"
      " vX = x, vT_x = tx, vT_cal = Tc,"
      " vCovParams_trans = p$trans, vCovParams_life = p$life,"
      " mCov_trans = mTrans, mCov_life = mLife)",
    **BGNBD,
)
def bgnbd_cet(p, d):
    """Expected transactions over the next 52 weeks."""
    return bgnbd.conditional_expected_transactions(
        *cbs(d), HORIZON_WEEKS, *bgnbd_scaled(p, d)
    )


@pair(
    id="bgnbd.staticcov.expectation",
    spec="M-05",
    tol=1e-11,
    r='cpp("bgnbd_staticcov_expectation")(r = p$model[1],'
      ' vAlpha_i = cpp("bgnbd_staticcov_alpha_i")(p$model[2], p$trans, mTrans),'
      ' vA_i = cpp("bgnbd_staticcov_a_i")(p$model[3], p$life, mLife),'
      ' vB_i = cpp("bgnbd_staticcov_b_i")(p$model[4], p$life, mLife),'
      " vT_i = rep(52, length(x)))",
    **BGNBD,
)
def bgnbd_expectation(p, d):
    """The unconditional expectation at 52 weeks."""
    r, alpha, a, b = bgnbd_scaled(p, d)
    return bgnbd.expectation(np.full(d["x"].shape, HORIZON_WEEKS), r, alpha, a, b)


@pair(
    id="bgnbd.staticcov.PMF",
    spec="PMF-01",
    tol=1e-11,
    r='cpp("bgnbd_staticcov_PMF")(r = p$model[1], x = 3,'
      ' vAlpha_i = cpp("bgnbd_staticcov_alpha_i")(p$model[2], p$trans, mTrans),'
      ' vA_i = cpp("bgnbd_staticcov_a_i")(p$model[3], p$life, mLife),'
      ' vB_i = cpp("bgnbd_staticcov_b_i")(p$model[4], p$life, mLife),'
      " vT_i = Tc)",
    **BGNBD,
)
def bgnbd_pmf(p, d):
    """P(X = 3) over each customer's own observation window."""
    r, alpha, a, b = bgnbd_scaled(p, d)
    return bgnbd.pmf(PMF_K, d["T.cal"], r, alpha, a, b)


# -- GGompertz/NBD ------------------------------------------------------------
#
# Every entry point below is called by no generator: the covariate arm of this
# family has had no equation-level check at all. Its parameter order is
# (r, alpha, b, s, beta) -- five, not four, and `b` sits in the middle.

GGOMNBD_GRID = {
    # From `ggomnbd_staticcov_fit.json`. `b` at 3.1e-06 and `beta` at 1.5e-04
    # are the real optimum: this family's `b` runs to the edge of the space on
    # apparel, which is exactly where a Hessian step of the wrong size sends it
    # negative -- the trap CLAUDE.md records for `inference.numerical_hessian`.
    "mle": {
        "model": (1.8368056462924693, 92.85442778955964, 3.0995944247600156e-06,
                  0.59277971715678, 0.0001540825058631308),
        "life": (-0.6423858371646177, 0.7892098608731487),
        "trans": (0.2859269939549405, 0.6238799164731486),
    },
    "offset": {
        "model": (0.8, 8.0, 0.01, 0.5, 2.0),
        "life": (0.2, -0.25),
        "trans": (-0.1, 0.35),
    },
}

GGOMNBD = {
    "family": "ggomnbd_staticcov",
    "prelude": "apparel_staticcov",
    "inputs": GGOMNBD_GRID,
}


def ggomnbd_scaled(p, d):
    """``(r, alpha_i, b, s, beta_i)`` -- the model at per-customer scale."""
    model, life, trans = blocks(p)
    m_life, m_trans = design(d)
    return (
        model[0],
        ggomnbd.alpha_i(model[1], trans, m_trans),
        model[2],
        model[3],
        ggomnbd.beta_i(model[4], life, m_life),
    )


@pair(
    id="ggomnbd.staticcov.alpha_i",
    spec="M-07",
    tol=1e-13,
    r='cpp("ggomnbd_staticcov_alpha_i")(alpha_0 = p$model[2],'
      " vCovParams_trans = p$trans, mCov_trans = mTrans)",
    **GGOMNBD,
)
def ggomnbd_alpha_i(p, d):
    """`alpha` carries the transaction process here too."""
    model, _life, trans = blocks(p)
    return ggomnbd.alpha_i(model[1], trans, design(d)[1])


@pair(
    id="ggomnbd.staticcov.beta_i",
    spec="M-07",
    tol=1e-13,
    r='cpp("ggomnbd_staticcov_beta_i")(beta_0 = p$model[5],'
      " vCovParams_life = p$life, mCov_life = mLife)",
    **GGOMNBD,
)
def ggomnbd_beta_i(p, d):
    """`beta` is the fifth parameter, not the fourth."""
    model, life, _trans = blocks(p)
    return ggomnbd.beta_i(model[4], life, design(d)[0])


@pair(
    id="ggomnbd.staticcov.LL_ind",
    spec="M-07",
    tol=1e-10,
    r='cpp("ggomnbd_staticcov_LL_ind")(vParams = c(log(p$model), p$life, p$trans),'
      " vX = x, vT_x = tx, vT_cal = Tc, mCov_life = mLife, mCov_trans = mTrans)",
    **GGOMNBD,
)
def ggomnbd_ll_ind(p, d):
    """The covariate likelihood, per customer, by the same composition."""
    return ggomnbd.log_likelihood_ind(*cbs(d), *ggomnbd_scaled(p, d))


@pair(
    id="ggomnbd.staticcov.LL_sum",
    spec="M-07",
    tol=1e-9,
    r='-cpp("ggomnbd_staticcov_LL_sum")(c(log(p$model), p$life, p$trans),'
      " x, tx, Tc, vN, mLife, mTrans)",
    **GGOMNBD,
)
def ggomnbd_ll_sum(p, d):
    """The sample log-likelihood the covariate fit maximises."""
    model, life, trans = blocks(p)
    m_life, m_trans = design(d)
    return ggomnbd.log_likelihood_staticcov(
        *cbs(d), *model, life, trans, m_life, m_trans
    )


@pair(
    id="ggomnbd.staticcov.PAlive",
    spec="M-07",
    tol=1e-10,
    r='cpp("ggomnbd_staticcov_PAlive")(r = p$model[1], alpha_0 = p$model[2],'
      " b = p$model[3], s = p$model[4], beta_0 = p$model[5],"
      " vX = x, vT_x = tx, vT_cal = Tc,"
      " vCovParams_trans = p$trans, vCovParams_life = p$life,"
      " mCov_life = mLife, mCov_trans = mTrans)",
    **GGOMNBD,
)
def ggomnbd_palive(p, d):
    """P(alive) at per-customer scale parameters."""
    return ggomnbd.probability_alive(*cbs(d), *ggomnbd_scaled(p, d))


@pair(
    id="ggomnbd.staticcov.CET",
    spec="M-09",
    # 1e-6, and the loosest bound in the paired oracle. Both sides integrate
    # numerically and the integrand's conditioning degrades as `b` goes to
    # zero, which is exactly where this family's optimum on apparel sits
    # (b = 3.1e-06). Measured against CLVTools by sweeping `b` with everything
    # else held at the fitted values:
    #
    #     b        max rel err      b        max rel err
    #     3.1e-06    1.47e-07       1e-03      2.02e-11
    #     1e-05      2.54e-07       1e-02      2.61e-11
    #     1e-04      4.66e-10       1e-01      1.30e-12
    #
    # Monotone in `b` and twelve digits clean away from the degenerate end,
    # which is what a quadrature difference looks like and not what a wrong
    # expression looks like. The bound is set above the worst of those rather
    # than at a round number, and the sweep is in the README's findings so the
    # next person to see 1e-6 here can tell it was measured.
    tol=1e-6,
    r='cpp("ggomnbd_staticcov_CET")(r = p$model[1], alpha_0 = p$model[2],'
      " b = p$model[3], s = p$model[4], beta_0 = p$model[5], dPeriods = 52,"
      " vX = x, vT_x = tx, vT_cal = Tc,"
      " vCovParams_trans = p$trans, vCovParams_life = p$life,"
      " mCov_life = mLife, mCov_trans = mTrans)",
    **GGOMNBD,
)
def ggomnbd_cet(p, d):
    """Expected transactions over the next 52 weeks.

    The loosest tolerance here at 1e-9: this CET integrates numerically on both
    sides, and the two quadratures are not the same one.
    """
    return ggomnbd.conditional_expected_transactions(
        *cbs(d), HORIZON_WEEKS, *ggomnbd_scaled(p, d)
    )


@pair(
    id="ggomnbd.staticcov.expectation",
    spec="M-07",
    tol=1e-9,
    r='cpp("ggomnbd_staticcov_expectation")(r = p$model[1], b = p$model[3],'
      " s = p$model[4],"
      ' vAlpha_i = cpp("ggomnbd_staticcov_alpha_i")(p$model[2], p$trans, mTrans),'
      ' vBeta_i = cpp("ggomnbd_staticcov_beta_i")(p$model[5], p$life, mLife),'
      " vT_i = rep(52, length(x)))",
    **GGOMNBD,
)
def ggomnbd_expectation(p, d):
    """The unconditional expectation at 52 weeks."""
    r, alpha, b, s, beta = ggomnbd_scaled(p, d)
    return ggomnbd.expectation(np.full(d["x"].shape, HORIZON_WEEKS), r, alpha, b, s, beta)


@pair(
    id="ggomnbd.staticcov.PMF",
    spec="PMF-01",
    tol=1e-9,
    r='cpp("ggomnbd_staticcov_PMF")(r = p$model[1], alpha_0 = p$model[2],'
      " b = p$model[3], s = p$model[4], beta_0 = p$model[5], x = 3,"
      " vCovParams_trans = p$trans, vCovParams_life = p$life,"
      " mCov_life = mLife, mCov_trans = mTrans, vT_i = Tc)",
    **GGOMNBD,
)
def ggomnbd_pmf(p, d):
    """P(X = 3) over each customer's own observation window."""
    r, alpha, b, s, beta = ggomnbd_scaled(p, d)
    return ggomnbd.pmf(PMF_K, d["T.cal"], r, alpha, b, s, beta)
