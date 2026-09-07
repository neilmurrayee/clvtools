"""The Pareto/NBD with time-varying covariates, S3.3 and S6.4.2.

The last block of CLVTools' per-customer surface to go onto the paired oracle,
and the one that needed a prelude rather than a parameter vector. Every entry
point here reaches its data through a *fitted object*: `pnbd_dyncov_palive`,
`_CET` and `_DECT` take one and read the coefficients off it, so on the face of
it they can only be checked at whatever optimum a fit happens to reach. That is
the "one point, and not a chosen one" problem the oracle fixtures exist to
escape, and it matters most here, where the fit is ten minutes and is
deselected from every ordinary run.

Two observations get around it, both in ``tools/oracle/run_pairs.R``:

* The walk structures do not depend on the parameters. Fitting with
  ``itnmax = 1`` gives a well-formed object whose walks are the real ones, in
  under six seconds.
* The parameters can then be replaced. ``at(p)`` returns a copy of that object
  carrying a declared vector, so the prediction expressions are evaluated where
  this module chooses rather than where an optimiser stopped.

**What these pairs do and do not check.** The walk arrays are 130,000 rows and
stay where they are, as committed CSV fixtures pinned by
``tests/test_pnbd_dyncov_walks.py``; the Python side reads them from there, as
``conftest`` already does. So a pair here checks the *likelihood and prediction
expressions built on the walks*, not the walk construction. The prelude records
a fingerprint of each array -- row count and column sums -- so that a
re-baselined walk fixture nobody re-recorded fails loudly rather than quietly
changing what these expressions are evaluated on. A fingerprint is not the
array and is not claimed to be.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from pairs import pair

from clvtools.pnbd import dyncov, dyncov_predict

NAMES = ["High.Season", "Gender", "Channel"]

#: The prediction settings the existing dyncov fixtures were generated with,
#: so that these pairs and `tests/test_pnbd_dyncov_predict.py` are talking
#: about the same horizon. `PERIOD_LENGTH` and `PREDICTION_END` must agree:
#: CLVTools uses the date to walk the covariate series and the period count in
#: the leading `(s - 1)` power, and a pair that gave it 52 weeks of periods
#: over four years of covariates would be comparing two different quantities.
PREDICTION_END = "2010-12-20"
PERIOD_LENGTH = 207.14285714285714

#: An *unscaled annual* rate, which is CLVTools' convention and one of the
#: traps CLAUDE.md records -- `clvtools.predict.discount_factor` is what
#: converts it per period. log(1.1), as the fixtures use.
CONTINUOUS_DISCOUNT_FACTOR = 0.09531017980432494

#: CLVTools' own fitted coefficients from `dyncov_fit.json`, and two points off
#: them. The optimum is included because it is where the model is used; the
#: other two because a pair that agreed only at an optimum would say very
#: little, and because `s` and `beta` move the F2 term through quite different
#: regimes.
GRID = {
    "mle": {
        "model": (1.9777409894092832, 115.1779145506357,
                  2.0127328567023923, 158.1817947718384),
        "life": (-2.482672230676113, -0.5124721777818542, 0.5057316203480627),
        "trans": (0.718271261172766, 0.2649019376433103, 0.6137395102298454),
    },
    # `s = 2.4` rather than the 1.5 this point was first written with, and the
    # reason is a performance cliff rather than an accuracy one. `DECT` forms
    # `U(s, s, z)`, and `scipy.special.hyperu` takes a slow path for
    # `1 < s < 2`: ~90x at s = 1.5, which turned one call over 600 customers
    # into 449 seconds. Correctness there is fine -- measured once against
    # CLVTools at 2.4e-08 -- but a seven-minute case does not belong in a
    # suite that runs on every push. `tests/test_performance.py` pins the
    # cliff itself, cheaply, and the README records it.
    "offset": {
        "model": (1.0, 50.0, 2.4, 60.0),
        "life": (0.2, -0.3, 0.4),
        "trans": (0.1, 0.25, -0.15),
    },
    #: Small `s`, where the lifetime distribution has a heavy tail and the
    #: hypergeometric arm of the likelihood is slowest to converge.
    "small.s": {
        "model": (0.8, 30.0, 0.35, 20.0),
        "life": (-0.5, 0.15, 0.3),
        "trans": (0.4, -0.2, 0.1),
    },
}

COMMON = {"family": "pnbd_dyncov", "prelude": "apparel_dyncov", "inputs": GRID}


def blocks(p):
    """``(r, alpha, s, beta), gamma_life, gamma_trans`` from a named input."""
    model = np.asarray(p["model"], dtype=float)
    return (
        model,
        np.asarray(p["life"], dtype=float),
        np.asarray(p["trans"], dtype=float),
    )


def params(p):
    """The input as the dataclass the prediction functions take."""
    from clvtools.pnbd.dyncov import PnbdDynCovParams

    model, life, trans = blocks(p)
    return PnbdDynCovParams(
        r=model[0], alpha=model[1], s=model[2], beta=model[3],
        gamma_life=life, gamma_trans=trans,
        names_cov_life=NAMES, names_cov_trans=NAMES,
        log_likelihood=float("nan"), converged=True, n_customers=600,
    )


@lru_cache(maxsize=1)
def walks():
    """The oracle's walk structures, from the committed fixtures.

    Built once. Cached rather than read at import so that this module stays
    importable -- and recordable -- without touching seven CSV files, and so
    that a replay run which needs none of them pays nothing.
    """
    from conftest import fixture_csv

    cbs = fixture_csv("dyncov_cbs")
    info = fixture_csv("dyncov_walkinfo")
    real_trans = fixture_csv("dyncov_walkinfo_real_trans")
    covdata = {
        kind: fixture_csv(f"dyncov_covdata_{kind}").to_numpy(dtype=float)
        for kind in ("aux_life", "real_life", "aux_trans", "real_trans")
    }
    return dyncov.DyncovWalks(
        ids=pd.Index(cbs["Id"], name="Id"),
        x=cbs["x"].to_numpy(dtype=float),
        t_x=cbs["t.x"].to_numpy(dtype=float),
        T_cal=cbs["T.cal"].to_numpy(dtype=float),
        d_omega=cbs["d_omega"].to_numpy(dtype=float),
        walkinfo_aux_life=info[["aux_life_from", "aux_life_to"]].to_numpy(float),
        walkinfo_real_life=info[["real_life_from", "real_life_to"]].to_numpy(float),
        walkinfo_aux_trans=info[
            ["aux_trans_from", "aux_trans_to", "aux_trans_d1", "aux_trans_tjk"]
        ].to_numpy(float),
        walkinfo_real_trans=real_trans[
            ["walk_from", "walk_to", "d1", "tjk"]
        ].to_numpy(float),
        real_trans_from=info["real_trans_from"].to_numpy(dtype=float),
        real_trans_to=info["real_trans_to"].to_numpy(dtype=float),
        covdata_aux_life=covdata["aux_life"],
        covdata_real_life=covdata["real_life"],
        covdata_aux_trans=covdata["aux_trans"],
        covdata_real_trans=covdata["real_trans"],
    )


@lru_cache(maxsize=1)
def holdout_data():
    """The S6.4.2 data object the prediction expressions take.

    Rebuilt from the packaged datasets rather than from the prelude, because
    ``CET`` and ``DECT`` walk the covariate *series* and not the flat arrays
    the likelihood sees. The two constructions agreeing is what the walk
    fixtures pin; what these pairs check is the expression on top.
    """
    from clvtools import ClvData, ClvDataDynCov, load_apparel_dyn_cov, load_apparel_trans

    return ClvDataDynCov(
        ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
        load_apparel_dyn_cov(),
        names_cov_life=NAMES,
        names_cov_trans=NAMES,
    )


@lru_cache(maxsize=1)
def prediction_end():
    """The horizon as a timestamp."""
    return pd.Timestamp(PREDICTION_END)


# -- The likelihood -----------------------------------------------------------


@pair(
    id="pnbd.dyncov.LL_ind",
    spec="DY-08",
    tol=1e-10,
    r='do.call(cpp("pnbd_dyncov_LL_ind"), llargs(p))[, 1]',
    **COMMON,
)
def ll_ind(p, d):
    """S3.3's time-varying likelihood, per customer.

    ``[, 1]`` on the R side is the total; the other thirty columns are the
    intermediates, which ``tests/test_pnbd_dyncov.py`` already holds to
    fixtures block by block. What this pair adds is that the *total* is checked
    at parameters chosen here, against a live oracle, without a ten-minute fit.
    """
    model, life, trans = blocks(p)
    return dyncov.log_likelihood_ind(walks(), *model, life, trans)


@pair(
    id="pnbd.dyncov.LL_sum",
    spec="DY-09",
    tol=1e-9,
    r='-do.call(cpp("pnbd_dyncov_LL_negsum"), llargs(p, list(vN = vN)))',
    **COMMON,
)
def ll_sum(p, d):
    """The sample log-likelihood, negated out of the minimiser's convention."""
    model, life, trans = blocks(p)
    return dyncov.log_likelihood(walks(), *model, life, trans)


# -- Prediction, S6.4.2 -------------------------------------------------------


@pair(
    id="pnbd.dyncov.PAlive",
    spec="DY-13",
    tol=1e-10,
    r='cpp("pnbd_dyncov_palive")(at(p))$palive',
    **COMMON,
)
def palive(p, d):
    """P(alive), one exponential away from the likelihood's own two halves."""
    model, life, trans = blocks(p)
    return dyncov.probability_alive(walks(), *model, life, trans)


@pair(
    id="pnbd.dyncov.CET",
    spec="DY-12",
    tol=1e-9,
    r='cpp("pnbd_dyncov_CET")(clv.fitted = at(p),'
      " predict.number.of.periods = 207.14285714285714,"
      ' prediction.end.date = as.Date("2010-12-20"),'
      " only.return.input.to.CET = FALSE)$CET",
    **COMMON,
)
def cet(p, d):
    """Expected transactions over the covariate path of the holdout period."""
    return dyncov_predict.conditional_expected_transactions(
        holdout_data(), params(p), prediction_end(), PERIOD_LENGTH
    ).to_numpy(dtype=float)


@pair(
    id="pnbd.dyncov.DECT",
    spec="DY-11",
    # The loosest bound in the dyncov family, and the only one above 1e-13.
    # `DECT` sums `U(s, s, .)` period by period over the horizon, and the two
    # implementations reach that function differently -- GSL's `hyperg_U`
    # against SciPy's `hyperu`. Measured at 1.4e-10 and 1.1e-10 at the two
    # points below; 1e-8 leaves room for a platform that rounds the other way.
    tol=1e-8,
    r='cpp("pnbd_dyncov_DECT")(clv.fitted = at(p),'
      " predict.number.of.periods = 207.14285714285714,"
      ' prediction.end.date = as.Date("2010-12-20"),'
      " continuous.discount.factor = 0.09531017980432494)$DECT",
    **COMMON,
)
def dect(p, d):
    """``CET`` with each period discounted back to the estimation end."""
    return dyncov_predict.discounted_expected_transactions(
        holdout_data(), params(p), prediction_end(),
        PERIOD_LENGTH, CONTINUOUS_DISCOUNT_FACTOR,
    ).to_numpy(dtype=float)
