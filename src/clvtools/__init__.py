"""A test-driven Python implementation of the CLVTools R package.

Follows Meierer, Bachmann, Naef, Schilter & Algesheimer, "Estimating Individual
Customer Lifetime Values with R: The CLVTools Package" (Journal of Statistical
Software, submission 5634), section by section.

Modules mirror the paper's structure:

``data``
    S6.1 - preparing and inspecting transaction data; the `clvdata()` analogue.
``special``
    The hypergeometric functions the Pareto/NBD expressions are built from.
``pnbd``
    S3.2 and Appendix A - the Pareto/NBD latent attrition model.
``gg``
    S3.5 - the Gamma-Gamma model of customer spending.
``bgnbd``
    The BG/NBD, one of Table 3's alternative latent attrition models.
``ggomnbd``
    The GGom/NBD, the other. Both take time-invariant covariates,
    equality constraints and regularization, per Table 3.
``timeunit``
    S5 - the unit of time, including the calendar arithmetic months and
    years need.
``predict``
    S6.3 - combining the two into predicted transactions, spending and CLV.
``diagnostics``
    S6.2.2 and S6.2.4 - the tracking, PMF and spending plots, as data.
``bootstrap``
    S6.3.3 - confidence intervals by resampling customers.
"""

from __future__ import annotations

from clvtools import bgnbd, bootstrap, diagnostics, gg, ggomnbd, pnbd
from clvtools.predict import discount_factor, predict
from clvtools.data import (
    ClvData,
    ClvDataDynCov,
    ClvDataStaticCov,
    load_apparel_dyn_cov,
    load_apparel_static_cov,
    load_apparel_trans,
    load_cdnow,
)

__all__ = [
    "ClvData",
    "bgnbd",
    "bootstrap",
    "diagnostics",
    "gg",
    "ggomnbd",
    "pnbd",
    "ClvDataDynCov",
    "ClvDataStaticCov",
    "discount_factor",
    "load_apparel_dyn_cov",
    "load_apparel_static_cov",
    "load_apparel_trans",
    "load_cdnow",
    "predict",
]
