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
``predict``
    S6.3 - combining the two into predicted transactions, spending and CLV.
"""

from __future__ import annotations

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
    "ClvDataDynCov",
    "ClvDataStaticCov",
    "discount_factor",
    "load_apparel_dyn_cov",
    "load_apparel_static_cov",
    "load_apparel_trans",
    "load_cdnow",
    "predict",
]
