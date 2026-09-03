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
    The BG/NBD, one of Table 4's alternative latent attrition models.
``ggomnbd``
    The GGom/NBD, the other. Both take time-invariant covariates,
    equality constraints and regularization, per Table 4.
``timeunit``
    S5 - the unit of time, including the calendar arithmetic months and
    years need.
``estimate``
    Table 2's two entry points - ``latent_attrition()`` and ``spending()``.
``predict``
    S6.3 - combining the two into predicted transactions, spending and CLV,
    including for prospective customers (S6.3.4).
``diagnostics``
    S6.1.2, S6.2.2 and S6.2.4 - the descriptive and model plots, as data.
``bootstrap``
    S6.3.3 - confidence intervals by resampling customers.
``inference``
    Table 2's model details - ``vcov()``, ``confint()``, ``summary()`` and the
    likelihood ratio test of S6.5.3.

Three modules answer to no section, because they are machinery the families
share rather than anything the paper names: ``_optimize`` (the optimiser setup
every fit goes through), ``_staticcov`` (the design matrices, constraints and
penalties common to all three covariate models) and ``_validate`` (the input
checks). They are private and may move.
"""

from __future__ import annotations

from clvtools import bgnbd, bootstrap, diagnostics, gg, ggomnbd, inference, pnbd
from clvtools.data import (
    ClvData,
    ClvDataDynCov,
    ClvDataStaticCov,
    load_apparel_dyn_cov,
    load_apparel_dyn_cov_future,
    load_apparel_static_cov,
    load_apparel_trans,
    load_cdnow,
)
from clvtools.estimate import latent_attrition, spending
from clvtools.inference import likelihood_ratio_test
from clvtools.predict import (
    discount_factor,
    newcustomer,
    newcustomer_dynamic,
    newcustomer_spending,
    newcustomer_static,
    predict,
)

__all__ = [
    "ClvData",
    "ClvDataDynCov",
    "ClvDataStaticCov",
    "bgnbd",
    "bootstrap",
    "diagnostics",
    "discount_factor",
    "gg",
    "ggomnbd",
    "inference",
    "latent_attrition",
    "likelihood_ratio_test",
    "load_apparel_dyn_cov",
    "load_apparel_dyn_cov_future",
    "load_apparel_static_cov",
    "load_apparel_trans",
    "load_cdnow",
    "newcustomer",
    "newcustomer_dynamic",
    "newcustomer_spending",
    "newcustomer_static",
    "pnbd",
    "predict",
    "spending",
]
