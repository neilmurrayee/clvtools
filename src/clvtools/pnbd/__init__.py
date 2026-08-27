"""The Pareto/NBD model of Schmittlein et al., as presented in S3.2.

Modules mirror the paper:

``individual``
    S3.2 - the attrition and transaction processes, and the individual-level
    likelihood of eq. (10), conditional on one customer's (lambda, mu).
``aggregate``
    Appendix A and S6.3 - the same quantities for a randomly chosen customer,
    obtained by marginalising over the two gamma distributions, plus the
    managerial expressions PAlive, CET and DERT.
``fit``
    S3.2 - maximum likelihood estimation of (r, alpha, s, beta).
``staticcov``
    S3.3 and S6.4.1 - the extension for time-invariant covariates.
"""

from __future__ import annotations

from clvtools.pnbd.aggregate import (
    conditional_expected_transactions,
    discounted_expected_residual_transactions,
    expectation,
    likelihood_appendix,
    log_likelihood,
    log_likelihood_ind,
    pmf,
    probability_alive,
)
from clvtools.pnbd.fit import PnbdParams, fit_pnbd
from clvtools.pnbd.staticcov import (
    PnbdStaticCovParams,
    alpha_i,
    beta_i,
    fit_pnbd_staticcov,
)
from clvtools.pnbd.individual import (
    gamma_pdf_lambda,
    gamma_pdf_mu,
    individual_likelihood,
    lifetime_pdf,
    lifetime_pdf_mixed,
    likelihood_alive_at_T,
    likelihood_died_at,
    log_individual_likelihood,
    nbd_pmf,
    poisson_pmf,
)

__all__ = [
    "PnbdParams",
    "PnbdStaticCovParams",
    "alpha_i",
    "beta_i",
    "conditional_expected_transactions",
    "discounted_expected_residual_transactions",
    "expectation",
    "fit_pnbd",
    "fit_pnbd_staticcov",
    "gamma_pdf_lambda",
    "gamma_pdf_mu",
    "individual_likelihood",
    "lifetime_pdf",
    "lifetime_pdf_mixed",
    "likelihood_alive_at_T",
    "likelihood_appendix",
    "likelihood_died_at",
    "log_individual_likelihood",
    "log_likelihood",
    "log_likelihood_ind",
    "nbd_pmf",
    "pmf",
    "poisson_pmf",
    "probability_alive",
]
