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
    S3.3 and S6.4.1 - time-invariant covariates, plus the equality
    constraints and L2 regularization of S6.5.
``correlation``
    S3.4 and S6.5.2 - Sarmanov-correlated transaction and attrition processes.
``dyncov_walks``
    S3.3 - the *walks* a customer's covariate path is cut into, which is the
    bookkeeping every time-varying expression is written in terms of.
``dyncov``
    S3.3 and S6.4.2 - the likelihood over those walks, and PAlive from it.
``dyncov_predict``
    S6.4.2 - CET and DECT, which need the covariate path ahead of the
    estimation period as well as the one already observed.
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
from clvtools.pnbd.correlation import (
    PnbdCorrelatedParams,
    correlated_log_likelihood,
    correlation_coefficient,
    fit_pnbd_correlated,
)
from clvtools.pnbd.fit import PnbdParams, fit_pnbd
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
from clvtools.pnbd.staticcov import (
    PnbdStaticCovParams,
    alpha_i,
    beta_i,
    fit_pnbd_staticcov,
)

__all__ = [
    "PnbdCorrelatedParams",
    "PnbdParams",
    "PnbdStaticCovParams",
    "alpha_i",
    "beta_i",
    "conditional_expected_transactions",
    "correlated_log_likelihood",
    "correlation_coefficient",
    "discounted_expected_residual_transactions",
    "expectation",
    "fit_pnbd",
    "fit_pnbd_correlated",
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
