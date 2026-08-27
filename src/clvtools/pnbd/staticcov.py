r"""S3.3 and S6.4.1 - the Pareto/NBD with time-invariant covariates.

S3.3: "heterogeneity has been solely modeled by Gamma distributions. However,
observable information, such as customer demographics, is often available. These
covariates may help to explain part of the heterogeneity among customers and,
therefore, increase the predictive accuracy of the model."

The extension scales each customer's two latent rates by an exponential in their
covariates,

.. math::
    \lambda_i = \lambda_0 \exp(\boldsymbol{\gamma}_{purch}'\mathbf{x}^P_i),
    \qquad
    \mu_i = \mu_0 \exp(\boldsymbol{\gamma}_{attr}'\mathbf{x}^A_i),

with :math:`\lambda_0 \sim \Gamma(r, \alpha)` and :math:`\mu_0 \sim
\Gamma(s, \beta)` as before. Scaling a gamma variate by a constant is the same
as dividing its rate by that constant, so nothing in Appendix A has to change:
every expression is reused verbatim with per-customer rates

.. math::
    \alpha_i = \alpha \exp(-\boldsymbol{\gamma}_{purch}'\mathbf{x}^P_i),
    \qquad
    \beta_i = \beta \exp(-\boldsymbol{\gamma}_{attr}'\mathbf{x}^A_i).

That is why :mod:`clvtools.pnbd.aggregate` broadcasts ``alpha`` and ``beta``
rather than taking them as scalars.

S3.3 also notes the nesting that makes this testable: "The standard model and
the extension for time-invariant covariates are nested within this model. With
covariate effects set to zero, we arrive at the standard model."
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from clvtools._optimize import options_for
from clvtools.pnbd.aggregate import log_likelihood_ind

__all__ = [
    "PnbdStaticCovParams",
    "alpha_i",
    "beta_i",
    "fit_pnbd_staticcov",
    "log_likelihood",
    "log_likelihood_staticcov_ind",
]

#: S6.4: "If not given, the start values are set to 0.1 for all covariates."
_DEFAULT_COV_START = 0.1


def alpha_i(
    alpha: float, gamma_trans: ArrayLike, covariates: ArrayLike
) -> NDArray[np.float64]:
    r"""Per-customer transaction rate parameter.

    .. math::
        \alpha_i = \alpha \exp(-\boldsymbol{\gamma}_{purch}'\mathbf{x}^P_i)

    The sign is negative because :math:`\alpha` is a *rate*: a covariate that
    raises the purchase rate :math:`\lambda_i` lowers :math:`\alpha_i`.

    Examples
    --------
    At the estimates of S6.4.1, a female customer acquired offline
    (``Gender = 1, Channel = 1``) has a smaller ``alpha`` -- so a higher
    purchase rate -- than the ``0, 0`` baseline:

    >>> import numpy as np
    >>> covariates = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> [round(float(v), 4) for v in alpha_i(92.9123, [0.2859, 0.6241], covariates)]
    [92.9123, 37.3995]
    """
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    gamma_trans = np.asarray(gamma_trans, dtype=float)
    if covariates.shape[1] != gamma_trans.size:
        raise ValueError(
            f"{covariates.shape[1]} transaction covariates but "
            f"{gamma_trans.size} parameters"
        )
    return alpha * np.exp(-(covariates @ gamma_trans))


def beta_i(
    beta: float, gamma_life: ArrayLike, covariates: ArrayLike
) -> NDArray[np.float64]:
    r"""Per-customer attrition rate parameter.

    .. math::
        \beta_i = \beta \exp(-\boldsymbol{\gamma}_{attr}'\mathbf{x}^A_i)

    >>> import numpy as np
    >>> covariates = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> [round(float(v), 4) for v in beta_i(49.6227, [-0.6430, 0.7907], covariates)]
    [49.6227, 42.809]
    """
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    gamma_life = np.asarray(gamma_life, dtype=float)
    if covariates.shape[1] != gamma_life.size:
        raise ValueError(
            f"{covariates.shape[1]} attrition covariates but "
            f"{gamma_life.size} parameters"
        )
    return beta * np.exp(-(covariates @ gamma_life))


def log_likelihood_staticcov_ind(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    r: float,
    alpha: float,
    s: float,
    beta: float,
    gamma_life: ArrayLike,
    gamma_trans: ArrayLike,
    cov_life: ArrayLike,
    cov_trans: ArrayLike,
) -> NDArray[np.float64]:
    r"""Per-customer log-likelihood with covariates.

    Appendix A's expression, evaluated at :math:`(\alpha_i, \beta_i)` instead of
    :math:`(\alpha, \beta)`.

    Examples
    --------
    Setting every covariate parameter to zero recovers the model without
    covariates, as S3.3 says it must:

    >>> import numpy as np
    >>> from clvtools.pnbd import log_likelihood_ind
    >>> cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> with_zeros = log_likelihood_staticcov_ind(
    ...     [6, 2], [93.285714, 99.571429], [104.0, 104.0],
    ...     1.4490, 48.6361, 0.5613, 46.8844,
    ...     gamma_life=[0.0, 0.0], gamma_trans=[0.0, 0.0],
    ...     cov_life=cov, cov_trans=cov)
    >>> plain = log_likelihood_ind([6, 2], [93.285714, 99.571429], [104.0, 104.0],
    ...                            1.4490, 48.6361, 0.5613, 46.8844)
    >>> bool(np.allclose(with_zeros, plain))
    True
    """
    return log_likelihood_ind(
        x,
        t_x,
        T,
        r=r,
        alpha=alpha_i(alpha, gamma_trans, cov_trans),
        s=s,
        beta=beta_i(beta, gamma_life, cov_life),
    )


def log_likelihood(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    r: float,
    alpha: float,
    s: float,
    beta: float,
    gamma_life: ArrayLike,
    gamma_trans: ArrayLike,
    cov_life: ArrayLike,
    cov_trans: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    """The sample log-likelihood maximised in S6.4.1."""
    ll = log_likelihood_staticcov_ind(
        x, t_x, T, r, alpha, s, beta, gamma_life, gamma_trans, cov_life, cov_trans
    )
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(ll * np.asarray(weights, dtype=float)))


@dataclass(frozen=True)
class PnbdStaticCovParams:
    r"""A fitted Pareto/NBD with time-invariant covariates.

    S6.4.1: "The covariate parameters are directly interpretable as rate
    elasticities: a 1% change in a covariate [...] changes the average purchase
    or the attrition rate by :math:`\gamma_{purch}\mathbf{x}^{P}` or
    :math:`\gamma_{life}\mathbf{x}^{A}` percent, respectively. When dummy
    variables are used as covariates, the interpretation is relative to the
    baseline, i.e., the state defined as 0."
    """

    r: float
    alpha: float
    s: float
    beta: float
    gamma_life: np.ndarray
    gamma_trans: np.ndarray
    names_cov_life: list[str]
    names_cov_trans: list[str]
    log_likelihood: float
    converged: bool
    n_customers: int
    hessian: np.ndarray | None = field(default=None, repr=False)

    def __iter__(self) -> Iterator[float]:
        yield from (self.r, self.alpha, self.s, self.beta)
        yield from self.gamma_life
        yield from self.gamma_trans

    @property
    def names(self) -> list[str]:
        """Coefficient names, in CLVTools' ``life.`` / ``trans.`` convention."""
        return (
            ["r", "alpha", "s", "beta"]
            + [f"life.{n}" for n in self.names_cov_life]
            + [f"trans.{n}" for n in self.names_cov_trans]
        )

    def coefficients(self) -> dict[str, float]:
        """Every estimate, keyed as ``summary()`` prints them in S6.4.1.

        >>> from clvtools import ClvData, ClvDataStaticCov
        >>> from clvtools import load_apparel_static_cov, load_apparel_trans
        >>> data = ClvDataStaticCov(
        ...     ClvData(load_apparel_trans(), estimation_split=104),
        ...     load_apparel_static_cov(),
        ...     names_cov_life=["Gender", "Channel"],
        ...     names_cov_trans=["Gender", "Channel"])
        >>> fit = fit_pnbd_staticcov(data, hessian=False)
        >>> list(fit.coefficients())
        ['r', 'alpha', 's', 'beta', 'life.Gender', 'life.Channel',
         'trans.Gender', 'trans.Channel']
        """
        return dict(zip(self.names, list(self)))

    @property
    def n_parameters(self) -> int:
        return 4 + self.gamma_life.size + self.gamma_trans.size

    @property
    def aic(self) -> float:
        return 2 * self.n_parameters - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.n_parameters * np.log(self.n_customers) - 2 * self.log_likelihood

    def standard_errors(self) -> dict[str, float]:
        """Standard errors from the inverse Hessian."""
        if self.hessian is None:
            raise ValueError("fit with hessian=True to obtain standard errors")
        return dict(
            zip(self.names, np.sqrt(np.diag(np.linalg.inv(self.hessian))))
        )

    def summary(self) -> "pd.DataFrame":  # noqa: F821
        r"""The coefficient table of S6.4.1, with z- and p-values.

        S6.4.1: "the four Pareto/NBD base parameters (:math:`r, \alpha, s,
        \beta`) are not reported with any z- and p-values. As these parameters
        are constrained to be strictly positive, the model definition fixes
        their lower bound at 0. Thus, a null hypothesis of :math:`\theta = 0`
        lies outside the admissible parameter space." They are ``NaN`` here for
        the same reason.
        """
        import pandas as pd

        errors = self.standard_errors()
        table = pd.DataFrame(
            {
                "Estimate": list(self),
                "Std. Error": [errors[n] for n in self.names],
            },
            index=self.names,
        )
        is_covariate = np.array(
            [n.startswith(("life.", "trans.")) for n in self.names]
        )
        z = np.where(
            is_covariate, table["Estimate"] / table["Std. Error"], np.nan
        )
        table["z-val"] = z
        table["Pr(>|z|)"] = 2 * (1 - stats.norm.cdf(np.abs(z)))
        return table


def fit_pnbd_staticcov(
    data: "ClvDataStaticCov",  # noqa: F821
    start: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    start_cov: float = _DEFAULT_COV_START,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = True,
    options: dict | None = None,
) -> PnbdStaticCovParams:
    r"""Estimate the model of S6.4.1 from a covariate-bearing data object.

    The search runs over
    ``[log r, log alpha, log s, log beta, gamma_life..., gamma_trans...]``:
    the four model parameters on the log scale because they must stay positive,
    the covariate parameters unconstrained because they may be either sign.

    Examples
    --------
    S6.4.1 prints ``r = 1.8378, alpha = 92.9123, s = 0.5920, beta = 49.6227``,
    ``life.Gender = -0.6430``, ``life.Channel = 0.7907``,
    ``trans.Gender = 0.2859``, ``trans.Channel = 0.6241`` and ``LL = -5821.0627``:

    >>> from clvtools import ClvData, ClvDataStaticCov
    >>> from clvtools import load_apparel_static_cov, load_apparel_trans
    >>> data = ClvDataStaticCov(
    ...     ClvData(load_apparel_trans(), estimation_split=104),
    ...     load_apparel_static_cov(),
    ...     names_cov_life=["Gender", "Channel"],
    ...     names_cov_trans=["Gender", "Channel"])
    >>> fit = fit_pnbd_staticcov(data, hessian=False)
    >>> round(fit.log_likelihood, 4)
    -5821.0627

    The covariate coefficients -- the ones S6.4.1 actually interprets -- come
    back to the printed precision:

    >>> {k: round(float(v), 2) for k, v in fit.coefficients().items() if "." in k}
    {'life.Gender': -0.64, 'life.Channel': 0.79,
     'trans.Gender': 0.29, 'trans.Channel': 0.62}

    The four base parameters land a little way off the published
    ``1.8378, 92.9123, 0.5920, 49.6227`` -- ``beta`` by 0.2%. As in the model
    without covariates this is the likelihood's flat ridge, not a disagreement
    about the likelihood: this fit attains a marginally *higher* value than the
    published one, and L-BFGS-B and Nelder-Mead independently agree with each
    other to four decimals.

    >>> [round(float(v), 2) for v in list(fit)[:4]]
    [1.84, 92.96, 0.59, 49.51]

    S6.4.1's reading of those coefficients: "female customers have a
    significantly higher purchase rate (``trans.Gender = 0.2859``) [...] Also,
    customers acquired offline, coded as 1, purchase more
    (``trans.Channel = 0.6241``) but drop out more quickly
    (``life.Channel = 0.7907``)."
    """
    cbs = data.customer_summary()
    x = cbs["x"].to_numpy(dtype=float)
    t_x = cbs["t_x"].to_numpy(dtype=float)
    T = cbs["T"].to_numpy(dtype=float)
    cov_life = data.design_life()
    cov_trans = data.design_trans()

    n_life, n_trans = cov_life.shape[1], cov_trans.shape[1]
    start_arr = np.asarray(start, dtype=float)
    if start_arr.shape != (4,):
        raise ValueError("start must give four values (r, alpha, s, beta)")
    if np.any(start_arr <= 0):
        raise ValueError("start values must be strictly positive")

    x0 = np.concatenate(
        [np.log(start_arr), np.full(n_life + n_trans, float(start_cov))]
    )

    def unpack(v: np.ndarray):
        r, alpha, s, beta = np.exp(v[:4])
        return r, alpha, s, beta, v[4 : 4 + n_life], v[4 + n_life :]

    def negative_ll(v: np.ndarray) -> float:
        r, alpha, s, beta, g_life, g_trans = unpack(v)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = log_likelihood(
                x, t_x, T, r, alpha, s, beta, g_life, g_trans, cov_life, cov_trans
            )
        return np.inf if not np.isfinite(value) else -value

    result = optimize.minimize(
        negative_ll,
        x0=x0,
        method=method,
        options=options_for(method, maxiter, x0, options),
    )
    r, alpha, s, beta, g_life, g_trans = unpack(result.x)

    hess = None
    if hessian:
        from clvtools.pnbd.fit import _numerical_hessian

        natural = np.concatenate([[r, alpha, s, beta], g_life, g_trans])

        def natural_nll(v: np.ndarray) -> float:
            return -log_likelihood(
                x, t_x, T, v[0], v[1], v[2], v[3],
                v[4 : 4 + n_life], v[4 + n_life :], cov_life, cov_trans,
            )

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hess = _numerical_hessian(natural_nll, natural)

    return PnbdStaticCovParams(
        r=float(r),
        alpha=float(alpha),
        s=float(s),
        beta=float(beta),
        gamma_life=np.asarray(g_life, dtype=float),
        gamma_trans=np.asarray(g_trans, dtype=float),
        names_cov_life=list(data.names_cov_life),
        names_cov_trans=list(data.names_cov_trans),
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        n_customers=int(x.size),
        hessian=hess,
    )
