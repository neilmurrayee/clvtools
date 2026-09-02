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

# `ClvDataStaticCov` appears only in an annotation, but it is imported for
# real rather than under `TYPE_CHECKING` so that `typing.get_type_hints()`
# resolves it -- `py.typed` promises the signatures are usable downstream.
# `clvtools.data` does not reach back here, so this closes no cycle.
from clvtools._staticcov import design
from clvtools.data import ClvDataStaticCov
from clvtools.inference import Fitted
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
    covariates, gamma_trans = design(covariates, gamma_trans, "transaction")
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
    covariates, gamma_life = design(covariates, gamma_life, "attrition")
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
class PnbdStaticCovParams(Fitted):
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
    unpenalised_log_likelihood: float | None = None
    names_cov_constr: list[str] = field(default_factory=list)
    reg_lambdas: tuple[float, float] | None = None
    hessian: np.ndarray | None = field(default=None, repr=False)

    def __iter__(self) -> Iterator[float]:
        """Estimates in the order :attr:`names` lists them."""
        yield from (self.r, self.alpha, self.s, self.beta)
        for name, value in zip(self.names_cov_life, self.gamma_life, strict=True):
            if name not in self.names_cov_constr:
                yield float(value)
        for name, value in zip(self.names_cov_trans, self.gamma_trans, strict=True):
            if name not in self.names_cov_constr:
                yield float(value)
        for name in self.names_cov_constr:
            yield float(self.gamma_life[self.names_cov_life.index(name)])

    @property
    def names(self) -> list[str]:
        """Coefficient names, in CLVTools' convention.

        A constrained covariate appears once as ``constr.<name>`` rather than
        twice, which is how S6.5.3's output reads.
        """
        constrained = set(self.names_cov_constr)
        return (
            ["r", "alpha", "s", "beta"]
            + [f"life.{n}" for n in self.names_cov_life if n not in constrained]
            + [f"trans.{n}" for n in self.names_cov_trans if n not in constrained]
            + [f"constr.{n}" for n in self.names_cov_constr]
        )

    @property
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
        >>> list(fit.coefficients)
        ['r', 'alpha', 's', 'beta', 'life.Gender', 'life.Channel',
         'trans.Gender', 'trans.Channel']
        """
        return dict(zip(self.names, list(self), strict=True))

    @property
    def _comparable_log_likelihood(self) -> float:
        """The true log-likelihood, whether or not a penalty was applied."""
        if self.unpenalised_log_likelihood is None:
            return self.log_likelihood
        return self.unpenalised_log_likelihood


def fit_pnbd_staticcov(
    data: ClvDataStaticCov,
    names_cov_constr: list[str] | None = None,
    reg_lambdas: tuple[float, float] | None = None,
    start: tuple[float, float, float, float] | None = None,
    start_cov: float | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = True,
    polish: bool = True,
    options: dict | None = None,
) -> PnbdStaticCovParams:
    r"""Estimate the model of S6.4.1 from a covariate-bearing data object.

    The search runs over
    ``[log r, log alpha, log s, log beta, gamma_life..., gamma_trans...]``:
    the four model parameters on the log scale because they must stay positive,
    the covariate parameters unconstrained because they may be either sign.

    Parameters
    ----------
    names_cov_constr
        Covariates whose coefficient is forced equal across the two processes,
        eq. (14): :math:`\boldsymbol{\gamma}_{purch} \equiv
        \boldsymbol{\gamma}_{attr}`. S6.5.3: "an additional model is estimated
        that forces the covariate gender to have the same parameter value for
        the transaction and attrition process. [...] In consequence, the model
        output only contains a single parameter value for gender." Such a
        coefficient is reported as ``constr.<name>``.
    reg_lambdas
        ``(life, trans)`` L2 penalties, as eq. (13). S6.5.1: "by specifying
        ``reg.lambdas = c(trans = 0.1, life = 0.1)``, a user sets the
        regularization weight to 0.1 for both processes. The larger this
        regularization weight, the stronger the effect of the regularization."

        .. warning::
           With regularization on, ``log_likelihood`` holds the *penalised
           mean* objective that was minimised, not a log-likelihood. CLVTools
           does the same -- its objective is
           ``LL / n + lambda_trans ||gamma_trans||^2 + lambda_life
           ||gamma_life||^2`` -- so ``logLik()`` on a regularized fit reports
           roughly -9.7 rather than roughly -5821 for this data. Eq. (13) in
           the paper shows the penalties applied to the summed likelihood, with
           no division by ``n``; the implementation divides. Use
           :attr:`~PnbdStaticCovParams.unpenalised_log_likelihood` for a value
           comparable across models.

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

    >>> {k: round(float(v), 2) for k, v in fit.coefficients.items() if "." in k}
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
    from clvtools._staticcov import SearchSettings, fit_static_covariates

    cbs = data.customer_summary()

    def objective(model, g_life, g_trans, cov_life, cov_trans):
        return log_likelihood(
            cbs["x"], cbs["t_x"], cbs["T"],
            model[0], model[1], model[2], model[3],
            g_life, g_trans, cov_life, cov_trans,
        )

    result = fit_static_covariates(
        x=cbs["x"], t_x=cbs["t_x"], T=cbs["T"],
        cov_life=data.design_life(), cov_trans=data.design_trans(),
        names_cov_life=data.names_cov_life,
        names_cov_trans=data.names_cov_trans,
        log_likelihood=objective,
        n_model_params=4, model_start=(1.0, 1.0, 1.0, 1.0),
        names_cov_constr=names_cov_constr,
        search=SearchSettings(
            start=start, start_cov=start_cov, reg_lambdas=reg_lambdas,
            method=method, maxiter=maxiter, hessian=hessian,
            polish=polish, options=options,
        ),
    )
    r, alpha, s, beta = (float(v) for v in result.model)
    return PnbdStaticCovParams(
        r=r, alpha=alpha, s=s, beta=beta,
        gamma_life=result.gamma_life,
        gamma_trans=result.gamma_trans,
        names_cov_life=result.names_cov_life,
        names_cov_trans=result.names_cov_trans,
        log_likelihood=result.log_likelihood,
        unpenalised_log_likelihood=result.unpenalised_log_likelihood,
        converged=result.converged,
        n_customers=result.n_customers,
        names_cov_constr=result.names_cov_constr,
        reg_lambdas=result.reg_lambdas,
        hessian=result.hessian,
    )
