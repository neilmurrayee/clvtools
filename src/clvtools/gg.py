r"""S3.5 and S6.2.3 - the Gamma-Gamma model of customer spending.

S3.5: "Latent attrition models usually only focus on customers' attrition and
transaction process. However, to derive a monetary value such as CLV, customers'
spending behavior must also be considered. Here, the Gamma-Gamma model is a
popular choice."

The model has no notion of time. S3.5: "this model does not consider time as a
dimension. The underlying "true" average transaction value is assumed to be
time-invariant." Each customer contributes only their transaction count
:math:`x` and their observed mean spend :math:`\bar{z}`.

Three transcription errors in the paper are worked around here, each noted at
the function concerned and each covered by a test:

* eq. (14) writes the per-transaction density as
  :math:`\nu^p z_i^{r-1} e^{-z_i\nu} / \Gamma(p)`, with the Pareto/NBD's
  :math:`r` where the shape :math:`p` belongs;
* the integral in eq. (17) writes :math:`\nu{q-1}` for :math:`\nu^{q-1}`;
* eq. (17)'s result drops the exponent :math:`px` from its final factor.

The last matters most: as printed, eq. (17) is not a density and does not agree
with what CLVTools maximises. :func:`mean_spending_pdf` restores the exponent.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special

from clvtools._optimize import options_for
from clvtools._validate import finished, spending_history, start_values
from clvtools.inference import Fitted, numerical_hessian

__all__ = [
    "GgParams",
    "expected_mean_spending",
    "fit_gg",
    "log_likelihood",
    "log_likelihood_ind",
    "mean_spending_pdf",
    "spending_pdf",
    "spending_pdf_given_x",
]

_NAMES = ("p", "q", "gamma")

#: All three parameters start at 1, as CLVTools does.
_DEFAULT_START = (1.0, 1.0, 1.0)


def _require_positive(**params: float) -> None:
    for name, value in params.items():
        if not np.all(np.asarray(value) > 0):
            raise ValueError(f"{name} must be strictly positive, got {value!r}")


# -- the spending process -----------------------------------------------------


def spending_pdf(z: ArrayLike, p: float, nu: float) -> NDArray[np.float64]:
    r"""Eq. (14) -- one transaction's value, gamma distributed.

    .. math::
        g(z_i) = \frac{\nu^{p} z_i^{p-1} e^{-z_i \nu}}{\Gamma(p)}

    S3.5: "the Gamma-Gamma model assumes that a customer's spending per
    transaction is Gamma distributed with :math:`p` as shape and :math:`\nu` as
    scale parameter."

    .. note::
       The paper prints the numerator as :math:`\nu^p z_i^{r-1} e^{-z_i\nu}`.
       The :math:`r` is the Pareto/NBD's transaction-process shape and has no
       meaning in this model; the exponent is :math:`p-1`, as the surrounding
       text ("with :math:`p` as shape") says.

    >>> import numpy as np
    >>> from scipy import stats
    >>> z = np.array([10.0, 50.0, 200.0])
    >>> bool(np.allclose(spending_pdf(z, 3.099, 0.05),
    ...                  stats.gamma.pdf(z, a=3.099, scale=1/0.05)))
    True
    """
    _require_positive(p=p, nu=nu)
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore"):
        log_pdf = (
            p * np.log(nu) + (p - 1) * np.log(z) - z * nu - special.gammaln(p)
        )
    return np.exp(log_pdf)


def spending_pdf_given_x(
    z_bar: ArrayLike, x: ArrayLike, p: float, nu: float
) -> NDArray[np.float64]:
    r"""Eq. (15) -- the mean of :math:`x` transactions, for a known :math:`\nu`.

    .. math::
        f(\bar{z} \mid p, \nu; x)
          = \frac{(\nu x)^{px} \bar{z}^{px-1} e^{-\nu x \bar{z}}}{\Gamma(px)}

    S3.5: "Following the Gamma distribution's scaling property, the density of
    customers' past average spending values :math:`\bar{z}` is" -- the sum of
    :math:`x` independent :math:`\Gamma(p, \nu)` draws is :math:`\Gamma(px,
    \nu)`, and dividing by :math:`x` scales the rate to :math:`\nu x`.

    >>> import numpy as np
    >>> from scipy import integrate
    >>> mass, _ = integrate.quad(spending_pdf_given_x, 0, np.inf, args=(4, 3.099, 0.05))
    >>> bool(np.isclose(mass, 1.0))
    True
    """
    _require_positive(p=p, nu=nu)
    z_bar = np.asarray(z_bar, dtype=float)
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        log_pdf = (
            p * x * np.log(nu * x)
            + (p * x - 1) * np.log(z_bar)
            - nu * x * z_bar
            - special.gammaln(p * x)
        )
    return np.exp(log_pdf)


def mean_spending_pdf(
    z_bar: ArrayLike, x: ArrayLike, p: float, q: float, gamma: float
) -> NDArray[np.float64]:
    r"""Eq. (17) -- the marginal density, with :math:`\nu` mixed out.

    .. math::
        f(\bar{z} \mid p, q, \gamma; x)
          = \frac{1}{\bar{z}\, B(px, q)}
            \left(\frac{\gamma}{\gamma + x\bar{z}}\right)^{q}
            \left(\frac{x\bar{z}}{\gamma + x\bar{z}}\right)^{px}

    S3.5: "To allow for heterogeneity in a customer's average spending value, we
    assume :math:`\nu` to be Gamma distributed with shape parameter :math:`q`
    and the scale parameter :math:`\gamma`."

    .. note::
       The paper's final factor is printed without its exponent, as
       :math:`\left(\frac{x\bar{z}}{\gamma+x\bar{z}}\right)`. As printed the
       expression is not a density -- it does not integrate to 1 -- and does not
       match the likelihood CLVTools maximises. The exponent is :math:`px`.
       ``tests/test_gg.py`` checks both the integral and the agreement.

    Examples
    --------
    It is a proper density, which the printed version is not:

    >>> import numpy as np
    >>> from scipy import integrate
    >>> mass, _ = integrate.quad(
    ...     mean_spending_pdf, 0, np.inf, args=(4, 3.099, 5.654, 56.504))
    >>> bool(np.isclose(mass, 1.0))
    True
    """
    _require_positive(p=p, q=q, gamma=gamma)
    z_bar = np.asarray(z_bar, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.exp(log_likelihood_ind(x, z_bar, p, q, gamma))


# -- estimation ---------------------------------------------------------------


def log_likelihood_ind(
    x: ArrayLike, z_bar: ArrayLike, p: float, q: float, gamma: float
) -> NDArray[np.float64]:
    r"""``log`` of eq. (17), per customer.

    .. math::
        \log f = q\log\gamma + (px-1)\log\bar{z} + px\log x
                 - (px+q)\log(\gamma + x\bar{z}) - \ln B(px, q)

    Customers with :math:`x = 0` contribute nothing: with no transaction there
    is no observed mean to explain. S6.2.3 puts it in terms of the default to
    drop the first transaction -- "customers with a single purchase are ignored
    during model estimation and their estimated average order value is the mean
    under the distribution".

    Examples
    --------
    The first two apparel customers at the estimates of S6.2.3:

    >>> import numpy as np
    >>> np.round(log_likelihood_ind([6, 2], [101.415, 43.9],
    ...                             3.099, 5.654, 56.504), 4)
    array([-7.2427, -4.358 ])

    A customer with no transactions is silent, not impossible:

    >>> float(log_likelihood_ind(0, 0.0, 3.099, 5.654, 56.504))
    0.0
    """
    _require_positive(p=p, q=q, gamma=gamma)
    x = np.asarray(x, dtype=float)
    z_bar = np.asarray(z_bar, dtype=float)
    x, z_bar = np.broadcast_arrays(x, z_bar)

    contributes = (x != 0) & (z_bar != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (
            q * np.log(gamma)
            + (p * x - 1) * np.log(z_bar)
            + p * x * np.log(x)
            - (p * x + q) * np.log(gamma + x * z_bar)
            - special.betaln(p * x, q)
        )
    return np.where(contributes, value, 0.0)


def log_likelihood(
    x: ArrayLike,
    z_bar: ArrayLike,
    p: float,
    q: float,
    gamma: float,
    weights: ArrayLike | None = None,
) -> float:
    r"""The sample log-likelihood maximised in S6.2.3.

    Examples
    --------
    The apparel cohort at the published estimates:

    >>> from clvtools import ClvData, load_apparel_trans
    >>> spend = ClvData(load_apparel_trans(), estimation_split=104).spending_summary()
    >>> round(log_likelihood(spend["x"], spend["Spending"],
    ...                      3.099, 5.654, 56.504), 4)
    -1670.663
    """
    ll = log_likelihood_ind(x, z_bar, p, q, gamma)
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(ll * np.asarray(weights, dtype=float)))


@dataclass(frozen=True)
class GgParams(Fitted):
    r"""A fitted Gamma-Gamma model.

    :math:`p` is the shape of the per-transaction gamma; :math:`q` and
    :math:`\gamma` are the shape and scale of the gamma mixing its rate
    :math:`\nu` across customers.
    """

    p: float
    q: float
    gamma: float
    log_likelihood: float
    converged: bool
    n_customers: int
    hessian: np.ndarray | None = field(default=None, repr=False)

    @property
    def names(self) -> list[str]:
        return list(_NAMES)

    def __iter__(self) -> Iterator[float]:
        r"""Yield :math:`(p, q, \gamma)` in the paper's order."""
        yield from (self.p, self.q, self.gamma)

    def as_dict(self) -> dict[str, float]:
        return {"p": self.p, "q": self.q, "gamma": self.gamma}

    @property
    def n_parameters(self) -> int:
        return 3

    @property
    def aic(self) -> float:
        return 2 * self.n_parameters - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.n_parameters * np.log(self.n_customers) - 2 * self.log_likelihood


def fit_gg(
    x: ArrayLike,
    z_bar: ArrayLike,
    weights: ArrayLike | None = None,
    start: tuple[float, float, float] = _DEFAULT_START,
    method: str = "L-BFGS-B",
    maxiter: int = 10_000,
    hessian: bool = True,
    options: dict | None = None,
) -> GgParams:
    r"""Maximise eq. (17) over :math:`(p, q, \gamma)`.

    S3.5: "The parameters :math:`p`, :math:`q`, and :math:`\gamma` of
    Eq. (17) are estimated using maximum likelihood."

    Examples
    --------
    S6.2.3 prints ``p = 3.099, q = 5.654, gamma = 56.504``:

    >>> from clvtools import ClvData, load_apparel_trans
    >>> spend = ClvData(load_apparel_trans(), estimation_split=104).spending_summary()
    >>> fit = fit_gg(spend["x"], spend["Spending"])
    >>> [round(v, 3) for v in fit]
    [3.099, 5.654, 56.504]
    >>> fit.converged
    True
    >>> round(fit.log_likelihood, 3)
    -1670.663
    """
    x, z_bar = (np.asarray(v, dtype=float).ravel() for v in (x, z_bar))
    x, z_bar = spending_history(x, z_bar)

    start_arr = start_values(start, count=3, parameters="values (p, q, gamma)")

    w = None if weights is None else np.asarray(weights, dtype=float).ravel()

    def negative_ll(log_params: np.ndarray) -> float:
        p, q, gamma = np.exp(log_params)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = log_likelihood(x, z_bar, p, q, gamma, weights=w)
        return np.inf if not np.isfinite(value) else -value

    result = optimize.minimize(
        negative_ll,
        x0=np.log(start_arr),
        method=method,
        options=options_for(method, maxiter, np.log(start_arr), options),
    )
    result = finished(result, "Gamma-Gamma")

    p, q, gamma = (float(v) for v in np.exp(result.x))
    hess = None
    if hessian:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hess = numerical_hessian(
                lambda v: -log_likelihood(x, z_bar, *v, weights=w),
                np.array([p, q, gamma]),
            )
    return GgParams(
        p=p,
        q=q,
        gamma=gamma,
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        n_customers=int(x.size if w is None else w.sum()),
        hessian=hess,
    )


# -- prediction ---------------------------------------------------------------


def expected_mean_spending(
    x: ArrayLike, z_bar: ArrayLike, p: float, q: float, gamma: float
) -> NDArray[np.float64]:
    r"""``predicted.mean.spending`` -- expected spend per future transaction.

    .. math::
        E[\zeta \mid x, \bar{z}] = \frac{(\gamma + x\bar{z})\, p}{px + q - 1}

    S6.3: metric (4), "the predicted average spending per transaction". It is a
    posterior mean: a weighted blend of the population mean and this customer's
    own observed average, with the weight moving toward the customer as
    :math:`x` grows.

    A customer with :math:`x = 0` falls back to the population figure
    :math:`\gamma p / (q-1)`, which is what S6.2.3 means by "their estimated
    average order value is the mean under the distribution". That is also the
    value S6.3.4 predicts for a prospective customer.

    Examples
    --------
    The first three apparel customers, on the full-data fit of S6.3.2, whose
    ``predicted.mean.spending`` the paper prints as 77.79363, 36.04491 and
    37.23417:

    >>> import numpy as np
    >>> from clvtools import ClvData, load_apparel_trans
    >>> spend = ClvData(load_apparel_trans()).spending_summary().set_index("Id")
    >>> rows = spend.loc[["1", "10", "100"]]
    >>> full = (2.476246215297599, 9.760124916255805, 133.4794545735068)
    >>> np.round(expected_mean_spending(rows["x"], rows["Spending"], *full), 5)
    array([77.79363, 36.04491, 37.23417])

    With no history, every customer gets the same population mean
    :math:`\gamma p/(q-1)`. S6.3.4 predicts exactly that for a prospective
    customer, and prints 39.1372 -- fitting the spending model on all orders,
    first purchases included, as that section requires:

    >>> with_first = (2.815721022227181, 7.094411331193874, 84.7094526802202)
    >>> float(round(expected_mean_spending(0, 0.0, *with_first), 4))
    39.1372
    """
    if p * np.min(np.asarray(x, dtype=float)) + q - 1 <= 0:
        raise ValueError(
            "expected spending is undefined unless p*x + q > 1 for every customer"
        )
    _require_positive(p=p, q=q, gamma=gamma)
    x = np.asarray(x, dtype=float)
    z_bar = np.asarray(z_bar, dtype=float)
    return (gamma + x * z_bar) * p / (p * x + q - 1.0)
