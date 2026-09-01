r"""S3.2 - the Pareto/NBD at the individual level.

"As with other latent attrition models, the Pareto/NBD consists of two parts.
The first process focuses on modeling customers' unobserved attrition. A second
process considers customer purchase frequency given the assumption that they are
still active."

Every expression here is conditional on one customer's own :math:`(\lambda,
\mu)`. Mixing them over the two gamma distributions gives the expressions in
:mod:`clvtools.pnbd.aggregate`, and the tests do exactly that numerically as a
check on both.

Equations, in the paper's numbering:

===============================  ==============================================
:func:`lifetime_pdf`             lifetime :math:`\omega \sim \mathrm{Exp}(\mu)`
:func:`gamma_pdf_mu`             heterogeneity in :math:`\mu`
:func:`lifetime_pdf_mixed`       the resulting Pareto of the second kind
:func:`poisson_pmf`              transactions while alive
:func:`gamma_pdf_lambda`         heterogeneity in :math:`\lambda`
:func:`nbd_pmf`                  the resulting negative binomial
:func:`individual_likelihood`    eq. (10) -- the two cases combined
===============================  ==============================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

__all__ = [
    "gamma_pdf_lambda",
    "gamma_pdf_mu",
    "individual_likelihood",
    "lifetime_pdf",
    "lifetime_pdf_mixed",
    "likelihood_alive_at_T",
    "likelihood_died_at",
    "log_individual_likelihood",
    "nbd_pmf",
    "poisson_pmf",
]


def _require_positive(**params: float) -> None:
    for name, value in params.items():
        if not np.all(np.asarray(value) > 0):
            raise ValueError(f"{name} must be strictly positive, got {value!r}")


# -- the attrition process ----------------------------------------------------


def lifetime_pdf(omega: ArrayLike, mu: ArrayLike) -> NDArray[np.float64]:
    r"""A customer's unobserved lifetime, exponentially distributed.

    .. math::
        f(\omega \mid \mu) = \mu e^{-\mu \omega}

    S3.2: "a customer's unobserved lifetime of length :math:`\omega` is assumed
    to be exponentially distributed with rate :math:`\mu`". Schmittlein et al.
    call :math:`\mu` the "average death rate".

    >>> import numpy as np
    >>> float(lifetime_pdf(0.0, 2.0))
    2.0
    >>> bool(np.isclose(lifetime_pdf(1.0, 2.0), 2 * np.exp(-2)))
    True
    """
    omega = np.asarray(omega, dtype=float)
    mu = np.asarray(mu, dtype=float)
    _require_positive(mu=mu)
    return mu * np.exp(-mu * omega)


def gamma_pdf_mu(mu: ArrayLike, s: float, beta: float) -> NDArray[np.float64]:
    r"""Heterogeneity in the attrition rate across customers.

    .. math::
        g(\mu) = \frac{\beta^{s} \mu^{s-1} e^{-\mu \beta}}{\Gamma(s)}

    S3.2: ":math:`\mu` [...] is itself assumed to follow a Gamma distribution
    with shape parameter :math:`s` and scale parameter :math:`\beta` to account
    for the cross-sectional heterogeneity".

    The paper calls :math:`\beta` a *scale* parameter, but it enters as
    :math:`e^{-\mu\beta}`, which is SciPy's rate: this equals
    ``scipy.stats.gamma.pdf(mu, a=s, scale=1/beta)``.

    >>> import numpy as np
    >>> from scipy import stats
    >>> mu = np.array([0.01, 0.1, 1.0])
    >>> bool(np.allclose(gamma_pdf_mu(mu, 0.5613, 46.8844),
    ...                  stats.gamma.pdf(mu, a=0.5613, scale=1/46.8844)))
    True
    """
    _require_positive(s=s, beta=beta)
    mu = np.asarray(mu, dtype=float)
    with np.errstate(divide="ignore"):
        log_pdf = (
            s * np.log(beta) + (s - 1) * np.log(mu) - mu * beta - special.gammaln(s)
        )
    return np.exp(log_pdf)


def lifetime_pdf_mixed(omega: ArrayLike, s: float, beta: float) -> NDArray[np.float64]:
    r"""The lifetime of a *randomly chosen* customer: a Pareto of the second kind.

    .. math::
        f(\omega) = \int_0^\infty f(\omega \mid \mu) g(\mu) \, d\mu
                  = \frac{s}{\beta}\left(\frac{\beta}{\beta+\omega}\right)^{s+1}

    S3.2: "Combining the Exponential distribution of :math:`\mu` with the Gamma
    distribution [...] results in a Pareto distribution of the second kind."

    >>> import numpy as np
    >>> from scipy import integrate
    >>> mass, _ = integrate.quad(lifetime_pdf_mixed, 0, np.inf, args=(0.5613, 46.8844))
    >>> bool(np.isclose(mass, 1.0))
    True
    """
    _require_positive(s=s, beta=beta)
    omega = np.asarray(omega, dtype=float)
    return (s / beta) * (beta / (beta + omega)) ** (s + 1.0)


# -- the transaction process --------------------------------------------------


def poisson_pmf(x: ArrayLike, lam: ArrayLike, t: ArrayLike) -> NDArray[np.float64]:
    r"""Transactions in :math:`(0, t]` while the customer is alive.

    .. math::
        P(X(t) = x \mid \lambda,\, t < \omega)
          = \frac{(\lambda t)^{x} e^{-\lambda t}}{x!}

    S3.2: "the Pareto/NBD model assumes that transactions follow a Poisson
    process with rate :math:`\lambda` for any given customer."

    >>> import numpy as np
    >>> bool(np.isclose(poisson_pmf(0, 0.5, 2.0), np.exp(-1.0)))
    True
    >>> bool(np.isclose(np.sum(poisson_pmf(np.arange(200), 0.5, 2.0)), 1.0))
    True
    """
    x = np.asarray(x, dtype=float)
    lam = np.asarray(lam, dtype=float)
    t = np.asarray(t, dtype=float)
    _require_positive(lam=lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pmf = x * np.log(lam * t) - lam * t - special.gammaln(x + 1.0)
    return np.where((t == 0) & (x == 0), 1.0, np.exp(log_pmf))


def gamma_pdf_lambda(
    lam: ArrayLike, r: float, alpha: float
) -> NDArray[np.float64]:
    r"""Heterogeneity in the purchase rate across customers.

    .. math::
        g(\lambda) = \frac{\alpha^{r} \lambda^{r-1} e^{-\lambda \alpha}}{\Gamma(r)}

    S3.2: ":math:`\lambda` follows a Gamma distribution with shape parameter
    :math:`r` and scale parameter :math:`\alpha`". As with
    :func:`gamma_pdf_mu`, :math:`\alpha` is a rate in SciPy's convention.

    Its mean is the "purchase rate of an average customer", :math:`r/\alpha` --
    S6.2.1 reports 0.030 for the apparel cohort:

    >>> round(1.4490 / 48.6361, 3)
    0.03
    """
    _require_positive(r=r, alpha=alpha)
    lam = np.asarray(lam, dtype=float)
    with np.errstate(divide="ignore"):
        log_pdf = (
            r * np.log(alpha) + (r - 1) * np.log(lam) - lam * alpha - special.gammaln(r)
        )
    return np.exp(log_pdf)


def nbd_pmf(x: ArrayLike, t: ArrayLike, r: float, alpha: float) -> NDArray[np.float64]:
    r"""Transactions by a randomly chosen customer while alive: negative binomial.

    .. math::
        P(X(t) = x \mid t < \omega)
          = \frac{\Gamma(r+x)}{\Gamma(r)\, x!}
            \left(\frac{\alpha}{\alpha+t}\right)^{r}
            \left(\frac{t}{\alpha+t}\right)^{x}

    S3.2: "This combination [...] results in a negative binomial distribution
    (NBD)."

    >>> import numpy as np
    >>> bool(np.isclose(np.sum(nbd_pmf(np.arange(500), 52.0, 1.4490, 48.6361)), 1.0))
    True
    """
    _require_positive(r=r, alpha=alpha)
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pmf = (
            special.gammaln(r + x)
            - special.gammaln(r)
            - special.gammaln(x + 1.0)
            + r * (np.log(alpha) - np.log(alpha + t))
            + x * (np.log(t) - np.log(alpha + t))
        )
    return np.where((t == 0) & (x == 0), 1.0, np.exp(log_pmf))


# -- the two processes combined -----------------------------------------------


def likelihood_alive_at_T(  # noqa: N802 - `T` is the paper's observation window
    x: ArrayLike, T: ArrayLike, lam: ArrayLike
) -> NDArray[np.float64]:
    r"""Eq. (8) -- the customer is still alive at the end of the window.

    .. math::
        L(\lambda \mid t_1, \ldots, t_x, T, \omega > T) = \lambda^{x} e^{-\lambda T}

    S3.2, quoting Fader & Hardie: if a customer is alive until :math:`T` the
    "likelihood function is simply the product of the (inter-transaction-time)
    exponential density functions and the associated survivor function".

    >>> import numpy as np
    >>> bool(np.isclose(likelihood_alive_at_T(2, 10.0, 0.3), 0.3**2 * np.exp(-3.0)))
    True
    """
    x = np.asarray(x, dtype=float)
    T = np.asarray(T, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return lam**x * np.exp(-lam * T)


def likelihood_died_at(
    x: ArrayLike, omega: ArrayLike, lam: ArrayLike
) -> NDArray[np.float64]:
    r"""Eq. (9) -- the customer became inactive at :math:`\omega \in (t_x, T]`.

    .. math::
        L(\lambda \mid t_1, \ldots, t_x, T, \text{inactive at } \omega)
          = \lambda^{x} e^{-\lambda \omega}

    S3.2: "The expressions are very similar and differ only in their
    conditioning on whether the customer "dies" before or after :math:`T`."

    >>> bool(likelihood_died_at(3, 5.0, 0.4) == likelihood_alive_at_T(3, 5.0, 0.4))
    True
    """
    return likelihood_alive_at_T(x, omega, lam)


def individual_likelihood(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    lam: ArrayLike,
    mu: ArrayLike,
) -> NDArray[np.float64]:
    r"""Eq. (10) -- the individual-level Pareto/NBD likelihood.

    .. math::
        L(\lambda, \mu \mid x, t_x, T)
        = \frac{\lambda^{x} \mu}{\lambda+\mu} e^{-(\lambda+\mu)t_x}
        + \frac{\lambda^{x+1}}{\lambda+\mu} e^{-(\lambda+\mu)T}

    The two terms are the two ways the observation could have arisen: the
    customer died somewhere in :math:`(t_x, T]`, or survived past :math:`T`.

    S3.2: "the exact timing of transactions, except the last one at
    :math:`t_x`, becomes irrelevant in this model due to the memoryless
    property of the exponential distribution" -- which is why the whole
    purchase history reduces to :math:`(x, t_x, T)`.

    .. note::
       Appendix A writes the integrand's second term as
       :math:`\frac{\lambda^{x+1}\mu}{\lambda+\mu} e^{-(\lambda+\mu)T}`, with a
       stray :math:`\mu` in the numerator. Eq. (10) in the body, and
       Schmittlein et al. and Fader & Hardie before it, have no :math:`\mu`
       there; with it the likelihood does not integrate to the closed form the
       appendix then states. The form here follows eq. (10).
       ``tests/test_pnbd_individual.py`` checks this by integration.

    Examples
    --------
    A customer observed for 10 periods whose last purchase was at 8:

    >>> float(individual_likelihood(2, 8.0, 10.0, 0.3, 0.05))
    0.0031113...

    With :math:`\mu \to 0` nobody ever dies, and only the survival term
    remains -- :math:`\lambda^{x+1}/\lambda \cdot e^{-\lambda T}`:

    >>> import numpy as np
    >>> bool(np.isclose(individual_likelihood(2, 8.0, 10.0, 0.3, 1e-12),
    ...                 0.3**2 * np.exp(-0.3 * 10.0), rtol=1e-6))
    True
    """
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    _require_positive(lam=lam, mu=mu)

    total = lam + mu
    died = lam**x * mu / total * np.exp(-total * t_x)
    survived = lam ** (x + 1.0) / total * np.exp(-total * T)
    return died + survived


def log_individual_likelihood(
    x: ArrayLike,
    t_x: ArrayLike,
    T: ArrayLike,
    lam: ArrayLike,
    mu: ArrayLike,
) -> NDArray[np.float64]:
    r"""``log`` of eq. (10), summing the two terms in log space.

    Equivalent to ``np.log(individual_likelihood(...))`` but usable where either
    term underflows.

    >>> import numpy as np
    >>> bool(np.isclose(log_individual_likelihood(2, 8.0, 10.0, 0.3, 0.05),
    ...                 np.log(individual_likelihood(2, 8.0, 10.0, 0.3, 0.05))))
    True
    """
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    _require_positive(lam=lam, mu=mu)

    total = lam + mu
    log_died = x * np.log(lam) + np.log(mu) - np.log(total) - total * t_x
    log_survived = (x + 1.0) * np.log(lam) - np.log(total) - total * T
    return np.logaddexp(log_died, log_survived)
