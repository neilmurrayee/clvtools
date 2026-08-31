r"""The model details of Table 2: standard errors, ``vcov``, ``confint``,
``summary`` and the likelihood ratio test.

S6.2.2: "Besides these plots and the ``summary()`` command, the canonical
generics ``coef()``, ``logLik()``, ``confint()``, ``vcov()``, ``nobs()``,
``AIC()``, ``BIC()`` are available for all fitted models to extract key
information."

Everything here follows from one object: the Hessian of the *negative*
log-likelihood in the natural parameters, evaluated at the optimum. Its inverse
is the asymptotic covariance matrix, whose diagonal gives the standard errors
and whose square roots give the Wald intervals CLVTools reports.

The Hessian is differenced in the natural parameters rather than the logarithms
the optimiser searches over, because that is what a standard error refers to.
A parameter's log-scale curvature would describe a different quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "Fitted",
    "LikelihoodRatioTest",
    "likelihood_ratio_test",
    "numerical_hessian",
]

#: Parameter-name prefixes CLVTools gives covariate coefficients. Only these
#: get a z- and a p-value; see :meth:`Fitted.summary`.
COVARIATE_PREFIXES = ("life.", "trans.", "constr.")


def numerical_hessian(
    fn, at: NDArray[np.float64], step: float = 1e-4, zero_tol: float = 1e-8
):
    """Central-difference Hessian of ``fn`` at ``at``.

    CLVTools uses ``numDeriv`` for the same purpose, and this follows its two
    important choices.

    The step is **relative** to each coordinate: parameters here sit on wildly
    different scales -- ``s`` near 0.56 against ``beta`` near 47, and the
    GGom/NBD's ``b`` near 8e-07 on this data -- and an absolute step wide
    enough for the largest sends the smallest negative, where the likelihood is
    not defined. A coordinate at (or indistinguishable from) zero has nothing
    to be relative to and takes ``step`` itself.

    ``step`` is 1e-4 rather than something smaller because these log-likelihoods
    are around -5800 while their second differences are around 1e-2. The
    subtraction cancels four significant figures before the division; halving
    the step again costs more to round-off than it gains in truncation. At 1e-4
    the standard errors agree with ``numDeriv``'s to about 1e-4 relative, and at
    1e-5 they are worse.
    """
    n = at.size
    magnitude = np.abs(at)
    h = step * np.where(magnitude > zero_tol, magnitude, 1.0)
    out = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = h[i]
            ej = np.zeros(n); ej[j] = h[j]
            out[i, j] = out[j, i] = (
                fn(at + ei + ej) - fn(at + ei - ej)
                - fn(at - ei + ej) + fn(at - ei - ej)
            ) / (4 * h[i] * h[j])
    return out


class Fitted:
    """The generics every fitted model shares. Cf. ``clv.fitted`` in Table 2.

    Mixed into each family's parameter class. It needs three things from one:
    ``names``, iteration over the estimates in that order, and ``hessian``.
    """

    @property
    def names(self) -> list[str]:  # pragma: no cover - each class provides it
        raise NotImplementedError

    @property
    def coefficients(self) -> dict[str, float]:
        """The estimates by name. Cf. ``coef()``."""
        return dict(zip(self.names, self))

    def _covariance(self) -> NDArray[np.float64]:
        if getattr(self, "hessian", None) is None:
            raise ValueError(
                "fit with hessian=True to obtain standard errors, a covariance "
                "matrix or confidence intervals"
            )
        return np.linalg.inv(self.hessian)

    def vcov(self) -> pd.DataFrame:
        """The asymptotic covariance matrix of the estimates. Cf. ``vcov()``.

        Examples
        --------
        >>> from clvtools import ClvData, load_apparel_trans
        >>> from clvtools.pnbd import fit_pnbd
        >>> cbs = ClvData(load_apparel_trans(), estimation_split=104).customer_summary()
        >>> fit = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"])
        >>> round(float(fit.vcov().loc["r", "r"]), 4)
        0.0593
        """
        return pd.DataFrame(self._covariance(), index=self.names, columns=self.names)

    def standard_errors(self) -> dict[str, float]:
        """Standard errors from the inverse Hessian, by name."""
        return dict(zip(self.names, np.sqrt(np.diag(self._covariance()))))

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        r"""Wald confidence intervals. Cf. ``confint()``.

        :math:`\hat\theta \pm z_{1-\alpha/2}\,\mathrm{se}(\hat\theta)`, on every
        parameter -- which is what CLVTools reports, and why an interval can
        run below zero for a parameter the model constrains to be positive.
        S6.4.1 makes the same point about the *hypothesis*: a null of
        :math:`\theta = 0` "lies outside the admissible parameter space".

        Examples
        --------
        >>> from clvtools import ClvData, load_apparel_trans
        >>> from clvtools.pnbd import fit_pnbd
        >>> cbs = ClvData(load_apparel_trans(), estimation_split=104).customer_summary()
        >>> fit = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"])
        >>> print(fit.confint().round(3).to_string())
                2.5 %   97.5 %
        r       0.972    1.926
        alpha  33.957   63.313
        s       0.030    1.093
        beta  -22.921  116.688
        """
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between 0 and 1")
        z = stats.norm.ppf(0.5 + level / 2)
        errors = np.array([self.standard_errors()[n] for n in self.names])
        estimates = np.array(list(self), dtype=float)
        tail = (1 - level) / 2
        return pd.DataFrame(
            {
                f"{tail * 100:g} %": estimates - z * errors,
                f"{(1 - tail) * 100:g} %": estimates + z * errors,
            },
            index=self.names,
        )

    def summary(self) -> pd.DataFrame:
        r"""The coefficient table of S6.2.1 and S6.4.1. Cf. ``summary()``.

        S6.4.1: "the four Pareto/NBD base parameters (:math:`r, \alpha, s,
        \beta`) are not reported with any z- and p-values. As these parameters
        are constrained to be strictly positive, the model definition fixes
        their lower bound at 0. Thus, a null hypothesis of :math:`\theta = 0`
        lies outside the admissible parameter space." Those rows are ``NaN``
        here for the same reason; covariate coefficients, which are not
        constrained, carry both.

        Examples
        --------
        >>> from clvtools import ClvData, load_apparel_trans
        >>> from clvtools.gg import fit_gg
        >>> spend = ClvData(load_apparel_trans(), estimation_split=104).spending_summary()
        >>> print(fit_gg(spend["x"], spend["Spending"]).summary().round(3).to_string())
               Estimate  Std. Error  z-val  Pr(>|z|)
        p         3.099       0.568    NaN       NaN
        q         5.654       0.846    NaN       NaN
        gamma    56.504      18.602    NaN       NaN
        """
        errors = self.standard_errors()
        table = pd.DataFrame(
            {
                "Estimate": list(self),
                "Std. Error": [errors[n] for n in self.names],
            },
            index=self.names,
        )
        is_covariate = np.array(
            [n.startswith(COVARIATE_PREFIXES) for n in self.names]
        )
        z = np.where(
            is_covariate, table["Estimate"] / table["Std. Error"], np.nan
        )
        table["z-val"] = z
        table["Pr(>|z|)"] = 2 * (1 - stats.norm.cdf(np.abs(z)))
        return table


@dataclass(frozen=True)
class LikelihoodRatioTest:
    """The result of :func:`likelihood_ratio_test`."""

    n_parameters_restricted: int
    n_parameters_unrestricted: int
    log_likelihood_restricted: float
    log_likelihood_unrestricted: float
    df: int
    statistic: float
    p_value: float

    def __repr__(self) -> str:
        return (
            f"LikelihoodRatioTest(df={self.df}, "
            f"chisq={self.statistic:.4g}, p={self.p_value:.4g})"
        )


def likelihood_ratio_test(restricted, unrestricted) -> LikelihoodRatioTest:
    r"""Compare two nested fits. Cf. ``lrtest()`` in S6.5.3.

    S6.5.3 uses it on an equality constraint: "A likelihood ratio test helps to
    evaluate if adding an equality constraint changes the model fit [...] the
    test results indicate whether the parameter of a covariate significantly
    differs between the attrition and transaction process."

    The statistic is :math:`2(\ell_u - \ell_r)`, referred to a :math:`\chi^2`
    with as many degrees of freedom as the constraint removes parameters.

    A regularized fit reports a penalised log-likelihood, which is not a
    likelihood; the unpenalised value is used when a fit carries one, so that
    the two sides are comparable.

    Parameters
    ----------
    restricted
        The fit with fewer free parameters -- the constrained one.
    unrestricted
        The fit that nests it.

    Examples
    --------
    >>> from clvtools import ClvData, ClvDataStaticCov
    >>> from clvtools import load_apparel_static_cov, load_apparel_trans
    >>> from clvtools.pnbd import fit_pnbd_staticcov
    >>> data = ClvDataStaticCov(
    ...     ClvData(load_apparel_trans(), time_unit="week", estimation_split=104),
    ...     load_apparel_static_cov(),
    ...     names_cov_life=["Gender", "Channel"],
    ...     names_cov_trans=["Gender", "Channel"])
    >>> free = fit_pnbd_staticcov(data, hessian=False)
    >>> tied = fit_pnbd_staticcov(data, names_cov_constr=["Gender"], hessian=False)
    >>> test = likelihood_ratio_test(tied, free)
    >>> test.df, round(test.statistic, 3), round(test.p_value, 6)
    (1, 10.943, 0.00094)
    """
    def value(fit) -> float:
        unpenalised = getattr(fit, "unpenalised_log_likelihood", None)
        return float(fit.log_likelihood if unpenalised is None else unpenalised)

    k_restricted = int(restricted.n_parameters)
    k_unrestricted = int(unrestricted.n_parameters)
    df = k_unrestricted - k_restricted
    if df <= 0:
        raise ValueError(
            "the unrestricted model must have more parameters than the "
            f"restricted one; got {k_unrestricted} against {k_restricted}"
        )
    statistic = 2 * (value(unrestricted) - value(restricted))
    return LikelihoodRatioTest(
        n_parameters_restricted=k_restricted,
        n_parameters_unrestricted=k_unrestricted,
        log_likelihood_restricted=value(restricted),
        log_likelihood_unrestricted=value(unrestricted),
        df=df,
        statistic=float(statistic),
        p_value=float(stats.chi2.sf(statistic, df)),
    )
