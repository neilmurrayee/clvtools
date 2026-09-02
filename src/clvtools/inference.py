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

import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from clvtools._validate import ConvergenceWarning

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
            ei = np.zeros(n)
            ei[i] = h[i]
            ej = np.zeros(n)
            ej[j] = h[j]
            out[i, j] = out[j, i] = (
                fn(at + ei + ej) - fn(at + ei - ej)
                - fn(at - ei + ej) + fn(at - ei - ej)
            ) / (4 * h[i] * h[j])
    return out


class Fitted:
    """The generics every fitted model shares. Cf. ``clv.fitted`` in Table 2.

    Mixed into each family's parameter class. It needs three things from one:
    ``names``, iteration over the estimates in that order, and ``hessian``.
    All three are declared below rather than left to prose, because
    ``py.typed`` says the annotations here can be relied on.
    """

    #: Curvature of the *negative* log-likelihood at the optimum, in the
    #: natural parameters, ordered as :attr:`names` lists them -- or ``None``
    #: when the fit was asked to skip it. Supplied by each family, usually as a
    #: dataclass field; an annotation rather than a property so a frozen
    #: dataclass can still assign to it.
    hessian: NDArray[np.float64] | None

    @property
    def names(self) -> list[str]:  # pragma: no cover - each class provides it
        raise NotImplementedError

    def __iter__(self) -> Iterator[float]:  # pragma: no cover - each class provides it
        raise NotImplementedError

    @property
    def coefficients(self) -> dict[str, float]:
        """The estimates by name. Cf. ``coef()``."""
        return dict(zip(self.names, self, strict=True))

    #: The maximised log-likelihood, and the number of customers it was
    #: maximised over. Annotations rather than properties, as :attr:`hessian`
    #: is and for the same reason: every family supplies them as dataclass
    #: fields, and a property here would shadow the field.
    log_likelihood: float
    n_customers: int

    @property
    def _comparable_log_likelihood(self) -> float:
        """The log-likelihood :attr:`aic` and :attr:`bic` should be built on.

        For most fits that is simply :attr:`log_likelihood`. A regularized one
        minimises eq. (13)'s **penalised mean**, which is not on the same scale
        as an unregularized model's sum and must not be compared with it, so
        the three covariate classes override this with the unpenalised value
        they also carry.
        """
        return self.log_likelihood

    @property
    def n_parameters(self) -> int:
        """How many estimates the fit reports, which is how many it names."""
        return len(self.names)

    @property
    def aic(self) -> float:
        """Akaike's criterion. Cf. ``AIC()``."""
        return 2 * self.n_parameters - 2 * self._comparable_log_likelihood

    @property
    def bic(self) -> float:
        """The Bayesian criterion, on the number of customers. Cf. ``BIC()``."""
        return (
            self.n_parameters * np.log(self.n_customers)
            - 2 * self._comparable_log_likelihood
        )

    def _covariance(self) -> NDArray[np.float64]:
        # `getattr` rather than `self.hessian`: the dyncov fit reports no
        # Hessian at all, so the attribute can be absent as well as None.
        hessian = getattr(self, "hessian", None)
        if hessian is None:
            raise ValueError(
                "fit with hessian=True to obtain standard errors, a covariance "
                "matrix or confidence intervals"
            )
        # A covariance matrix is the inverse of a Hessian that is positive
        # definite. Where it is not, the inversion still returns something and
        # ``sqrt`` of a negative diagonal entry is ``nan``, which is how the
        # BG/NBD covariate fit shipped `life.Gender` = nan beside
        # `life.Channel` = 0.594 with ``converged = True`` -- on the ridge the
        # README documents, where `a + b` runs to hundreds of thousands and one
        # direction is genuinely flat. The number is not wrong so much as
        # absent, and it should say so. Finding 9 of the outside review.
        # A regularized fit's curvature is mostly the penalty's, not the
        # data's, and neither this package nor CLVTools said so anywhere. At
        # lambda = 10 on the apparel cohort every covariate standard error here
        # is within 4% of the penalty-only 1/sqrt(2*lambda), and CLVTools'
        # four are identical to twelve significant figures. Whoever reads the
        # number should be told what it is made of. See the README's findings
        # and docs/backlog.md item 22.
        lambdas = getattr(self, "reg_lambdas", None)
        if lambdas is not None and any(lambdas):
            warnings.warn(
                f"these standard errors come from the regularized objective "
                f"(reg_lambdas={tuple(lambdas)}), whose curvature is dominated "
                "by the penalty rather than by the data: they are ridge "
                "standard errors, they shrink towards 1/sqrt(2*lambda), and "
                "they are not comparable with an unregularized fit's",
                ConvergenceWarning,
                stacklevel=3,
            )

        matrix = np.asarray(hessian, dtype=float)
        if not np.all(np.isfinite(matrix)):
            # The GGom/NBD covariate fit reaches parameters where differencing
            # the likelihood gives non-finite second derivatives -- its `b` is
            # 8.1e-07 on this data, and the surface around it is not resolvable
            # at any step this package uses. `eigvalsh` raises `LinAlgError`
            # from inside numpy on such a matrix, which is not an answer a
            # caller can act on, so say what happened instead.
            bad = [
                name
                for name, row in zip(self.names, matrix, strict=True)
                if not np.all(np.isfinite(row))
            ]
            warnings.warn(
                "the Hessian has non-finite entries, so no standard error is "
                f"trustworthy; the rows involved are {bad}. The likelihood "
                "could not be differenced there -- usually a parameter pinned "
                "at a boundary.",
                ConvergenceWarning,
                stacklevel=3,
            )
            return np.full_like(matrix, np.nan)

        eigenvalues = np.linalg.eigvalsh(matrix)
        if not np.all(eigenvalues > 0):
            flat = [
                name
                for name, value in zip(self.names, np.diag(matrix), strict=True)
                if not value > 0
            ]
            warnings.warn(
                "the Hessian is not positive definite (smallest eigenvalue "
                f"{eigenvalues.min():.3g}), so these standard errors are not "
                "trustworthy and some may be NaN"
                + (f"; flat directions include {flat}" if flat else "")
                + ". This usually means a parameter is not identified by the "
                "data -- see the README on the BG/NBD's beta parameters under "
                "covariates.",
                ConvergenceWarning,
                stacklevel=3,
            )
        return np.linalg.inv(hessian)

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
        return dict(zip(self.names, np.sqrt(np.diag(self._covariance())), strict=True))

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
        r  0.97...  1.92...
        alpha  33.95...  63.31...
        s  0.03...  1.09...
        beta  -22.9...  116.6...
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
        p  3.09...  0.5...  NaN  NaN
        q  5.65...  0.8...  NaN  NaN
        gamma  56.50...  18.6...  NaN  NaN
        """
        errors = self.standard_errors()
        table = pd.DataFrame(
            {
                "Estimate": list(self),
                "Std. Error": [errors[n] for n in self.names],
            },
            index=self.names,
        )
        # A z-value is reported where a null of zero is admissible. That is
        # every covariate coefficient, and also the Sarmanov correlation
        # ``m``: S6.5.2's whole question is whether the two processes are
        # independent, which is exactly ``m = 0``, and CLVTools prints one for
        # it. The four model parameters are "constrained to be strictly
        # positive" (S6.4.1), so a null of zero lies outside the space and
        # their rows stay NaN. Finding 8 of ``docs/review-2026-09-02.md``.
        testable = np.array(
            [n.startswith(COVARIATE_PREFIXES) or n == "m" for n in self.names]
        )
        z = np.where(
            testable, table["Estimate"] / table["Std. Error"], np.nan
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
    n_r = getattr(restricted, "n_customers", None)
    n_u = getattr(unrestricted, "n_customers", None)
    if n_r is not None and n_u is not None and n_r != n_u:
        raise ValueError(
            "a likelihood ratio test compares two fits of the same data; "
            f"these were fitted on {n_r} and {n_u} customers"
        )

    statistic = 2 * (value(unrestricted) - value(restricted))
    if statistic < -1e-6:
        # A restricted model is a special case of the unrestricted one, so it
        # cannot fit better. A negative statistic means either that the two are
        # not nested -- in which case the chi-square has no meaning and this
        # used to return one anyway (finding 12 of the outside review) -- or
        # that the unrestricted fit stopped somewhere worse, which is worth
        # knowing before reading a p-value off it.
        raise ValueError(
            f"the restricted model fits better than the unrestricted one "
            f"({value(restricted):.6f} against {value(unrestricted):.6f}), so "
            "they are not nested in the order given, or the unrestricted fit "
            "did not converge; a likelihood ratio test needs a genuine "
            "restriction"
        )
    return LikelihoodRatioTest(
        n_parameters_restricted=k_restricted,
        n_parameters_unrestricted=k_unrestricted,
        log_likelihood_restricted=value(restricted),
        log_likelihood_unrestricted=value(unrestricted),
        df=df,
        statistic=float(statistic),
        p_value=float(stats.chi2.sf(statistic, df)),
    )
