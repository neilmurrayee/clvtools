r"""Table 2's two entry points: ``latentAttrition()`` and ``spending()``.

Every code chunk in the paper's walkthrough estimates a model through one of
these two, naming a ``family`` and -- where there are covariates -- a formula:

.. code-block:: r

    est.pnbd <- latentAttrition(family = pnbd, data = clv.apparel)
    est.pnbd.static <- latentAttrition(
        formula = ~ Gender + Channel | Gender + Channel,
        family = pnbd, data = clv.static)
    est.gg <- spending(family = gg, data = clv.apparel)

The per-family ``fit_*`` functions underneath take exactly what they need --
``(x, t_x, T)``, or a covariate-bearing data object -- which is the right shape
for testing an estimator but makes the caller pick the function *and* remember
which of three covariate variants applies. These two pick for them, from the
data object's own type: plain data gets the plain fit, a
:class:`~clvtools.data.ClvDataStaticCov` gets the time-invariant covariate fit,
a :class:`~clvtools.data.ClvDataDynCov` the time-varying one.
"""

from __future__ import annotations

from typing import Any

from clvtools import bgnbd, ggomnbd
from clvtools.data import ClvData, ClvDataDynCov, ClvDataStaticCov
from clvtools.gg import GgParams, fit_gg
from clvtools.pnbd import fit_pnbd, fit_pnbd_correlated, fit_pnbd_staticcov
from clvtools.pnbd.dyncov import fit_pnbd_dyncov

__all__ = ["latent_attrition", "parse_formula", "spending"]

#: The three latent attrition families of Table 4, by the name the paper gives
#: them, with their plain and time-invariant-covariate estimators.
FAMILIES: dict[str, dict[str, Any]] = {
    "pnbd": {"plain": fit_pnbd, "staticcov": fit_pnbd_staticcov},
    "bgnbd": {"plain": bgnbd.fit_bgnbd, "staticcov": bgnbd.fit_bgnbd_staticcov},
    "ggomnbd": {
        "plain": ggomnbd.fit_ggomnbd, "staticcov": ggomnbd.fit_ggomnbd_staticcov,
    },
}


def _split_terms(side: str) -> list[str]:
    """Split one side of a formula on ``+``, but not inside parentheses.

    ``I(log(Channel + 2))`` is one term; the ``+`` it contains belongs to the
    expression. Anything unbalanced is left to the caller to reject by name.

    >>> _split_terms("Gender + Channel")
    ['Gender ', ' Channel']
    >>> _split_terms("Gender + I(log(Channel + 2))")
    ['Gender ', ' I(log(Channel + 2))']
    """
    terms, depth, start = [], 0, 0
    for i, character in enumerate(side):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "+" and depth == 0:
            terms.append(side[start:i])
            start = i + 1
    terms.append(side[start:])
    return terms


def parse_formula(formula: str) -> tuple[list[str] | None, list[str] | None]:
    """Split S6.4's covariate formula into attrition and transaction names.

    S6.4 writes the two processes either side of a bar, attrition first:
    ``~ Gender + Channel | Gender + Channel``. A bare ``.`` on either side
    means every covariate the data object carries, as it does in R, and comes
    back as ``None`` for the fit to fill in.

    A term wrapped in ``I(...)`` is an expression rather than a column name,
    as it is in R, and survives splitting whole -- the ``+`` inside it is
    arithmetic, not a term separator. :meth:`ClvDataStaticCov.with_covariates`
    evaluates it.

    Examples
    --------
    >>> parse_formula("~ Gender + Channel | Gender")
    (['Gender', 'Channel'], ['Gender'])
    >>> parse_formula("~ . | .")
    (None, None)
    >>> parse_formula("~ Gender | I(log(Channel + 2))")
    (['Gender'], ['I(log(Channel + 2))'])
    """
    body = formula.strip()
    body = body.removeprefix("~")
    parts = body.split("|")
    if len(parts) != 2:
        raise ValueError(
            "a covariate formula needs both processes, attrition first: "
            "'~ Gender + Channel | Gender + Channel'"
        )

    def names(side: str) -> list[str] | None:
        side = side.strip()
        if side == ".":
            return None
        found = [n for n in (t.strip() for t in _split_terms(side)) if n]
        if not found:
            raise ValueError(f"no covariates named in {side!r}")
        return found

    return names(parts[0]), names(parts[1])


def _family_name(family) -> str:
    """Accept the module, or its name, as CLVTools accepts the generic."""
    name = getattr(family, "__name__", family)
    name = str(name).rsplit(".", 1)[-1]
    if name not in FAMILIES:
        raise ValueError(
            f"unknown family {name!r}; Table 4 has {', '.join(FAMILIES)}"
        )
    return name


def latent_attrition(
    family,
    data: ClvData,
    formula: str | None = None,
    use_cor: bool = False,
    **kwargs,
):
    """Estimate a latent attrition model. Cf. ``latentAttrition()``.

    Parameters
    ----------
    family
        ``clvtools.pnbd``, ``clvtools.bgnbd`` or ``clvtools.ggomnbd`` -- or the
        name of one.
    data
        The transaction data. Its type chooses the estimator: covariate data
        gets the covariate model, as it does in S6.4.
    formula
        S6.4's ``~ life | trans``. Only meaningful with covariate data, where
        omitting it uses every covariate the data object carries.
    use_cor
        S6.5.2's ``use.cor``: fit the Sarmanov correlation between the two
        processes. Pareto/NBD without covariates only, as Table 4 marks it.
    **kwargs
        Passed through to the underlying fit -- ``names_cov_constr``,
        ``reg_lambdas``, ``start``, ``method``, ``maxiter``, ``hessian``.

    Examples
    --------
    S6.2.1's first estimation, and S6.4.1's with covariates:

    >>> from clvtools import ClvData, ClvDataStaticCov, pnbd
    >>> from clvtools import load_apparel_static_cov, load_apparel_trans
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> fit = latent_attrition(family=pnbd, data=data, hessian=False)
    >>> [round(v, 3) for v in fit]
    [1.449, 48.635, 0.561, 46.88...]

    >>> covariates = ClvDataStaticCov(
    ...     data, load_apparel_static_cov(),
    ...     names_cov_life=["Gender", "Channel"],
    ...     names_cov_trans=["Gender", "Channel"])
    >>> with_covariates = latent_attrition(
    ...     formula="~ Gender + Channel | Gender + Channel",
    ...     family=pnbd, data=covariates, hessian=False)
    >>> round(with_covariates.coefficients["trans.Channel"], 3)
    0.624
    """
    name = _family_name(family)
    names_life, names_trans = (
        parse_formula(formula) if formula is not None else (None, None)
    )

    if isinstance(data, ClvDataDynCov):
        if name != "pnbd":
            raise ValueError(
                f"Table 4 gives time-varying covariates to the Pareto/NBD "
                f"alone; {name} takes time-invariant ones"
            )
        if use_cor:
            raise ValueError("process correlation is for the plain Pareto/NBD")
        if names_life is not None or names_trans is not None:
            data = data.with_covariates(names_life, names_trans)
        return fit_pnbd_dyncov(
            data.walks(),
            names_cov_life=data.names_cov_life,
            names_cov_trans=data.names_cov_trans,
            **kwargs,
        )

    if isinstance(data, ClvDataStaticCov):
        if use_cor:
            raise ValueError("process correlation is for the plain Pareto/NBD")
        if names_life is not None or names_trans is not None:
            data = data.with_covariates(names_life, names_trans)
        return FAMILIES[name]["staticcov"](data, **kwargs)

    if formula is not None and (names_life or names_trans):
        raise ValueError(
            "the data has no covariates: build it with ClvDataStaticCov or "
            "ClvDataDynCov before naming any in a formula"
        )
    cbs = data.customer_summary()
    fit = fit_pnbd_correlated if use_cor else FAMILIES[name]["plain"]
    if use_cor and name != "pnbd":
        raise ValueError(
            f"Table 4 gives process correlation to the Pareto/NBD alone, "
            f"not to the {name}"
        )
    return fit(cbs["x"], cbs["t_x"], cbs["T"], **kwargs)


def spending(
    family, data: ClvData, remove_first_transaction: bool = True, **kwargs
) -> GgParams:
    """Estimate a spending model. Cf. ``spending()``.

    S6.2.3: CLVTools "by default does not use the first transaction when
    estimating a spending model because in many cases this transaction has been
    found to be atypical for future purchases". S6.3.4 turns that off when the
    prediction it feeds counts the first transaction too.

    Examples
    --------
    >>> from clvtools import ClvData, gg, load_apparel_trans
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> [round(v, 3) for v in spending(family=gg, data=data, hessian=False)]
    [3.099, 5.654, 56.504]
    """
    name = getattr(family, "__name__", family)
    if str(name).rsplit(".", 1)[-1] != "gg":
        raise ValueError(
            "the Gamma-Gamma is the only spending model in the paper (S3.5)"
        )
    spend = data.spending_summary(remove_first_transaction=remove_first_transaction)
    return fit_gg(spend["x"], spend["Spending"], **kwargs)
