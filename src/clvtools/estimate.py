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

from typing import Any, TypeVar

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

    A side that is exactly ``.`` comes back as ``None`` -- "everything the data
    carries". Mixed with other terms it survives as a ``"."`` term, because what
    it expands *to* depends on the data and this function does not see it;
    :func:`_narrowed` does the expanding:

    >>> parse_formula("~ . | . + I(Gender + 1)")
    (None, ['.', 'I(Gender + 1)'])

    ``.`` also takes exclusions, written as R writes them:

    >>> parse_formula("~ . - Gender | .")
    (['.', '-Gender'], None)
    """
    body = formula.strip()
    # A left-hand side used to be swallowed into the first covariate name:
    # `y ~ Gender | Channel` parsed to `(['y ~ Gender'], ['Channel'])` and then
    # failed complaining about a covariate called "y ~ Gender". Spec `FI-13`
    # asks for it to fail, and it did -- for the wrong reason.
    #
    # The rule is "nothing before the tilde", not "starts with a tilde": the
    # tilde is optional here and `Gender | Gender` is a valid formula, which
    # `TestFormula::test_the_tilde_is_optional` caught the first draft of this
    # breaking.
    if "~" in body and body.split("~", 1)[0].strip():
        raise ValueError(
            f"a covariate formula has no left-hand side: got {formula!r}, with "
            f"{body.split('~', 1)[0].strip()!r} before the '~'. The response is "
            f"the data, not a column"
        )
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
        terms = [t.strip() for t in _expand_exclusions(_split_terms(side))]
        if not [t for t in terms if t]:
            raise ValueError(f"no covariates named in {side!r}")
        # A `+` with nothing after it used to be dropped silently, so
        # `~ Gender + | Gender` fitted on Gender alone -- the shape a half-
        # finished edit leaves behind, answering as though it were finished.
        if not all(terms):
            raise ValueError(
                f"empty covariate term in {side!r}: a '+' with nothing after it"
            )
        constrained = [t for t in terms if t.startswith("constraint(")]
        if constrained:
            # R names tied covariates inside the formula; here they are an
            # argument, so `constraint(Gender)` used to be read as a column of
            # that name. Spec `FI-15`.
            raise ValueError(
                f"{constrained[0]!r}: R's constraint() has no counterpart in a "
                f"formula here. Pass names_cov_constr=['Gender'] to the fit "
                f"instead -- see S6.5.3 and the README's findings"
            )
        return terms

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


def _reject_use_cor(name: str, covariates: bool) -> None:
    """S6.5.2's ``use.cor`` is offered on the plain Pareto/NBD, and only there.

    Two ways to be outside that, and each is worth naming separately: the data
    carries covariates, or the family is not the Pareto/NBD. Table 4 marks both
    as unavailable. Which of the two the caller hit is the caller's to know --
    it is already dispatching on the data's type -- so the covariate case
    arrives as a flag rather than being re-derived here.
    """
    if covariates:
        raise ValueError("process correlation is for the plain Pareto/NBD")
    if name != "pnbd":
        raise ValueError(
            f"Table 4 gives process correlation to the Pareto/NBD alone, "
            f"not to the {name}"
        )


#: The two covariate data classes, as a constrained type variable rather than
#: PEP 695's ``def _narrowed[Data: (...)]``. That syntax is what this was
#: written as, and it resolves under 3.12.11 but raises ``NameError: name
#: 'Data' is not defined`` from ``typing.get_type_hints()`` under 3.12.3 --
#: the type-parameter scope and ``from __future__ import annotations`` did not
#: cooperate until later in the 3.12 series. ``requires-python`` says ">=3.12",
#: `py.typed` promises these annotations are usable, and
#: ``TestTy.test_the_shipped_annotations_resolve`` is what holds us to it, so
#: the spelling that works on every supported version is the right one.
_CovariateData = TypeVar("_CovariateData", ClvDataStaticCov, ClvDataDynCov)


def _expand_exclusions(terms: list[str]) -> list[str]:
    """Split R's ``a - b`` exclusions into separate ``-b`` terms.

    ``_split_terms`` splits on ``+`` only, so ``~ . - Gender | .`` arrived as
    the single term ``". - Gender"`` and was looked up as a column of that
    name. Spec `FI-15` names "``.`` with exclusions" as one of its twelve
    claims.

    Only ``.`` can be excluded *from*, which is R's rule too: subtracting from
    an explicit list is the same as not listing it, so there is nothing to
    express. That is checked in :func:`_expanded`, where the names are known.
    """
    out: list[str] = []
    for term in terms:
        # `I(...)` may contain a minus of its own -- `I(Channel - 1)` -- and is
        # an expression rather than a list of terms, so it is left alone.
        if term.strip().startswith("I("):
            out.append(term)
            continue
        head, *excluded = term.split("-")
        out.append(head)
        out.extend(f"-{name.strip()}" for name in excluded)
    # Empty terms are *kept*: `~ Gender + | Gender` has to reach the "a '+' with
    # nothing after it" check in `parse_formula`, and filtering them here
    # silently disabled it -- `TestAnEmptyCovariateTerm` caught that.
    return out


def _require_clv_data(data, entry_point: str) -> None:
    """Refuse a raw frame before it reaches a method it does not have.

    Spec `FI-13` and `FI-14` both ask that the entry points fail on data that
    is not a ``clv.data``. They did -- with
    ``AttributeError: 'DataFrame' object has no attribute 'customer_summary'``,
    which names an internal method rather than the thing the caller got wrong,
    and which reads like a bug in the library rather than in the call.
    """
    if not isinstance(data, ClvData):
        raise TypeError(
            f"{entry_point}() needs a ClvData, not {type(data).__name__}: "
            f"wrap the transactions first, as ClvData(transactions, "
            f"time_unit=..., estimation_split=...)"
        )


def _narrowed(  # noqa: UP047 - PEP 695 here breaks get_type_hints on 3.12.3; see above
    data: _CovariateData,
    names_life: list[str] | None,
    names_trans: list[str] | None,
) -> _CovariateData:
    """The covariate data a formula asks for, or all of it when none was given.

    Generic over the two covariate data classes because each narrows to its own
    type -- a :class:`~clvtools.data.ClvDataDynCov` must still be one
    afterwards, or the time-varying fit has nothing to walk.
    """
    if names_life is None and names_trans is None:
        return data
    return data.with_covariates(
        _expanded(names_life, data.names_cov_life),
        _expanded(names_trans, data.names_cov_trans),
    )


def _expanded(named: list[str] | None, available: list[str]) -> list[str] | None:
    """A side's terms with ``.`` replaced by every covariate the data carries.

    ``~ . | . + I(Gender + 1)`` is one of spec `FI-04`'s three claims: every
    covariate on the attrition side, and every covariate *plus* a transformed
    term on the transaction side. A bare ``.`` never reaches here -- it arrives
    as ``None`` and means the same thing -- but a ``.`` beside other terms used
    to be passed through as a **literal column name**, so the formula failed
    looking for a covariate called ``"."``.

    Order is the data's own, with the named terms after it, and a covariate
    named twice is selected once -- ``~ . + Gender | .`` asks for nothing more
    than ``~ . | .``.
    """
    excluded = [t[1:] for t in (named or []) if t.startswith("-")]
    if named is None or "." not in named:
        if excluded:
            raise ValueError(
                f"cannot exclude {excluded[0]!r} from a side that does not use "
                f"'.': list the covariates you want instead"
            )
        return named
    unknown = [name for name in excluded if name not in available]
    if unknown:
        raise ValueError(
            f"cannot exclude {unknown[0]!r}: the data carries "
            f"{', '.join(available)}"
        )
    out = [name for name in available if name not in excluded]
    out.extend(
        term for term in named
        if term != "." and not term.startswith("-") and term not in out
    )
    return out


def _fit_dyncov(name: str, data: ClvDataDynCov, **kwargs):
    """S6.4.2's time-varying covariates -- the Pareto/NBD alone, per Table 4."""
    if name != "pnbd":
        raise ValueError(
            f"Table 4 gives time-varying covariates to the Pareto/NBD "
            f"alone; {name} takes time-invariant ones"
        )
    return fit_pnbd_dyncov(
        data.walks(),
        names_cov_life=data.names_cov_life,
        names_cov_trans=data.names_cov_trans,
        **kwargs,
    )


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
    _require_clv_data(data, "latent_attrition")
    names_life, names_trans = (
        parse_formula(formula) if formula is not None else (None, None)
    )
    if use_cor:
        _reject_use_cor(name, isinstance(data, ClvDataStaticCov | ClvDataDynCov))

    if isinstance(data, ClvDataDynCov):
        return _fit_dyncov(
            name, _narrowed(data, names_life, names_trans), **kwargs
        )

    if isinstance(data, ClvDataStaticCov):
        return FAMILIES[name]["staticcov"](
            _narrowed(data, names_life, names_trans), **kwargs
        )

    if formula is not None:
        # `names_life or names_trans` was the old test, which let `~ . | .`
        # through: it parses to (None, None), the "use every covariate" marker,
        # and there are none to use. A formula on plain data is a mistake
        # whether it names covariates or asks for all of them.
        raise ValueError(
            "the data has no covariates, so the formula has nothing to select: "
            "build it with ClvDataStaticCov or ClvDataDynCov first"
        )
    cbs = data.customer_summary()
    fit = fit_pnbd_correlated if use_cor else FAMILIES[name]["plain"]
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
    _require_clv_data(data, "spending")
    spend = data.spending_summary(remove_first_transaction=remove_first_transaction)
    return fit_gg(spend["x"], spend["Spending"], **kwargs)
