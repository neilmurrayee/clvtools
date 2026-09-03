r"""S6.3.3 - quantifying uncertainty by bootstrapping.

S6.3.3: "Confidence intervals help users to quantify the uncertainty of a
statistical model […] **CLVTools** facilitates the derivation of confidence
intervals based on a bootstrapping routine. […] Customers, together with their
entire purchasing history, are sampled with replacement. A new model is
estimated on the sampled data with the same specification and optimization
options as the given fitted model."

Two details of that description shape the implementation.

A customer drawn twice must count as two customers, not one seen twice. So the
duplicates are given distinct identifiers while the sampling is going on, and
the suffix is stripped again before the quantiles are taken -- which is how a
customer ends up with several draws to form an interval from.

And the periods are pinned. S6.3.3: "simply sampling customers with their orders
and creating a data object may yield different estimation and holdout periods
because the end of the data is determined by the last order. This method makes
sure that the estimation and holdout periods are preserved as in the original
data. In this way, the model inputs […] in each iteration remain for each
customer the same as in the original data."

What the intervals do and do not cover is worth keeping in view. S6.3.3:
"bootstrapping only accounts for uncertainty in model parameters (epistemic
uncertainty), and not sampling variability in the actual outcomes (aleatoric
uncertainty)."
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from clvtools._validate import ConvergenceWarning
from clvtools.data import ClvData, ClvDataDynCov, ClvDataStaticCov

__all__ = [
    "BOOTSTRAP_SUFFIX",
    "bootstrap_apply",
    "bootstrap_data",
    "confidence_intervals",
    "predict_intervals",
]

#: Marks a customer's duplicate draws so they are distinct while resampling.
#: Stripped again before quantiles are taken.
BOOTSTRAP_SUFFIX = "_BOOTSTRAP_ID_"


def bootstrap_data(data: ClvData, ids: Sequence[str]) -> ClvData:
    r"""Rebuild ``data`` from a (possibly repeated) list of customer ids.

    Repeats become separate customers, suffixed with :data:`BOOTSTRAP_SUFFIX`.
    The estimation and holdout boundaries are carried over from ``data`` rather
    than re-derived, since S6.3.3 requires each customer's
    :math:`(x, t_x, T)` to be unchanged by resampling -- and re-deriving them
    would move the window whenever the last transaction failed to be drawn.
    """
    transactions = data.transactions
    known = set(transactions["Id"].unique())
    missing = [i for i in ids if i not in known]
    if missing:
        raise ValueError(
            f"{len(missing)} sampled ids are not in the data, e.g. {missing[:3]}"
        )

    # One pass over the frame, then positional lookups. Filtering
    # ``transactions["Id"] == customer`` inside the loop walked all 6,696 CDNOW
    # rows once per drawn customer: 0.965 s a draw against 0.134 s for the
    # summary and the Pareto/NBD fit together, so a hundred draws spent about
    # 95 seconds rebuilding data and 13 fitting it. Finding 11 of
    # ``docs/review-2026-09-02.md``.
    positions = transactions.groupby("Id", sort=False).indices

    seen: dict[str, int] = {}
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for customer in ids:
        occurrence = seen.get(customer, 0)
        seen[customer] = occurrence + 1
        where = positions[customer]
        rows.append(where)
        label = (
            customer if not occurrence
            else f"{customer}{BOOTSTRAP_SUFFIX}{occurrence}"
        )
        labels.append(np.full(where.size, label, dtype=object))

    resampled = transactions.iloc[np.concatenate(rows)].copy()
    resampled["Id"] = np.concatenate(labels)
    resampled = resampled.reset_index(drop=True)

    return ClvData(
        resampled,
        time_unit=data.time_unit,
        estimation_split=data.estimation_end,
        # Pinned, so the resampled window matches the original even if the last
        # transaction was not drawn.
        data_end=data.data_end,
        name_price="Price" if data.has_spending else None,
    )


def _resample_covariates(
    data: ClvDataStaticCov, rebuilt: ClvData, ids: Sequence[str]
) -> ClvDataStaticCov:
    """Carry the covariate rows across to the resampled customers."""
    frames, seen = [], {}
    for customer in ids:
        occurrence = seen.get(customer, 0)
        seen[customer] = occurrence + 1
        block_life = data._cov_life.loc[[customer]].reset_index()
        block_trans = data._cov_trans.loc[[customer]].reset_index()
        if occurrence:
            new_id = f"{customer}{BOOTSTRAP_SUFFIX}{occurrence}"
            block_life["Id"] = new_id
            block_trans["Id"] = new_id
        frames.append((block_life, block_trans))

    cov_life = pd.concat([f[0] for f in frames], ignore_index=True)
    cov_trans = pd.concat([f[1] for f in frames], ignore_index=True)
    return ClvDataStaticCov(
        rebuilt, cov_life, cov_trans,
        names_cov_life=data.names_cov_life,
        names_cov_trans=data.names_cov_trans,
    )


def _drawer(sample, rng) -> Callable[[np.ndarray], np.ndarray]:
    """The customer sampler, given the generator ``seed`` produced.

    A caller's own sampler used to be invoked with the pool alone, so ``seed``
    did nothing whenever one was passed: the runs were not reproducible and
    nothing said so. It is offered the generator when it can take one, and a
    one-argument sampler still works -- that is the shape
    ``?clv.bootstrapped.apply``'s own example has. Finding 11 of
    ``docs/review-2026-09-02.md``.
    """
    if sample is None:
        return lambda pool: rng.choice(pool, size=len(pool), replace=True)
    if len(inspect.signature(sample).parameters) >= 2:
        return lambda pool: sample(pool, rng)
    return sample


def _kept(results: list, failures: list[str], num_boots: int) -> list:
    """The draws that survived, having said what happened to the ones that did not.

    An exception on draw 3 of 5 used to discard the two that had already
    succeeded. A resample is a random object: some of them are degenerate, and
    losing the whole run to one of those is the wrong trade. What is not
    acceptable is losing it silently, so every failure is counted and named
    here -- as an exception when nothing survived, and as a
    :class:`~clvtools._validate.ConvergenceWarning` when something did. Only
    one of the two: if nothing survived, the exception has already said so and
    a warning beside it is noise.

    >>> _kept([1, 2], [], num_boots=2)
    [1, 2]
    """
    if not results:
        raise ValueError(
            f"all {num_boots} bootstrap draws failed. First: {failures[0]}"
        )
    if failures:
        warnings.warn(
            f"{len(failures)} of {num_boots} bootstrap draws failed and were "
            f"dropped; {len(results)} were kept. First: {failures[0]}",
            ConvergenceWarning,
            stacklevel=3,
        )
    return results


def bootstrap_apply(
    data: ClvData,
    apply: Callable[[ClvData], object],
    num_boots: int = 100,
    # Either ``sample(pool)`` or ``sample(pool, rng)``; the second is offered
    # so that ``seed`` reaches a caller's own sampler, and the first is what
    # ``?clv.bootstrapped.apply``'s example has. ``...`` rather than a union of
    # two signatures because that is exactly what is accepted here.
    sample: Callable[..., np.ndarray] | None = None,
    seed: int | None = None,
) -> list:
    r"""Resample customers, rebuild the data, and apply a function each time.

    The counterpart of ``clv.bootstrapped.apply``. S6.3.3: "Given an estimated
    model, it samples new data from the transactions stored in it, re-fits the
    model on it, and then applies a user-given method on the newly estimated
    model."

    Here ``apply`` receives the *resampled data*, and is expected to do its own
    fitting -- which keeps this independent of any one model family, and lets
    the caller decide what to compute. The paper's own example returns the
    tracking-plot data from each iteration and builds a ribbon from the
    quantiles.

    Parameters
    ----------
    apply
        Called once per iteration with the resampled :class:`ClvData`.
    sample
        Draws the ids for one iteration. Defaults to sampling with replacement
        to the original size, as S6.3.3 describes.
    seed
        For reproducibility. The paper's example sets ``set.seed(1)``.

    Examples
    --------
    Ten iterations, recording each resample's repeat-transaction count:

    >>> from clvtools import ClvData, load_apparel_trans
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> totals = bootstrap_apply(
    ...     data, lambda d: int(d.customer_summary()["x"].sum()),
    ...     num_boots=10, seed=1)
    >>> len(totals)
    10

    Each draw has the same number of customers as the original, and its total
    varies around the observed 1266:

    >>> import numpy as np
    >>> bool(1000 < np.mean(totals) < 1500)
    True
    """
    if num_boots < 1:
        raise ValueError("num_boots must be at least 1")
    if not callable(apply):
        # Checked here rather than left to the first draw. Spec `B-15` asks
        # that `fn.boot.apply` be a function; a non-callable used to run every
        # draw and then report "all 100 bootstrap draws failed", which buries
        # the one thing that was wrong under a hundred symptoms of it -- and
        # costs a hundred resamples to say so.
        raise TypeError(
            f"apply must be callable, not {type(apply).__name__}: it is given "
            f"one resampled ClvData per draw and returns what to collect"
        )

    ids = np.array(sorted(data.transactions["Id"].unique()))
    draw = _drawer(sample, np.random.default_rng(seed))

    if isinstance(data, ClvDataDynCov):
        # `ClvDataDynCov` subclasses `ClvData` rather than `ClvDataStaticCov`,
        # so the covariate branch below never fired for it: a dyncov object
        # went in and `apply` received a plain `ClvData` with every covariate
        # silently gone, then refitted a model that is *defined* by those
        # covariates without them. Reproduced before this guard existed.
        #
        # Resampling a time-varying covariate series is not the static case
        # with more rows: a customer drawn twice needs its whole per-period
        # series duplicated under the new id and re-aligned to the resampled
        # window. Until that exists, refusing is the only honest answer -- the
        # alternative is an interval computed from the wrong model. Finding A2
        # of `docs/spec-audit.md`.
        raise NotImplementedError(
            "bootstrapping time-varying covariate data is not supported: the "
            "covariate series would have to be resampled with the customers, "
            "and until it is, the refit would silently drop every covariate. "
            "Bootstrap the static-covariate or plain model instead."
        )

    results = []
    failures: list[str] = []
    for attempt in range(num_boots):
        drawn = list(draw(ids))
        try:
            rebuilt = bootstrap_data(data, drawn)
            if isinstance(data, ClvDataStaticCov):
                rebuilt = _resample_covariates(data, rebuilt, drawn)
            results.append(apply(rebuilt))
        except Exception as error:
            failures.append(f"draw {attempt + 1}: {type(error).__name__}: {error}")

    return _kept(results, failures, num_boots)


def confidence_intervals(
    draws: Sequence[pd.DataFrame],
    level: float = 0.9,
    by: str = "Id",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    r"""Percentile intervals across bootstrap draws.

    Each draw is a frame indexed by ``by`` -- typically the prediction table of
    S6.3. A customer drawn several times in one iteration contributes several
    rows, and they all count: the suffix that kept them apart during resampling
    is stripped here so they pool.

    The bounds are the ordinary percentiles at :math:`(1-\text{level})/2` and
    :math:`1-(1-\text{level})/2`, named ``<column>.CI.<percent>`` as CLVTools
    names them. S6.3.3's example uses the 5% and 95% bounds.

    Examples
    --------
    >>> import pandas as pd
    >>> draws = [
    ...     pd.DataFrame({"CET": [float(i), float(i) + 1]}, index=["a", "b"])
    ...     for i in range(20)]
    >>> for frame in draws:
    ...     frame.index.name = "Id"
    >>> intervals = confidence_intervals(draws, level=0.9)
    >>> list(intervals.columns)
    ['CET.CI.5', 'CET.CI.95']
    >>> [round(v, 2) for v in intervals.loc["a"]]
    [0.95, 18.05]
    """
    if not 0 < level < 1:
        raise ValueError("level must lie strictly between 0 and 1")
    if not draws:
        raise ValueError("no bootstrap draws to summarise")

    stacked = pd.concat(list(draws))
    if stacked.index.name != by:
        stacked = stacked.set_index(by)
    stacked.index = stacked.index.str.replace(
        rf"{BOOTSTRAP_SUFFIX}\d+$", "", regex=True
    )

    if columns is None:
        columns = [
            c for c in stacked.columns
            if pd.api.types.is_numeric_dtype(stacked[c])
        ]

    lower, upper = (1 - level) / 2, 1 - (1 - level) / 2
    grouped = stacked.groupby(level=0)
    out = {}
    for column in columns:
        out[f"{column}.CI.{lower * 100:g}"] = grouped[column].quantile(lower)
        out[f"{column}.CI.{upper * 100:g}"] = grouped[column].quantile(upper)
    result = pd.DataFrame(out)
    result.index.name = by
    return result


def predict_intervals(
    data: ClvData,
    fit_and_predict: Callable[[ClvData], pd.DataFrame],
    num_boots: int = 100,
    level: float = 0.9,
    columns: Sequence[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    r"""Bootstrap confidence intervals for a prediction table.

    S6.3.3: "To calculate confidence intervals for all predicted metrics, the
    ``predict()`` command accepts arguments ``uncertainty`` and ``num.boots``.
    If the former argument is set to ``"boots"``, the latter defines the number
    of bootstrap samples."

    :func:`bootstrap_apply` followed by :func:`confidence_intervals`, named for
    the operation the paper names. ``fit_and_predict`` must both estimate and
    predict, because every draw needs its own fit -- that is the uncertainty
    being measured.

    Examples
    --------
    >>> from clvtools import ClvData, load_apparel_trans
    >>> from clvtools.gg import fit_gg
    >>> from clvtools.pnbd import fit_pnbd
    >>> from clvtools.predict import predict
    >>> data = ClvData(load_apparel_trans(), time_unit="week", estimation_split=104)
    >>> def refit(sample):
    ...     cbs, spend = sample.customer_summary(), sample.spending_summary()
    ...     return predict(
    ...         sample,
    ...         fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False),
    ...         fit_gg(spend["x"], spend["Spending"]))
    >>> intervals = predict_intervals(
    ...     data, refit, num_boots=5, columns=["CET"], seed=1)
    >>> list(intervals.columns)
    ['CET.CI.5', 'CET.CI.95']
    >>> bool((intervals["CET.CI.5"] <= intervals["CET.CI.95"]).all())
    True
    """
    draws = bootstrap_apply(data, fit_and_predict, num_boots=num_boots, seed=seed)
    return confidence_intervals(draws, level=level, columns=columns)
