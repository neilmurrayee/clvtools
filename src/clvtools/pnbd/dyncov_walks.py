r"""S3.3 - the walks each customer's covariate path is cut into.

S3.3 makes the transaction and attrition rates functions of time by letting
the covariates vary, "parameterized by assuming equidistant time intervals
(e.g., one week) during which the covariates are assumed to be constant". Two
things follow. Every integral of a rate over a span of time becomes a sum over
the covariate intervals that span crosses; and, since "the times between
purchases are no longer independently and identically distributed", the
sufficient statistic :math:`(x, t_x, T)` of the standard model is no longer
enough -- "this extension considers the exact date of all transactions".

The bookkeeping that makes this tractable is the *walk*: for a span of time,
the sequence of :math:`\exp(\boldsymbol{\gamma}'\mathbf{x}_k)` values over the
covariate intervals it crosses. Each customer has

``real_walks_trans``
    one per repeat transaction, covering the interval since the previous one;
``aux_walk_trans``
    from the last transaction to the end of the estimation period;
``real_walk_life``
    from coming alive to the last transaction;
``aux_walk_life``
    from the last transaction to the end of the estimation period.

A walk carries two extra numbers: ``d1``, the distance from its start to the
end of the first covariate interval it touches, and ``tjk``, its total length.

This module is the data structure and its construction from a transaction log
and a covariate table; :mod:`clvtools.pnbd.dyncov` is the likelihood written
over it. The dependency runs one way -- the likelihood knows about walks,
nothing here knows about :math:`r, \alpha, s, \beta` -- which is why the walks
can be built once and reused across every evaluation of a fit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

# `build_walks` annotates its argument with `ClvData`, and `py.typed`
# promises that annotation resolves -- so this is a real import, not a
# `TYPE_CHECKING` one. The cycle stays broken at the other end:
# `ClvDataDynCov` imports `build_walks` inside the method that calls it,
# and must keep doing so.
from clvtools.data import ClvData

__all__ = [
    "EMPTY_WALK",
    "Customer",
    "DyncovWalks",
    "TransactionWalk",
    "Walk",
    "build_walks",
]


# -- the walk primitives ------------------------------------------------------


@dataclass(frozen=True)
class Walk:
    r"""A span of time, as the covariate multipliers over the intervals it crosses.

    ``values`` holds :math:`\exp(\boldsymbol{\gamma}'\mathbf{x}_k)` for each
    covariate interval :math:`k` the span touches, first to last.
    """

    values: NDArray[np.float64]

    @property
    def n_elem(self) -> int:
        return int(self.values.size)

    @property
    def first(self) -> float:
        return float(self.values[0])

    @property
    def last(self) -> float:
        return float(self.values[-1])

    def elem(self, i: int) -> float:
        """Zero-based element access."""
        return float(self.values[i])

    def sum_middle(self) -> float:
        """Everything but the first and last element. Needs at least three."""
        if self.n_elem < 3:
            raise ValueError(
                f"sum_middle needs at least 3 elements, walk has {self.n_elem}"
            )
        return float(self.values[1:-1].sum())

    def sum_from_to(self, start: int, stop: int) -> float:
        """Zero-based, inclusive of both ends -- Armadillo's ``subvec``."""
        return float(self.values[start : stop + 1].sum())


@dataclass(frozen=True)
class TransactionWalk(Walk):
    r"""A walk over a transaction interval.

    ``d1`` is the distance from the walk's start to the end of the covariate
    interval it starts in; ``tjk`` is the walk's total length. Both are in the
    data's time units.
    """

    d1: float = float("nan")
    tjk: float = float("nan")


EMPTY_WALK = Walk(np.empty(0))


@dataclass(frozen=True)
class Customer:
    """One customer's sufficient statistics and walks."""

    x: float
    t_x: float
    T_cal: float
    d_omega: float
    real_walks_trans: list[TransactionWalk]
    aux_walk_trans: TransactionWalk
    real_walk_life: Walk
    aux_walk_life: Walk


@dataclass(frozen=True)
class DyncovWalks:
    r"""The walk structures for a whole customer base.

    Rather than one array per customer, the covariate rows for every walk of a
    kind are stacked into one matrix and each customer holds a ``[from, to]``
    index into it. That is how CLVTools stores them, and it keeps
    :math:`\exp(\text{covariates} \cdot \gamma)` a single matrix product for the
    entire sample instead of one per customer.

    Indices are one-based and inclusive at both ends.
    """

    ids: pd.Index
    x: NDArray[np.float64]
    t_x: NDArray[np.float64]
    T_cal: NDArray[np.float64]
    d_omega: NDArray[np.float64]

    #: ``(n_customers, 2)``: from, to.
    walkinfo_aux_life: NDArray[np.float64]
    walkinfo_real_life: NDArray[np.float64]
    #: ``(n_customers, 4)``: from, to, d1, tjk.
    walkinfo_aux_trans: NDArray[np.float64]
    #: ``(n_walks, 4)``, with each customer owning a contiguous block.
    walkinfo_real_trans: NDArray[np.float64]
    #: ``(n_customers,)`` block bounds into ``walkinfo_real_trans``; NaN for a
    #: zero-repeater, who has no repeat-purchase intervals at all.
    real_trans_from: NDArray[np.float64]
    real_trans_to: NDArray[np.float64]

    covdata_aux_life: NDArray[np.float64]
    covdata_real_life: NDArray[np.float64]
    covdata_aux_trans: NDArray[np.float64]
    covdata_real_trans: NDArray[np.float64]

    @property
    def n_customers(self) -> int:
        return int(self.x.size)

    @property
    def n_cov_life(self) -> int:
        return int(self.covdata_aux_life.shape[1])

    @property
    def n_cov_trans(self) -> int:
        return int(self.covdata_aux_trans.shape[1])

    def customers(
        self, gamma_life: ArrayLike, gamma_trans: ArrayLike
    ) -> list[Customer]:
        r"""Build every customer's walks at the given covariate parameters."""
        gamma_life = np.asarray(gamma_life, dtype=float)
        gamma_trans = np.asarray(gamma_trans, dtype=float)
        if gamma_life.size != self.n_cov_life:
            raise ValueError(
                f"{self.n_cov_life} attrition covariates but {gamma_life.size} "
                "parameters"
            )
        if gamma_trans.size != self.n_cov_trans:
            raise ValueError(
                f"{self.n_cov_trans} transaction covariates but "
                f"{gamma_trans.size} parameters"
            )

        adj_aux_life = np.exp(self.covdata_aux_life @ gamma_life)
        adj_real_life = np.exp(self.covdata_real_life @ gamma_life)
        adj_aux_trans = np.exp(self.covdata_aux_trans @ gamma_trans)
        adj_real_trans = np.exp(self.covdata_real_trans @ gamma_trans)

        out = []
        for i in range(self.n_customers):
            al_from, al_to = self.walkinfo_aux_life[i]
            aux_life = Walk(adj_aux_life[int(al_from) - 1 : int(al_to)])

            rl_from, rl_to = self.walkinfo_real_life[i]
            if np.isfinite(rl_from) and np.isfinite(rl_to):
                real_life = Walk(adj_real_life[int(rl_from) - 1 : int(rl_to)])
            else:
                real_life = EMPTY_WALK

            at_from, at_to, at_d1, at_tjk = self.walkinfo_aux_trans[i]
            aux_trans = TransactionWalk(
                adj_aux_trans[int(at_from) - 1 : int(at_to)], d1=at_d1, tjk=at_tjk
            )

            real_trans: list[TransactionWalk] = []
            if np.isfinite(self.real_trans_from[i]):
                block = self.walkinfo_real_trans[
                    int(self.real_trans_from[i]) - 1 : int(self.real_trans_to[i])
                ]
                for w_from, w_to, d1, tjk in block:
                    real_trans.append(
                        TransactionWalk(
                            adj_real_trans[int(w_from) - 1 : int(w_to)],
                            d1=d1, tjk=tjk,
                        )
                    )

            out.append(
                Customer(
                    x=float(self.x[i]), t_x=float(self.t_x[i]),
                    T_cal=float(self.T_cal[i]), d_omega=float(self.d_omega[i]),
                    real_walks_trans=real_trans,
                    aux_walk_trans=aux_trans,
                    real_walk_life=real_life,
                    aux_walk_life=aux_life,
                )
            )
        return out


# -- building the walks from raw data -----------------------------------------


def _interval_index(grid: NDArray[np.int64], when: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""Which covariate interval each timepoint falls in.

    S3.3: the covariate functions "return the covariates :math:`x^P_k, x^A_k` of
    the interval :math:`k` in which :math:`s` lies". Intervals are half-open,
    :math:`[\text{grid}_k, \text{grid}_{k+1})`, because S6.1 has the covariate
    date marking the *start* of the period it describes.
    """
    return np.searchsorted(grid, when, side="right") - 1


def _distance_to_interval_end(
    grid: NDArray[np.int64], when: int, span: Callable[[int, int], float]
) -> float:
    r"""``d`` -- from a timepoint to the end of the covariate interval it is in.

    A timepoint sitting exactly on an interval boundary gets a whole period, not
    zero: CLVTools' comment is "d shall be 1 if it is exactly on the time unit
    lower boundary". Without that, a transaction on a boundary would contribute
    nothing to its own interval and the walk arithmetic would double-count.

    ``span`` measures a gap between two grid days in time units, so this works
    for a calendar unit as well as a fixed one.
    """
    k = int(np.searchsorted(grid, when, side="right") - 1)
    if grid[k] == when:
        return 1.0
    return span(when, int(grid[k + 1]))


@dataclass(frozen=True)
class _WalkSpec:
    """One walk, before the flat arrays are assembled."""

    customer: int
    lo: int
    hi: int
    d1: float = float("nan")
    tjk: float = float("nan")


def _to_days(values) -> NDArray[np.int64]:
    """Whole days since the epoch, so every span is exact.

    All of :func:`build_walks`' arithmetic is done in whole days and converted
    to time units only at the end. Working in nanoseconds instead leaves ``d1``
    and ``tjk`` off by around 4e-13, which is enough to break the exact
    cancellation that makes ``F2.2`` vanish.
    """
    return (
        pd.to_datetime(pd.Series(values))
        .to_numpy(dtype="datetime64[D]")
        .astype("int64")
    )


def _prepare_covariates(
    frame: pd.DataFrame, names: list[str] | None, name_date_cov: str
) -> tuple[dict, dict, list[str]]:
    """Split a covariate table into a per-customer date grid and value matrix."""
    out = frame.copy()
    out["Id"] = out["Id"].astype(str)
    if name_date_cov not in out.columns:
        raise ValueError(f"covariate data has no {name_date_cov!r} column")
    chosen = list(
        names
        if names is not None
        else [c for c in out.columns if c not in ("Id", name_date_cov)]
    )
    missing = [n for n in chosen if n not in out.columns]
    if missing:
        raise ValueError(f"covariate data is missing columns: {missing}")
    out = out.sort_values(["Id", name_date_cov], kind="stable")
    grids, rows = {}, {}
    for cid, group in out.groupby("Id", sort=True):
        grids[cid] = _to_days(group[name_date_cov])
        rows[cid] = group[chosen].to_numpy(dtype=float)
    return grids, rows, chosen


def _check_covariate_coverage(ids, grids_life: dict, grids_trans: dict) -> None:
    """Every customer covered, and the two grids aligned where they overlap.

    Every walk's interval indices are derived once, from the lifetime grid, and
    then used to slice *both* covariate matrices (see :func:`_stack`). That is
    sound only while the two grids agree on the intervals a walk can reach.
    The two series may legitimately run to different dates -- whether they
    reach far enough to predict over is checked at prediction time -- so only
    the overlapping prefix has to match here. A misalignment within it would
    silently shift the transactional walks and return a wrong likelihood.
    """
    missing_cov = (set(ids) - set(grids_life)) | (set(ids) - set(grids_trans))
    if missing_cov:
        raise ValueError(
            f"{len(missing_cov)} customers have no covariate data, "
            f"e.g. {sorted(missing_cov)[:3]}"
        )
    mismatched = [
        cid
        for cid in ids
        if not np.array_equal(
            grids_life[cid][: min(len(grids_life[cid]), len(grids_trans[cid]))],
            grids_trans[cid][: min(len(grids_life[cid]), len(grids_trans[cid]))],
        )
    ]
    if mismatched:
        raise ValueError(
            "lifetime and transaction covariates must share one date grid; "
            f"{len(mismatched)} customers differ, e.g. {mismatched[:3]}"
        )


@dataclass
class _CustomerSpecs:
    """The per-customer summary, and which covariate intervals each walk spans."""

    x: NDArray[np.float64]
    t_x: NDArray[np.float64]
    T_cal: NDArray[np.float64]
    d_omega: NDArray[np.float64]
    aux_life: list[_WalkSpec]
    aux_trans: list[_WalkSpec]
    real_life: dict[int, _WalkSpec]
    real_trans: list[_WalkSpec]


def _customer_specs(
    trans: pd.DataFrame,
    ids: list,
    grids_life: dict,
    end: int,
    span: Callable[[int, int], float],
) -> _CustomerSpecs:
    """Walk out each customer's transactions, recording the four kinds of walk."""
    index_of = {cid: i for i, cid in enumerate(ids)}
    n = len(ids)
    out = _CustomerSpecs(
        x=np.zeros(n), t_x=np.zeros(n), T_cal=np.zeros(n), d_omega=np.zeros(n),
        aux_life=[], aux_trans=[], real_life={}, real_trans=[],
    )

    for cid, group in trans.groupby("Id", sort=True):
        i = index_of[cid]
        grid = grids_life[cid]
        dates = np.sort(_to_days(group["Date"]))
        first, last = int(dates[0]), int(dates[-1])

        out.x[i] = len(dates) - 1
        out.t_x[i] = span(first, last)
        out.T_cal[i] = span(first, end)
        out.d_omega[i] = _distance_to_interval_end(grid, first, span)

        k_first = int(_interval_index(grid, np.array([first]))[0])
        k_last = int(_interval_index(grid, np.array([last]))[0])
        k_end = int(_interval_index(grid, np.array([end]))[0])

        # The auxiliary walks run from the last transaction to the window end.
        out.aux_life.append(_WalkSpec(i, k_last, k_end))
        out.aux_trans.append(_WalkSpec(
            i, k_last, k_end,
            d1=_distance_to_interval_end(grid, last, span),
            tjk=span(last, end),
        ))

        # The real lifetime walk stops short of the auxiliary walk's first
        # interval, so the two never overlap.
        if k_first <= k_last - 1:
            out.real_life[i] = _WalkSpec(i, k_first, k_last - 1)

        # One transaction walk per repeat purchase, over the interval since the
        # previous one.
        for previous, this in pairwise(dates):
            out.real_trans.append(_WalkSpec(
                i,
                int(_interval_index(grid, np.array([previous]))[0]),
                int(_interval_index(grid, np.array([this]))[0]),
                d1=_distance_to_interval_end(grid, int(previous), span),
                tjk=span(int(previous), int(this)),
            ))
    return out


def _stack(
    specs: list[_WalkSpec], rows: dict[str, NDArray[np.float64]], ids: list
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Flatten walks into one matrix plus one-based inclusive indices."""
    blocks, info, cursor = [], [], 1
    for spec in specs:
        block = rows[ids[spec.customer]][spec.lo : spec.hi + 1]
        if len(block) != spec.hi - spec.lo + 1:
            # A short covariate series would slice to fewer rows than the walk
            # spans, shifting every later walk in the stacked matrix.
            raise ValueError(
                f"customer {ids[spec.customer]!r} has covariate data for "
                f"{len(block)} of the {spec.hi - spec.lo + 1} periods its walk spans"
            )
        blocks.append(block)
        info.append((cursor, cursor + len(block) - 1, spec.d1, spec.tjk))
        cursor += len(block)
    covdata = np.vstack(blocks) if blocks else np.empty((0, 0))
    return covdata, np.array(info, dtype=float)


def _real_trans_bounds(
    specs: list[_WalkSpec], n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """First and last transaction walk belonging to each customer, one-based."""
    real_from = np.full(n, np.nan)
    real_to = np.full(n, np.nan)
    for position, spec in enumerate(specs, start=1):
        if np.isnan(real_from[spec.customer]):
            real_from[spec.customer] = position
        real_to[spec.customer] = position
    return real_from, real_to


def build_walks(
    clv_data: ClvData,
    covariates_life: pd.DataFrame,
    covariates_trans: pd.DataFrame | None = None,
    names_cov_life: list[str] | None = None,
    names_cov_trans: list[str] | None = None,
    name_date_cov: str = "Cov.Date",
) -> DyncovWalks:
    r"""Assemble every customer's walks from transaction and covariate data.

    Takes a :class:`~clvtools.data.ClvData` rather than a raw log, so the
    day-level aggregation of S6.1 and the estimation split are already applied.
    Passing the raw log instead would double-count a customer who bought twice
    on one day, which changes both ``x`` and the number of transaction walks.

    The four kinds of walk, and the span each covers:

    ``real_walks_trans``
        :math:`[t_{j-1}, t_j]` for each repeat transaction :math:`j`. The first
        transaction gets none -- there is no preceding interval for it to
        depend on.
    ``aux_walk_trans``
        :math:`[t_x, T]`, from the last transaction to the end of the
        estimation period.
    ``real_walk_life``
        the covariate intervals from coming alive up to, but not including,
        the one holding the last transaction. A customer whose first and last
        transactions share an interval has none.
    ``aux_walk_life``
        the same span as ``aux_walk_trans``, over the attrition covariates.

    ``real_walk_life`` and ``aux_walk_life`` deliberately do not overlap: the
    interval containing the last transaction belongs to the auxiliary walk
    alone, which is what lets
    :func:`~clvtools.pnbd.dyncov.d_i` treat the pair as one continuous span.
    """
    if covariates_trans is None:
        covariates_trans = covariates_life

    trans = clv_data.transactions
    trans = trans[trans["Date"] <= clv_data.estimation_end]
    if trans.empty:
        raise ValueError("no transactions fall within the estimation period")

    unit = clv_data.time
    ids = sorted(trans["Id"].unique())

    def span(day_a: int, day_b: int) -> float:
        """The gap between two epoch-days, in time units."""
        return unit.elapsed(
            pd.Timestamp(day_a, unit="D"), pd.Timestamp(day_b, unit="D")
        )

    grids_life, rows_life, names_cov_life = _prepare_covariates(
        covariates_life, names_cov_life, name_date_cov
    )
    grids_trans, rows_trans, names_cov_trans = _prepare_covariates(
        covariates_trans, names_cov_trans, name_date_cov
    )
    _check_covariate_coverage(ids, grids_life, grids_trans)

    end = int(_to_days([clv_data.estimation_end])[0])
    specs = _customer_specs(trans, ids, grids_life, end, span)

    covdata_aux_life, wi_aux_life = _stack(specs.aux_life, rows_life, ids)
    covdata_aux_trans, wi_aux_trans = _stack(specs.aux_trans, rows_trans, ids)

    ordered_real_life = [specs.real_life[i] for i in sorted(specs.real_life)]
    covdata_real_life, wi_real_life_packed = _stack(ordered_real_life, rows_life, ids)
    # Customers without a real lifetime walk carry NaN, which the walk builder
    # reads as "empty".
    wi_real_life = np.full((len(ids), 2), np.nan)
    for spec, row in zip(ordered_real_life, wi_real_life_packed, strict=True):
        wi_real_life[spec.customer] = row[:2]

    covdata_real_trans, wi_real_trans = _stack(specs.real_trans, rows_trans, ids)
    real_from, real_to = _real_trans_bounds(specs.real_trans, len(ids))

    return DyncovWalks(
        ids=pd.Index(ids, name="Id"),
        x=specs.x, t_x=specs.t_x, T_cal=specs.T_cal, d_omega=specs.d_omega,
        walkinfo_aux_life=wi_aux_life[:, :2],
        walkinfo_real_life=wi_real_life,
        walkinfo_aux_trans=wi_aux_trans,
        walkinfo_real_trans=wi_real_trans,
        real_trans_from=real_from,
        real_trans_to=real_to,
        covdata_aux_life=covdata_aux_life,
        covdata_real_life=covdata_real_life,
        covdata_aux_trans=covdata_aux_trans,
        covdata_real_trans=covdata_real_trans,
    )
