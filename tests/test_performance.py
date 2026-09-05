"""The performance invariants, asserted by counting operations.

``docs/performance.md`` profiles the package and finds nothing slow: the
vectorised models sit at the floor of what SciPy costs. What is unguarded is
*becoming* slow later, and the regressions that would actually cost minutes are
structural -- de-vectorising an inner function, waking a scalar fallback,
doubling the number of likelihood evaluations a fit needs. Every one of them is
invisible to the rest of the suite, because every one of them leaves the
numbers exactly right.

So this module gates efficiency the way ``test_code_quality.py`` gates
tidiness: deterministically. **Nothing here looks at a clock.** A wall-clock
assertion would be the first gate in this repo that fails for reasons unrelated
to the change under test, and the first response to a flaky gate is to loosen
it -- which the backlog forbids anyway. Counting operations catches the
same regressions and cannot be flaky.

Each limit below carries the value measured on 2026-09-01 in a comment beside
it, in the same style as the ruff limits in ``pyproject.toml``: measured
against this codebase, set just past what the code needs, so a regression trips
it.

The subject is ``fit_pnbd`` on the apparel data -- 600 customers, 0.065 s -- and
``cdnow`` at two sizes for the O(*n*) check. Nothing marked ``slow`` runs here,
and the dyncov fit (13.5 minutes) certainly does not: these tests are part of
every ``uv run pytest``, so they cost under a second in total.

A note on the instrumentation, because getting it wrong is silent.
``clvtools/pnbd/aggregate.py`` does ``from clvtools.special import
hyp2f1_ratio``, which binds the function object into ``aggregate``'s namespace
at import time. Patching ``clvtools.special.hyp2f1_ratio`` therefore leaves the
bound name untouched and records **zero** -- a green test that measured
nothing. The counters below are installed in the module that *calls* the
function, and :meth:`Count.fired` makes a counter that never ran a failure
rather than a pass.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from conftest import fixture_json

import clvtools.pnbd.aggregate as pnbd_aggregate
import clvtools.special
from clvtools import ClvData, load_cdnow
from clvtools.pnbd import dyncov
from clvtools.pnbd.fit import fit_pnbd

pytestmark = pytest.mark.performance


@dataclass
class Count:
    """How often a function ran, and over how many array elements."""

    calls: int = 0
    elements: int = 0

    @property
    def fired(self) -> bool:
        """Whether the counter saw anything at all.

        An invariant that expects zero calls has to be able to tell "the code
        never did this" from "the counter was installed in the wrong place";
        every test below that expects a zero asserts a companion counter's
        :attr:`fired` in the same breath.
        """
        return self.calls > 0


@contextlib.contextmanager
def counted(module: ModuleType, name: str) -> Iterator[Count]:
    """Count calls to ``module.name``, and the elements each returns.

    The patch goes on the module that *uses* the function, not the one that
    defines it -- see this module's docstring.

    >>> import numpy as np, clvtools.pnbd.aggregate as agg
    >>> with counted(agg, "hyp2f1_ratio") as count:
    ...     _ = agg.hyp2f1_ratio(np.full(7, 2.0), 1.0, 0.0)
    ...     _ = agg.hyp2f1_ratio(np.full(7, 2.0), 1.0, 0.5)
    >>> count.calls, count.elements, count.fired
    (2, 14, True)

    And it is put back afterwards:

    >>> agg.hyp2f1_ratio is clvtools.special.hyp2f1_ratio
    True

    A function that returns a tuple is measured by its first element. The
    :math:`F_2` arms return ``(log magnitude, sign)`` since backlog item 28,
    and it is the magnitudes that are one per covariate interval; counting the
    signs as well would double every number here for no change in the work
    done.
    """
    original: Callable = getattr(module, name)
    count = Count()

    def counting(*args, **kwargs):
        result = original(*args, **kwargs)
        count.calls += 1
        count.elements += int(np.size(result[0] if type(result) is tuple else result))
        return result

    setattr(module, name, counting)
    try:
        yield count
    finally:
        setattr(module, name, original)


@pytest.fixture(scope="module")
def apparel_cbs(apparel_trans) -> pd.DataFrame:
    """The 600-customer estimation-period summary of S6.2.1."""
    return ClvData(apparel_trans, estimation_split=104).customer_summary()


@dataclass
class FitCounts:
    """What one instrumented ``fit_pnbd`` did."""

    n_customers: int
    evaluations: int
    log_likelihood: float
    ratio: Count
    series: Count


def instrumented_fit(cbs) -> FitCounts:
    """Fit the Pareto/NBD with both counters installed.

    The Hessian is skipped: it is a separate 80 evaluations of the likelihood
    (measured), and this is about the search, not about standard errors.
    """
    with (
        counted(pnbd_aggregate, "hyp2f1_ratio") as ratio,
        counted(clvtools.special, "_hyp2f1_series") as series,
    ):
        fit = fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)
    return FitCounts(
        len(cbs), fit.n_evaluations, fit.log_likelihood, ratio, series
    )


@pytest.fixture(scope="module")
def apparel_fit(apparel_cbs) -> FitCounts:
    """One instrumented fit, shared by the tests that only read its counts."""
    return instrumented_fit(apparel_cbs)


@pytest.fixture(scope="module")
def cdnow_cbs() -> pd.DataFrame:
    """CDNOW's 2,357 customers, split as the R package's ``?pmf`` does."""
    return ClvData(
        load_cdnow(), time_unit="week", estimation_split=37
    ).customer_summary()


class TestHyp2f1StaysVectorised:
    r""":func:`clvtools.special.hyp2f1_ratio` is the hot spot, and is fine.

    ``docs/performance.md``: it is half the self-time of a ``fit_pnbd``, but
    580 bare ``scipy.special.hyp2f1`` calls on 600-element arrays cost all of
    that on their own, so there is no overhead to remove. What there *is* to
    protect is the shape of the call: one SciPy call per :math:`A` term over
    the whole sample. Rewriting the ``np.where`` in ``log_likelihood_ind`` as a
    loop over customers would keep every number identical and cost about 100x.
    """

    def test_a_bounded_number_of_calls_per_likelihood_evaluation(self, apparel_fit):
        """Two per evaluation, not two per customer per evaluation.

        Measured: 210 evaluations, 420 calls -- the :math:`A_1` and :math:`A_2`
        terms of Appendix A, once each. The gate is a bound rather than an
        equality so that folding them into a single call stays legal.
        """
        assert apparel_fit.ratio.fired, (
            "the hyp2f1_ratio counter never fired, so this test measured "
            "nothing. It patches the name in clvtools.pnbd.aggregate, which is "
            "where the call lives; patching clvtools.special instead records "
            "zero, because aggregate.py binds the function at import time."
        )
        per_evaluation = apparel_fit.ratio.calls / apparel_fit.evaluations
        assert per_evaluation <= 2.0, (  # measured: exactly 2.0
            f"hyp2f1_ratio ran {per_evaluation:.1f} times per likelihood "
            f"evaluation ({apparel_fit.ratio.calls} calls over "
            f"{apparel_fit.evaluations} evaluations). Two is the whole sample "
            "twice; anything near the customer count means the call has been "
            "moved inside a loop."
        )

    def test_every_call_covers_the_whole_sample(self, apparel_fit):
        """Each call is one array of length *n*, not *n* scalars.

        Measured: 252,000 elements over 420 calls on 600 customers, i.e. 600
        elements per call exactly. This is the assertion that a per-customer
        loop fails hardest -- it would report one element per call.
        """
        assert apparel_fit.ratio.fired
        assert apparel_fit.ratio.elements == (
            apparel_fit.ratio.calls * apparel_fit.n_customers
        ), (
            f"{apparel_fit.ratio.elements} elements over "
            f"{apparel_fit.ratio.calls} calls on {apparel_fit.n_customers} "
            "customers; every call should cover the whole sample."
        )

    def test_the_scalar_series_fallback_stays_cold(self, apparel_fit):
        """``_hyp2f1_series`` runs for 0 of 252,000 elements (measured).

        SciPy returns ``nan`` for large first parameter with ``z`` near 1, and
        :func:`~clvtools.special.hyp2f1_ratio` then sums the Euler series in
        Python, one entry at a time. That is the correct answer and an
        expensive way to get it. If a change moves the optimiser into that
        corner the fits get much slower and every number stays right, so
        nothing else in the suite would notice.
        """
        assert apparel_fit.ratio.fired, "instrumentation dead; see the message above"
        assert apparel_fit.series.calls == 0, (  # measured: 0 of 252,000
            f"the scalar series fallback ran {apparel_fit.series.calls} times "
            f"during a fit that evaluated {apparel_fit.ratio.elements} "
            "hypergeometrics. SciPy is now returning non-finite values "
            "somewhere the search visits; find out where before relaxing this."
        )

    def test_the_series_counter_is_wired(self):
        """The zero above is a fact about the fit, not a dead counter.

        The one corner ``scipy.special.hyp2f1`` cannot do is the one
        :func:`~clvtools.special.hyp2f1_ratio`'s docstring names, so it makes a
        positive control: with the counter installed, that call must record.
        """
        from scipy import special

        assert not np.isfinite(special.hyp2f1(200.0, 20.0, 201.0, 0.999))
        with counted(clvtools.special, "_hyp2f1_series") as series:
            value = clvtools.special.hyp2f1_ratio(200.0, 20.0, 0.999)
        assert series.calls == 1
        assert np.isfinite(value)

    def test_the_counters_are_installed_where_the_calls_happen(self, apparel_cbs):
        """Patching ``clvtools.special`` would measure nothing. Proof.

        This is the trap the module docstring describes, pinned so that nobody
        "simplifies" the counters onto the defining module and gets a suite
        that passes while watching an empty room. If this test fails because
        ``aggregate.py`` now reaches the function through the module, move the
        counters -- do not delete the check.
        """
        head = apparel_cbs.head(20)
        args = (head["x"], head["t_x"], head["T"], 1.449, 48.636, 0.561, 46.884)
        with (
            counted(clvtools.special, "hyp2f1_ratio") as at_definition,
            counted(pnbd_aggregate, "hyp2f1_ratio") as at_call_site,
        ):
            pnbd_aggregate.log_likelihood(*args)
        assert at_call_site.fired, (
            "the counter on clvtools.pnbd.aggregate saw nothing, so every "
            "count in this file is meaningless. aggregate.py no longer calls "
            "the name it imported."
        )
        assert not at_definition.fired, (
            "clvtools.special.hyp2f1_ratio was reached through the module, so "
            "the counters in this file are now in the wrong place: move them "
            "to clvtools.special and check they still fire."
        )


class TestEvaluationsPerFit:
    """The only way to make a Pareto/NBD fit faster is fewer evaluations.

    ``docs/performance.md``: an evaluation is already all SciPy, so the count
    of them *is* the run time. That makes it the thing to watch, and it is a
    property of the optimiser setup and the start values rather than of any
    equation -- so a regression in either changes it while leaving every
    published number intact.
    """

    #: A coarse sanity band only. The exact count is **not portable**: 210 on
    #: macOS/ARM and 185 on x86-64 Linux for the same code, because a line
    #: search on a finite-differenced gradient follows a slightly different
    #: path on a different libm. The first CI run failed the old (200, 400)
    #: band at 185, which was the band being wrong rather than the code. The
    #: two tests below carry the real weight, and both compare configurations
    #: *within* whatever platform is running them.
    BAND = (120, 500)

    def test_pnbd_on_apparel_stays_in_the_coarse_band(self, apparel_fit):
        """210 on macOS/ARM, 185 on x86-64 Linux; 120 to 500 allowed."""
        low, high = self.BAND
        assert low <= apparel_fit.evaluations <= high, (
            f"fit_pnbd took {apparel_fit.evaluations} likelihood evaluations "
            f"against 210 measured on macOS/ARM and 185 on x86-64 Linux. This "
            "band is deliberately wide because the count is not portable; the "
            "two comparisons below are the ones that mean something."
        )

    @pytest.mark.oracle
    def test_the_tolerance_buys_a_better_optimum_than_scipy_s_default(
        self, apparel_cbs
    ):
        """Why ``_optimize`` tightens ``ftol`` at all, asserted rather than said.

        The old lower bound existed to catch a relaxed ``ftol``. It caught it
        by *counting*, which made it a statement about one libm. The same
        regression is visible in the objective, and that comparison is
        portable: run the same fit at SciPy's own 1e-8 and it stops earlier and
        lands worse.

        The comparison is from CLVTools' own all-ones start, which is where the
        tolerance still earns its keep and which is not the default any more --
        see :func:`~clvtools._optimize.start_scale`, and
        :meth:`test_a_scaled_start_reaches_the_same_optimum_at_either_tolerance`
        for what changed. Measured on macOS/ARM from that start: 210
        evaluations against 150, and -5848.097826903 against -5848.097841543 --
        1.5e-5 better, on a ridge where 1e-10 of log-likelihood is 3e-5 of
        ``beta``.
        """
        ones = (1.0, 1.0, 1.0, 1.0)
        tight = fit_pnbd(
            apparel_cbs["x"], apparel_cbs["t_x"], apparel_cbs["T"],
            hessian=False, start=ones,
        )
        loose = fit_pnbd(
            apparel_cbs["x"], apparel_cbs["t_x"], apparel_cbs["T"],
            hessian=False, start=ones, options={"ftol": 1e-8},
        )
        assert tight.log_likelihood > loose.log_likelihood + 1e-6
        assert tight.n_evaluations > loose.n_evaluations

    def test_a_scaled_start_reaches_the_same_optimum_at_either_tolerance(
        self, apparel_cbs, apparel_fit
    ):
        """And what the scaled start took away from the test above.

        Started at ``alpha = beta = mean(T)`` rather than at 1, SciPy's own
        loose ``ftol`` lands on the same optimum -- and in 125 evaluations on
        macOS/ARM, fewer than the 150 the loose fit from all-ones needed. The
        tolerance is still worth keeping, because a start is a convention and
        some data will defeat any convention; but on this data it is no longer
        what finds the optimum, and pretending otherwise would leave a test
        asserting something that had stopped being true.

        The bound is 1e-6, which is a *thousand times* tighter than the 1.5e-5
        the all-ones start costs and so still separates the two claims, and
        loose enough to be a statement about the optimum rather than about one
        libm. It was 1e-8 for one commit, and x86-64 Linux on 3.13 put the two
        fits 1.8e-8 apart -- the same lesson the evaluation-count bounds above
        were rewritten to learn, made twice. The counts are quoted here rather
        than asserted for exactly that reason.
        """
        loose = fit_pnbd(
            apparel_cbs["x"], apparel_cbs["t_x"], apparel_cbs["T"],
            hessian=False, options={"ftol": 1e-8},
        )
        assert loose.log_likelihood == pytest.approx(
            apparel_fit.log_likelihood, abs=1e-6
        )

    def test_the_search_is_not_nelder_mead(self, apparel_cbs, apparel_fit):
        """The old upper bound, made portable the same way.

        Nelder-Mead is the fallback S6.2.1 recommends and a plausible
        accidental default. It reaches the same optimum and pays about 2.3x
        for it -- 489 evaluations against 210 on macOS/ARM -- so the ordering,
        rather than either number, is what to assert.
        """
        simplex = fit_pnbd(
            apparel_cbs["x"], apparel_cbs["t_x"], apparel_cbs["T"],
            hessian=False, method="Nelder-Mead",
        )
        assert apparel_fit.evaluations < simplex.n_evaluations

    def test_the_band_is_measuring_the_search(self, apparel_fit):
        """A count of 1 would pass a lower bound of 0, so the bound binds."""
        assert self.BAND[0] > 0
        assert apparel_fit.evaluations > 0

    @pytest.mark.oracle
    def test_the_search_still_arrives_where_it_should(self, apparel_fit):
        """The band's blind spot, closed.

        A count inside the band says nothing about the *quality* of the
        optimum: a change that stops early somewhere equally good and one that
        stops early somewhere worse are the same number. The published
        log-likelihood is already pinned under ``-m paper`` and against the
        oracle, but in another file -- so a reader of the band above has to
        know to go and look. Asserting it here makes the pair one thing: this
        many evaluations, *and* this optimum.
        """
        want = fixture_json("pnbd_nocov_fit")["logLik"]
        assert apparel_fit.log_likelihood == pytest.approx(want, abs=1e-4), (
            "the fit no longer reaches the oracle's optimum, so the "
            "evaluation band above is counting a different search"
        )


class TestCostIsFlatInN:
    """O(*n*) by operation count, not by stopwatch.

    The quantity that must not grow is hypergeometric evaluations per customer
    per likelihood evaluation. It is 2 at every size: two calls per evaluation,
    each over all *n* customers. An accidental O(*n*^2) -- a per-customer loop
    that re-evaluates the whole array, say -- shows up here as a figure that
    scales with *n*, and shows up at 2,357 customers whether or not the machine
    running the test is fast.

    The number of *evaluations* is deliberately not compared across sizes: it
    is a property of the optimiser's path, not of *n* (measured: 165 for cdnow,
    200 for its first half -- the smaller problem took *more* of them).
    """

    def test_hypergeometric_work_per_customer_does_not_grow(self, cdnow_cbs):
        """Two element-evaluations per customer per likelihood evaluation.

        Measured: 2,357 customers -> 330 calls over 777,810 elements in 165
        likelihood evaluations; 1,178 customers -> 400 calls over 471,200
        elements in 200. Both give exactly 2.0.
        """
        big = instrumented_fit(cdnow_cbs)
        small = instrumented_fit(cdnow_cbs.iloc[: len(cdnow_cbs) // 2])
        assert big.n_customers > small.n_customers  # 2,357 against 1,178

        def per_customer_per_evaluation(counts: FitCounts) -> float:
            assert counts.ratio.fired
            return counts.ratio.elements / (counts.n_customers * counts.evaluations)

        large, little = (
            per_customer_per_evaluation(counts) for counts in (big, small)
        )
        assert large == pytest.approx(little, rel=1e-12), (
            f"hypergeometric elements per customer per likelihood evaluation "
            f"is {little:.3f} at {small.n_customers} customers and "
            f"{large:.3f} at {big.n_customers}. It should not depend on the "
            "sample size at all; that it does means the work per customer is "
            "growing with n."
        )
        assert large <= 2.0  # measured: exactly 2.0 at both sizes


@dataclass
class DyncovCounts:
    """What one evaluation of the time-varying likelihood dispatched."""

    n_customers: int
    intervals: int
    between: int
    ge: Count
    gt: Count
    batch: Count


@pytest.fixture(scope="module")
def dyncov_counts(dyncov_walks) -> DyncovCounts:
    """One instrumented evaluation, at CLVTools' own fitted parameters.

    The parameters matter. Which hypergeometric arm a covariate interval takes
    depends on where in the parameter space the evaluation happens: at a
    convenient starting vector every one of the 39,754 intervals takes the
    ``beta > alpha`` arm and the other is never entered, so a broken arm would
    pass unnoticed. At the fitted point both run. See ``docs/performance.md``.
    """
    coefficients = fixture_json("dyncov_fit")["coefficients"]
    names = fixture_json("dyncov_fit")["names.cov"]
    with (
        # Installed on the module that calls them -- `_hyp_terms` and the two
        # arms are looked up as globals of `clvtools.pnbd.dyncov`, so this is
        # where a patch is seen. See this module's docstring.
        counted(dyncov, "_hyp_alpha_ge_beta") as ge,
        counted(dyncov, "_hyp_beta_gt_alpha") as gt,
        counted(dyncov, "_hyp_terms") as batch,
    ):
        dyncov.log_likelihood(
            dyncov_walks,
            *(coefficients[k] for k in ("r", "alpha", "s", "beta")),
            gamma_life=[coefficients[f"life.{n}"] for n in names],
            gamma_trans=[coefficients[f"trans.{n}"] for n in names],
        )
    info = dyncov_walks.walkinfo_aux_trans
    lengths = (info[:, 1] - info[:, 0] + 1).astype(int)
    return DyncovCounts(
        n_customers=dyncov_walks.n_customers,
        intervals=int(lengths.sum()),
        between=int(np.maximum(lengths - 2, 0).sum()),
        ge=ge, gt=gt, batch=batch,
    )


class TestDyncovStaysVectorised:
    r"""The time-varying likelihood is batched over covariate intervals.

    ``docs/performance.md``: this was the one real finding of the profile --
    0.33 s and 600,000 Python-level calls for one number, because every one of
    a customer's ~66 covariate intervals took its own scalar trip through
    :func:`~clvtools.pnbd.dyncov._hyp_term`. Backlog item 9 replaced that inner
    loop with array work and the evaluation fell to 0.097 s.

    Nothing else in the suite would notice it going back: the oracle fixtures
    check the numbers, and a scalar loop would produce the same ones. So the
    shape of the call is gated here, in the same way and for the same reason as
    :class:`TestHyp2f1StaysVectorised` gates the plain model's.
    """

    def test_the_arms_are_called_a_bounded_number_of_times_per_customer(
        self, dyncov_counts
    ):
        """Four per customer, not one per covariate interval.

        Measured: 2,385 dispatches over 600 customers, i.e. 3.98 each --
        :math:`Y_1` and :math:`Y_{k_T}` one apiece, and one batched call per
        arm for everything in between. Before the rewrite it was 39,754, i.e.
        66.3 per customer, and it scaled with the length of the walks.
        """
        dead = (
            "the hypergeometric arm counters never fired, so this test "
            "measured nothing. They patch the names in clvtools.pnbd.dyncov, "
            "which is where `_hyp_terms` looks them up."
        )
        assert dyncov_counts.ge.fired, dead
        assert dyncov_counts.gt.fired, dead
        calls = dyncov_counts.ge.calls + dyncov_counts.gt.calls
        per_customer = calls / dyncov_counts.n_customers
        assert per_customer <= 4.0, (  # measured: 3.975
            f"the hypergeometric arms ran {per_customer:.1f} times per "
            f"customer ({calls} calls over {dyncov_counts.n_customers} "
            f"customers) against {dyncov_counts.intervals} covariate "
            "intervals. Anything near the interval count means the batched "
            "call in `_f2_middle` has been unrolled back into a loop."
        )

    def test_every_covariate_interval_is_still_evaluated_once(self, dyncov_counts):
        """Batched, but not fewer terms: 39,754 elements for 39,754 intervals.

        The count above would also fall if the rewrite quietly dropped
        intervals. This is the other half: the arms between them see one
        element per interval the auxiliary walks cross, and
        :func:`~clvtools.pnbd.dyncov._hyp_terms` sees exactly the 38,555 that
        lie between the first and the last.
        """
        assert dyncov_counts.ge.elements + dyncov_counts.gt.elements == (
            dyncov_counts.intervals
        ), (
            f"{dyncov_counts.ge.elements + dyncov_counts.gt.elements} "
            f"hypergeometric evaluations for {dyncov_counts.intervals} "
            "covariate intervals; there should be exactly one each."
        )
        assert dyncov_counts.batch.elements == dyncov_counts.between
        assert dyncov_counts.batch.calls <= dyncov_counts.n_customers  # 593

    def test_both_arms_are_exercised_at_the_fitted_parameters(self, dyncov_counts):
        r"""Which is true here and false at any convenient starting vector.

        Measured: 1,212 intervals take :math:`\alpha \ge \beta` and 38,542 take
        :math:`\beta > \alpha`. The split itself is not asserted -- it is a
        property of where in the parameter space the evaluation sits, and the
        comparison it comes from is between two floats. That *both* are
        non-empty is asserted, because a batched implementation that handles
        one arm and mangles the other is the failure this rewrite invited, and
        a profile or a test taken at ``r=0.5, alpha=10, s=0.6, beta=12`` would
        never enter the smaller one.
        """
        assert dyncov_counts.ge.elements > 0, (  # measured: 1,212
            "no covariate interval took the alpha >= beta arm, so this "
            "evaluation cannot say whether it works. Check the parameters: "
            "away from the optimum every interval takes the other arm."
        )
        assert dyncov_counts.gt.elements > 0  # measured: 38,542


class TestDyncovDeduplicatesItsHypergeometrics:
    r"""Backlog item 30: 93.3% of the hypergeometrics are duplicate arguments.

    ``docs/performance.md``: one evaluation over the apparel cohort asks for
    **79,508** :math:`{}_2F_1` values and only **5,303** are distinct, because
    the covariates are categorical, so :math:`\exp(\gamma'x)` takes few values
    and so does :math:`z`. The duplication is entirely *across* customers --
    within one customer's call every :math:`z` differs -- which is why the memo
    is opened once for the whole sweep in ``log_likelihood_ind`` and why a
    narrower scope would catch nothing.

    Gated here for the same reason as :class:`TestDyncovStaysVectorised` above:
    the oracle fixtures check the *numbers*, and removing the memo would produce
    exactly the same ones, only slowly. Counted rather than timed, which is this
    module's rule.
    """

    def test_scipy_sees_only_the_distinct_arguments(self, dyncov_walks):
        """The count SciPy is asked for, against the count the model wants."""
        from clvtools.pnbd import dyncov

        asked, distinct = [0], set()
        real = dyncov.special.hyp2f1

        def counting(a, b, c, z):
            z_arr = np.atleast_1d(np.asarray(z, float))
            a_arr = np.broadcast_to(np.atleast_1d(np.asarray(a, float)), z_arr.shape)
            b_arr = np.broadcast_to(np.atleast_1d(np.asarray(b, float)), z_arr.shape)
            asked[0] += z_arr.size
            distinct.update(zip(a_arr.ravel(), b_arr.ravel(), z_arr.ravel(),
                                strict=True))
            return real(a, b, c, z)

        coefficients = fixture_json("dyncov_fit")["coefficients"]
        names = fixture_json("dyncov_fit")["names.cov"]
        dyncov.special.hyp2f1 = counting
        try:
            dyncov.log_likelihood(
                dyncov_walks,
                *(coefficients[k] for k in ("r", "alpha", "s", "beta")),
                gamma_life=[coefficients[f"life.{n}"] for n in names],
                gamma_trans=[coefficients[f"trans.{n}"] for n in names],
            )
        finally:
            dyncov.special.hyp2f1 = real

        # Every argument SciPy is handed is one the memo had not seen: no
        # duplicate reaches it. This is the invariant -- the absolute counts
        # below are the measurement that motivated it.
        assert asked[0] == len(distinct), (
            f"SciPy was asked for {asked[0]} hypergeometrics but only "
            f"{len(distinct)} distinct arguments: the memo is not being reached"
        )
        # And the saving is the order of magnitude recorded, not a rounding.
        assert asked[0] < 12_000, (
            f"{asked[0]} distinct arguments; ~5,300 was the measurement, and "
            "79,508 is what an unmemoised evaluation asks for"
        )

    def test_the_memo_does_not_outlive_one_evaluation(self):
        """Scope is load-bearing in both directions.

        Narrower catches nothing, since the duplication is across customers.
        *Wider* is worse than useless: the parameters move every evaluation, so
        a key from the last one can never hit, and a fit's ~1,900 evaluations
        would grow an unbounded dictionary of pure misses.
        """
        from clvtools.pnbd.dyncov import _HYP_MEMO, _memoised_hypergeometrics

        assert _HYP_MEMO.get() is None
        with _memoised_hypergeometrics():
            assert _HYP_MEMO.get() == {}
            _HYP_MEMO.get()["sentinel"] = 1.0
        assert _HYP_MEMO.get() is None, "the memo outlived its evaluation"

    def test_and_it_returns_what_scipy_would_have(self):
        """Bit-exact, since it is the same function on the same arguments."""
        from scipy import special

        from clvtools.pnbd.dyncov import _hyp2f1, _memoised_hypergeometrics

        a = np.array([4.0, 4.0, 5.0, 4.0])
        z = np.array([0.5, 0.5, 0.9, 0.5])
        want = special.hyp2f1(a, 3.0, a + 1.0, z)
        assert np.array_equal(_hyp2f1(a, 3.0, z), want)
        with _memoised_hypergeometrics():
            assert np.array_equal(_hyp2f1(a, 3.0, z), want)
            # Second pass is all hits, and must still be identical.
            assert np.array_equal(_hyp2f1(a, 3.0, z), want)
