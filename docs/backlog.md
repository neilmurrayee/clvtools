# Backlog

The paper and the R package's documentation are both covered — `docs/audit.md`
records both rounds, and every item in them is closed. What remains is not
model work. It is the difference between "the port is correct" and "the package
is in good shape", and each item below was found by measurement, not by
speculation.

**This file is the loop's state.** Work the topmost unchecked item, tick it, and
stop. Do not add items without evidence, and do not work an item marked
`[needs-decision]` — those need the maintainer.

**Round 3 is open**, from a review on 2026-09-02. Items 1-9 are closed, item 9
last, and both audit rounds with them. What the review found is below, item 10
first: the port is correct and gated, and none of it has ever left this machine.
Item 10 blocks the rest — it is the only one that becomes impossible to do
properly once something is pushed.

Definition of done for every item: `uv run pytest` green (906 tests at the time
of writing), `uv run ruff check src tests tools docs` clean, and 100% line
coverage of `src/`. Anything that changes behaviour also needs a test and, if it
deviates from CLVTools, a README findings entry — the house rule.

---

## 1. `[x]` Continuous integration

There is no `.github/workflows/`. Every gate this repo has — 888 tests, 100%
coverage, ruff, the complexity and size limits — runs only when someone
remembers to run it. That is the largest structural gap in the repo and the
cheapest to close.

*Done when:* a workflow runs `uv run pytest --cov=clvtools --cov-fail-under=100`
and `uv run ruff check src tests tools docs` on push and pull request, on the
supported Python versions (`requires-python = ">=3.12"`, so 3.12 and 3.13 at
least). The R oracle must **not** be required: the suite is committed-fixture
based precisely so it runs without R, and CI is where that promise gets tested.

**Done:** `.github/workflows/ci.yml` — `uv sync --locked`, then
`uv run ruff check src tests tools docs` and
`uv run pytest --cov=clvtools --cov-fail-under=100`, on push and on pull
request, over a 3.12/3.13 matrix on `ubuntu-latest` with `astral-sh/setup-uv`
caching on `uv.lock`. No R is installed, so the committed-fixture promise is
what the job actually tests. A step asserts the interpreter really is the
matrix one, because `.python-version` pins 3.12 and a matrix that silently ran
it twice would look identical to a passing one. `dyncov_fit` stays deselected
by `addopts`; scheduling it is item 6.

Verified locally on both interpreters before landing: 888 passed, 1 deselected,
`TOTAL 2647 0 100%`, ruff clean — under 3.12 in 271s and under 3.13 in a
throwaway environment built from the same lock file. The workflow YAML was
parsed with `uvx --with pyyaml`; no project dependency was added for it.

## 2. `[x]` Make `py.typed` true

`src/clvtools/py.typed` ships, which tells every downstream type checker that
this package's annotations are meant to be relied on. Nothing verifies them.
`uvx ty check src/` reports **56 diagnostics**. Most are pandas-stub noise
(`Timestamp | NaTType`), but four groups are real contract defects:

- **Annotations that do not resolve at runtime.** `bgnbd.py`, `ggomnbd.py`,
  `pnbd/staticcov.py` and `pnbd/dyncov.py` annotate parameters with
  `ClvDataStaticCov`, `ClvData` and `StaticCovResult`, which are imported
  *inside* the function to break an import cycle. At module scope the name does
  not exist — `dyncov.build_walks` already carries a `# noqa: F821` admitting
  it. `typing.get_type_hints()` on those signatures raises for anyone
  downstream. Fix with `if TYPE_CHECKING:` imports and string annotations.
- **`Fitted` uses an attribute it never declares.** `inference.py:101` reads
  `self.hessian`; the mixin's contract (per CLAUDE.md: a family provides
  `names`, `__iter__` and `hessian`) is documented in prose and nowhere in the
  types.
- **Family dispatch is typed `object`.** `predict.py` calls `.expectation`,
  `.probability_alive`, `.conditional_expected_transactions` and
  `.discounted_expected_residual_transactions` on a value the checker knows
  nothing about. A `Protocol` would state the interface that
  `clvtools.predict` actually requires of a family.
- **`_search` returns `None | Unknown`.** `_staticcov.py` initialises `result`
  to `None` and fills it in a loop the checker cannot prove runs.

*Done when:* a type checker runs in the same gate as ruff — extend
`tests/test_code_quality.py`, which is already the place where static analysis
is a test — and the four groups above are fixed rather than suppressed.
Stub-driven pandas noise may be ignored by rule, with the reason recorded in
`pyproject.toml` next to the ruff ignores, in the same style.

**Done:** `ty==0.0.77` is a dev dependency and `tests/test_code_quality.py`
runs `ty check src` beside `ruff check`. 56 diagnostics down to 0: 31 fixed, 25
suppressed by three rules in `[tool.ty.rules]`, each with a note saying which
stub idiom it covers and how many sites it accounts for. The version is pinned
exactly rather than with `>=`, because ty is pre-1.0 and its inference moves
between releases.

The four groups were fixed, not suppressed:

- The annotations now resolve, but **not** by the `TYPE_CHECKING` import this
  item proposed. `TYPE_CHECKING` satisfies a checker and leaves
  `get_type_hints()` raising exactly as before, since the name never enters the
  module namespace -- so it would have fixed the diagnostic and not the defect.
  `ClvDataStaticCov`, `ClvData` and `StaticCovResult` are imported for real at
  module scope in `bgnbd`, `ggomnbd`, `pnbd/staticcov` and `pnbd/dyncov`, each
  with a note saying why that closes no cycle. The runtime-only local imports
  (`fit_static_covariates`, `build_walks` inside `ClvDataDynCov`) stay exactly
  where they were; they are what keeps the graph one-way.
- `Fitted` declares `hessian` and `__iter__` alongside `names`. `hessian` is an
  annotation rather than a property so that a frozen dataclass can still supply
  it as a field. `_covariance` narrows through a local, which turned up a
  second thing worth keeping: the dyncov fit reports no Hessian at all, so the
  `getattr` there guards absence, not just `None`.
- `predict.Family` is a `Protocol` over the three expressions every family
  provides, its members read-only properties because a module supplies them.
  `DERT` is Pareto/NBD only, so `_FAMILIES` now carries the expression itself
  rather than a `has_dert` flag -- the branch narrows on `is not None` and no
  cast is needed.
- `_search` runs every candidate and takes a `min`, so there is no point at
  which the winner is `None`, and its return type says `OptimizeResult`.

Four more real defects fell out of turning the checker on, all fixed:
`_require_positive(**params: float)` was annotated for a scalar and only ever
called with arrays; `log_likelihood_ind` returns a table only with
`intermediates=True`, which is now two `@overload`s rather than a union its
callers had to index blindly; `predict()` returns a `float`, not a
`DataFrame`, for an S6.3.4 new-customer scenario, and accepts one as
`clv_data`; and `np.linalg.inv` was reached with a possibly-`None` Hessian.

A second test, `test_the_shipped_annotations_resolve`, walks every module in
the package and calls `typing.get_type_hints()` on every function and class it
defines -- the thing a checker never does. Reintroducing the old local import
in `pnbd/staticcov.py` was confirmed to fail it with
`NameError: name 'ClvDataStaticCov' is not defined`, which `ty` alone reports
as a diagnostic that a `TYPE_CHECKING` block would have silenced.

Verified: 891 passed (888 plus the two new tests and a doctest), 1 deselected,
`TOTAL 2659 0 100%`, ruff clean over `src tests tools docs`, `ty check src`
clean, and `import clvtools` still works both as a fresh import and via
`importlib.reload`, and with `clvtools.data` or `clvtools.pnbd.dyncov`
imported first.

## 3. `[x]` Split `pnbd/dyncov.py`

At 655 code lines against the 700 limit, it is the largest module in `src/` and
nearly twice the next (`ggomnbd.py`, 365). Its complexity metrics are green
since `build_walks` was decomposed, so this is not urgent — but it is the file
the size gate will stop first, and it has a clean seam: walk primitives and
construction (`Walk`, `TransactionWalk`, `Customer`, `DyncovWalks`,
`_to_days`, `_prepare_covariates`, `_customer_specs`, `_stack`, `build_walks`)
on one side, likelihood and estimation on the other.

*Done when:* both halves are under the limit, `dyncov.py` re-exports whatever
moved so no import outside the package changes, and the module docstrings still
carry S3.3/S6.4. Beware the import cycle: `DyncovWalks` is used by the
likelihood, so the primitives must move *with* it to keep the dependency
one-way.

**Done:** `src/clvtools/pnbd/dyncov_walks.py` — the walk primitives (`Walk`,
`TransactionWalk`, `EMPTY_WALK`, `Customer`, `DyncovWalks`) and everything that
builds them from a transaction log and a covariate table (`build_walks` and its
helpers `_interval_index`, `_distance_to_interval_end`, `_WalkSpec`, `_to_days`,
`_prepare_covariates`, `_check_covariate_coverage`, `_CustomerSpecs`,
`_customer_specs`, `_stack`, `_real_trans_bounds`). `dyncov.py` keeps the
likelihood and the fit and goes from **655 code lines to 382**; the new module
is **309**. Nothing outside the package changed: `dyncov.py` imports the five
public names for real and re-exports them, so `tests/`, `docs/paper.md`,
`clvtools.data`, `clvtools.estimate`, `clvtools.predict` and
`pnbd/dyncov_predict.py` are untouched. Its `__all__` gains one entry,
`EMPTY_WALK` — the tests already imported it from `dyncov`, so it was public in
practice and is now declared, which is also what keeps ruff from calling the
re-export an unused import.

The dependency runs one way, likelihood → walks, and the import that closes no
cycle is the one that was already there: `dyncov_walks` imports `ClvData` at
module scope (so `build_walks`' annotation still resolves, per item 2), while
`ClvDataDynCov` keeps importing `build_walks` inside the method that calls it.
Nothing new is imported inside a function. Checked by importing the package in
four orders, `dyncov_walks` first among them.

The paper stays where the code went: `dyncov_walks` carries S3.3's covariate
intervals and the four kinds of walk, and `dyncov` keeps S3.3's rate equations
and the S6.4.2 likelihood, with a pointer to where the walks now live.

Verified: 891 passed, 1 deselected, `TOTAL 2666 0 100%`, ruff clean over
`src tests tools docs`, and the `quality` tests green — including
`test_the_shipped_annotations_resolve` and `test_the_limit_still_binds`, which
still binds because the largest file the gate sees is now `test_families.py` at
662. The module's own tests, run on their own, are 106 passed, 1 deselected
(the `dyncov_fit` MLE, deselected by `addopts` as always).

## 4. `[x]` Packaging metadata

`pyproject.toml` has name, version, description, readme, authors and
`requires-python`, and nothing else. No `classifiers`, no `[project.urls]`, no
`keywords`, no `license` field.

*Done when:* the metadata is complete enough that `uv build` produces a
distribution whose PyPI page is intelligible. Note this depends on item 5.

**Done:** an SPDX `license` with `license-files`, plus keywords, classifiers
and `[project.urls]`. The licence *classifier* is deliberately absent -- PEP
639 supersedes it with the expression, and `uv` warns when both are present.
Verified by building rather than by reading: `uv build` produces both artifacts
with no warnings, and the wheel's metadata carries `License-Expression:
GPL-3.0-only` and `License-File: LICENSE`, with the text under
`dist-info/licenses/` and `py.typed` shipped alongside.

## 5. `[x]` Licensing

There is no `LICENSE` file and no `license` field. CLVTools itself is GPL-3.
This is a from-scratch port written against the paper rather than a translation
of that source, so it is not automatically a derivative work — but that is a
judgement for the maintainer to make and record, not for an agent to guess.

*Done when:* the maintainer picks a licence. Do not choose one; if this is the
topmost unchecked item, report it and stop.

**Done:** the maintainer was asked and delegated the choice, so GPL-3.0-only,
matching CLVTools 0.12.1. The code alone would not have forced it -- it is
written against the paper, not translated -- but `data/` redistributes the four
datasets CLVTools bundles, and a permissive licence over GPL-3 data is an
inconsistency nobody needs. `LICENSE` is the unmodified FSF text taken from R's
own `share/licenses` rather than retyped. No per-file headers: recommended by
the GPL, not required, and noise in files whose docstrings are curated this
closely.

## 6. `[x]` Run the time-varying MLE somewhere

`-m 'not dyncov_fit'` is in `addopts`, so the one fit that takes ~13.5 minutes —
the time-varying covariate MLE, the most intricate estimator in the package —
never runs unless asked for by name. Its likelihood and its prediction are
tested at fixed parameters; the fit itself is not, in any routine run.

*Done when:* it runs on a schedule in CI (a nightly or weekly job), so a
regression in the dyncov optimiser is caught within a day rather than whenever
someone next types `-m dyncov_fit`.

**Done:** `.github/workflows/dyncov.yml`, nightly at 03:17 UTC plus
`workflow_dispatch`, one interpreter, `timeout-minutes: 120`. Nightly rather
than weekly because this item's own condition is "within a day", and a week of
commits is a bad bisect for a fit that costs a quarter of an hour to evaluate
once.

The trick is that a second `-m` overrides the `-m 'not dyncov_fit'` in
`addopts`; confirmed by collection (`1/892 tests collected, 891 deselected`)
rather than assumed. The job runs without `-q` on purpose: `addopts` already
carries one, so adding another gives `-qq` and suppresses the `1 passed` line
that makes a CI log worth reading.

The fit was run end to end twice, independently, agreeing to two seconds:
**13:29.96** and 13:27.25, both passing. That also retired a stale figure --
"about 17 minutes" appeared in nine places and is now 13.5 -- and corrected
`docs/performance.md`, which had derived "roughly 3,500 evaluations" from the
old time when the test docstring recorded 1,870 all along. The real average is
0.43s per evaluation, not the 0.29s measured at a single parameter vector.

Note the schedule is inert until this reaches the default branch: GitHub fires
`schedule:` only from the workflow file on `main`. On a side branch
`workflow_dispatch` is the only live trigger.

## 7. `[x]` Guard the performance invariants

Correctness and tidiness are gated; efficiency is not. `docs/performance.md`
profiles the package and finds nothing slow — but nothing stops it becoming
slow. The regressions that would actually cost minutes are all structural and
all invisible to the current suite.

Gate them by **counting operations, not seconds**. A wall-clock assertion would
be the first flaky gate here and the first one someone loosens, which this file
forbids anyway.

*Done when:* `tests/test_code_quality.py` — already the home for "static
analysis is a test" — or a sibling module asserts, on the standard fits:

- `special.hyp2f1_ratio` is called a bounded number of times per likelihood
  evaluation, on arrays of length *n*, rather than *n* times on scalars.
  De-vectorising it is a ~100x regression no current test would notice.
- its scalar series fallback stays cold: 0 of 348,000 elements during a
  `fit_pnbd` on the apparel data today.
- likelihood evaluations per fit stay in a measured band — 290 for `fit_pnbd`
  on the apparel data — which catches an optimiser or start-value regression.
- cost per customer is flat in *n*, asserted by operation counts at two input
  sizes so it is O(*n*) by construction rather than by stopwatch.

Choose each band by measurement, exactly as the ruff limits were chosen, and
record the measured value in a comment beside it.

**Done:** `tests/test_performance.py`, a sibling of `test_code_quality.py`
rather than an addition to it — these count operations rather than read source
— under a new `performance` marker. Eight tests and a doctest, **1.0 s** on
every `uv run pytest`. The subject is `fit_pnbd` on the apparel data (600
customers, no Hessian, 0.065 s) and `cdnow` at two sizes; nothing marked `slow`
or `dyncov_fit` runs. No assertion looks at a clock.

The invariants, each with what it measured on 2026-09-01:

- **`hyp2f1_ratio` is called a bounded number of times per likelihood
  evaluation** — measured 580 calls over 290 evaluations, i.e. exactly two, the
  `A_1` and `A_2` terms of Appendix A. The gate is `<= 2`, a bound rather than
  an equality, so folding the two into one stays legal.
- **Every call covers the whole sample** — measured 348,000 elements over 580
  calls on 600 customers, so `elements == calls * n` exactly. This is the
  assertion a per-customer loop fails hardest: it reports one element per call.
- **The scalar series fallback stays cold** — 0 of 348,000 elements.
- **Likelihood evaluations per fit stay in a band** — 290 measured, 200 to 400
  allowed. The width is set from the two regressions worth catching rather than
  from a round percentage: relaxing `ftol` to SciPy's own 1e-8 gives **150**,
  and Nelder-Mead — the fallback S6.2.1 recommends, and a plausible accidental
  default — gives **489**. Both bounds sit ~1.4x from 290, far more room than a
  SciPy point release moves a converged line search and far less than either
  regression needs.
- **Cost per customer is flat in *n*** — hypergeometric elements per customer
  per likelihood evaluation is 2.0 at 2,357 customers (330 calls, 777,810
  elements, 165 evaluations) and 2.0 at 1,178 (400 calls, 471,200 elements,
  200). The *evaluation* count is deliberately not compared across sizes: it is
  a property of the optimiser's path, not of *n*, and the smaller problem here
  took more of them.

The instrumentation is the part that can silently measure nothing.
`pnbd/aggregate.py` does `from clvtools.special import hyp2f1_ratio`, binding
the function at import time, so patching `clvtools.special.hyp2f1_ratio`
records **zero** and every test passes while watching an empty room. The
counters go on the module that *calls* the function; `Count.fired` makes a
counter that never ran a failure; and a test named for the trap pins it, by
asserting that a counter on
`clvtools.special` sees nothing while the one on `aggregate` fires.

Every gate was watched to fail, by breaking what it guards and reverting:

| Break | Test | Result |
|---|---|---|
| the two `hyp2f1_ratio` calls looped scalar-wise | bounded calls; whole sample | 348,000 calls of 1 element (1,200 per evaluation, against a bound of 2) |
| `unresolved` forced to all-true in `special` | fallback stays cold | series ran 324,000 times |
| `ftol` 1e-16 → 1e-8 | evaluation band | 150 evaluations, below 200 |
| default method → Nelder-Mead | evaluation band | 489 evaluations, above 400 |
| one `hyp2f1_ratio` call repeated `n // 100` times | flat in *n* | 12.0 per customer at 1,178 against 24.0 at 2,357 |
| `aggregate` reaching the function through the module | counters installed where the calls happen | the call-site counter recorded nothing |

Every one of those breaks leaves the fitted parameters and the log-likelihood
exactly right, which is the whole argument for the module: nothing else in the
suite moves.

Verified: 900 passed, 1 deselected, `TOTAL 2666 0 100%` in 4:36 with coverage,
ruff clean over `src tests tools docs`, `ty check src` clean. `README.md`,
`CLAUDE.md` and `docs/performance.md` carry the new marker and the new count.

## 8. `[x]` A committed profile report

`tools/benchmark.py` reports Appendix B's wall-clock. Nothing reports *where*
the time goes, so every question about it starts from scratch — as
`docs/performance.md` had to.

*Done when:* a `tools/profile.py` sibling emits a cProfile summary of the
standard paths (`summary()`, `fit_pnbd`, `fit_pnbd_staticcov`, one dyncov
likelihood evaluation) in a form that can be pasted into `docs/performance.md`
and diffed between versions. Informational, not a gate — it must not be able to
fail CI.

**Done:** `tools/profile.py`, a sibling of `tools/benchmark.py` in structure,
argument style and register. **6.0-6.6 s** for all four paths. It emits markdown —
a header naming the machine, the interpreter and the numpy/scipy/pandas
versions, then one section per path with its unprofiled median wall clock, its
profiled total, and a table of the hottest functions — so its output goes into
`docs/performance.md` verbatim.

Built to be *diffed*, which drove three choices. Rows carry **call counts and
shares of `tottime`**, not seconds: a call count is a property of the code and
moves only when the code does. Labels drop the absolute path and the line
number and keep `clvtools/pnbd/dyncov.py:_hyp_beta_gt_alpha`, so a diff survives
both another machine and an edit above the function; entries that then share a
label are summed. And ordering is `tottime` descending with **ties broken by
name**, without which the cool rows shuffle every run.

Where a path has a natural denominator the table reports calls *per likelihood
evaluation* as well as calls, taken from the profile's own count for the
likelihood rather than from the fitter, so it works the same for the static
covariate fit, which exposes none. `fit_pnbd` reports 290 evaluations and
`hyp2f1_ratio` at 2 per evaluation, matching `tests/test_performance.py`
exactly. The dyncov path is **one** likelihood evaluation and never the fit;
`build_walks` is setup, timed and reported separately at 0.432 s.

Nothing about it can fail CI: no test imports it, no workflow runs it, it
asserts nothing and exits 0. It is in `tools/`, which ruff checks, and it is
clean. Its four doctests are correct but *not* collected -- `testpaths` is
`src tests docs` -- so they were run once by hand through `doctest.testmod`;
wiring them into the suite would have made the tool a test, which this item
forbids.

One trap, and it is fatal rather than subtle: **`tools/profile.py` shadows the
standard library's `profile`**, which `cProfile` imports and reads at import
time. Running the script puts `tools/` first on `sys.path`, so `import
cProfile` re-executes this very file as `profile` and dies with
`AttributeError: partially initialized module 'cProfile' has no attribute
'Profile'` -- confirmed against a stub before writing a line of the tool.
`_import_cprofile` drops the script's own directory for the duration of that
one import and puts it back, which is what makes the filename this item asks
for safe. A local import rather than a `# noqa: E402`, in the house style
already recorded for `PLC0415`.

Running it immediately paid for itself: `docs/performance.md` credited
`_hyp_beta_gt_alpha` with **39,754** calls per dyncov evaluation, which is
`_hyp_term`'s count -- the dispatcher, splitting 38,542 onto `beta > alpha` and
1,212 onto `alpha >= beta`. Every other count in that table reproduced to the
digit (`elem` 155,418, `sum_from_to` 77,110, `first` 99,647, `n_elem` 114,978,
`d_i` and `b_i` 39,755), so the document was right except where it named a
function, and the profile it was written from no longer existed to check. That
is the item in miniature. The document now carries the correction and says the
table is regenerable.

Two figures changed with the denominator rather than with the code. The old
"57% of a `fit_pnbd` is `hyp2f1_ratio`" divided a *profiled* `tottime` by an
*unprofiled* wall clock; against the profiled run's own total it is 50%. The
tool prints both totals on every path so the two cannot be mixed again. Wall
clocks moved 0-13% (`fit_pnbd` 0.065 -> 0.063 s, `summary()` 0.111 -> 0.103 s,
dyncov 0.290 -> 0.328 s, `build_walks` 0.454 -> 0.432 s) under numpy 2.5, scipy
1.18 and pandas 3.0; all refreshed from one run, median of five, so the
document is a single coherent paste rather than a stitch of several.

Verified: 901 passed, 1 deselected, `TOTAL 2666 0 100%`, ruff clean over
`src tests tools docs`, `ty check src` clean.

## 9. `[x]` Spike: vectorise the dyncov likelihood over covariate intervals

`docs/performance.md` measures one evaluation of the time-varying covariate
likelihood at ~0.33 s, and roughly **600,000 Python-level calls into this
package for one number** — 39,754 dispatches to `_hyp_term`, 155,418 to
`Walk.elem`, all of it interpreter overhead rather than library work. The
computation is a sum over covariate intervals, which is the shape that
vectorises. A fit is 1,870 evaluations, so this is the whole 13.5 minutes.

**This is a spike, not a commitment.** The upside is plausible and unproven.
Abandoning it with a written finding — "vectorising the inner loop gains X%,
here is where the rest goes" — is a perfectly good outcome and worth more than
the guess in `performance.md` is now.

Deliberately deferred until items 3 and 7 existed. It is safe to attempt only
because the likelihood is pinned against oracle fixtures expression by
expression at several parameter vectors, and because there are now counters
that can show a rewrite changed no arithmetic.

*Done when:* either the inner loop is vectorised with every oracle fixture
still green **expression by expression, not merely in total**, and the measured
gain is recorded; or the attempt is abandoned and `docs/performance.md` records
what was learned about why. Three constraints, each of which has already caught
something:

- **Order of operations is load-bearing.** `CLAUDE.md`: whole-day arithmetic
  shifts `d1` and `tjk` by ~4e-13 if done in nanoseconds, which breaks the
  exact cancellation that makes `F2.2` vanish. Preserve the order, not just the
  algebra.
- **Both hypergeometric arms must be exercised.** The `alpha >= beta` arm is
  reached only near the optimum — at a convenient starting vector it takes
  **zero** of 39,754 dispatches (see `performance.md`), so a rewrite validated
  there could ship a broken arm and pass. Measure and test at the *fitted*
  parameters.
- **The gated invariants in `tests/test_performance.py` cover `fit_pnbd`, not
  this path.** Nothing currently counts the dyncov likelihood, so a regression
  here is invisible to them; consider whether the spike should leave one behind.

**Done: the spike paid, and it also corrected the guess that motivated it.**
`_f2`'s per-interval loop is now `_f2_middle`, which builds `B_i`, `D_i` and
both hypergeometric arms as arrays over all ~66 of a customer's covariate
intervals at once; the arms take batches and dispatch per element, so a customer
whose intervals straddle the `alpha >= beta` branch is handled in one call.
39,754 scalar arm dispatches became 2,385, and 232,528
`Walk.elem`/`Walk.sum_from_to` calls became one `cumsum` per walk.

**Per evaluation, 3.3-5.1x. On the fit, 1.33x — 13:27 to 10:07.** The second
number is the finding. Timing all 1,925 evaluations of a fit shows the cost
stepping up sevenfold at the fourth decile and staying there, and at the vector
the search then dwells on (`life.High.Season = -8.12`, the coefficient this
implementation's optimum is already known to differ on) **84% of self-time is
inside `scipy.special.hyp2f1`** and the rewrite buys 1.5x rather than 5x. So the
time-varying likelihood is Python-bound in the easy part of the parameter space
and library-bound in the part the optimiser spends two thirds of its time in.
`docs/performance.md` has the deciles, the four-vector comparison and the
profile. The next lever on this fit is `hyp2f1`, not Python.

Against the three constraints:

- **Order of operations.** 27 of the 30 per-customer intermediates are
  **bit-identical** at both oracle parameter vectors, `F2.1` and `F2.2` among
  them. `F2.3` moves by up to 2.2e-15 relative and carries `F2` and `LL` with
  it. One cause, isolated by rerunning with the prefixes summed the old way:
  `ndarray.sum` adds pairwise, `np.cumsum` left to right. Substituting a
  pairwise `_prefix_sums` makes all thirty columns bit-identical again, which
  proves nothing else moved. It was not substituted: the shift is an order of
  magnitude *smaller* than the package's existing disagreement with CLVTools
  (2.3e-13 on `F2.3`), pairwise is an artefact of numpy's blocking rather than
  anyone's specification, and it costs 37% of a single evaluation. It is a close
  call — over a whole fit the two are within 3% — and `_prefix_sums` is one
  function with the argument in its docstring.
- **Both arms.** Exercised, and now asserted. The `mle` grid case splits
  1,212 / 38,542 and the `offset` case 12,331 / 27,423, so the oracle suite was
  already covering both — the *profile* was the thing that wasn't.
- **Fixtures green expression by expression.** All 30 columns, both vectors,
  unchanged tolerances.

Left behind: `tests/test_performance.py::TestDyncovStaysVectorised` (three
counted invariants, demonstrated to fail at 66.3 dispatches per customer against
a deliberately unrolled `_hyp_terms`) and
`test_the_batched_middle_sum_matches_the_scalar_one`, which holds `_f2_middle`
against the loop it replaced for all 600 customers at both vectors, with both of
`d_i`'s branches asserted to have been taken.

One figure was left stale on purpose: `CLAUDE.md` still says the dyncov fit is
`~13.5 min`. It is 10:07 now, but `CLAUDE.md` is the maintainer's file.


---

# Round 3 — publication

Added 2026-09-02, from a review of the whole repo against its own records. The
first two rounds asked whether the port is *correct*; the nine items above asked
whether the package is in *good shape*. This round asks the remaining question:
whether it is fit to leave this machine. Nothing here is model work.

What the review measured, so that the items below rest on evidence rather than
on a plan:

- `uv run pytest` — 906 passed, 1 deselected, exit 0. `ruff check src tests
  tools docs` clean. Both audit rounds closed; items 1-9 above closed.
- **Nothing has ever been pushed.** There is no git remote, and
  `gh repo view neilmurrayee/clvtools` answers *"Could not resolve to a
  Repository"*. So neither workflow has ever run, and `dyncov.yml` says so in
  its own header.
- 17 commits sit on `interface-layer`; `master` is an ancestor of it and has
  none of them.
- 13 of the 30 commits carry a personal email address as author and committer.
- `clvtools` is **unclaimed on PyPI** — `/pypi/clvtools/json` is 404 on both
  the real index and TestPyPI.

## 10. `[x]` Take the personal address out of the history

`git log --format='%ae'` gives two identities: the first thirteen commits carry
a personal name and a personal mailbox — not written out here, because a task
file that quotes the address it exists to remove has put it straight back into
a blob — and the other seventeen carry
`neilmurrayee <132654876+neilmurrayee@users.noreply.github.com>`, which is what
`git config user.email` has held since. (That trap was sprung while writing
this item, and the commit was amended before the rewrite rather than after.) Nothing has been pushed, so nobody holds
a copy of the old hashes and the rewrite costs a force-push to precisely no one.
This is the one item that must happen *before* anything reaches a remote: after
a push, the address is on someone else's disk and no rewrite here can recall it.

The working tree is already clean of it — the only address anywhere in the
repository is the GitHub noreply in `pyproject.toml`'s `authors`, which is the
anonymised form by construction: it routes nowhere and exists so that GitHub can
attribute a commit without publishing an inbox.

*Done when:* every commit on every ref reports the noreply address and the
single name `neilmurrayee` as both author and committer; `git log` over all refs
matches no `gmail`; `refs/original/`, the reflogs and the loose objects the
rewrite orphans are gone, so the address is not recoverable from the clone that
gets pushed; and the address is pinned in the repository's own config, not only
in the global one, so a later change to `--global` cannot leak it back into this
repo.

**Done:** `git filter-branch --env-filter` over `interface-layer` and `master`,
setting all four identity variables unconditionally rather than mapping the old
address to the new one -- a substitution leaves whatever it did not think to
match, and there is no reason for this history to carry more than one identity.
All **31** commits now report `neilmurrayee
<132654876+neilmurrayee@users.noreply.github.com>` as author *and* committer.
The GitHub noreply is the anonymised form rather than a compromise: it is what
attributes a commit to the account, and it reaches no inbox.

The rewrite changed identities and nothing else, which was checked rather than
assumed: `HEAD^{tree}` is **bit-identical** to the pre-rewrite tree
(`a242a6c2`), `git diff refs/original/... HEAD` is empty, the 31 subjects
`diff` clean against the originals, and author dates are preserved end to end
(2026-08-27 to 2026-09-02). The suite was not re-run for it, deliberately: an
identical tree cannot test differently, and saying so is worth more than three
and a half minutes spent proving arithmetic that did not move.

Then the paths back were cut, in this order: the safety branch deleted,
`refs/original/` removed, `reflog expire --expire=now --expire-unreachable=now
--all`, `gc --prune=now`. Verified afterwards by looking rather than by
trusting: the pre-rewrite commit is unreachable (`git cat-file -e` fails),
`git log --all` matches the old name and address zero times, **every reachable
blob** was piped through `cat-file --batch` and matches zero times, and
`grep -ril` over the whole of `.git/` finds nothing. `user.name` and
`user.email` are now set in `.git/config` as well as globally, so a later change
to `--global` cannot put a personal address back into this repository.

One trap, sprung and disarmed within the item: the first draft of this section
quoted the address it exists to remove, which would have written it into a blob
of the very commit that announces its removal -- and `git log -S` found it
exactly because the rewrite was preceded by a search rather than started
straight away. The commit was amended before the rewrite, so the blob never
outlived the pre-`gc` object store.

Nothing was pushed at any point, so this cost a force-push to nobody. That
window is now closed by item 13.

## 11. `[x]` Correct three counts that are wrong in the workflow comments

Found by counting rather than by reading. None of them is a gate — all three are
comments — but they are the first thing a reader meets in the file that claims
to define green, and one of them was wrong on the day it was written.

- `ci.yml:87` says "888 tests". It is 906.
- `ci.yml:11` says `tests/fixtures/` holds "118 files". It holds **123**, and
  `git log 4d5a117..HEAD -- tests/fixtures` is empty, so none were added after
  that workflow landed: 118 was never right.
- `dyncov.yml:89` cites collection as `1/892 tests collected (891 deselected)`.
  It is `1/907 (906 deselected)` today.

*Done when:* each figure is what the command it describes actually prints, and
the counts are quoted from a run rather than from the last time someone looked.

**Done, and it was four rather than three.** Counting again before editing --
which is the whole method of this item -- turned up one the review had missed:
`dyncov.yml:5` describes the job as "~1,870 likelihood evaluations", which is
the count from *before* item 9 vectorised the likelihood. `docs/performance.md`
already records the vectorised fit converging in 1,925 and 1,562, and the test
docstring already says "some 1,900"; only the workflow still quoted the old
number, in the file whose entire purpose is to run that fit.

The four, each replaced with what the command prints today:

| Where | Was | Is |
|---|---|---|
| `ci.yml:11` | `tests/fixtures/` "118 files" | **123** |
| `ci.yml:87` | "888 tests" | **906** |
| `dyncov.yml:5` | "~1,870 likelihood evaluations" | **some 1,900** |
| `dyncov.yml:89` | `1/892 tests collected (891 deselected)` | `1/907 (906)` |

`ci.yml`'s note now also carries `TOTAL 2694 0 100%` beside the test count and
the date both were read, so the next reader can tell staleness from
disagreement. Measured rather than reasoned: `ls tests/fixtures | wc -l`,
`pytest --collect-only`, and `pytest -m dyncov_fit --collect-only` for the line
that quotes a deselection count the job itself never prints.

Also reflowed a three-line comment in `dyncov.yml` that an earlier edit had
left wrapped mid-sentence ("A GitHub-hosted / runner is / markedly slower").

Verified: both files still parse (`yaml.safe_load` under `uvx --with pyyaml`,
no dependency added) and both still name the job they did; `pytest -m quality`
green. Nothing executable changed -- every edit is inside a comment, which is
also why this item could not be verified by a run and had to be verified by
counting.

## 12. `[x]` Point `CLAUDE.md`'s layout at the two documents it omits

`CLAUDE.md`'s Layout block lists `docs/paper.md`, `docs/vignette.md` and
`docs/audit.md`. It does not mention `docs/backlog.md` — *this file*, which
declares itself the loop's state and is where an agent is supposed to start —
nor `docs/performance.md`, which the README cites four times. An instruction
file that does not name the file holding the instructions' queue is a gap a
new session pays for, not the maintainer.

*Done when:* both appear in the Layout block with a one-line description in the
register of the entries around them.

**Done:** two lines in the Layout block --

    docs/backlog.md     what is left once the port is right — the work queue; start here
    docs/performance.md where the time goes, and why nothing asserts a wall clock

-- and, under the block, two sentences stating the loop's rule where a new
session will actually meet it: work the topmost unchecked item, tick it with
what was measured, leave `[needs-decision]` alone. The rule lived only inside
the file it governs, which is no use to someone who has not opened it.
`AGENTS.md` is a symlink to `CLAUDE.md`, so it followed for free.

Item 11's method paid again on the way past: counting the marker totals that
`CLAUDE.md` and `README.md` both print found `-m oracle` at **231**, not the
229 both claimed. The other five are right -- `paper` 24, `rdoc` 22, `slow`
138, `quality` 7, `performance` 13 -- and are now confirmed rather than
inherited. Both files corrected in the same commit, since it is the same drift
by the same cause.

## 13. `[ ]` Push, and let the gates run for the first time

Everything item 1 built is unexercised. Four things need settling in the same
breath, because they contradict each other today:

- The repository does not exist yet under `neilmurrayee`.
- `pyproject.toml`'s `[project.urls]` point at `blob/main/...` while the local
  default branch is `master`, so the Documentation and Changelog links would
  404 on the branch name even once the repository exists.
- `dyncov.yml`'s schedule fires only from the default branch, so the nightly
  fit stays inert until this history lands there.
- **`[needs-decision]` The paper is in the repository.** `git ls-files` tracks
  40 files under `arXiv-2602.09845v1/` plus `2602.09845v1.pdf` — 3.2 MB, the
  complete LaTeX source and typeset PDF of somebody else's article. Locally
  that is a working copy of the specification. Pushed to a public repository it
  is redistribution, and the arXiv package declares no licence: `00README.json`
  lists source files and nothing about terms, and no `LICENSE`, `COPYING` or
  copyright line appears in `jss5634.tex`. arXiv's default submission licence
  grants arXiv a non-exclusive right to distribute; it grants a third party
  none. This does not touch the port itself — quoting a section number, an
  equation and a sentence in a docstring is citation, and the paper's *numbers*
  are facts — only the wholesale copy. The options are to untrack both and
  `.gitignore` them, keeping the local copy and citing the arXiv id and DOI
  from the README; to keep them if the licence turns out to be a CC one, which
  is worth thirty seconds on the arXiv abstract page; or to make the repository
  private, which forecloses items 1, 6 and 15. Settle it *before* the first
  push and the answer costs a `git rm --cached`; settle it after and it needs a
  history rewrite, exactly like item 10.

  **Settled, 2026-09-02, and done.** The licence is the question and arXiv
  answers it: 2602.09845 carries `arxiv.org/licenses/nonexclusive-distrib/1.0/`,
  whose whole text is *"I grant arXiv.org a perpetual, non-exclusive license to
  distribute this article."* arXiv, and nobody else — so "keep them if it is a
  CC licence" was not available, and of the remaining two, untracking is the one
  that does not foreclose items 1, 6 and 15.

  Two commits, because untracking is only half of it. The first stops the files
  being tracked (they stay on disk, `.gitignore`d, with the two fetch commands
  in the comment) and repoints `CLAUDE.md`, `README.md`, `docs/paper.md` and
  `tests/paper_values.py` at arXiv:2602.09845 — `docs/paper.md`'s relative link
  to the `.tex` would have 404'd on GitHub, and a link check over all of `docs/`
  says nothing else would. The second is the half that is easy to miss: **a
  push publishes history**, and all 36 commits still carried the paper, 40 blobs
  and 3.1 MB of a 4.6 MB `.git`. Untracking at the tip would have redistributed
  it anyway. `filter-branch --index-filter` over both branches, then
  `refs/original` dropped, reflogs expired, `gc --prune=now`.

  Verified as item 10 was: `HEAD^{tree}` **bit-identical** before and after
  (`7f80caa4`, which is the test that this removed only what the tip had already
  untracked), 36 subjects `diff` clean, the single identity intact,
  `git log --all --name-only` matching the paths **zero** times,
  `git rev-list --objects --all` matching **zero** objects, and `.git` down from
  4.6 MB to **2.0 MB**. The suite was not re-run against the rewritten history
  for the same reason as item 10: an identical tree cannot test differently.
  `--prune-empty` was passed in case a commit had done nothing but add those
  files; none had, and the count stayed 36.

*Done when:* the repository exists, the default branch and the URLs in
`pyproject.toml` agree with each other, `ci.yml` has gone green on 3.12 and
3.13 against a runner that has no R installed — which is the committed-fixture
promise finally being tested rather than asserted — and `dyncov.yml` has been
dispatched by hand once, so its first run is watched rather than nocturnal.

## 14. `[ ]` Spike: the cost of `hyp2f1` where the dyncov search dwells

Item 9 ended by naming its own successor and this is it, with the measurement
already taken. After vectorising, the fit fell only 1.33x — 13:27 to 10:07 —
because at the vector the optimiser spends two thirds of its time near
(`life.High.Season = -8.12`), **83.8% of self-time is inside
`scipy.special.hyp2f1`** over 948 calls. The interpreter overhead is gone;
what is left is the library doing real work, in one region of the parameter
space.

`docs/performance.md` names the two levers and does not choose between them:
the cost of `hyp2f1` in that region, or keeping the search out of it. The
second is the more interesting one, because that region is also where this
implementation's optimum is known to differ most from CLVTools'.

**A spike, on the same terms as item 9.** Abandoning it with a written finding
is a good outcome. The oracle fixtures pin the likelihood expression by
expression at both parameter vectors, and `TestDyncovStaysVectorised` counts
the dispatches, so an attempt cannot quietly change the arithmetic.

*Done when:* either an approach is measured and adopted with the fixtures still
green expression by expression and the gain recorded, or the attempt is
abandoned and `docs/performance.md` records what was learned about why.

## 15. `[needs-decision]` Publish to PyPI

Item 4 got the metadata to where `uv build` is clean, and `dist/` holds a
`clvtools-0.1.0` wheel and sdist that have never been uploaded. The name is
free — 404 on PyPI and on TestPyPI — so the decision is not being forced by
anyone else.

What it needs from the maintainer: whether to publish at all; whether `0.1.0`
and `Development Status :: 4 - Beta` are the version and the status to go out
under; and whether uploads happen from a laptop with a token or from a GitHub
Actions job with Trusted Publishing, which is the safer shape and which item 13
has to land before anyone can configure.

Note the ordering trap: the name is claimed by the first upload, and the
metadata that upload carries is what PyPI shows until the next release. Item 13
first, so the URLs on that page resolve.

## 16. `[needs-decision]` `bgbb`

The one model-level gap against CLVTools, recorded in `docs/audit.md`'s "Not
gaps" and unchanged since: `bgbb` is exported by the R package and absent from
the paper, whose Table 4 lists three families. It is out of scope by a decision
that was correct for a port *of the paper*, and it is the only thing left that
would change the answer to "does this cover CLVTools?".

Not an item to work. A scope question, listed so it stops being invisible.

---

## Closed

Both audit rounds. See `docs/audit.md`:

- **Round 1** — fourteen items, the paper and CLVTools' `NAMESPACE`.
- **Round 2** — the R package's own documentation: five published tables and
  the CDNOW dataset, two features built (`summary(ids=)`, `I(...)` formula
  terms), one withdrawn as not a gap, and `docs/vignette.md`.
- **Static analysis** — ruff with measured limits, wired into `pytest` as
  `tests/test_code_quality.py`.
