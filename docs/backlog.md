# Backlog

The paper and the R package's documentation are both covered — `docs/audit.md`
records both rounds, and every item in them is closed. What remains is not
model work. It is the difference between "the port is correct" and "the package
is in good shape", and each item below was found by measurement, not by
speculation.

**This file is the loop's state.** Work the topmost unchecked item, tick it, and
stop. Do not add items without evidence, and do not work an item marked
`[needs-decision]` — those need the maintainer.

**Nothing is open.** All eight items are closed, item 8 last; the loop has no
topmost unchecked item to work. A ninth needs evidence, in the sense the
paragraph above means it — something measured, not something imagined.

Definition of done for every item: `uv run pytest` green (901 tests at the time
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

---

## Closed

Both audit rounds. See `docs/audit.md`:

- **Round 1** — fourteen items, the paper and CLVTools' `NAMESPACE`.
- **Round 2** — the R package's own documentation: five published tables and
  the CDNOW dataset, two features built (`summary(ids=)`, `I(...)` formula
  terms), one withdrawn as not a gap, and `docs/vignette.md`.
- **Static analysis** — ruff with measured limits, wired into `pytest` as
  `tests/test_code_quality.py`.
