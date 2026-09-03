# Backlog

The paper and the R package's documentation are both covered — `docs/audit.md`
records both rounds, and every item in them is closed. What remains is not
model work. It is the difference between "the port is correct" and "the package
is in good shape", and each item below was found by measurement, not by
speculation.

**This file is the loop's state.** Work the topmost unchecked item, tick it, and
stop. Do not add items without evidence, and do not work an item marked
`[needs-decision]` — those need the maintainer.

**Rounds 3 and 4 are closed.** Every item, 1 to 33, is done. Both reviews of
2026-09-02, the spec-derived audit and the two audit rounds before them have no
open work left.

**Round 5 is open**, and it is the one class `docs/spec-audit.md` left behind:
the **76 `weak` verdicts** in its Appendix 3 — spec claims the suite touches but
does not pin. That document's own caveat calls them its least certain class, and
it never worked them individually. Item 34. Everything else is closed, and item
16 was the last one carrying `[needs-decision]` — no item does now. The phrase
survives on two settled sub-bullets inside closed items 13 and 22, which say so
where they stand.

**Both gates are green, which they had never been when this paragraph was first
written.** Read on 2026-09-03 with `gh run list`: `ci.yml` has 29 runs, 5 of
them successes, and the latest —
[33735879159](https://github.com/neilmurrayee/clvtools/actions/runs/33735879159)
— succeeded on `2506c67`, which is `main`'s tip. `dyncov.yml`, the nightly
time-varying MLE, has run twice and succeeded both times, most recently
[33731273502](https://github.com/neilmurrayee/clvtools/actions/runs/33731273502)
on `a8f35e1`. The earlier text here said CI "has run four times and **has never
been green**", citing run 33602019629; that was true when written and is the
reason the rule below exists.

**A rule this file did not have, and should have.** A claim about an external
system — CI, PyPI, GitHub — carries the run id, URL or command output that
established it. Item 17 was ticked with "green on GitHub on both interpreters"
while that run was still in progress and then failed; the tick was written from
a local suite and an expectation. Ticking on an expectation is how a record
becomes wrong, and correcting it is cheaper than trusting it afterwards.

Definition of done for every item: `uv run pytest` green (1,244 passed and 1
deselected as of 2026-09-03; `TOTAL 2874 0 100%`), `uv run ruff check src tests
tools docs` clean, and 100% line coverage of `src/`. Anything that changes
behaviour also needs a test and, if it deviates from CLVTools, a README findings
entry — the house rule.

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

## 13. `[x]` Push, and let the gates run for the first time

Everything item 1 built is unexercised. Four things need settling in the same
breath, because they contradict each other today:

- ~~The repository does not exist yet under `neilmurrayee`.~~ **Done:** created
  public on 2026-09-02, <https://github.com/neilmurrayee/clvtools>.
- ~~`pyproject.toml`'s URLs point at `blob/main/...` while the default branch is
  `master`.~~ **Done:** `master` was renamed to `main`, fast-forwarded to the
  tip, and `interface-layer` retired into it, so the URLs resolve.
- ~~`dyncov.yml`'s schedule fires only from the default branch.~~ It is now on
  `main`, so **the nightly cron is live and has never been watched**. Dispatching
  it once by hand is the outstanding half of this item.
- **Done: CI is green.** Run
  [33612912362](https://github.com/neilmurrayee/clvtools/actions/runs/33612912362),
  success on 3.12 and 3.13, ten steps each, on a runner with no R — which is
  the committed-fixture promise tested rather than asserted.
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

**Eleven runs to get there, and every failure was a real defect.** Worth
recording, because the argument for CI is usually made in the abstract and this
is the concrete version: an action version that could not resolve (only a runner
resolves those); convergence tolerances asking for better than machine
precision, which made Linux report `converged = False` at the *same* optimum;
five rounds of doctests asserting digits no libm fixes — including the rule
"log-likelihoods are portable", which is false at a regularized optimum; and
three assertions written badly while fixing the others.

Two cautions learned on the way. `cancel-in-progress` means a push cancels the
run before it — with CI at 7-9 minutes and pushes every 10-20, three runs in a
row read `cancelled` rather than reaching a verdict, which looks like failure
and is not. And a failing doctest aborts its whole file, so each red run hid
whatever came after the first failure; sweeping a class of defect at once beats
fixing the one the log happens to show.

*Done when:* the repository exists, the default branch and the URLs in
`pyproject.toml` agree with each other, `ci.yml` has gone green on 3.12 and
3.13 against a runner that has no R installed — which is the committed-fixture
promise finally being tested rather than asserted — and `dyncov.yml` has been
dispatched by hand once, so its first run is watched rather than nocturnal.

## 14. `[x]` Spike: the cost of `hyp2f1` where the dyncov search dwells

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

**Done, 2026-09-03 — the first lever is closed and the second turned out not to
be the choice.** `docs/performance.md` carries the measurements; the short
version:

The claim reproduces. One evaluation is 0.480 s at `life.High.Season = -8.12`
against 0.120 s at CLVTools' fit, **85.2% of it inside `hyp2f1`** (the item said
83.8%). Capturing every argument at both vectors shows why, and it is narrower
than "SciPy does real work": the *same* 79,508 hypergeometrics in the same 4,770
calls, `a` and `b` with identical ranges, and only `z` moving — 27.6% of the
calls cross `z > 0.999`, max 0.9999, where the fast vector's max is 0.9844.

**Both exact rewrites fail, and the pair is the finding.** `c = a + 1` in both
arms, so two classical routes open up. The `1-z` connection formula collapses
beautifully — `2F1(a,b;b;x) = (1-x)^-a` kills one of its hypergeometrics — and
is **4.8x faster and wrong by 5.7e35 relative**: `(1-z)^(1-b)` reaches 1e88
against an O(1) answer, which is genuine cancellation and so is not reachable
by the log-space treatment that saved item 28. Euler's transformation has one
term, cannot cancel, is accurate to 5e-15, and is **not faster** — SciPy is
already doing it. So the fast transformation and the accurate one are the same
transformation, and it cannot be both.

**What does work is structural, and it contradicts this repo's own written
prediction.** The covariates are categorical, so `z` takes few values: of 79,508
hypergeometrics per evaluation there are **5,303 distinct `(a,b,z)` — 93.3%
duplicates**, from 1,570 distinct `z` and 31 distinct `(a,b)`. Collapsing them
is *bit-exact* — `np.array_equal`, since it is the same function on the same
arguments — and, with `np.unique`'s own cost charged to it, takes the
hypergeometric from 0.409 s to 0.067 s at the dwell vector. It costs 0.045 s →
0.059 s at the fast one, where there is nothing to save.

It is **not a local change**: within one customer's call 0% is removable, the
93.3% is entirely across customers, and the median call is one element wide. So
it needs the likelihood batched over the cohort — item 9 one level up. That is
item 30, and `docs/performance.md`'s closing bullet argued *against* exactly
that refactor on the grounds that it "would not touch" the hypergeometric cost.
The reasoning was sound and the conclusion wrong: batching does not merely
remove Python dispatch, it puts the duplicate arguments in one array where they
can be collapsed before SciPy sees them. That bullet is now corrected in place
rather than deleted.

Nothing in `src/` changed, so the fixtures are untouched by construction; the
suite is green and this item's whole output is measurement and two documents.

## 15. `[x]` Publish to PyPI — decided: no

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

**Decided, 2026-09-02: not publishing.** The maintainer's call, and it closes
the item rather than deferring it. What the work for it bought stays and was
worth having on its own terms: the wheel now carries its datasets (item 18,
where `pip install` of the built artifact raised `FileNotFoundError` on the
README's first line), the metadata is complete, and `CHANGELOG.md` says plainly
that 0.1.0 was built and never published. `pip install
git+https://github.com/neilmurrayee/clvtools` is the supported route and is
verified end to end from a clean environment — the README's Installation
section leads with it.

If this is ever revisited, the name was free on both indexes as of today and
Trusted Publishing from Actions is the shape to use; nothing here blocks it.

## 16. `[x]` `bgbb` — decided: nothing to port, the gap does not exist

The one model-level gap against CLVTools, recorded in `docs/audit.md`'s "Not
gaps" and unchanged since: `bgbb` is exported by the R package and absent from
the paper, whose Table 4 lists three families. It is out of scope by a decision
that was correct for a port *of the paper*, and it is the only thing left that
would change the answer to "does this cover CLVTools?".

Not an item to work. A scope question, listed so it stops being invisible.

**Closed 2026-09-03, and it was never a scope question.** Asked of CLVTools
0.12.1 itself rather than of its `NAMESPACE`:

```
> bgbb(clvdata(apparelTrans, date.format="ymd", time.unit="w",
+               estimation.split=104))
Error: This model has not yet been implemented!
```

`man/bgbb.Rd` agrees, and says so three times over: the title is "BG/BB models
- **Work In Progress**", the description is "Fits BG/BB models on transactional
data with static and without covariates. **Not yet implemented.**", and the
value is "**No value is returned.**" Three S4 methods are registered — for
`clv.data`, `clv.data.static.covariates` and `clv.data.dynamic.covariates` —
and all three raise. `NEWS.md` has never mentioned BG/BB.

So the paper's "not currently included in CLVTools" is **accurate, not
outdated**, this port is not behind the reference implementation, and there is
no feature to port. The question this item existed to ask — "does this cover
CLVTools?" — is answered yes, with nothing outstanding.

**How the wrong premise got in is the part worth keeping.** The session that
raised it checked `args(bgbb)`, saw a full fitting signature carrying
`start.params.model`, `reg.lambdas` and the covariate arguments, and read that
as an implementation. The signature is real; the body is a `stop()`. That is the
same shape as two defects already in the README's findings — the wheel that
passed every test in a checkout because a checkout always has the files, and
`tools/benchmark.py` documented in the README while raising on every
invocation — where what was checked was not what mattered. It reached the right
answer by the wrong route, which is the kind of correctness that stops being
correct when the circumstances move.

Corrected in three places rather than one, since the claim had spread:
`README.md`'s *Deliberately not ported* (whose premise it was), `docs/audit.md`'s
*Not gaps*, and `docs/spec-audit.md`'s Appendix 4 on M-13 — that last as a note
in place rather than a rewrite, on the same principle as
`docs/performance.md`'s corrected prediction.

**One condition on this closure.** It is a statement about CLVTools **0.12.1**
(dated 2025-11-06), the version this port targets and the one in `.Rlib/`. If
the oracle is ever re-baselined against a later release, re-run the call above
rather than assuming this still holds.


## 17. `[x]` Make the suite pass off macOS/ARM

Opened and closed by the first CI run, which is what item 13 was for. Seven
failures on both interpreters, none of which reproduce on this machine, in four
distinct classes -- and the repo had no way to know, because every number in it
had only ever been produced by one libm on one architecture.

**The one that is a real defect, not a test artefact.** `_optimize` set
`ftol = 1e-16` and `gtol = 1e-14` for L-BFGS-B. SciPy turns `ftol` into
`factr = ftol / eps`, so `1e-16` asks for a relative reduction of **0.45 eps** --
better than machine precision, which no line search can report satisfying. And
`gtol = 1e-14` is unreachable on an objective of order 5e3, whose gradient
cannot be resolved below about 1e-9. Both exits were therefore closed, leaving
only line-search failure. macOS/ARM happened to reach a reduction of exactly
zero and reported success; x86-64 Linux failed the line search and returned
**the same optimum to twelve significant figures with `converged = False`**.
Since the package is now published and Linux is where most of its users are,
that is a user-facing bug rather than a CI inconvenience: `spending()` on CDNOW
returns `p = 7.4875, q = 3.5829, gamma = 12.2457` on both platforms and calls it
a failure on one. Now `ftol = 1e-14` (`factr = 45`, still 200,000x tighter than
SciPy's default `1e7`) and `gtol = 1e-10`, with the arithmetic written out at
the definition so the next person to tighten them has to argue with it.

**The last printed digit of a Pareto/NBD estimate is not portable.** The ridge
moves `beta` by ~3e-4 between platforms -- 46.8837 here, 46.8834 there -- which
is a different third decimal. Nine printed values across six doctests asserted
exactly that digit. They now elide it: `[1.449, 48.635, 0.561, 46.88...]`,
`beta  -22.92...  116.68...`. `ELLIPSIS` was already in `doctest_optionflags`,
so this needed no configuration, and it is better than rounding to two decimals
because it keeps every stable digit visible and marks precisely where agreement
stops. The tolerance-based checks in `paper_values.py` and `rdoc_values.py` did
not move at all, which is the argument for having written them that way.

**A tolerance tighter than the thing it compares.** `test_families` asserted
"at least as good as CLVTools" to `1e-9` and missed by 1.7e-9 on Linux. Now
`1e-6`: 5e-11 relative on a log-likelihood of -5857, and still far below the
`1e-5` agreement asserted on the line above it, so a genuinely worse optimum
still trips it.

**A claim that is true on one platform only.** `test_warm_start_avoids_the_bad_basin`
asserted `default > cold` strictly. The policy is "run both starts, keep the
better", so what holds everywhere is that the default is never *worse*; whether
a cold start actually falls into the `s = 0.069` basin is platform-dependent,
and on Linux it reaches the same optimum to two ulps. The unconditional
assertion is now `>=` within 1e-9, and the strict one fires only when the cold
fit did land in the bad basin (`cold.s < 0.2`), so the guard keeps its power
where the basin exists without asserting a falsehood where it does not.

*Done when:* CI is green on 3.12 and 3.13, the fits still reproduce every
published number, and both deviations are in the README's findings.

**Done, over four CI runs rather than the one this claimed.** All four classes
fixed, README findings written for the two that are deviations rather than test
hygiene, and a fifth class found afterwards: printed digits in the docs, which
took three more rounds because a failing doctest aborts its whole file and hides
what follows it.

The original text of this paragraph said "green on GitHub on both interpreters"
while [run 33599631277](https://github.com/neilmurrayee/clvtools/actions/runs/33599631277)
was still in progress. It then failed, seven tests. That is corrected here
rather than quietly: the local suite is not evidence about CI, and the tick
should have waited for the run id. See the rule added to this file's header.

Runs, in order: 33597654955 (could not resolve `setup-uv@v10`), 33597787901 (9
failures), 33599631277 (7), 33602019629 (1). Each fixed a real class of defect
and each was ticked only after the *next* run reported.


# Round 4 — an outside review

`docs/review-2026-09-02.md`, commissioned from a second model at commit
`faea7f6` and reviewed on 2026-09-02: four independent read-only passes over
the models, the API and data layer, the test suite, and the docs, with every
ranked claim re-run before it was ranked. Twenty findings. Its own summary of
what is sound is worth keeping in view — the oracle discipline, the log-space
Pareto/NBD path, the typing and lint gates, and that **nothing in it suggests a
wrong equation**. What it found is the layer built after the port: packaging,
CI, input validation, and the regimes the oracle fixtures never visit.

Items below follow its recommended order rather than its numbering; the finding
number is given for each so the two can be read together. `[x]` items were done
in the same session that received the review.

## 18. `[x]` Ship the datasets inside the package — finding 1

**Blocker, and the one that would have shipped.** `DATA_DIR` was
`Path(__file__).parent.parent.parent / "data"`: the repository root in a
checkout, and a directory that does not exist under `site-packages`. The built
wheel contained **zero CSVs**, so `load_apparel_trans()` — the first statement
of the README's usage block — raised `FileNotFoundError` on any installed copy.
Every test passed throughout, because they all run from the checkout.

**Done:** `data/` moved to `src/clvtools/data/`, `DATA_DIR` resolved inside the
package, and `tools/extract_data.R`, `tests/conftest.py`, `CLAUDE.md`,
`README.md` and `pyproject.toml` follow. `uv build` now puts five CSVs in the
wheel, and in a throwaway environment the installed package loads CDNOW and
fits the Pareto/NBD to `[1.449, 48.635, 0.561, 46.884]`. Two tests guard it:
`DATA_DIR` must be inside the package, and every dataset must be reachable
through `importlib.resources`, which is how an installed package reaches it.

## 19. `[x]` One precision rule, everywhere — findings 2 and 13

The suite pinned digits the platforms do not fix, in more places than the CI
failures showed. **Done:** the rule is written at the top of
`tests/test_pnbd_fit.py` and applied across the suite — estimates to no better
than 1e-3 relative, log-likelihoods tightly, "at least as good as the oracle"
with 1e-6 of slack, and no assertion on a printed digit of an estimate. The
1e-4 paper check (which was using 80% of its allowance here and 89% on Linux),
three `>= oracle - 1e-9` checks, the dyncov PAlive count and every fitted table
in `docs/` and `src/` are covered.

## 20. `[x]` Say only what a run id supports — finding 3

Six statements about the repository's own state were false, and the worst was
this file's: item 17 was ticked "green on GitHub on both interpreters" while
that run was in progress, and it then failed. **Done:** all six corrected, and
the header carries the rule they were missing — a claim about an external
system carries the run id, URL or command output that established it.

## 21. `[x]` Make bad input loud — findings 5, 6, 7, 12

The largest cluster, and the review's estimate is "perhaps a hundred lines" for
a shared validator and a shared optimiser-result helper:

- `t_x <= T + 1e-9` is accepted where the likelihood needs `t_x <= T` exactly,
  so one customer a hair over collapses a whole fit to its start values with
  `-inf` and no exception (finding 5);
- NaN covariates and NaN prices flow through `astype(float)` into the
  likelihoods and come back as plausible numbers — a customer with all-NaN
  prices is predicted at 7.72 mean spending against 88.65 with real ones
  (finding 6);
- **there is no `warnings.warn` anywhere in `src/`**, so no fit says anything
  when it fails to converge; CLVTools warns (finding 7);
- and seven smaller silent paths: duplicate covariate rows, timezone-aware
  dates, `predict()` ignoring arguments it was given, a `name_price` typo
  silently disabling spending, the README's own snippet discounting at the
  unscaled annual rate, `likelihood_ratio_test` accepting non-nested models,
  and `ClvDataDynCov` validating nothing at construction (finding 12).

*Done when:* one validator and one result helper are called from all six fits,
each of the above raises or warns with the offending id or column named, and
each has a test. The README snippet is fixed in the same commit.

**Done:** `src/clvtools/_validate.py`. `customer_history()` is one validator
where three families had three, and clamps the 1e-9 slack that used to collapse
a fit; `finished()` warns through its own `ConvergenceWarning` category and
raises when the objective is not finite at the returned point. Non-finite
prices and covariates, duplicate covariate ids, a `name_price` typo, `predict()`
ignoring arguments, and a non-nested likelihood ratio test all raise, each
naming what is wrong. The README snippet discounts properly.

Every one was reproduced first, and the reproduction is in the test's
docstring: `t_x = T + 1e-10` gave `[1.0, 1.0, 1.0, 1.0]` with `-inf`; six
purchases with NaN prices gave `Spending = 0.0`; duplicate ids gave a 601-row
design matrix for 600 customers. 927 passed, `TOTAL 2731 0 100%`, ruff clean.

## 22. `[x]` Standard errors that exist and can be trusted — findings 8 and 9

`latent_attrition`'s docstring forwards `hessian`, which three estimators do
not accept, so `summary()` on a correlated fit raises with advice that cannot
be followed. And `_covariance` inverts without checking definiteness, so the
BG/NBD covariate fit on the apparel ridge ships `nan` standard errors with
`converged = True`.

*Done when:* `hessian` reaches the correlated, GGom covariate and dyncov fits;
`_covariance` checks `eigvalsh > 0` and warns naming the flat directions; `m`
gets its z-value as CLVTools prints one; and the regularized-fit question — the
Hessian is differenced on the unpenalised sum while the objective is the
penalised mean — is answered, documented, and covered by one oracle fixture.

**Mostly done.** `hessian=` now reaches the correlated fit (whose `summary()`
used to raise advice naming an argument the function did not have) and the GGom
covariate fit, which hard-coded `False`. `m` carries a z-value: `m = 0` is an
admissible null and is precisely S6.5.2's question, unlike the four strictly
positive parameters. `_covariance` warns when the Hessian is not positive
definite, naming the flat directions and the smallest eigenvalue — the BG/NBD
covariate fit on the apparel ridge gives `life.Gender = nan` beside
`life.Channel = 0.594` at `converged = True`, eigenvalue −2.2 — and warns rather
than letting `numpy` raise `LinAlgError` when the Hessian is not finite, which
is what the GGom covariate fit produces at `b = 8.1e-07`.

**Two parts left, both `[needs-decision]` rather than work:**

- *The dyncov Hessian.* **Done:** the argument exists and defaults to
  `False`, alone among the fits, with the cost written at the site. What that
  buys is that `summary()`'s advice -- "fit with `hessian=True`" -- now names
  something the function accepts. Eight customers make the branch testable in
  the default suite, at 1.5 s.
- *What a regularized standard error refers to.* **Settled by asking the
  oracle, which is the only reason this was ever a question.** Nothing had ever
  requested a regularized `vcov` from CLVTools. Requested, it returns four
  covariate variances identical to twelve significant figures
  (0.007580647473) beside off-diagonals that differ, and standard errors that
  are not monotone in the penalty -- 0.1303, 0.0871, 0.0913, 0.0853 at
  `lambda` = 1, 10, 40, 100. Neither property is possible for a curvature
  computed from data, so the oracle cannot be followed here, and both are now
  asserted **in R** by `generate_interface_fixtures.R` before it writes the
  fixture.

  The Hessian is therefore differenced on the penalised mean -- the objective
  actually minimised, so that the estimates and their standard errors describe
  one function -- and the disagreement with CLVTools is pinned rather than
  hidden, exactly as the regularized AIC and BIC already are. This package's
  0.2231 is `1/sqrt(2*lambda)` to within 4% and moves with `lambda`; CLVTools'
  0.0871 corresponds to no `lambda` and does not. The two agree on everything
  else: log-likelihood to 1e-3, model parameters to 1%.

  And `standard_errors()` now warns on a regularized fit that its numbers are
  ridge standard errors dominated by the penalty and not comparable with an
  unregularized fit's -- which neither CLVTools nor the paper says anywhere,
  and is the part that protects whoever reads the table.

## 23. `[x]` Log-domain integrals for the regimes the oracle cannot see — findings 4 and 10

The GGom/NBD forms its integrand as `(alpha + tau)^(r + x)` in the direct
domain: at the fitted parameters `x = 140` gives `CET 0.00` and `x = 160` gives
`CET nan` with `PAlive` exactly 1.0, because the integral underflows and
`logaddexp` keeps the alive branch alone. On daily data that starts near
`x = 105`. The dyncov `F2` terms divide by `alpha**(r + s + x)`, which is
exactly 0.0 by `x = 160`; CLVTools arranges it the same way, so **fixture
agreement proves nothing here**.

*Done when:* both are evaluated in log space with a signed log-sum-exp, the
dyncov CET guards `s` near 1 as the aggregate module does, the `pmf` call falls
back on a non-finite hypergeometric, and each carries a heavy-buyer test —
against the Pareto/NBD limit the README documents where no oracle can reach.

**The GGom/NBD half is done.** Both integrals are now scaled by the integrand's
value at the lower limit, where it is largest, and the `CET`'s divisor is formed
in log space with an asymptotic branch for where the product overflows. At the
fitted parameters `CET` was `nan` from `x = 140` and `PAlive` exactly 1.0 from
`x = 160`; both are now finite and monotone out to `x = 400`, `CET` running
3.57 → 123.28 and `PAlive` declining 0.9947 → 0.9818.

The check is the one this item asked for, because no oracle reaches here: as
`b → 0` with `beta = b·beta_P` the GGom/NBD **is** the Pareto/NBD, and the two
agree to 1.2e-3 relative on `CET` and 1e-6 absolute on `PAlive` at
`x = 5, 50, 160, 400` — including the two values where the old code returned
`nan`. Fourteen tests, and the 33 GGom oracle checks unmoved.

**Closed 2026-09-03, both halves.** The GGom/NBD half is above. The dyncov
half was finished by item 28, and the two pieces of finding 10 that this item
explicitly carried forward were closed after it: the unguarded `s - 1` divide
by item 29, and the `pmf`'s cancellation by items 29 and 32 — the latter
finding that the arrangement had been wrong by 1e-3 at `k = 16` well before it
was visibly `NaN`. Nothing under findings 4 or 10 is outstanding.

**The dyncov half was finished by item 28.** Each `F2` term was
`value / alpha**(r+s+x)`, whose divisor passes the top of float64 by `x = 160`
while the quotient is about 1e-370. It became `exp(log value - a·log alpha)`
here, so a representable quotient was no longer lost through an overflowing
intermediate — 2.4e-279 at `x = 120` was being computed through 4.5e+278 —
and at `x = 160` the term was then honestly 0, with `log_likelihood_customer`
still taking the alive-only branch there without saying so. Item 28 stopped
exponentiating the terms at all; see it for what that was worth.

## 24. `[x]` Bootstrap: report failures, and rebuild once — finding 11

One exception on draw 3 of 5 loses every draw; non-converged refits are pooled
silently; a user-supplied `sample` never receives the seeded generator, so
`seed` is ignored with it; and `bootstrap_data` filters the whole frame per
drawn customer — 0.95 s a draw on CDNOW against 0.16 s for the summary and fit
together.

**Done, all four.** The rebuild does one `groupby(...).indices` pass and then
positional lookups: **0.965 s a draw on CDNOW becomes 0.016 s**, so a hundred
draws stop spending 95 seconds rebuilding data against 13 fitting it. Measured
rather than gated, because this repo counts operations and not seconds.

A failed draw is collected and counted rather than propagating — losing five
minutes of refits to one degenerate resample is the wrong trade, and losing
them silently is worse — and only if *every* draw fails does it raise, with the
first failure quoted either way. A caller's own sampler is offered the seeded
generator, so `seed` is no longer ignored whenever `sample` is passed (runs that
looked reproducible were not); a one-argument sampler, which is the shape
`?clv.bootstrapped.apply`'s example has, still works. Non-converged refits now
announce themselves from inside the refit, through item 21's
`ConvergenceWarning`, which is a better place for it than the pooling code.

## 25. `[x]` Suite hygiene — findings 14, 15, 16, 17

`tools/benchmark.py` crashes (`fit_static_covariates() got an unexpected
keyword argument 'method'`) and nothing imports it; a user's own `-m` flag
replaces `addopts` and re-selects the ten-minute dyncov fit; seven fixtures have
no reader and two have no generator; one R self-check compares `coef(fit)` with
itself; some `paper` and `oracle` marks are on classes that do not earn them;
and two tests assert wall-clock seconds while the README says nothing does.

**Findings 14 and 15 done, and the useful half of 16.**

`tools/benchmark.py` runs again — its optimiser arguments moved into a
`SearchSettings` when the covariate fits were unified, and it had been raising
on every invocation since, while the README documents running it. Two smoke
tests now cover `tools/`: the benchmark at twenty customers, and
`tools/profile.py`'s doctests, which had never run because `testpaths` does not
include `tools/`. That is the same shape of gap as the wheel that shipped
without data — a documented entry point nothing exercised.

The `dyncov_fit` deselection moved from `addopts` to `conftest.py`'s
`pytest_collection_modifyitems`. pytest does not compose two `-m` expressions —
the caller's wins — so `pytest -m "not slow"` collected the ten-minute fit,
which is the slowest test here and the clearest thing that phrase excludes. It
now composes, `--run-dyncov-fit` is offered as a second way in, and both
directions are asserted: a user's `-m` does not re-select it, and
`dyncov.yml`'s `-m dyncov_fit` still selects exactly it.

`inference_pnbd_staticcov.json` carried CLVTools' full-precision standard
errors while the test compared against the paper's printed four decimals at 5%
— written, committed, never read. It is wired in at 2e-3, 25 times tighter.

**Closed 2026-09-03**, and two of finding 17's claims did not survive being
checked. What was left is worked below; the paragraph after it is the original
list, kept for the trail.

**The four orphaned fixtures, each decided on its own evidence.**

* `hyp2f0.csv` — **wired in.** There is no `hyp2f0` in `src/`, so it looked
  like data for something unported. But GSL defines :math:`{}_2F_0` *through*
  the confluent :math:`U`, which this package does have as `kummer_u`:
  `2F0(a,b;;x) = (-1/x)^a U(a, 1+a-b, -1/x)`, and the identity holds on all
  eighteen rows. That makes them an oracle for `kummer_u` **from a different
  library than SciPy**, which is worth more than deleting the file and more
  than a SciPy-against-SciPy check. At `rtol` 1e-12: seventeen rows agree to
  better than 1e-14 and one, `a=6, b=3, z=-0.5`, differs by 4.2e-13, which is
  `hyperu` against `hyp2f0` rather than either being wrong — and the fixture
  itself only carries about fourteen digits.
* `dyncov_palive.csv` — **deleted.** Its 600 values are *bit-identical*
  (`np.array_equal`) to the `PAlive` column of `dyncov_predict_holdout`, which
  the suite already checks at `rtol` 1e-11. Wiring it in would have added a
  duplicate wearing the clothes of a second oracle. The generator keeps the
  `check()` that establishes the two R entry points agree — that is the part
  with content — and simply stops writing the file.
* `dyncov_future_covariates.json` — **wired in.** Row counts and the covariate
  window's bounds, asserted nowhere: the frame was loaded and used, so a
  truncated export would have surfaced as a prediction disagreeing with the
  oracle rather than as a dataset of the wrong size.
* `pnbd_staticcov_lrtest.json` — **wired in**, and it is the stronger of the
  two LRT fixtures. `lrtest_pnbd_staticcov.json` pins the *derived* quantities;
  this one pins the two log-likelihoods they come from. A chi-squared statistic
  comes out right from two wrong fits that differ by the right amount.

**The self-comparing R check was real.** `cf <- coef(fit)` sits a few lines
above `check("coef r", cf[["r"]], coef(fit)[["r"]])`, so it could not fail. It
now compares against `exp()` of optimx's own `log.r`, `log.alpha`, ... —
verified in R to agree at 1e-10. That checks the log-scale convention the whole
port is built on, so a transformation or ordering slip between the search and
`coef()` is what it would catch.

**Finding 17, with two claims overturned.**

* *Three `paper` marks were wrong* — none compares against a number the paper
  prints. Two read oracle fixtures and are now `oracle`; one compares two of
  this package's own likelihoods and is now unmarked. `-m paper` 25 → **22**.
* *The `pragma: no cover` claim is **wrong**.* Finding 17 says it "covers a
  branch the suite already exercises". Nothing simulates a missing matplotlib —
  `pytest.importorskip` *skips* when it is absent rather than faking absence —
  so the `ImportError` arm is genuinely unreachable and the pragma is right.
* *The unmarked slow fit is real, in a different test than cited.* Not
  `test_estimate.py:166` at twelve seconds but
  `TestDispatch::test_the_other_families_dispatch_too[clvtools.ggomnbd]` at
  **18.4 s**, the slowest unmarked test in the suite. Now `slow`.
* *Both wall clocks are gone.* `assert elapsed < 0.5` and `assert elapsed <
  10.0` in `test_special.py` contradicted the README and `docs/performance.md`,
  which both say nothing here asserts one — and a half-second bound on a shared
  runner is the first gate to go flaky. They now assert what they were really
  guarding: that the series decides its term count up front, and that the
  degenerate fit is bounded by `maxiter` rather than by the clock.
* *The coverage bar* stays enforced in CI rather than in `addopts`, so that an
  ordinary local run is not made to pay for `--cov`. `CLAUDE.md` prints the
  command.

**Still not done, and it needs R:** `time_elapsed.csv` and
`time_add_periods.csv` have no generator, so they cannot be re-baselined.
`CLAUDE.md`, `README.md` and `tests/conftest.py` each say so at the point where
they would otherwise claim every fixture comes from `tools/oracle/*.R`. Writing
that generator is the one piece of this item carried forward, as **item 33**.

*Counts re-measured after the marker changes*, since three files quote them:
default **1,244**, `paper` 22, `rdoc` 22, `literature` 14, `oracle` 249, `slow`
157, `quality` 14, `performance` 18; `TOTAL 2874 0 100%` in 3:57.

**Left:** four genuinely orphaned fixtures (`dyncov_palive`,
`dyncov_future_covariates`, `hyp2f0`, `pnbd_staticcov_lrtest`) to delete or
use — `fitted_pnbd` is no longer one of them, wired in at rtol 1e-10 by
finding B3 of `docs/spec-audit.md`; `time_elapsed.csv` and `time_add_periods.csv`, which have no generator at
all, so CLAUDE.md's "generated by tools/oracle/*.R" is not true of them; the
CDNOW generator's `check("coef r", cf[["r"]], coef(fit)[["r"]])`, which
compares a value with itself; and finding 17's marker taxonomy. All cleanup
with no behavioural consequence.

## 26. `[x]` Onboarding — finding 18

No install instruction anywhere, no API index for 24 exports, a `Changelog` URL
pointing at `docs/audit.md`, and an empty code block in `docs/paper.md` where
the `newcustomer_static` example belongs.

**Done, and three of the four had already been done in passing** by items 4, 18
and the spec-audit work; this pass is what checked them rather than assuming
them, which is item 11's method and the rule at the top of this file.

All four verified on 2026-09-03 at `2506c67`:

- **Installation.** `README.md` has a section, between the not-ported list and
  Usage: `pip install git+...` and the `[plot]` extra, the Python floor, the
  statement that the five datasets ship inside the package, and `uv sync` for
  development.
- **The API index.** `README.md`'s "The public API" table names all 24 exports
  — checked by joining the table against `__all__` rather than by reading, and
  the answer is *none missing*. Each row says which section of the paper the
  name comes from.
- **The `Changelog` URL.** `pyproject.toml:45` points at `CHANGELOG.md`, which
  exists and carries an `Unreleased` section in Keep a Changelog form.
- **The empty code block.** Gone; `docs/paper.md:475-478` has the
  `newcustomer_static` example, and a scan for fences less than two lines apart
  finds no empty block anywhere in the file.

**What this pass did change**, all of it documentation that had drifted from
what a run prints, and all of it measured rather than inherited:

| Claim | Said | Is |
|---|---|---|
| default run, `CLAUDE.md` | 1,152 | **1,153** passed, 1 deselected, `TOTAL 2752 0 100%`, 3:44 |
| default run, `README.md` | 1,146 | 1,153 |
| `-m literature`, both files | 13 | **14** |
| `-m oracle`, `README.md` | 242 | **245** |
| `-m slow`, both files | 152 | **153** |
| `ci.yml`'s comment | 1,152 / `TOTAL 2751` | 1,153 / `TOTAL 2752` |
| `dyncov.yml`'s collection | `1/907 (906 deselected)` | `1/1154 (1153 deselected)` |
| `docs/performance.md`'s opening | 906 tests | 1,153 |
| the quickstart, `README.md` | twenty cells, ~4 s | seventeen code cells, ~5 s |

Three files still said the `dyncov_fit` deselection lives in `addopts` —
`CLAUDE.md`, `ci.yml` and `dyncov.yml`, the last twice. Item 25 moved it to
`tests/conftest.py` precisely so a caller's `-m` composes with it, and
`dyncov.yml`'s comment called the old behaviour "the whole trick" while running
a command that now works for a different reason. `conftest.py` and
`test_code_quality.py` had it right in the past tense all along; the three that
were wrong now name the hook and both ways back in.

Two smaller repairs on the way past, both cross-references that did not resolve:

- `README.md`'s Installation says the wheel "for a while" could not load its own
  data "and that is recorded in the findings below" — and it was not. The
  defect has a test (`TestTheDatasetsShipWithThePackage`) and a comment on
  `DATA_DIR`, so it had the test half of the house rule and not the README
  half. It is now a finding.
- `CLAUDE.md`'s Layout named ten paths and omitted `docs/spec.md`,
  `docs/spec-audit.md`, `docs/review-2026-09-02.md`, `CHANGELOG.md` and
  `.github/workflows/` — between them the specification, its audit, the review
  that produced items 18-29, and the two gates. A root-level `TASK.md` was
  named there too, as a *finished handoff* rather than a queue; it has since
  been folded into `docs/spec-audit.md` as Appendix 4 and deleted, which is the
  better fix for a filename that reads as a live queue to a new session.

And one thing this pass found that it did not fix, recorded because the claim it
falsifies is in three places. `tests/fixtures/time_elapsed.csv` and
`time_add_periods.csv` — the 840 spans and 280 additions `tests/test_timeunit.py`
reads, and the "Time arithmetic, §5" row of the README's table — are produced by
no generator in `tools/oracle/`. They came from CLVTools' `clv.time` classes in
commit `e385a16`, by a script that was never committed, so they are the one
thing here that cannot be re-baselined. Item 25 already carries writing it;
`CLAUDE.md`, `README.md` and `tests/conftest.py` now say so where each of them
claims the fixtures come from `tools/oracle/*.R`.

*Definition of done met:* `uv run pytest` 1,153 passed, 1 deselected,
`TOTAL 2752 0 100%` in 3:44; `uv run ruff check src tests tools docs` clean.

## 27. `[x]` Reconcile the families — findings 19 and 20

Three validators, two result shapes, two weight conventions, `ClvData(` in
subclass reprs, `scipy.stats` imported at module scope for 0.36 s of a 0.4 s
import, and `.claude/settings.local.json` tracked when its name says otherwise.

**Done, 2026-09-03.** Nine distinct claims sat in those two findings. Each was
reproduced before being acted on, and **two of the nine were already closed** by
items 21 and 25 — which is the argument for reproducing rather than working from
the review text:

| | Claim | Found |
|---|---|---|
| a | three different validators | **already fixed** — all three families give byte-identical errors for `T <= 0`, `t_x < 0` and `x = 0` with `t_x > 0` |
| f | `.claude/settings.local.json` tracked | **already fixed** — `git ls-files .claude/` is empty |
| b | dyncov reports the unweighted count | confirmed and fixed |
| c | two result shapes | confirmed and fixed |
| d | subclass reprs print `ClvData(` | confirmed and fixed |
| e | `cbs.T` is the transpose | confirmed and documented |
| g | four inputs accepted or bare `TypeError` | confirmed and fixed |
| h | `scipy.stats` at module scope | confirmed and fixed |
| i | ~13 covariate refits across four modules | **largely already done** — see below |

**(b) The weight convention.** `fit_pnbd` reports `sum(weights)` as
`n_customers`; the dyncov fit accepted `weights`, multiplied the per-customer
log-likelihoods by them, and then reported `walks.n_customers`. So a bootstrap
draw's BIC was computed against a different `n` from its own likelihood — the
two halves of `k·ln(n) − 2L` describing different samples. Now weighted, like
its sibling, and the test asserts BIC against its definition with the `n` each
fit reports rather than against an algebraic identity (the first draft's
identity was simply wrong: `BIC_d − 2·BIC_p` is `k(ln2 − ln n)`, not `k·ln2`).

**(c) The result shapes.** The nesting itself is deliberate and documented on
`DelegatesToCovariates` — the Pareto/NBD's covariate class predates
`StaticCovResult`. What was not deliberate is that the mixin forwarded ten
properties and not `names_cov_constr` or `reg_lambdas`, **both of which were on
`StaticCovResult` already**. So "which covariates are tied?" and "are these the
ridge standard errors the README warns about?" could be asked of a Pareto/NBD
fit and of neither other family. Two properties; all three families now answer
the same twelve questions, asserted as a list to extend rather than a test to
add.

**(g) Four inputs.** `~ Gender + | Gender` fitted on Gender alone, because the
parser dropped empty terms — the shape a half-finished edit leaves behind,
answering as though it were finished. `~ . | .` on plain data was accepted and
ignored, because the guard tested the *parsed* names and `~ . | .` parses to
`(None, None)`. `reg_lambdas=1.0` said "'float' object is not iterable" and
`ids=1` said "'int' object is not iterable", both naming Python's difficulty
rather than the caller's; each now names the fix, and `ids` suggests the quotes.

**(h) The import.** Measured here, worse than the review had it: `scipy.stats`
was **0.55 s of a 0.70 s** `import clvtools`, and it is wanted by three
expressions — two normal tails and a chi-squared — and by nothing in a fit, a
prediction or a diagnostic. Deferred behind one helper, `import clvtools` is
**0.436 s against 0.719 s, a 39% saving**. Gated by absence from `sys.modules`
rather than by a clock, which is this repo's rule: the saving is a property of
what gets imported, and only the seconds move with the machine. Three tests —
not imported by `import clvtools`, not by a fit, *and* still imported by
`summary()`, so the deferral cannot silently become a removal.

**(i)** A shared module-scoped `static_data` fixture already exists and six test
modules use it. Of the four that still build their own, three need a *different*
covariate selection (`["Zero"]` for the nesting invariant, `["Nonexistent"]` for
a validation test, `["Gender"]` alone) and `test_predict.py`'s hits construct
result objects from fixture coefficients rather than fitting at all. The few
seconds the review estimated are no longer there to save.

**A mistake worth recording, because it is the one this item was about.** The
first draft of the parity tests *fitted* three families with constraints and
regularization over the 600-customer apparel cohort — and a regularized fit
searches from a cold and a warm start, so that was six full fits. **It took the
suite from 3:45 to 9:32.** Rewritten over twelve synthetic customers it was
still 1:57, because a cohort that small leaves the covariates unidentified and
Nelder-Mead wanders: a single *unregularized* Pareto/NBD covariate fit there was
**10.6 s**. The code under test was the forwarding, not the optimiser, so the
tests now build the result objects the way `TestCovariateResultAccessors`
already did: **18 tests in 0.42 s**. Adding a redundant-refit problem to the
suite while closing an item about redundant refits is the shape of thing this
file exists to catch.

No README findings entry: every change here is internal reconciliation between
this package's own families rather than a divergence from CLVTools, which is
what that section is for. Each has a test instead.

*Verified:* 1,200 passed (47 new), 1 deselected, `TOTAL 2769 0 100%`, in
**4:04** against the 3:45 baseline; `ruff check src tests tools docs` clean.

**One thing this item concluded was wrong**, and item 31 carries the
correction: chasing why a capped fit still ran long, it blamed `maxfun` being a
Nelder-Mead near-miss and raised item 31 on that. All the call sites in question
use L-BFGS-B, where `maxfun` is correct. The real cause was the polish stage
ignoring the caller's options entirely. This commit's message states the wrong
version; item 31 states what is true.


## 28. `[x]` Combine `F1·F2 + F3` in log space — the rest of finding 10

Split from item 23 because it changes what the oracle fixtures compare, which
is not a thing to do at the end of a long session.

The `F2` terms were formed in log space individually (item 23) but still
*combined* as values: `_f2` returned a float, and `log_likelihood_customer`
branched on its sign. When a customer's `F2` underflowed to zero — genuinely,
at `x >= 160` on the arguments the review names — the `else` branch quietly
returned `log_F0 + log_F3`, the alive-only likelihood, with no signal. That is
the same silent-degradation shape as findings 4 and 5, in the one estimator
whose fixtures cannot see it: CLVTools arranges the arithmetic the same way, so
the 30 committed intermediates agree with the broken version by construction.

**Attempted and reverted, 2026-09-02 — the spike's negative result.** The
cheap version of this does not work. Scaling every term of one customer's
:math:`F_2` by a single offset, :math:`(r+s+x)\log\alpha_1` of the first term,
keeps the sum O(1) and looks like it should be equivalent. It is not: customer
93 at `dyncov_fit_full`'s parameters has `x = 52` and a true `F2` of
**5.57e-165** — comfortably representable, computed correctly by the existing
code — and the scaled version moved its likelihood by 2.4e-3, disagreeing with
CLVTools where the two had agreed.

**Done, per-term, and the bug was worse than the item says.** The arms, the
middle sum and `_f2` all report `(log magnitude, sign)`; `_log_diff_exp` takes
each arm's difference of two hypergeometrics with `expm1`, and
`_signed_logsumexp` combines terms against the largest of them — inside an
arm, across `F2.1/F2.2/F2.3`, and finally as `logaddexp(log_F1 + log_F2,
log_F3)`. Per-term logs and a per-group offset are what the spike lacked: the
600 apparel customers move by **at most 3e-14 relative** on every one of the
thirty intermediates and 1.4e-14 absolute on `PAlive`, against the spike's
2.4e-3.

What that buys is not a rounding improvement. The heavy-buyer check is the
nesting S3.3 asserts — zero coefficients make this model the standard
Pareto/NBD, whose likelihood `clvtools.pnbd` computes in closed form at any
`x` — and against it the old arrangement was wrong by **225 log-units** at
`x = 200`, with `PAlive` reported as **exactly 1.0** where the truth is
1.6e-98. Certainty, for a customer with hundreds of transactions and none for
two years. Both are now exact to 1e-12 out to `x = 400`, swept in steps of two
across the crossing so no step survives between the sampled points.

The fixture comparison did not have to move. `F2.1`, `F2.2`, `F2.3` and `F2`
keep their value form in the intermediates table — `_value()` — because that
is what the thirty columns are and what CLVTools reports; a term below float64
still *prints* as zero, so does CLVTools', and only the likelihood is formed
from the logs. All 231 oracle checks unmoved, at both grid vectors.

It costs **26% of an evaluation** — 0.104 s to 0.132 s on the apparel cohort,
best of three runs of five. Not quoted as a fit time: `-m dyncov_fit` passed in
7:31 against the 10:07 on record, which says only that a fit's wall clock is
the optimiser's path and not the arithmetic. Hoisting the `errstate` context
managers out of the per-term loops was tried first, on CLAUDE.md's note that
the context manager was once an eighth of the runtime here, and bought nothing
measurable; it was reverted rather than kept as a plausible-looking
non-improvement. `docs/performance.md` records both.

Two things fell out of it. `F2 == 0` now means something: an auxiliary walk of
no length, whose two hypergeometrics cancel term for term. Customer 262 of the
apparel cohort is the only one at either grid vector — they bought on the last
day of the estimation period — and `log_F0 + log_F3` is the right answer for
them. And `test_performance.py`'s `counted()` measured `np.size` of a return
value, which doubled when the arms began returning a pair; it now measures the
first element, which is the one that is per-interval.

Also still open from finding 10, and untouched here: the dyncov `CET` divides
by `s - 1` unguarded where `aggregate.py` raises near `s = 1`, and
`aggregate.py`'s `pmf` calls `hyp2f1` with no fallback, returning `NaN` for
`k >= 23` at `alpha = 500, beta = 1`. Carried to item 29.

## 29. `[x]` The two cheap halves of finding 10 — one of which was not cheap

Left behind by items 23 and 28, both small and neither touching the
likelihood:

- `dyncov_predict.py`'s `CET` divides by `s - 1` unguarded, where
  `aggregate.py` raises near `s = 1`.
- `aggregate.py`'s `pmf` calls `hyp2f1` with no fallback and returns `NaN` for
  `k >= 23` at `alpha = 500, beta = 1`.

**Done, 2026-09-03. The first half was cheap; the second was misdiagnosed, and
the misdiagnosis is the more useful half of this item.**

**The guard.** Both places in `dyncov_predict.py` that divide by `s - 1` —
`conditional_expected_transactions` and the prospective-customer expression —
now call one `_reject_unit_s`, raising the *same message* as
`aggregate.conditional_expected_transactions` has raised since it was written.
Asserted as string equality between the two, since two spellings of one refusal
would be its own small trap, and `np.isclose` decides "near", so both shoulders
(0.999 and 1.001) are checked to pass.

**The `pmf`, where the finding is wrong twice over.** `hyp2f1` does **not**
fail: it returns a finite value at every one of those arguments. What fails is
the subtraction `b1 - b2` in the closed form, where the two are each of order
**1e-7** and their difference of order **1e-22** — fifteen digits of
cancellation, after which the difference lands on zero or goes negative and
`np.log` of that is a `NaN`. And it starts at **`k = 18`**, not 23.

A fallback, which is what the item asked for, would have been *wrong*. Measured
against a 60-digit evaluation of the same closed form at
`alpha=500, beta=1, s=1.5, T=52`: the true pmf at `k = 18` is **8.805978e-22**
and the surviving first term alone is **8.012608e-22**. The term that cannot be
computed is **9% of the answer**, not a rounding correction, so dropping it
trades a visible `NaN` for a quiet 9% error — the exact trade this repository
keeps finding on the wrong side of.

So the value is unchanged and the silence is not. `pmf` now raises a
`PrecisionWarning` — a new category in `_validate.py`, distinct from
`ConvergenceWarning` because that one is about where a *search* stopped and this
is about arithmetic — naming the `k`, the magnitude that cancelled, and item 32.
That matters because **a `NaN` is contagious**: one negligible term takes
`sum(pmf(k) for k in range(...))` with it, so a caller summing over `k` needs to
know where to stop trusting it. Four tests pin that it stays quiet at `k <= 17`,
that the paper's own parameters are untouched (400 terms, still summing to 1 to
1e-6, no warning), and that `part1` alone would have been 9% low.

*Verified:* 15 new tests; `uv run pytest` green; ruff clean.

## 30. `[x]` Collapse the dyncov likelihood's duplicate hypergeometrics

Item 14 measured the prize and named this. `log_likelihood_customer` runs once
per customer, and the 79,508 hypergeometrics one evaluation asks for are 93.3%
duplicates — but the duplication is *across* customers, so nothing local can
reach it.

**Done, 2026-09-03 — and not by the restructure this item specified, which
turned out to be unnecessary.** The item assumed that reaching duplicates
*across* customers required batching the likelihood *over* customers. It does
not: a memo shared by one evaluation reaches exactly the same duplicates, and it
is cheaper, smaller and safer than the restructure.

Both arms want one shape, :math:`{}_2F_1(a, b; a{+}1; z)`. Both now call
`_hyp2f1`, which consults a `ContextVar` memo when one is open, gathers the
misses and evaluates them in a **single** vectorised call, so item 9's batching
is not handed back one element at a time. `log_likelihood_ind` opens the memo
once for the whole sweep, beside the `errstate` manager already hoisted there.
`log_likelihood_customer` keeps its shape and its thirty-column table entirely.

| | evaluation before | after | `hyp2f1` before | after |
|---|---|---|---|---|
| CLVTools' fitted parameters | 0.120 s | **0.110 s** | 0.045 s | 0.032 s |
| the vector the search dwells on | 0.480 s | **0.119 s** | 0.409 s | 0.042 s |

The dwell vector is now as cheap as the easy one, which was the object of both
items: two thirds of a fit ran at 4x the cost of the other third, and that gap
is gone rather than narrowed. `-m dyncov_fit` passed in **2:53** against the
7:31 item 28 recorded and the 10:07 item 9 did — quoted as corroboration, not
as the measurement, because a fit's wall clock is the optimiser's path (item 28
saw 7:31 for a change that made every evaluation 26% *slower*). It does
establish that the optimiser still arrives: what passed asserts this
implementation reaches at least CLVTools' optimum.

**A memo beats the `np.unique` dedup item 14 measured.** Sorting 79,508×3 values
to find 5,303 distinct ones cost more than the hypergeometrics saved wherever
they were cheap — 6.0x at the dwell vector but **0.72x, a loss**, at CLVTools'
fit. A dict pays per lookup rather than per sort and wins at both, 1.6x and
9.9x. Item 14's table recorded the losing arrangement; `docs/performance.md`
now carries this one beside it.

**Bit-identical, not merely close.** All **30 intermediate columns over all 600
customers at both oracle grid vectors** compare equal under `np.array_equal` —
a memo returns the same function's value for the same arguments, so there is no
rearrangement to lose a digit to. Stronger than item 9's rewrite could manage
(27 of 30) or item 28's (3e-14 relative), and the reason to prefer this over
the specified restructure, which would have re-associated the arithmetic and
had to argue about the last two bits.

*Done when* asked for a gate: `TestDyncovDeduplicatesItsHypergeometrics` counts
that SciPy is asked for no argument twice, and asserts the memo does not outlive
its evaluation — scope being load-bearing in both directions, since a wider one
would grow an unbounded dictionary of misses across a fit's ~1,900 evaluations.

## 31. `[x]` Validate optimiser overrides — and the claim that raised it was wrong

Found while working item 27, and **the specific instance it named does not
exist.** This item, as written, said:

> The live instance is `maxfun`. [...] Seven call sites across
> `tests/test_families.py` and `tests/test_pnbd_dyncov.py` pass
> `options={"maxiter": N, "maxfun": M}` to Nelder-Mead fits believing they cap
> function evaluations. `maxiter` works; `maxfun` does nothing.

**All seven call sites use L-BFGS-B, where `maxfun` is the correct key.**
`fit_pnbd`, `fit_ggomnbd`, `fit_pnbd_dyncov` and the covariate fits all default
to `method="L-BFGS-B"`; only the polish stage inside `_staticcov` is
Nelder-Mead. Nothing was dead. The inference came from watching a capped fit
run long and reaching for the nearest explanation without checking which method
the call actually used — the same shape as the `args(bgbb)` mistake in item 16,
made by me one item earlier.

**What was really defeating the cap is worse, and is now fixed.** The polish
stage ran `optimize.minimize(..., options={"maxiter": 20_000, "maxfev": 20_000,
...})` — its own budget, hard-coded, ignoring the caller's `options` entirely.
So a caller who bounded a fit got an unbounded second stage:

| `fit_pnbd_staticcov(options={"maxiter": 3})` | |
|---|---|
| `polish=False` | 0.005 s |
| `polish=True` (the default) | **10.527 s** |

A factor of **2,100** between what was asked for and what ran, on twelve
customers. The polish now carries the caller's overrides too, narrowed to the
keys Nelder-Mead reads — the search above it is usually L-BFGS-B, so `maxfun`
would be forwarded into a stage that rejects it. Capped, the same fit is now
**0.008 s**. With no overrides `_polish_overrides` returns `{}` and the polish
options are byte-identical to before, so the default path is unchanged.

**And the general claim behind the item holds, so the validation was built.**
`options_for` merged anything: `overrides={"nonsense": 1}` passed straight
through, and SciPy's response is a `UserWarning: Unknown solver options` and
then silently dropping the key. R errors. So does this now, naming the method,
the near-miss where there is one (`maxfun` ↔ `maxfev`, which cap the same thing
under different spellings), and the keys the method does accept. That is finding
20's last bullet and spec `V-03`.

The accepted keys are **asked of SciPy** rather than listed here — the keyword
parameters of `_minimize_neldermead` and `_minimize_lbfgsb` *are* the contract,
and a hard-coded copy would drift from it. An unrecognised method validates
nothing rather than refusing something SciPy might accept.

## 32. `[x]` Form the `pmf`'s difference without cancelling — the rest of item 29

Item 29 found that `aggregate.pmf` loses its second term to fifteen digits of
cancellation from `k = 18` at `alpha = 500, beta = 1`, and that the lost term is
**9% of the answer** rather than a rounding correction, so no fallback fixes it.
It warns; it is still `NaN`.

**Done, 2026-09-03 — and the planned fix would not have worked.** This item said
"the shape of the fix is known, because item 28 did it for the dyncov `F2`:
carry the two sides as `(log magnitude, sign)`". That cannot work here, and the
reason is the distinction the item glossed: **item 28's problem was underflow**
— values below float64 but perfectly well determined — and **this one is
cancellation**, values representable whose *difference* is not determined by
them. Measured at 50 digits: sixteen leading digits cancel by `k = 18` and
twenty-three by `k = 25`, against float64's sixteen. No rearrangement of `b1`
and `b2` recovers information that is not in them.

**What works came from the structure.** `b2` is the first `k+1` terms of a
convergent series whose full sum is `b1`, so `b1 - b2` is the **tail** of that
series — and a tail of positive terms has nothing to cancel. `_series_tail`
sums it in log space with `logsumexp`, taking 26–92 terms in the regime that
needs it.

**The larger finding, which this item did not know about.** The `NaN` from
`k = 18` was only where the rot became *visible*. Against the 50-digit
reference, the subtraction was already wrong by:

| `k` | 10 | 12 | 14 | 16 | ≥ 18 |
|---|---|---|---|---|---|
| before | 2.4e-8 | 9.0e-7 | 6.5e-5 | **1.0e-3** | `NaN` |
| after | **exact** | 3.9e-14 | 3.7e-14 | 7.7e-15 | 1e-12..1e-14 |

Wrong in the third decimal at `k = 16`, finite, and silent. That is the
README-findings class, and it now has an entry there.

**No published number moves.** The subtraction is kept where little cancels:
`_CANCELLATION_LIMIT = 1e-4` keeps about twelve of the sixteen digits, and at
the paper's own parameters the surviving share stays above 2.5e-3 out to
`k = 20`, so every published PMF table is answered exactly as before. All 47
`-m paper` and `-m rdoc` tests pass unmoved, and the 400-term doctest still sums
to 1 within 1e-6 with no warning.

**Two corrections made on the way.** The `PrecisionWarning` fired on 72
parameter sets whose answers were *right*: a difference of exactly zero is both
terms underflowing, which is correct for sixty purchases in a window of 0.001,
where only a **negative** difference is impossible. Narrowed to that, the sweep
warns nowhere. And a non-raw heredoc wrote literal control characters
(`\alpha` → `\a`) into the new docstring — the same defect the README already
records against `dyncov.py`'s `PAlive`, caught by ruff and repaired.

The guard that remains is unreachable by any of a 2,430-point sweep over `k`,
the four model parameters and `T`, spanning 1e-8 to 1e8. It is extracted into
`_warn_if_unresolved` and tested directly rather than pragma'd or deleted: an
unreachable guard that quietly stops being unreachable is the shape of defect
this function already has a finding about.

The oracle cannot see any of this — CLVTools arranges the arithmetic the same
way and cancels in the same place — so the reference is `mpmath` at 50 digits,
the same standing as items 23 and 28.

## 33. `[x]` A generator for the two time fixtures — the rest of item 25

`tests/fixtures/time_elapsed.csv` and `time_add_periods.csv` — the 840 spans
and 280 additions `tests/test_timeunit.py` reads, and the "Time arithmetic, §5"
row of the README's table — were produced by no script in `tools/oracle/`. They
came from CLVTools' `clv.time` classes in commit `e385a16`, by a generator that
was never committed, so they were the one thing here that could not be
re-baselined.

**Done, 2026-09-03. `tools/oracle/generate_time_fixtures.R` reproduces both
files byte for byte** — `git status` is clean after running it, which is the
only acceptable proof for a reconstruction: anything less and the committed
fixtures and the script that claims to make them are two different oracles.

The grid was recovered from the files themselves rather than guessed: four
units (no `months` — CLVTools has no month unit at all, which is its own
finding), ten start dates, twenty-one day offsets and seven period counts,
giving 4 × 10 × 21 = 840 and 4 × 10 × 7 = 280. The starts are the calendar's
awkward cases — two leap days, the day after one, a 31st, a year boundary, a
century leap year, 28 February in a non-leap year, and one ordinary day as a
control.

Two things had to match that no amount of correct arithmetic would have given:

* **The row order.** `STARTS` is in the committed files' own order, which is
  not the order anyone would choose. A regenerated fixture identical in content
  but differing by 616 reordered lines is a diff nobody reads, and the next
  person to re-baseline would have had to decide whether that mattered.
* **The timestamp format.** `fwrite(dateTimeAs = "ISO")`, which the sibling
  generators use, writes `2005-01-02T00:00:00Z`; the committed files have
  `2005-01-02 00:00:00`. And `dateTimeAs = "write.csv"` renders a `POSIXct` at
  exactly midnight as a bare date, dropping the time from every `start`. Both
  columns are therefore formatted explicitly.

Five self-checks run before it writes, in the register of its siblings: 365
days over a year, 52 weeks over 364 days, 24 hours over a day, a whole
anniversary as exactly 1 — the property that separates a calendar unit from a
division by a fixed length — and `add` inverting `elapsed` over 52 weeks.

The caveats this required in `CLAUDE.md`, `README.md` and `tests/conftest.py`
are gone, and "generated by `tools/oracle/*.R`" is now true of every fixture in
the directory.

## 34. `[ ]` Re-verdict Appendix 3, then work what is genuinely weak — round 5

`docs/spec-audit.md`'s Appendix 3 carries 205 per-item verdicts, **76 of them
`w`**: the suite touches the claim but does not pin it. Its own caveat says so —
"treat `weak` verdicts as the least certain class ... reasonable readers will
disagree on some" — and the audit closed without working them one by one.

**The first job is triage, not test-writing, and the reason is structural.**
Appendix 3 is a snapshot taken on 2026-09-02, at commit `61cd5ba`. Everything
in items 21-33 happened *after* it and much of it was aimed at exactly these
claims — but no row was ever re-verdicted. So an unknown fraction of the 76 are
already closed and the document does not know it.

Measured before opening this: **17 of the 76 cite a finding that has since been
worked** (A3, A4, A6, A7, B1-B6, D3, D5, D6). Five were checked against the
suite as it stands, and **all five are stale**:

| Spec | Verdict said | Today |
|---|---|---|
| `D-14` | the column rename is never exercised (B4) | `TestTheColumnRenameActuallyRenames` |
| `T-04` | all timezone variants absent (A6) | `test_timezone_aware_dates_are_refused_rather_than_half_supported` |
| `NC-06` | `test_covariates_separate_the_scenarios` never calls `predict` (B1) | it predicts four scenarios and asserts the spread |
| `I-09` | `fitted_pnbd` reachable from no test (B3) | read at `test_diagnostics.py:217` |
| `X-06` | one scalar at `abs=1e-4` (B6) | the coefficient vector and the summary table |

Five of five is not proof about the other 71, which is the point of doing this
properly rather than by inference — the last two rounds punished exactly that
kind of extrapolation four separate times.

*Done when:* every `w` row in Appendix 3 has been re-checked against the suite
as it stands and carries either a corrected verdict or a note saying why it is
still weak; the ones that are genuinely weak are either strengthened or
recorded as a deliberate limit with a reason; and the section counts in that
document are recomputed, since they are tallied from the per-item rows and will
move.

Batched by spec section, largest first: `T` 8, `DY` 8, `FI` 8, `I` 7, `C` 6,
`X` 6, `B` 6, `V` 6, `PR` 5, `S` 4, `NC` 4, `PMF` 3, `D` 2, `F` 2, `M` 1.
Items in a section share fixtures and mechanisms, so they are cheaper together
than apart.

### Batch 1 — the 17 rows citing a worked finding

**14 of 17 were stale**, closed by items 21-33 without their verdict being
updated: `D-14`, `T-04`, `S-13`, `C-10`, `X-06`, `X-09`, `DY-10`, `DY-15`,
`PR-11`, `NC-06`, `B-02`, `I-09`, `FI-12` and `V-06`. Each was checked against
the suite as it stands rather than inferred from the finding being closed —
several of those tests name the spec item or the finding in their own
docstrings, which is what made them findable. All fourteen are re-verdicted `c`
in Appendix 3, with what closed them.

*The first draft of this paragraph said fifteen, listed `DY-15` twice and left
`I-05` out.* `I-05` is the one of the seventeen that is **still weak**, and
checking it is what caught the miscount: `fit_pnbd_dyncov` takes no constraint,
regularization or correlation argument, so 12 of the 29 configurations the spec
names are unreachable — exactly as the verdict said a year of work ago. That is
a capability gap rather than a test gap, so it becomes item 35 rather than being
closed here.

**Two were real, and neither was what the verdict said.**

`T-22` — *"`14.4` and `14` give different windows; nothing pins it"*. The
divergence is genuine and the README's findings record it, ending "Spec T-22".
**It had no test.** That is half of this repository's own rule — "deviations get
a test, not a comment ... add to both" — so the behaviour could have reverted to
R's truncation with nothing to catch it. Now pinned three ways: the window keeps
its fraction, the fraction reaches the *date* (0.4 of a week to the second), and
`CET` moves with it, so the extra 0.4 of a period cannot be carried in the dates
and dropped in the arithmetic.

`DY-20` — *"matrices match; the round-trip is never asserted"*. Correct. The
real and auxiliary life walks were each compared against oracle tables, which
pins their values and not the claim that they **partition one covariate series**
— exactly the seam an off-by-one opens. Asserted now for all 600 customers, bit
for bit (`np.array_equal`, no tolerance: both sides are the same products of the
same floats), at :math:`\gamma \ne 0` — with :math:`\gamma = 0` every
multiplier is 1 and the reconstruction holds for reasons having nothing to do
with the walks, which is how DY-15's `d_omega` oracle came to be degenerate.

### Batch 3 — the `DY` section, and a filter that under-selected

**Two more were stale, and batch 1's filter had missed them.** `DY-03` and
`DY-06` were closed by finding **B7** — all 600 customers and all four walks,
`test_pnbd_dyncov.py:374` and `test_pnbd_dyncov_predict.py:125`. Batch 1
selected rows whose *note cited a finding tag*, and these two cite `file:line`
instead. So that filter was a useful first cut and not a sound one: the
remaining rows have to be checked individually, which is what this round's
*done when* already says and what the next batches will do.

**`DY-17` needed building rather than finding.** Its second claim is that an
auxiliary walk is exactly **2 periods** when `T` sits on a week start and the
customer comes alive shortly before it with no real life walk. No apparel
customer can show it — all 600 have an auxiliary life walk of 12 — so the case
was constructed: birth at +21 days from the grid start, `T` on 2005-01-31, a
Monday. Two periods is the length worth reaching, being the shortest walk with
an interior: at one period the `d_omega` and `d1` corrections that scale the
first and last intervals coincide. The surrounding lengths (5, 4, 3, 2, 2 as
the customer arrives later) are asserted too, so the 2 is not a coincidence.

**`DY-24`'s three uncovered claims** — 1- and 2-period horizons (CLVTools'
issue #128), a 20-customer sample, and a horizon past where the covariates
reach. All correct, including the last raising `_require_coverage`'s named
error rather than stopping the walk short. Short horizons are where a covariate
grid off-by-one shows, which is why they were worth reaching.

**`DY-19` is decided, not covered**, and it is the one the audit itself left
open: "the epsilon-apart claim is unreachable (day aggregation), undecided".
Both halves of that are right. S6.1 collapses the log to one record per
customer-day *before* walks are built, so two purchases an epsilon apart are
**one** transaction and there is no second walk to lose. Asserting "the walk
survives" would assert something the data layer has already made vacuous, so
the aggregation step is asserted instead — the thing that could actually
regress. Re-verdicted `o`, the same disposition `T-01` earned for the same
reason.

### Batch 2 — the `T` section, and a bare `IndexError` behind it

All six remaining `T` rows were genuinely untested, and **five turned up no
defect** — they are now pinned and re-verdicted `c`:

* `T-17` floor idempotence, for all five units, plus that `week` floors to the
  *day* rather than a calendar week, which `Weeks.ceiling` documents and
  nothing asserted.
* `T-15` the `data_end` prediction window, against the spec's own two dates
  (1998-07-16 / 1998-07-30) to the day. The +1 day rule was carried by a paper
  example in a configuration the paper never uses.
* `T-19` one- and two-period horizons, numeric and as a date, and that the two
  spellings agree — one-period horizons are where a grid off-by-one shows.
* `T-11` is satisfied by a **stricter** rule than the spec states: any
  `data_end` before the last purchase is refused, which subsumes one before the
  split. Pinned as the stricter rule, since that is what the code promises.
* `T-01` is a real observation with no consequence. The spec wants a 1-second
  epsilon on datetime units and `holdout_start` steps a whole hour — but
  `_aggregate_to_day` floors every transaction to the unit before anything
  reads it, so **nothing can land in the gap** and a finer epsilon would select
  the same rows. What is asserted instead is the property the epsilon exists
  for: the two samples partition the log exactly.

**`T-18` found a defect.** All four boundary combinations build correctly, as
the claim asks. But a covariate grid that stops *short* of the estimation end
raised `IndexError: index 4 is out of bounds for axis 0 with size 4` from inside
`_distance_to_interval_end` — a numpy index error for "your covariate data does
not cover your transaction data". The model already had the right words at the
other end: `dyncov_predict._require_coverage` refuses a prediction horizon the
covariates cannot reach. Construction had no equivalent, and now does, worded to
match: it names which grid, which customer, that customer's last covariate date
and the estimation end. A grid ending *exactly* at the estimation end is still
accepted, because a covariate date describes the period starting there.

**The first draft of that check was too broad, and an existing test said so.**
It checked both grids, and
`test_rejects_a_series_too_short_for_the_walks_it_must_cover` failed: a short
*transaction* series was already refused, later and more specifically, by
`_stack` with "periods its walk spans". The two checks divide the work because
every walk's interval indices come from the **lifetime** grid and then slice
both matrices — so only that one can index past its end, and only that one
needed the earlier guard. Narrowed to it, and the boundary is now pinned from
both sides rather than left to whichever check happens to fire first.

**And writing batch 1 turned up the structure that verdict had not.** The first draft
asserted `real > 0`, which is false for **214 of the 600**: the real life walk
spans birth to the last repeat purchase *in a later covariate interval*, so it
is empty for the 213 customers with `x = 0` — and for customer **129**, who buys
again at `t_x = 0.43` weeks and never leaves the interval they were born in.
That is the same customer finding B2 turned up, reached from the other
direction. The test states that correspondence rather than the naive bound.

## 35. `[ ]` The dyncov fit takes no constraint, regularization or correlation

Spec `I-05` asks that `hessian()` match the optimiser's own for **29
configurations**: the four families without covariates and with correlation;
the three transaction families with static covariates × {default, constrained,
regularized, both}; and dyncov × {default, correlation, constrained,
regularized}. Round 5 re-checked it and it is the one weak verdict of the
seventeen that survives — but not as a testing gap.

`fit_pnbd_dyncov`'s parameters are `walks, names_cov_life, names_cov_trans,
start, start_cov, method, maxiter, hessian, options, weights`. There is no
`names_cov_constr`, no `reg_lambdas` and no `use_cor`, so **12 of the 29
configurations cannot be constructed at all**, never mind pinned. Every other
family reaches all four of its static-covariate combinations through
`_staticcov`; the time-varying one reaches one.

CLVTools does offer at least part of this: `getMethod("pnbd",
"clv.data.dynamic.covariates")` carries `use.cor` and `start.param.cor`.
Whether it also accepts `names.cov.constr` and `reg.lambdas` there was not
established — its Rd lists them for the dynamic-covariate method of `bgbb`,
which is a stub, so that is not evidence about `pnbd`. **Establish it in R
before building anything**, the way item 16 should have been.

*Done when:* either the three arguments exist on `fit_pnbd_dyncov` and route
through the same `_staticcov` machinery the other families use, with the
configurations `I-05` names pinned; or each absence is a recorded decision with
CLVTools' own behaviour checked rather than assumed, and `I-05` is re-verdicted
`o` rather than `w`.

Note the cost before starting: a dyncov fit is ~2:53 since item 30, so a test
matrix over four configurations is not something to put on the default path —
`dyncov_fit` and its nightly workflow are where it belongs.

---

## Closed

Both audit rounds. See `docs/audit.md`:

- **Round 1** — fourteen items, the paper and CLVTools' `NAMESPACE`.
- **Round 2** — the R package's own documentation: five published tables and
  the CDNOW dataset, two features built (`summary(ids=)`, `I(...)` formula
  terms), one withdrawn as not a gap, and `docs/vignette.md`.
- **Static analysis** — ruff with measured limits, wired into `pytest` as
  `tests/test_code_quality.py`.
