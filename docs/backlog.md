# Backlog

The paper and the R package's documentation are both covered — `docs/audit.md`
records both rounds, and every item in them is closed. What remains is not
model work. It is the difference between "the port is correct" and "the package
is in good shape", and each item below was found by measurement, not by
speculation.

**This file is the loop's state.** Work the topmost unchecked item, tick it, and
stop. Do not add items without evidence, and do not work an item marked
`[needs-decision]` — those need the maintainer.

Definition of done for every item: `uv run pytest` green (891 tests at the time
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

## 7. `[ ]` Guard the performance invariants

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

## 8. `[ ]` A committed profile report

`tools/benchmark.py` reports Appendix B's wall-clock. Nothing reports *where*
the time goes, so every question about it starts from scratch — as
`docs/performance.md` had to.

*Done when:* a `tools/profile.py` sibling emits a cProfile summary of the
standard paths (`summary()`, `fit_pnbd`, `fit_pnbd_staticcov`, one dyncov
likelihood evaluation) in a form that can be pasted into `docs/performance.md`
and diffed between versions. Informational, not a gate — it must not be able to
fail CI.


---

## Closed

Both audit rounds. See `docs/audit.md`:

- **Round 1** — fourteen items, the paper and CLVTools' `NAMESPACE`.
- **Round 2** — the R package's own documentation: five published tables and
  the CDNOW dataset, two features built (`summary(ids=)`, `I(...)` formula
  terms), one withdrawn as not a gap, and `docs/vignette.md`.
- **Static analysis** — ruff with measured limits, wired into `pytest` as
  `tests/test_code_quality.py`.
