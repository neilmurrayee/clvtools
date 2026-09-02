# Audit of the suite against `docs/spec.md`

Date: 2026-09-02, at commit `61cd5ba`. `docs/spec.md` states what the sources
demand, derived without reading `src/` or `tests/`. This file is the join:
for each spec item, what the suite actually pins.

**Status: every finding worked**, 2026-09-02. A1–A7, B1–B7, C, D1–D6, the six
`out-of-scope` items that needed a recorded decision, and D-17 and NC-13, the
two the audit never reached. `TASK.md` records what each one turned out to be —
fourteen confirmed on the first pass, one overturned, two kept as deliberate
divergences, and four defects that no test here could see: standard errors 24.5×
too large at a zero penalty, a dyncov bootstrap refitting without covariates, a
`NaN` horizon becoming a `NaN` prediction, and a mistyped covariate name being
dropped. The suite went from 906 tests to 1,146, still at 100% line coverage of
`src/`. What remains of this document is the `weak` verdicts it did not
individually list, which its own caveat below calls its least certain class.

**How this differs from `docs/audit.md`.** That audit indexed the paper and the
R `NAMESPACE` *against this implementation* — for each thing the sources have,
does a counterpart exist here. It found missing features, and closed all
fourteen. This one runs the other way, so it can also find a claim that is
present but wrongly pinned. Every finding below is of that second kind or of a
kind the first direction structurally could not reach.

**New source.** CLVTools' own testthat suite, from the CRAN source tarball
(`CLVTools_0.12.1.tar.gz`, 2025-11-06) — 81 files, 12,647 lines, 645 named
claims. It is not installed with the package, so nothing in this repository had
ever consulted it. It supplies most of what follows, and it is the only source
that says what must happen on bad input.

## Verdicts

Of 222 spec items, 220 were verdicted (D-17 and NC-13 were not reached):

| | Count |
|---|---|
| `covered` — the claim is genuinely pinned | 64 |
| `weak` — touched but not pinned | 76 |
| `absent` — nothing covers it | 75 |
| `out-of-scope` — deliberately not ported | 6 |

Read those numbers carefully. `absent` is not a defect list: much of it is
behaviour this port has and simply never asserts, and a good deal of it passes
today. The four sections below are the part that matters.

---

# A. Divergences from R that nothing records

The house rule is "deviations get a test, not a comment — add to both"
(`CLAUDE.md`), and the README's Findings section does this well for the
deviations it knows about. These are the ones it does not. Each was confirmed
against the running code. Several are *pinned in the opposite direction* — a
test asserts the divergent behaviour — which means they cannot drift, but also
that nobody chose them on the record.

**A1. Zero-length prediction windows are refused where R returns zero.**
Three places, one root: `predict(prediction_end=0)` raises rather than
returning `CET = 0` (`test_predict.py:349` pins the refusal), the dyncov path
raises `"does not reach past"` (`test_pnbd_dyncov_predict.py:385`), and
`newcustomer(0)` raises `ValueError("strictly positive")` where R returns
**1** (`predict.py:144,157,175`; refusal pinned at `test_predict.py:564` and
`test_pnbd_dyncov_predict.py:415`). Spec PR-05, DY-12, T-20, NC-02. In none of
README Findings, `docs/audit.md`, `docs/backlog.md`.

**A2. The dyncov bootstrap silently refits a model with no covariates.**
`ClvDataDynCov` subclasses `ClvData`, not `ClvDataStaticCov`, so the resample
branch at `bootstrap.py:213` never fires; the caller's `apply` receives a plain
`ClvData` with the dynamic covariates gone. A correctly written refit closure
still cannot produce a dyncov fit. The result is confidence intervals that are
plausible, wrong, and unflagged. Spec B-09. **This is the most serious item in
the audit** — everything else is a missing assertion; this is a wrong answer.

**A3. The discount factor has no upper bound and rejects zero.** R admits
`[0, 1)`. This port rejects `0` (`aggregate.py:401`) and silently returns a
number for `1.5` and `100` — I ran it. The parameter carries CLVTools' exact
semantics (`DEFAULT_DISCOUNT_FACTOR = log(1.1)`), so R's range does transfer.
`predict()` itself performs no validation at all. Spec PR-11.

**A4. Five silent acceptances where R errors.** Each probe-confirmed:
`NA` in `Id` or `Date` **silently drops the transaction** (the groupby in
`_aggregate_to_day`, `data.py:262`, drops NaN keys — a 3-row frame returns 2
rows, no warning, customer summary quietly wrong); a single-category covariate
yields a `(600, 0)` design, i.e. a covariate model with no covariates
(`data.py:746`); covariate `Id`s absent from the transaction data are dropped
by `.loc` rather than rejected (`data.py:731`); re-setting covariates on an
object that already has them overwrites without complaint (`data.py:669`); an
empty DataFrame is accepted and a non-DataFrame gives `AttributeError` rather
than a named error. Spec V-06, C-09, C-10, C-13. The first is the one to fix
first: it is exactly the silent-wrong class commit `b75c15e` was written
against, and the one input path it missed.

**A5. Time-unit spellings R accepts are rejected.** `"w"`, `"weeks"`,
`"Weeks"` all fail; only five exact lowercase singulars are admitted
(`timeunit.py:TIME_UNITS`, pinned at `test_timeunit.py:292`). R accepts cases,
plurals and full names. Spec T-07.

**A6. Timezone-aware input is half-broken.** A date or string split raises
`TypeError: Cannot compare tz-naive and tz-aware timestamps`; a *numeric* split
builds a fully usable object with a tz-aware `estimation_end`, and spans are
computed from `total_seconds()`, so a DST transition inside the window shifts
`t_x`/`T` by up to an hour against the tz-naive answer. Usable and silently
inconsistent. Spec T-04.

**A7. Smaller, each unrecorded.** `prediction_end=14.4` and `14` give different
windows where R makes them agree (T-22). Tracking-plot periods between the last
transaction and `data_end` report `0.0` where R gives `NA` (S-12) — a zero and
a missing value are not the same thing on a plot. The `"10+"` frequency row is
emitted with zero customers when the bins already cover everyone (S-13).
Covariate column names are never coerced to valid identifiers; `my var!`
survives verbatim (C-05). Correlation is refused with static covariates
(`estimate.py:193`) — README:39 says "Pareto/NBD only", which is about families,
not covariates (X-09). `fit_pnbd_dyncov` takes no constraint, regularization or
correlation argument, removing 12 of I-05's 29 Hessian configurations. The
formula parser accepts `Gender*Channel`, `.-Gender` and bare `log(x+2)`, then
fails on each with a misleading "not in the data" (FI-06, FI-07). There is no
`date.format` argument at all, and `time_unit` has a default where R has none.

---

# B. Tests that do not test what they claim

The class you cannot find by reading coverage, because every one of these lines
executes.

**B1. `test_covariates_separate_the_scenarios`** (`test_predict.py:537`).
Docstring: *"S6.3.4's 'region A versus region B' comparison."* Body: reads four
numbers out of `newcustomer_static.json` and asserts `len(set(...)) == 4`.
**It never calls `predict`.** It asserts that R's fixture file contains four
distinct numbers. The port's own ability to separate scenarios follows only
indirectly, from four parametrized oracle comparisons, and only for
static-covariate Pareto/NBD.

**B2. `test_real_and_auxiliary_lifetime_walks_do_not_overlap`**
(`test_pnbd_dyncov.py:373`). Named and documented for the walk non-overlap
invariant. Body asserts `(cbs.loc[has_real, "x"] > 0).all()` — that customers
with a real lifetime walk have repeat purchases. A different claim entirely,
and the invariant in the name is pinned nowhere.

**B3. `fitted_data`'s doctest** (`diagnostics.py:338`) is reachable from no
test file, and its printed values come from this implementation rather than
from R's `fitted()`. Self-referential: it cannot fail if the function is wrong,
only if it changes.

**B4. The `name_id` / `name_date` / `name_price` rename is only ever exercised
as identity.** The three uses in the suite are the default value passed
explicitly, a misspelling checked for raising, and `None`. The rename mapping
in `data.py:__init__` never actually renames anything, so a broken non-identity
mapping would pass all 906 tests. Spec D-14.

**B5. Two degenerate oracles.** `d_omega ≡ 1` for all 600 customers and
`d1 ≡ 1` throughout the ABCD table, so comparing them against the oracle cannot
discriminate, and the non-boundary branch of `_distance_to_interval_end`
(`dyncov_walks.py:275`) is never reached through `d_omega`. Spec DY-01, DY-15.

**B6. `test_zero_weight_reproduces_the_unpenalised_fit`**
(`test_pnbd_advanced.py:454`) is three lines and compares one scalar at
`abs=1e-4` — about `1.7e-8` relative on a log-likelihood near −5821. R's claim
is that λ=0 reproduces the coefficient vector *and* the summary coefficient
table, `expect_equal`, no loosened tolerance. Spec X-06.

**B7. Restricted samples presented as general.** `test_zero_coefficients_make_
every_multiplier_one` checks the two auxiliary walks over `customers[:20]`;
real walks are unchecked and the `Bbar_i = Dbar_i = 0` half of the claim is not
checked at all (DY-03). `i` and the window start are compared only over
`settings["sample.ids"]` (DY-06). Bootstrap resampling is verified by four
hand-picked ids and `x`/`t_x`/`T` only (B-02), and covariate resampling by a
`(600, 2)` shape assertion (B-08).

---

# C. Published oracles that are absent — and that already pass

The highest value-per-line finding in the audit. **No literature value is
pinned anywhere**; a literal grep for each hits only `docs/spec.md`. The R
suite asserts them at `estimation.split = "1997-09-30"`, a different split from
the 37-week default this port fits, which is why `paper_values.py` does not
already cover them.

`audit-models` fitted cdnow at that split and the port reproduces every one:

| Source | Published | This port |
|---|---|---|
| Fader/Hardie/Lee 2005, Pareto/NBD | `r=0.553, α=10.578, s=0.606, β=11.669`, LL `−9595.0` | `0.5533, 10.5777, 0.6062, 11.6687`, LL `−9594.9762` |
| Fader/Hardie/Lee 2005, BG/NBD | `r=0.243, α=4.414, a=0.793, b=2.426`, LL `−9582.4` | `0.2425948, 4.4136079, 0.7929248, 2.4259154`, LL `−9582.4292` |
| Fader/Hardie 2013, Gamma-Gamma | `p=6.25, q=3.74, γ=15.44`, LL `−4055.9177` | `6.2496, 3.7442, 15.4435`, LL `−4055.9177` |
| CLVTools' own Pareto/NBD SEs | `0.0476264, 0.8427222, 0.1872594, 6.2105448` | `0.0476278, 0.8427222, 0.187277, 6.2109628` — all inside R's 0.001 |

Three published papers, five oracles, roughly sixty lines of a slow module.
BG/NBD and GGompertz/NBD are **never fitted on cdnow at all** (F-04, F-06).

Two more that pass today and are pinned nowhere:

- **The Pareto/NBD `PAlive` numerical-stability case** (M-04). `x=(221,254,161,204)`,
  `T.cal=(103.57,97.29,98.00,99.43)` at `r=0.5143, α=2.8845, s=0.2856, β=14.1087`
  — inputs that produced `NaN` in an earlier implementation. Verified finite:
  `[0.99960, 0.99956, 0.74949, 0.99426]`. A named regression case, exact
  inputs, no R needed.
- **`P(X=0)` strictly decreasing in `T.cal`** (PMF-04). Verified to hold, for
  no family asserted.
- **`dyncov_palive.csv` is already committed and orphaned**, slated for
  deletion at `docs/backlog.md:1101`. Wiring it up is DY-13 for free rather
  than a fixture to throw away.

---

# D. Invariants checkable with no R at all

These need no fixture and no oracle. They are the cheapest tests in the audit
and they cover the mechanisms most likely to break silently.

1. **DY-07, the static-as-dynamic cross-check.** Feed constant covariates as a
   dynamic series; require `A_i`/`C_i` to equal the static values, `Dbar_i = 0`,
   `Bbar_i = −T·A_i` in the CET table. The only whole-machinery validation of
   the walk construction, and absent entirely.
2. **X-07 / X-08, permuted covariate data.** Nothing in the suite shuffles
   covariate rows or reverses covariate columns; every oracle frame arrives in
   the order the implementation sorts into. A design matrix mis-joined to the
   customer summary, or `names_cov_*` drifting from column position, is
   invisible — and `get_dummies` moving dummies to the end (C-03/C-04) is
   exactly that mechanism.
3. **DY-10's `α = β` arm.** `dyncov.py:355` and `:374` branch on
   `alpha_1 >= beta_1`; only one side is ever taken. R runs both, at
   `α = β = 1.234`.
4. **DY-22, seven weekday splits.** Every dyncov test in both files uses
   `estimation_split=104`, so all 600-customer oracle comparisons run at one
   alignment against the weekly covariate grid.
5. **PR-08, X-01..X-05, FI-12** — the nesting and cross-model invariants:
   γ=0 recovering the plain model in `predict`/`plot`/`pmf` (not only in the
   likelihood), and a spending model's cbs `x` equalling the Pareto/NBD's.
   Both sides of FI-12 are separately oracle-pinned; the agreement between them
   is never stated, and they come from genuinely different code paths
   (`data.py:355` vs `:395`).
6. **B-02 / B-11, the bootstrap identity.** Draw every id once; expect the
   original cbs, covariates and coefficients back. The strongest available test
   of the whole resampling path, and the one that would have caught A2.

---

# Section verdicts

| Spec section | covered | weak | absent | out-of-scope |
|---|---|---|---|---|
| S1 Transaction data (D) | 7 | 2 | 8 | 0 |
| S2 Time and splits (T) | 6 | 8 | 8 | 0 |
| S3 Descriptives (S) | 9 | 4 | 2 | 1 |
| S4 Covariate data (C) | 1 | 6 | 7 | 0 |
| S5 Model expressions (M) | 10 | 1 | 1 | 1 |
| S6 PMF (PMF) | 1 | 3 | 2 | 0 |
| S7 Estimation (F) | 3 | 2 | 10 | 0 |
| S8 Covariate fits (X) | 2 | 6 | 7 | 0 |
| S9 Dyncov (DY) | 4 | 8 | 13 | 0 |
| S10 Prediction (PR) | 7 | 5 | 2 | 2 |
| S11 newcustomer (NC) | 3 | 4 | 5 | 0 |
| S12 Bootstrapping (B) | 5 | 6 | 3 | 0 |
| S13 Inference (I) | 2 | 7 | 2 | 0 |
| S14 Formula interface (FI) | 4 | 8 | 3 | 0 |
| S15 Input validation (V) | 0 | 6 | 2 | 0 |

Where the suite is strongest: `DY-08` compares all 30 intermediate columns
across 2 parameter vectors and 600 customers at `rtol 1e-8`; `M-01..M-08` pin
every family's expressions against per-customer oracles; the constrained-Hessian
alignment that README Findings records is genuinely pinned. Fixture provenance
is clean — all 124 come from R generators that self-check against a public
generic before writing, so there is no self-referential oracle anywhere.

Where it is thinnest: **S15 has no `covered` item at all.** Roughly 25–30% of
the applicable validation claims are pinned, discounting ~20 that cannot cross
the language boundary. The split is sharp — the *numeric* input surface is
solid and "Make bad input loud" clearly landed there, but the *argument-shape*
surface is unguarded: optimiser overrides accept anything (`_optimize.py:82`),
no single-logical argument is validated anywhere, and a `NaN` start parameter
passes `np.any(start <= 0)` and then misreports as a data fault.

---

# Suggested order of work

1. **A2** — the dyncov bootstrap. The only wrong answer in the audit. Raise on
   `ClvDataDynCov` in `bootstrap_apply` until resampling exists.
2. **A4's first item** — reject `NA` in `Id`/`Date` instead of dropping the
   row; empty-frame and non-DataFrame input in the same change.
3. **C** — the literature tier. Five published oracles from three papers that
   the port already reproduces, for ~60 lines.
4. **D1, D2, D3** — the no-R invariants: DY-07, permuted covariates, DY-10's
   `α = β` arm.
5. **B1, B2, B4, B6** — fix the four tests that do not test what they claim.
6. **A1, A3, A5, A7** — decide each divergence and record it, or close it.
   Several are pinned in the divergent direction already; they need the
   Findings entry, not new code.

Items marked `out-of-scope` in `docs/spec.md` need a recorded decision rather
than a test — the audit could not otherwise tell a gap from a choice, and six
of the divergences above sat in that ambiguity.

---

# Appendix 1 — Reproducing the sources

The R test suite is not installed with the package and is not committed here
(CRAN's terms cover CRAN's redistribution, not ours). To re-derive or check any
`Rtest:` citation in `docs/spec.md`:

```bash
curl -sSLO https://cran.r-project.org/src/contrib/CLVTools_0.12.1.tar.gz
tar xzf CLVTools_0.12.1.tar.gz          # -> CLVTools/tests/testthat/ (81 files)
```

Verify you have the same version the spec was derived from: `CLVTools/DESCRIPTION`
must read `Version: 0.12.1`, `Date: 2025-11-06`. The man pages cited as `Rdoc:`
are in `CLVTools/man/` (91 files) — the installed `.Rlib/CLVTools/help/` holds
them only in binary `.rdb` form.

# Appendix 2 — Provenance of the findings

Not every verdict below carries the same weight, and a checker should know which
is which.

**Verified by running the code, in this session:** the dyncov bootstrap
returning a plain `ClvData` (A2); the discount factor accepting `1.5` and `100`
and rejecting `0` (A3); `test_covariates_separate_the_scenarios` never calling
`predict` (B1); the `name_*` rename never being exercised (B4);
`test_zero_weight_reproduces_the_unpenalised_fit` asserting one scalar (B6);
the absence of every literature value by literal grep (C); all 124 fixtures
coming from R generators via `paste0()` naming.

**Verified by section audits running read-only probes against the installed
package:** the port reproducing the literature values at
`estimation.split="1997-09-30"` (C); `PAlive` finite on the M-04 inputs;
`P(X=0)` monotone in `T.cal`; the five silent acceptances in A4; the timezone
behaviour in A6; the formula-parser acceptances in A7.

**Reported from reading, not executed:** the remaining per-item verdicts in
Appendix 3. Treat `weak` verdicts as the least certain class — they are a
judgement about whether an assertion pins a claim, and reasonable readers will
disagree on some.

**Two spec items were corrected during the audit** and the corrections are
recorded in place in `docs/spec.md` rather than silently patched: DY-02 (the R
test's *title* contradicts its body) and DY-04 (two tables in the R file share
column names and disagree at `i = 1`).

# Appendix 3 — Per-item verdicts

`c` covered · `w` weak · `a` absent · `o` out-of-scope · `!` divergence from R
that nothing in README Findings / `docs/audit.md` / `docs/backlog.md` records.

## S1 Transaction data

| | | evidence / note |
|---|---|---|
| D-01 | c | `test_data.py:174` |
| D-02 | c | `test_data.py:185`, exact `[10.0, 25.0]` |
| D-03 | a | every synthetic frame is midnight; the hour unit's floor path is untested end to end |
| D-04 | a | no test varies input row order; all four orderings verified to agree |
| D-05 | a | the synthetic duplicate is on the customer's *second* day; no fixture has two records on a first day |
| D-06 | c | `test_data.py:152`, `test_descriptives.py:327` |
| D-07 | c | `test_data.py:124`, rtol 1e-12 |
| D-08 | a ! | no numeric/categorical `Id` test; `astype(str)` gives `"1.0"` for a float id where R gives `"1"` |
| D-09 | a | integer `Price` never tested |
| D-10 | a | no shuffled-input test; verified equal |
| D-11 | c | `test_data.py:338`, `test_descriptives.py:178` — flag *and* descriptives consequence |
| D-12 | c | `test_descriptives.py:67`, rel 1e-12 |
| D-13 | a | `.copy()` holds; nothing asserts it |
| D-14 | w | `test_data.py:262` passes the *default* `name_id="Id"`; rename never exercised (**B4**) |
| D-15 | a | strings and `datetime.date` work, untested |
| D-16 | c | `test_cdnow.py:73,95` — incidental but genuine |
| D-17 | — | not reached by the audit |
| D-18 | w | NA-for-one-transaction pinned via descriptives; the sort-order half untested |

## S2 Time and splits

| | | evidence / note |
|---|---|---|
| T-01 | w | `test_cdnow.py:211`; datetime epsilon untested, and the hour unit uses 1 hour, not 1 second |
| T-02 | c | `test_data.py:107` |
| T-03 | c | `test_data.py:115`, `test_cdnow.py:206` |
| T-04 | w ! | `test_data.py:280`; all timezone variants absent — tz-aware raises on a date split, silently builds on a numeric one (**A6**) |
| T-05 | a | no partial-period warning; `split=37.5` silently gives a mid-day end |
| T-06 | c | |
| T-07 | a ! | `"w"`/`"weeks"`/`"Weeks"` rejected; pinned at `test_timeunit.py:292` (**A5**) |
| T-08 | c | |
| T-09 | a | no minimum-holdout check; a 1-day holdout is accepted |
| T-10 | a | only `end <= estimation_start` rejected; `split=0.5` accepted |
| T-11 | w | `test_data.py:268`, only without `data_end` |
| T-12 | a | nothing compares `data_end=<last transaction>` with omitting it |
| T-13 | a | works, untested |
| T-14 | c | |
| T-15 | w | `test_predict.py:131` pins the +1 day rule via the paper; the `data_end` configuration is untested (the port does emit 1998-07-16/1998-07-30) |
| T-16 | a | suite only ever passes `datetime64[ns]`; none of the 8 conversions pinned |
| T-17 | w | `test_timeunit.py:245-286`; boundary idempotence implied, never asserted |
| T-18 | w | `test_pnbd_dyncov.py:456`; only the on/on combination |
| T-19 | w | `test_predict.py:329,:112`; 1- and 2-period horizons untested |
| T-20 | a ! | `prediction_end=0` raises; pinned opposite at `test_predict.py:349` (**A1**) |
| T-21 | c | |
| T-22 | w ! | `14.4` and `14` give different windows; nothing pins it (**A7**) |

## S3 Descriptives

| | | evidence / note |
|---|---|---|
| S-01 | c | |
| S-02 | c | |
| S-03 | w | `test_descriptives.py:145` asserts 2 cells where R asserts the whole object (which does match) |
| S-04 | c | |
| S-05 | w | `test_descriptives.py:100`; the no-holdout customer only appears aggregated |
| S-06 | c | |
| S-07 | a | copy holds on pandas 3.0.5, nothing asserts it |
| S-08 | c | |
| S-09 | o | no coercion generic; `ClvData()` is the only constructor |
| S-10 | c | |
| S-11 | c | |
| S-12 | a ! | periods between last transaction and `data_end` report `0.0` where R gives `NA` (**A7**) |
| S-13 | w ! | the `"10+"` row is emitted with 0 customers when bins already cover everyone (**A7**) |
| S-14 | c | |
| S-15 | c | |
| S-16 | w | `ids` honoured; `annotate.ids` has no counterpart |

## S4 Covariate data

| | | evidence / note |
|---|---|---|
| C-01 | w | `test_pnbd_staticcov.py:112`; one string column, no factor arm, no holdout arm; expected names are pandas' convention, not R's |
| C-02 | w | only 3-cat→2-dummy; no 2-cat→1-dummy, no dynamic case |
| C-03 | w | only the no-numeric arm; the mixed arm is where `get_dummies` reorders |
| C-04 | w | `test_pnbd_staticcov.py:99`; the with-categoricals half absent |
| C-05 | a ! | names never coerced; `my var!` survives verbatim (**A7**) |
| C-06 | a | `frame.copy()` untested; `with_covariates` uses `copy.copy`, so `_cov_life` is shared when no term is derived |
| C-07 | a | nothing feeds dyncov data longer than needed |
| C-08 | w | `test_pnbd_dyncov.py:574,593` approximate it; claim never stated |
| C-09 | a ! | single-category covariate silently accepted → `(600, 0)` design (**A4**) |
| C-10 | w ! | "covers every customer" covered; extra ids silently dropped by `.loc` (**A4**) |
| C-11 | a | dyncov has no duplicate, NA, or (Id, Date) completeness check |
| C-12 | c | |
| C-13 | a ! | re-setting covariates silently overwrites (**A4**) |
| C-14 | a | covariate `name_id` has no test; works |

## S5 Model expressions · S6 PMF · S7 Estimation

| | | evidence / note |
|---|---|---|
| M-01..M-03 | c | per-customer oracles |
| M-04 | a | the `PAlive` NaN regression case exists nowhere; verified finite `[0.99960, 0.99956, 0.74949, 0.99426]` (**C**) |
| M-05..M-08 | c | |
| M-09 | w | `test_families.py:296` pins the post-erratum CET at 1e-6, but no `erratum`/`#206` note anywhere |
| M-10..M-12 | c | |
| M-13 | o | BG/BB not ported |
| PMF-01 | w | `test_cdnow.py:198`; pnbd only, column means not row sums |
| PMF-02 | w | `test_pnbd_aggregate.py:254`; one scalar `T`, total only |
| PMF-03 | c | holds structurally — `pmf(k,T,params)` cannot see `x`/`t_x` |
| PMF-04 | a | `P(X=0)` monotone in `T` asserted for no family; verified to hold (**C**) |
| PMF-05 | a | no fitted-object `pmf()` generic, no `pmf.x.<k>` frame |
| PMF-06 | w | negative `k` rejected; non-integer silently truncated — `pmf(2.7,…) == pmf(2,…)` |
| F-01 | a | the port's only cdnow fit uses `estimation_split=37`, not `"1997-09-30"` |
| F-02..F-06 | a | no literature value pinned; BG/NBD and GGomNBD never fitted on cdnow at all (**C**) |
| F-07 | a ! | Bemmaor/Glady comparison absent, and the port's own `b`/`β` divergence unrecorded |
| F-08 | a | `test_gg.py:183` pins the paper's *apparel* fit, not cdnow (**C**) |
| F-09 | w | no family × dataset finiteness sweep; the kkt2 half is `o` — no KKT flag exists, only `converged` |
| F-10 | c | |
| F-11 | w | only 2 optimiser methods ever run |
| F-12 | a | hourly tested at the timeunit layer; no fit runs on hourly data |
| F-13 | a | nothing fits without spending then predicts on spending-bearing data |
| F-14, F-15 | c | |

## S8 Covariate fits

| | | evidence / note |
|---|---|---|
| X-01 | a | no fit ever run on identically-zero covariate data |
| X-02 | w | `test_pnbd_staticcov.py:267` uses *fixed* γ, not a random draw — the randomness is the claim; pnbd only; summed form not asserted |
| X-03 | c | |
| X-04 | a | no γ=0 `predict()` comparison in any of the three forms |
| X-05 | a | no γ=0 plot/pmf/pmf-plot comparison |
| X-06 | w | `test_pnbd_advanced.py:454`, one scalar at abs=1e-4 (**B6**) |
| X-07 | a | nothing shuffles covariate rows (**D2**) |
| X-08 | a | nothing reverses covariate columns (**D2**) |
| X-09 | w ! | 5 of 6; correlation + covariates raises (`estimate.py:193`) — README:39 is about families (**A7**) |
| X-10 | c | |
| X-11 | a | no syntactically illegal covariate name |
| X-12 | w | refusals covered; `start_m` never given a non-default value |
| X-13 | a | `start_m` has no single-value/NA/[-1,1] check, no warning when `use_cor=False` |
| X-14 | w | 2 of 6; NaN escapes and misreports as "objective is not finite"; the (life, trans) order *is* pinned |
| X-15 | w | 1 of 6; `names_cov_constr=["Nope"]` gives a misleading "covariate of both" message |

## S9 Dyncov

| | | evidence / note |
|---|---|---|
| DY-01 | a | `d_omega = d_1` never asserted, and `d_omega ≡ 1` makes the oracle degenerate (**B5**) |
| DY-02 | a | nothing sets covariate *data* to zero |
| DY-03 | w | `test_pnbd_dyncov.py:362`; aux walks only, `customers[:20]`, `Bbar=Dbar=0` half unchecked |
| DY-04 | a | nothing asserts at `i = 1` in the expectation table |
| DY-05 | a | same γ never passed to both processes, so `Bbar_i = Dbar_i` never holds |
| DY-06 | w | `test_pnbd_dyncov_predict.py:98,:107`; sample ids only |
| DY-07 | a | the static-as-dynamic cross-check does not exist (**D1**) |
| DY-08 | c | `test_pnbd_dyncov.py:194` — 30 columns × 2 parameter vectors × 600 customers, rtol 1e-8. The strongest item in the audit |
| DY-09 | a | LL never evaluated on both a split and an unsplit build |
| DY-10 | w | `test_pnbd_dyncov.py:411`; only the `α ≠ β` arm — `dyncov.py:355,374` branch on `alpha_1 >= beta_1` (**D3**) |
| DY-11 | a | no γ=0 predict-vs-nocov comparison |
| DY-12 | a ! | zero-length window refused where R gives `CET = 0` (**A1**) |
| DY-13 | a | untested, and its oracle `dyncov_palive.csv` is committed and orphaned (`backlog.md:1101`) |
| DY-14 | c | `test_pnbd_dyncov.py:476,495,456`, atol 1e-12 |
| DY-15 | w | degenerate: `d_omega ≡ 1`, so the non-boundary branch is never reached (**B5**) |
| DY-16 | c | incidental — exactly one of 600 customers has `t_x == T_cal` with `x > 0` |
| DY-17 | w | general splitting covered; the named 2-period edge case never constructed |
| DY-18 | a | the setup exists but asserts a predict-time error, not that aux walks survived |
| DY-19 | w | claims 1–2 pinned; the epsilon-apart claim is unreachable (day aggregation), undecided |
| DY-20 | w | matrices match; the round-trip is never asserted, and the test named for it asserts `x > 0` (**B2**) |
| DY-21 | c | `test_pnbd_dyncov.py:484,495` |
| DY-22 | a | every dyncov test uses `estimation_split=104` — zero of seven weekdays (**D4**) |
| DY-23 | a | no epsilon-interval construction; `_to_days` collapses to whole days |
| DY-24 | w | 2 of 5 |
| DY-25 | a | no partially-empty estimation/holdout, and no dyncov plotting test anywhere |

## S10 Prediction

| | | evidence / note |
|---|---|---|
| PR-01 | c | `test_predict.py:78-126`; CLVTools emits `actual.period.spending` — the Rdoc name in the spec is stale |
| PR-02 | w | the `log(1.1)` default *is* pinned (`test_predict.py:304`); `num_boots`/`level` defaults are not |
| PR-03 | o | no `newdata` parameter — the data object is the first positional argument |
| PR-04 | w | `test_predict.py:372`; `PAlive` only, 3 hand-picked ids, no static-cov arm |
| PR-05 | a ! | raises where R gives `CET = 0`; refusal pinned at `test_predict.py:349` (**A1**) |
| PR-06 | c | `test_predict.py:100` — genuinely pinned, but incidentally: apparel id 262 has a transaction on the estimation end, and `>=` would give 21 not 20 |
| PR-07 | c | `test_predict.py:267`, strict `<` for every customer |
| PR-08 | c | `test_predict.py:78`, rtol 1e-12 — though structurally unfalsifiable: one code path, not two |
| PR-09 | o | `predict.spending=TRUE` and `verbose` forwarding not ported |
| PR-10 | c | `test_predict.py:335` |
| PR-11 | w ! | only the `<= 0` half; no upper bound (**A3**) |
| PR-12 | c | `test_predict.py:354` |
| PR-13 | w | 2 of 3; differently-named covariates surface as a bare pandas `KeyError` |
| PR-14 | a | nothing checks for `NaN` parameters at predict time; returns an all-`NaN` table silently |
| PR-15 | w | date/period agreement pinned; not-single, NA and disallowed types untested |
| PR-16 | c | `test_predict.py:684-720` |

## S11 newcustomer

| | | evidence / note |
|---|---|---|
| NC-01 | c | all four constructors exercised |
| NC-02 | a ! | raises where R returns **1**; refusal pinned at `test_predict.py:564` (**A1**) |
| NC-03 | a | γ=0 nesting pinned for the likelihood, never for the new-customer path — which is a second, independent rate builder (`predict.py:262`) |
| NC-04 | a | same gap for zero covariate data |
| NC-05 | a | holds by construction (`row()` looks up by name); no test perturbs column order |
| NC-06 | w | `test_predict.py:537` never calls `predict` (**B1**) |
| NC-07 | a | no test varies `first_transaction` on an effectively static path |
| NC-08 | c | `test_pnbd_dyncov_predict.py:277,334`, rel 1e-12, both branches |
| NC-09 | w | the `< 1` case is dyncov only |
| NC-10 | w | both tests are the *refusal* side; the working cases are not asserted |
| NC-11 | w | only `str` ever passed as `first_transaction` |
| NC-12 | c | |
| NC-13 | — | not reached by the audit |

## S12 Bootstrapping

| | | evidence / note |
|---|---|---|
| B-01 | w | `num_boots=100` default never pinned |
| B-02 | w | `test_diagnostics.py:425`; 4 hand-picked ids, `x`/`t_x`/`T` only, never the full id list (**D6**) |
| B-03 | c | |
| B-04 | a | holdout rows do survive; nothing asserts it |
| B-05 | c | |
| B-06 | c | |
| B-07 | c | |
| B-08 | w | `test_diagnostics.py:546` asserts a `(600, 2)` shape only; alignment is correct but unchecked |
| B-09 | a ! | **`ClvDataDynCov` subclasses `ClvData`, so the resample branch never fires — the dyncov bootstrap silently refits without covariates** (**A2**) |
| B-10 | o | the library never holds a specification: `apply` does its own fitting, so nothing can drop `use_cor` or lambdas. Architectural, not tested — and B-09 breaks the dyncov arm regardless |
| B-11 | a | no all-customers refit compared to the original coefficients (**D6**) |
| B-12 | w | pinned at data level for one id |
| B-13 | c | |
| B-14 | w | one combination end to end; static-cov shape-tested; dyncov broken per B-09 |
| B-15 | w | 2 of 6 |

## S13 Inference

| | | evidence / note |
|---|---|---|
| I-01 | w | `test_inference.py:174,184`; coef↔vcov for plain pnbd only (**D**) |
| I-02 | w | no covariate fit has its `vcov()` index checked; under constraints only `names` is compared |
| I-03 | a | `confint` takes only `level`; no `parm` argument exists |
| I-04 | w | columns and the NaN/z pattern pinned; no structural check against R, printing never exercised |
| I-05 | w ! | ~6 of 29 configurations reach an oracle; `fit_pnbd_dyncov` takes no constraint/regularization/correlation argument, removing 12 (**A7**) |
| I-06 | c | warns rather than raising — recorded in README Findings |
| I-07 | o | no named-parameter accessor exists; all likelihood functions take positional parameters |
| I-08 | w | `ClvData.nobs()` pinned; fitted objects have no `nobs()` |
| I-09 | w | `fitted_data` reachable from no test; only its own doctest, self-referential (**B3**) |
| I-10 | w | one model only; "runs for all models" untested |
| I-11 | c | |

## S14 Formula interface

| | | evidence / note |
|---|---|---|
| FI-01..FI-03 | c | |
| FI-04 | w | 2 of 3; `~ . \| . + I(...)` unsupported — `'.'` is looked up as a literal column |
| FI-05 | a | duplicate name yields `['Gender','Gender']` and a rank-deficient `(600, 2)` design — a defect, untested |
| FI-06 | a ! | naming-by-term-text *is* recorded (`audit.md:489`); bare `log(x+2)` unsupported is not (**A7**) |
| FI-07 | a ! | `Gender*Channel` and `.-Gender` parse then fail as column names (**A7**) |
| FI-08 | w | the *type* claim never asserted |
| FI-09 | w | plain and `.` cases share storage via `copy.copy` |
| FI-10 | c | |
| FI-11 | w | only `not np.allclose(...)`, not "every coefficient"; nothing on cbs `x` or the id set |
| FI-12 | w | both sides separately oracle-pinned; the invariant never stated, and the two come from different code paths (**D5**) |
| FI-13 | w | 4 of 11; `**kwargs` forwards anything; a LHS parses silently and fails with the wrong message |
| FI-14 | w | 1 of 4 |
| FI-15 | w | ~4 of 12; `constraint()` syntax does not exist here |

## S15 Input validation

| | | evidence / note |
|---|---|---|
| V-01 | w | length and `<= 0` pinned for all 5 fits; a `NaN` start passes `np.any(start <= 0)` and misreports as a data fault |
| V-02 | w | covariate start is one scalar, so 5 of 7 claims cannot arise; the API divergence is unrecorded |
| V-03 | a | `options_for` merges overrides with zero validation; unknown keys reach SciPy as a *warning* where R errors |
| V-04 | w | Python's `TypeError` covers it, but `**kwargs` forwarding lands it at the inner signature; never asserted |
| V-05 | a | no single-logical validation exists anywhere |
| V-06 | w ! | ~10 of 42; `NA` in `Id`/`Date` silently drops the row; empty frame accepted; non-DataFrame gives `AttributeError` (**A4**) |
| V-07 | w | a few defaults pinned; `time_unit` has a default where R has none |
| V-08 | w | `sample`/`ids`/bins/Price pinned; `label`, `other.models`, `annotate.ids` have no counterpart |
