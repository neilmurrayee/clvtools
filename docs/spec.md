# Validation specification, derived from the sources alone

**What this is.** An independent statement of what a correct port of CLVTools
0.12.1 must do, derived *only* from the primary sources — the paper, the R
package's documentation, and the R package's own test suite. It was written
without reading `src/` or `tests/` of this repository, so that it can only
describe what the sources demand and never what this implementation happens to
do. `docs/spec-audit.md` is the separate pass that joins it against the suite
we actually have.

**Why derive it this way.** `docs/audit.md` indexed the paper and the R
`NAMESPACE` *against this implementation*: for each thing the sources contain,
does a counterpart exist here. That direction finds missing features. It cannot
find a claim that is present but wrongly pinned, because the implementation is
what supplies the index. Inverting the index is the whole point of this file.

## Sources

| Tag | Source | Extent |
|---|---|---|
| `P` | Meierer et al., JSS 5634, `arXiv-2602.09845v1/jss5634.tex` | 1,956 lines; 34 `CodeInput` chunks, 16 `CodeOutput`, 11 numbered equations |
| `Rdoc` | `CLVTools/man/*.Rd`, `vignettes/CLVTools.Rmd` | 91 man pages |
| `Rtest` | `CLVTools/tests/testthat/*.R` from the CRAN source tarball, version 0.12.1, dated 2025-11-06 | 81 files, 12,647 lines, 645 named `test_that` claims |
| `Lit` | Literature values the R suite itself asserts against | Fader/Hardie/Lee 2005; Bemmaor/Glady 2012; Fader/Hardie 2013 |

The R test suite is **not installed** with the package (`.Rlib/CLVTools` has no
`tests/`); it comes from `https://cran.r-project.org/src/contrib/CLVTools_0.12.1.tar.gz`.
Nothing in this repository fetches it, so before this file it had never been
consulted. It is the densest of the four sources and the only one that states
what must happen on *bad* input.

## How to read an item

    ID  claim — source — oracle, and the tolerance the claim deserves

`oracle` names the thing that decides the answer. Where the source fixes an
expected value, that value is quoted. Where the claim is an invariant (two paths
must agree), the invariant is the oracle and no external number is needed —
these are the items worth having most, because they are checkable without R.

Tolerances follow the house rule: at *published or fixed* parameters an
expression should agree to 1e-9..1e-14; where an optimiser runs, assert on the
order the R suite itself uses (`tolerance = 0.001` on coefficients, testthat's
default 1.5e-8 on expressions).

---

# S1 — Transaction data construction (`clvdata`)

**D-01** Two transactions by the same `Id` at the same timepoint are aggregated
to one row. — `Rtest:test_correctness_clvdata_clvdata.R` — 6 rows over 2 ids →
3 rows; exact.

**D-02** When spending is present, aggregation *sums* `Price` within
(`Id`, timepoint). — `Rtest` — constructed frame, prices `1,2,3 / 4,5 / 6` →
`6 / 9 / 6`; exact.

**D-03** Aggregation applies to `Date` and to `POSIXct` timepoints alike. —
`Rtest` — same two cases run twice, once per type.

**D-04** Repeat transactions drop exactly the first transaction per customer,
whatever order the input rows arrive in. — `Rtest` — asserted for input sorted
by `Date` ascending, `Date` descending, `Id` ascending, `Id` descending; all
four give the identical set.

**D-05** When a customer's first two transactions share a timepoint, only one is
removed. — `Rtest` — `2019-01-01 ×2, 2019-01-02` → `2019-01-01, 2019-01-02`.

**D-06** Zero-repeaters (a single transaction) appear in the transaction data
but not in the repeat transactions. — `Rtest` — id `"2"` absent from repeat, id
`"1"` present.

**D-07** Aggregating first and removing first-transactions after removes *all*
first transactions, and `clvdata()` end-to-end gives the same result as the two
steps composed. — `Rtest` — exact set equality.

**D-08** `Id` given as character, factor, or numeric yields equal objects. —
`Rtest` — `expect_equal` on the whole object with `call` normalised.

**D-09** `Price` given as integer or numeric yields equal objects. — `Rtest`.

**D-10** Row order of the input transactions does not affect the object. —
`Rtest` — shuffled cdnow vs original, equal.

**D-11** `name.price = NULL` produces an object with no spending: `has.spending`
false, no `Price` column in transactions or repeat transactions, and **no
Spending rows in the descriptives table**. — `Rtest` — the descriptives
consequence is asserted, not just the flag.

**D-12** `name.price = "Price"` produces the converse, including Spending rows
present in descriptives. — `Rtest`.

**D-13** The input transaction table is copied, not referenced. — `Rtest` —
address inequality. *(Python analogue: mutating the caller's frame afterwards
must not change the object.)*

**D-14** `name.id`, `name.date`, `name.price` default to `"Id"`, `"Date"`,
`"Price"`. — `Rdoc:clvdata.Rd`, `Rtest`.

**D-15** Transaction dates accept `Date`, `POSIXct`, `POSIXlt`, character, and
character-including-time. — `Rtest:test_runability_clvdata_clvdata.R`.

**D-16** An unneeded extra column in the input is tolerated. — `Rtest`.

**D-17** `clv.data.get.repeat.transactions.in.estimation.period()` equals
building repeat transactions from the estimation-period transactions. —
`Rtest` — set equality. *An ordering/nesting invariant that needs no oracle
data.*

**D-18** Mean interpurchase time is `NA` for a customer with one transaction,
and the mean of interval differences otherwise; the result must not depend on
the input being sorted by (`Id`, `Date`). — `Rtest` — computed on data sorted by
`Price` and compared to the reference implementation on data sorted by
(`Id`, `Date`); testthat default tolerance.

# S2 — Time, splits, and `data.end`

**T-01** Date-based time units have an epsilon of one day; datetime-based units
an epsilon of one second. — `Rtest:helper_testthat_correctness_clvtime.R`.

**T-02** Estimation start is the first transaction in the data. — `Rtest`.

**T-03** `estimation.split = NULL` sets estimation end = holdout start = holdout
end = the last transaction date, i.e. no holdout. — `Rtest` — all four
timepoints asserted on cdnow.

**T-04** `estimation.split` given as a number of periods, a `Date`, a `POSIXct`,
or a character string produce the same split. — `Rtest` — asserted separately
for input data held as `Date`, `POSIXct` UTC, `POSIXct` Australia/Darwin, and
character. **The non-UTC timezone case is a distinct claim.**

**T-05** A numeric `estimation.split` that implies a partial period warns. —
`Rtest`.

**T-06** Different `time.unit`s with the same calendar split date produce the
same four timepoints. — `Rtest` — days vs weeks on `"1997-09-17"`.

**T-07** `time.unit` accepts different cases, full names, and plurals
(`"w"`, `"week"`, `"weeks"`, `"Weeks"`). — `Rtest:test_runability_clvdata_clvdata.R`.

**T-08** `data.end` fails if before the last transaction. — `Rtest`.

**T-09** `data.end` fails if it leaves a holdout period shorter than 2 periods. — `Rtest`.

**T-10** `data.end` fails if it leaves an estimation period shorter than 1 period. — `Rtest`.

**T-11** `data.end` fails if before `estimation.split`. — `Rtest`.

**T-12** `data.end` set to the last transaction timepoint gives the same object
as omitting it. — `Rtest` — object equality.

**T-13** With `data.end` given, `estimation.split` may lie after the last
transaction. — `Rtest`.

**T-14** `data.end` moves only holdout end, when estimation split is before it. — `Rtest`.

**T-15** `data.end` moves the start of the prediction period: with
`estimation.split = NULL, data.end = "1998-07-15"`, `predict(prediction.end =
"1998-07-30")` returns `period.first == "1998-07-16"` and `period.last ==
"1998-07-30"`. — `Rtest:helper_testthat_correctness_transactions.R` — exact
dates.

**T-16** Conversion is exact and total across `character → POSIX`, `character →
Date`, `Date → POSIX`, `IDate → POSIX`, `Date → Date`, `IDate → Date`, `POSIXct
→ POSIXct`, `POSIXct → Date`. — `Rtest` — eight separate claims.

**T-17** Floor-date rounds down, and is idempotent when already on a boundary. — `Rtest`.

**T-18** Covariate date ranges are correct for all four combinations of
{start on, start off} × {end on, end off} a period boundary. — `Rtest` — four
separate claims.

**T-19** The expectation/prediction period table is valid for `prediction.end`
as numeric and as date, and works for a single period and for two. — `Rtest`.

**T-20** With no prediction length, `period.first` and `period.last` are the
same date and the length is 0. — `Rtest`.

**T-21** A `prediction.end` before the estimation end stops. — `Rtest`.

**T-22** `prediction.end` as number, `Date`, `POSIXct`, or character give the
same values; `14.4` and `14` agree. — `Rtest:helper_s3_fitted_plot.R`,
`helper_s3_fitted_predict.R`.

# S3 — Descriptives and the data-level interface

**S-01** `summary(clv.data)` reports the descriptives table; no cell is `NA`. —
`Rtest:test_correctness_clvdata_s3.R`.

**S-02** Zero repeaters are counted correctly in the summary. — `Rtest`.

**S-03** `summary(ids = NULL)` equals `summary(ids = <all ids>)`. — `Rtest`.

**S-04** `summary(ids = <subset>)` selects those ids and gives different output
from the full summary. — `Rtest` — two claims.

**S-05** The holdout column shows `-` for a customer with no holdout
transactions. — `Rtest`.

**S-06** `nobs`, `print`, `show`, `as.data.frame`, `as.data.table`, `subset`
all work on `clv.data`. — `Rtest:test_runability_clvdata_s3.R`,
`Rdoc:nobs.clv.data.Rd, as.data.frame.clv.data.Rd, subset.clv.data.Rd`.

**S-07** `as.data.frame`/`as.data.table` return a **copy** every time. —
`Rtest` — asserted twice, once per method.

**S-08** `subset(sample = "full"|"estimation"|"holdout")` selects the correct
data; when there is no holdout, full and estimation coincide; results are the
same when argument positions are swapped. — `Rtest`, `Rdoc:subset.clv.data.Rd`.

**S-09** `as.clv.data()` exists and `clvdata` works when called through it. —
`Rdoc:as.clv.data.Rd`, `Rtest`.

**S-10** `plot(clv.data, which=)` accepts `"tracking"`, `"frequency"`,
`"spending"`, `"interpurchasetime"`, `"timings"` — five kinds. —
`Rdoc:plot.clv.data.Rd`.

**S-11** Tracking plot without `data.end`: the last period is `NA`, and the plot
emits no warning. — `Rtest`.

**S-12** Tracking plot with `data.end`: values are `NA` after the last
transaction through to `data.end`, with no warning. — `Rtest`.

**S-13** Frequency plot: actual transactions never include a 0 bin; the
remaining label is the highest level and disappears when not needed. — `Rtest`
— two claims. Defaults: `trans.bins = 0:9`, `count.repeat.trans = TRUE`,
`count.remaining = TRUE`, `label.remaining = "10+"`. — `Rdoc`.

**S-14** Spending plot: different sample → different data; `mean.spending`
toggles; correct number plotted. — `Rtest`.

**S-15** Interpurchase-time plot removes zero-repeaters. — `Rtest`.

**S-16** Timings plot honours `ids` and `annotate.ids`. — `Rtest`,
`Rdoc` (default `annotate.ids = FALSE`, `ids = c()`).

# S4 — Covariate data (`SetStaticCovariates`, `SetDynamicCovariates`)

**C-01** Character and factor covariates produce the same dummies — asserted
both with and without a holdout period. — `Rtest:test_correctness_clvdata_setstaticcov.R`.

**C-02** A 2-category variable produces 1 dummy; a 3-category variable produces
2. — `Rtest` — two separate claims, static and dynamic each.

**C-03** Categories convert to dummies both when there are no numeric covariates
and when there are. — `Rtest` — two claims.

**C-04** Numeric covariates stay numeric, both with and without categorical
covariates present. — `Rtest` — two claims.

**C-05** Covariate column names are coerced to syntactically valid names. —
`Rtest` — static and dynamic.

**C-06** Covariate data is copied, not referenced. — `Rtest`.

**C-07** Dynamic covariates: data longer than needed before estimation start is
cut to the correct range. — `Rtest:test_correctness_clvdata_setdynamiccov.R`.

**C-08** Dynamic covariates: if one covariate's data is longer than the other's,
all data must be that long. — `Rtest`.

**C-09** A covariate with a single category is rejected. — `Rtest` — static and
dynamic.

**C-10** Covariate `Id`s must cover every transaction-data customer, and may not
introduce customers absent from the transaction data. — `Rtest` — two claims,
static and dynamic.

**C-11** Dynamic covariates must cover every (`Id`, `Date`) pair, with no
duplicates and no `NA` in `Id`, `Date`, or any covariate column. — `Rtest` —
five claims.

**C-12** Dynamic covariate data must not end before `data.end`. — `Rtest`.

**C-13** Setting covariates on data that already has covariates fails. —
`Rtest` — static and dynamic.

**C-14** Covariates work with `Id` as factor, numeric, or character, and with a
non-standard `Id` column name. — `Rtest:test_runability_clvdata_setstaticcov.R`.

# S5 — Model expressions

For each family the paper's equations and the R package's per-customer entry
points must agree at fixed parameters. These are the items where 1e-9..1e-14 is
the right bar.

**M-01** Pareto/NBD `LL` per customer, no covariates. — `P §2.2`, `Rdoc:pnbd_LL.Rd`.

**M-02** Pareto/NBD `PAlive`, `CET`, `DERT`, expectation, `pmf`. —
`Rdoc:pnbd_PAlive.Rd, pnbd_CET.Rd, pnbd_DERT.Rd, pnbd_expectation.Rd, pnbd_pmf.Rd`.

**M-03** Pareto/NBD expectation matches the closed form
`(r β)/(α (s-1)) · (1 - (β/(β+t))^(s-1))` evaluated in R, for both no-cov and
static-cov (`α_i = α exp(-x'γ_trans)`, `β_i = β exp(-x'γ_life)`). —
`Rtest:test_correctness_pnbd_nocovstaticcov.R` — testthat default tolerance.
**Note the covariate roles: `α` carries transaction, `β` carries lifetime.**

**M-04** Pareto/NBD `PAlive` is finite for the extreme inputs that produced
`NaN` in an earlier implementation: `x = (221, 254, 161, 204)`,
`t.x = (103.42857, 97.14286, 94.71429, 98.57143)`,
`T.cal = (103.57143, 97.28571, 98.00000, 99.42857)` at
`r=0.5143, α=2.8845, s=0.2856, β=14.1087`. — `Rtest` — **a named regression
case with exact inputs; the assertion is finiteness, not a value.**

**M-05** BG/NBD expectation matches
`((a+b-1)/(a-1)) · (1 - (α/(α+t))^r · ₂F₁(r, b; a+b-1; t/(α+t)))`, no-cov and
static-cov. — `Rtest:test_correctness_bgnbd.R`. **BG/NBD covariate roles differ
from Pareto/NBD: `α_i = α exp(-x'γ_trans)` but `a_i = a exp(+x'γ_life)` and
`b_i = b exp(+x'γ_life)` — sign positive, and the *same* lifetime index enters
both `a` and `b`.**

**M-06** GGompertz/NBD `LL` per customer matches the Bemmaor–Glady MATLAB
formulation, no-cov exactly and static-cov to 1e-4 (the difference verified to
be numerical integration). — `Rtest:test_correctness_ggomnbd.R` — the MATLAB
integral is quoted in the test and is reproducible without R.

**M-07** GGompertz/NBD `PAlive` matches the same MATLAB formulation, no-cov
exactly and static-cov to 1e-4. — `Rtest`.

**M-08** GGompertz/NBD expectation matches the MATLAB cumulative expectation
`(r/α)·[ (β/(β+e^{bt}-1))^s · t + b s β^s ∫₀^t τ e^{bτ} (β+e^{bτ}-1)^{-(s+1)} dτ ]`,
no-cov and static-cov (static to 1e-6). — `Rtest`. **GGomNBD covariate roles:
`α_i = α exp(-x'γ_trans)`, `β_i = β exp(-x'γ_life)`.**

**M-09** GGompertz/NBD `CET` changed after Adler's erratum (issue #206) and no
longer matches the original MATLAB code. — `Rtest` — recorded as a comment in
the R suite; **a deviation to pin, not a value to match.**

**M-10** Gamma-Gamma `LL`. — `Rdoc:gg_LL.Rd`, `P §2.5`.

**M-11** `ggomnbd_PMF` exists as a documented entry point. — `Rdoc:ggomnbd_PMF.Rd`.

**M-12** `vec_gsl_hyp2f0_e` and `vec_gsl_hyp2f1_e` are exported and used inside
the model expressions. — `Rdoc:vec_gsl_hyp2f0_e.Rd, vec_gsl_hyp2f1_e.Rd`. The
Python port needs an equivalent to the same accuracy wherever ₂F₁ or ₂F₀ appears.

**M-13** `bgbb` is documented. — `Rdoc:bgbb.Rd`. **Scope question: is the BG/BB
model in or out?**

# S6 — PMF invariants

All four hold for every family that has a PMF, and are checkable with no oracle
data at all.

**PMF-01** `pmf(x=0:6)` row sums strictly exceed `pmf(x=0:5)` row sums. —
`Rtest:helper_testthat_correctness_transactions.R`.

**PMF-02** For `x = 0:20`: every value in `[0, 1]`, every row sum `≤ 1`, no
`NA`. — `Rtest`.

**PMF-03** PMF values depend *only* on `T.cal`: within a `T.cal` there is
exactly one distinct value per `x`, and across `T.cal` there are as many
distinct values as there are distinct `T.cal`. — `Rtest` — **both halves are
asserted; the second half is what catches a PMF that ignores `T.cal`.**

**PMF-04** `P(X = 0)` is strictly decreasing in `T.cal`. — `Rtest` — with the
derivation quoted: Pareto/NBD `dP(X=0)/dt = -λ/e^{(λ+μ)t} < 0`, BG/NBD
`-μ/e^{μt} < 0`.

**PMF-05** `pmf()` accepts integer and numeric `x`, and a single `x = 0`; the
returned frame has one `Id` column plus one `pmf.x.<k>` column per `x`. —
`Rtest:helper_s3_fittedtransactions_pmf.R`, `Rdoc:pmf.Rd` (default `x = 0:5`).

**PMF-06** `x` must be a valid non-negative integer vector; otherwise error. —
`Rtest:test_inputchecks_pmf.R`.

# S7 — Estimation and published parameter recovery

**F-01** Pareto/NBD on cdnow, `estimation.split = "1997-09-30"`, start
`(r=1, α=1, s=1, β=1)` → `r=0.5532, α=10.5763, s=0.6063, β=11.6715`,
`LL = -9594.976`. — `Rtest:test_correctness_pnbd_nocovstaticcov.R` — tolerance
0.001.

**F-02** The same fit reproduces the Fader/Hardie/Lee (2005) published values
`r=0.553, α=10.578, s=0.606, β=11.669`, `LL = -9595.0`. — `Lit` via `Rtest`.

**F-03** Pareto/NBD standard errors on cdnow from start `(r=1, α=2, s=1, β=2)`:
`r=0.0476264, α=0.8427222, s=0.1872594, β=6.2105448`. — `Rtest` — tolerance
0.001. **`β`'s standard error is 6.21 against an estimate of 11.67: a flat
ridge, and the item most likely to move between platforms.**

**F-04** BG/NBD on cdnow, start `(r=1, α=3, a=1, b=3)` →
`r=0.2425945, α=4.4136019, a=0.7929199, b=2.4258881`, `LL = -9582.429`. —
`Rtest:test_correctness_bgnbd.R`.

**F-05** The same fit reproduces Fader/Hardie/Lee (2005) `r=0.243, α=4.414,
a=0.793, b=2.426`, `LL = -9582.4`. — `Lit`.

**F-06** GGompertz/NBD on cdnow, start `(r=0.5, α=2, b=0.5, s=0.5, β=0.5)` →
`r=0.55313, α=10.5758, b=0.0000011, s=0.60682, β=0.000013`, `LL = -9594.9762`.
— `Rtest:test_correctness_ggomnbd.R`.

**F-07** The same fit is compared to Bemmaor/Glady (2012) Table 2, p. 1018:
`r=0.553, α=10.578, b=0.0002, s=0.603, β=0.0026`, `LL = -9594.98`. — `Lit`.
**`b` and `β` differ from CLVTools' own estimate by two orders of magnitude
while the log-likelihood agrees to 5 significant figures — a flat direction
that any port will also land differently on. This is a deviation to pin, not a
target to hit.**

**F-08** Gamma-Gamma on cdnow reproduces Fader/Hardie (2013) `p=6.25, q=3.74,
γ=15.44`, `LL = -4055.9177`. — `Lit` via `Rtest:test_correctness_gg.R`.

**F-09** "Flawless results out of the box": for each family on cdnow, on apparel
without covariates, and on apparel with static covariates — no non-finite value
in `Estimate` or `Std. Error`; no non-finite value anywhere in `predict()`; no
non-finite value in `plot()` except the last (partial) period; `kkt1` true;
`kkt2` true for Pareto/NBD, BG/NBD and Gamma-Gamma but **not** for
GGompertz/NBD. — `Rtest:helper_testthat_correctness_clvfitted.R` — **the kkt2
exception for GGomNBD is an explicit claim about which model fails second-order
conditions.**

**F-10** `zval` and `pval` are `NA` for the main model parameters on purpose,
while `Estimate` and `Std. Error` are finite. — `Rtest` — same helper.

**F-11** Fits work from custom `start.params.model`, custom `optimx.args`, and
across all optimx methods. — `Rtest:helper_testthat_runability_clvfitted.R`.

**F-12** Fits work on hourly data. — `Rtest`.

**F-13** A fit works with no spending data present, and can then predict on
newdata that *does* have spending. — `Rtest:helper_testthat_runability_nocov.R`.

**F-14** A spending model cannot be fit without spending data, and cannot
predict on newdata lacking spending. — `Rtest:helper_testthat_inputchecks_nocov.R`.

**F-15** Gamma-Gamma cannot be fit on data with negative spending. —
`Rtest:test_inputchecks_gg.R`.

# S8 — Covariate fits: constraints, regularization, correlation

**X-01** Fitting with covariate data that is identically 0 gives model
parameters nearly equal to the no-covariate fit. — `Rtest:helper_testthat_consistency.R`
— tolerance 0.001.

**X-02** With covariate data identically 0, the static-cov individual LL equals
the no-cov individual LL **for arbitrary random covariate parameters**
(`rnorm`) — both element-wise and summed. — `Rtest` — the random draw is the
point: it must hold for any γ.

**X-03** With covariate parameters γ = 0, the static-cov individual LL equals
the no-cov individual LL at the same model parameters. — `Rtest`.

**X-04** With γ = 0, `predict()` is identical to the no-cov fit — asserted
three ways: out of the box, with `prediction.end = 6`, and with
`continuous.discount.factor = 0.25`. — `Rtest`.

**X-05** With γ = 0, `plot()` values and `pmf()` (`x = 0:10`) and the PMF plot
are identical to the no-cov fit. — `Rtest` — three claims.

**X-06** Regularization with `reg.lambdas = c(trans=0, life=0)` gives exactly
the coefficients *and the summary coefficient table* of the unregularized fit. —
`Rtest:helper_testthat_correctness_transactions.R` — `expect_equal`, no
loosened tolerance.

**X-07** Row-shuffled covariate data gives an identical fitted object (call and
timing aside) and identical `predict`, `plot`, `summary`. — `Rtest`.

**X-08** Column-reversed covariate data gives an identical fitted object and
identical `predict`, `plot`, `summary`. — `Rtest`.

**X-09** Fits work with 2 constraints; with 1 constraint and 1 free covariate;
with regularization; with zero regularization lambdas; with combined interlayers
with and without correlation. — `Rtest:helper_testthat_runability_staticcov.R`
— six claims.

**X-10** Estimation reduces to the relevant covariates only. — `Rtest`.

**X-11** Static covariates with syntactically illegal names work. — `Rtest`.

**X-12** `use.cor = TRUE` works, alone and with start parameters. —
`Rtest:helper_testthat_runability_common.R`. Correlation is Pareto/NBD only;
other families must reject `use.cor`. — `Rtest:helper_testthat_inputchecks_*.R`
("Fails for use.cor").

**X-13** Correlation start parameter must be a single non-`NA` numeric in
`[-1, 1]`; giving it with `use.cor = FALSE` warns. — `Rtest`.

**X-14** Regularization lambdas must be a named numeric of exactly two entries
named `life` and `trans`, non-negative, no duplicates, no `NA`. — `Rtest`
— six separate failure claims.

**X-15** Constraint names must be a non-empty character vector, present among
the covariates, without duplicates or `NA`; start parameters may not be given
for a constrained covariate as if it were free. — `Rtest` — six claims.

# S9 — Time-varying covariates (dyncov)

**DY-01** `d_omega = d_1`. — `Rtest:test_correctness_pnbd_dyncov.R`.

**DY-02** With covariate data 0: `A_i = C_i = 0` and `Dbar_i = Bbar_i = 0`. —
`Rtest` — two claims.

**DY-03** With all covariate parameters 0: `A_i = C_i = 1` and
`Bbar_i = Dbar_i = 0`. — `Rtest`. **Note this differs from DY-02: zero *data*
gives 0, zero *parameters* give 1.**

**DY-04** For all `i = 1`, `Bbar_i = 0` and `Dbar_i = 0`. — `Rtest`.

**DY-05** For identical life and transaction covariate data, `Bbar_i = Dbar_i`. — `Rtest`.

**DY-06** `i` is an integer with the same maximum for every customer, and all
customers start and end on the same date. — `Rtest` — two claims.

**DY-07** With static covariate data supplied as dynamic: `A_i` and `C_i` equal
the static values, `Dbar_i = 0`, and `Bbar_i = -T·A`. — `Rtest` — three claims,
and the cleanest available cross-check of the dyncov machinery.

**DY-08** The dyncov LL yields the correct intermediate results. — `Rtest`.

**DY-09** The dyncov LL is the same whether or not there is a holdout period —
i.e. whether or not there are more covariates than required. — `Rtest`.

**DY-10** With γ = 0 the dyncov individual LL equals the no-cov individual LL,
asserted for both `α ≠ β` and `α = β` (`α = β = 1.234`). —
`Rtest:test_consistency_pnbd.R` — **both arms of the branch.**

**DY-11** With γ = 0, dyncov `predict` matches no-cov after renaming `DECT` →
`DERT` and `predicted.period.CLV` → `predicted.CLV`, comparing all other
columns. — `Rtest`.

**DY-12** `CET = 0` for a zero-length prediction period. — `Rtest`.

**DY-13** The numerically improved `PAlive` gives the same result as the old
one. — `Rtest`.

**DY-14** Walk creation: `d_x` arithmetic is correct (checked against a
spreadsheet), and `d_x` changes correctly on the lower boundary. —
`Rtest:test_correctness_dynamiccov_walkcreation.R` — two claims.

**DY-15** `d_omega` is correct in the cbs. — `Rtest`.

**DY-16** `tjk` is correct, including for auxiliary transactions where
`t.x = T.cal`. — `Rtest` — two claims.

**DY-17** The auxiliary walk splitting method is correct; an aux walk is 2
periods when `T` is on a week start and the customer is alive one day before
`T-1` with no real life walk. — `Rtest` — two claims.

**DY-18** Aux walks are not lost when covariates exist only for the calibration
period (issue #134). — `Rtest`.

**DY-19** Real transaction walks are correct; none exist when there are no
repeat transactions; and no walk is lost when transactions are only one epsilon
apart. — `Rtest` — three claims.

**DY-20** Real life walk plus aux life walk reconstruct the original covariate
data. — `Rtest` — **an exact round-trip invariant.**

**DY-21** For repeat buyers, the life walk and the first transaction walk start
on the same timepoint. — `Rtest`.

**DY-22** All walks are basically correct for an `estimation.split` on *every
day of the week*. — `Rtest` — seven cases.

**DY-23** An interval created with epsilon equals one created with `shift()`. — `Rtest`.

**DY-24** Dyncov predict/plot works with newdata: predicting the original data,
predicting a sample of it, predicting further ahead than the fitting data
allows, plotting further ahead, and with ≤ 2 periods (issue #128). —
`Rtest:helper_testthat_runability_dynamiccov.R` — five claims.

**DY-25** Fit, plot and predict work with a partially empty estimation or
holdout period. — `Rtest:test_runability_pnbd_dynamiccov.R`.

# S10 — Prediction

**PR-01** `predict()` with no arguments on a fit with holdout returns
`Id, period.first, period.last, period.length, actual.x, actual.total.spending,
PAlive, CET, DERT, predicted.mean.spending, predicted.CLV`. —
`Rdoc:predict.clv.fitted.transactions.Rd`, `Rtest:helper_s3_fitted_predict.R`
("Formal correct").

**PR-02** Defaults: `newdata = NULL`, `prediction.end = NULL`,
`predict.spending = gg`, `continuous.discount.factor = log(1 + 0.1)`,
`uncertainty = "none"`, `level = 0.9`, `num.boots = 100`. — `Rdoc`. **The
discount factor default is `log(1.1) ≈ 0.0953`, not `0.1`.**

**PR-03** `predict(newdata = <the fitting data>)` equals `predict()`. —
`Rtest:helper_testthat_correctness_transactions.R`.

**PR-04** Fitting on a 100-customer sample and predicting with `newdata = <full
data>` returns one row per customer in the full data, and the sampled
customers' rows are identical to predicting the sample alone. — `Rtest` —
`nrow` asserted on both sides. Also asserted for a 300-customer apparel sample
with static covariates.

**PR-05** `CET = 0` when `prediction.end = 0`. — `Rtest`.

**PR-06** `actual.x` counts holdout transactions correctly, and a transaction
falling *on* the split date is not counted: with
`estimation.split = "1998-01-01"` on cdnow, `Id "1" → 0`, `Id "1000" → 3`,
`Id "1056" → 3` (id 1056 has a transaction on 1998-01-01). — `Rtest` —
**exact per-customer values; the boundary case is the point.**

**PR-07** A higher `continuous.discount.factor` gives a strictly smaller `DERT`
for every customer: asserted at 0.001 vs 0.06 vs 0.99. —
`Rtest:test_correctness_pnbd_nocovstaticcov.R`.

**PR-08** The spending prediction inside `predict()` on a transaction model
equals a standalone Gamma-Gamma fit's `predicted.mean.spending`. — `Rtest` —
**a cross-model invariant needing no oracle.**

**PR-09** `predict.spending` accepts a logical, a fitted spending model, and a
spending-model method; `verbose` is forwarded to the spending fit it makes. —
`Rtest:helper_s3_fitted_predict.R` — four claims.

**PR-10** `predict()` fails when `prediction.end` is absent and there is no
holdout period. — `Rtest:test_inputchecks_predict_transactions.R`.

**PR-11** `predict()` fails for a discount factor outside `[0, 1)`. — `Rtest`.

**PR-12** `predict()` fails for `prediction.end` before the estimation end. — `Rtest`.

**PR-13** `predict(newdata=)` fails if newdata is not a `clv.data`, is the wrong
kind of `clv.data`, or has differently-named covariates. —
`Rtest:helper_testthat_inputchecks_predict_plot.R` — three claims.

**PR-14** `predict()` fails if any prediction parameter is `NA` — separately for
model parameters and for life/trans covariate parameters. — `Rtest` — two claims.

**PR-15** `prediction.end` must be single, non-`NA`, of an allowed type, and in
the original `date.format`; when `newdata` is given it is interpreted relative
to `newdata`. — `Rtest` — five claims.

**PR-16** Unused arguments in `...` are an error, not silently ignored — for
`predict` (transactions and spending) and for `plot`. — `Rtest` — four
occurrences across files.

# S11 — Predictions for prospective customers (`newcustomer`)

**NC-01** `newcustomer(num.periods)`, `newcustomer.static(num.periods,
data.cov.life, data.cov.trans)`, `newcustomer.dynamic(num.periods,
data.cov.life, data.cov.trans, first.transaction)`, and
`newcustomer.spending()` all exist. — `Rdoc:newcustomer.Rd`, `P §6.3.4`.

**NC-02** `predict(newdata = newcustomer(num.periods = 0)) == 1` — for no-cov,
static cov, and dyncov. — `Rtest` — three claims. **The expected value is 1,
not 0.**

**NC-03** With γ = 0, the static-cov new-customer prediction equals the no-cov
one; likewise for dyncov via `newcustomer.dynamic`. —
`Rtest:helper_testthat_consistency.R`, `test_consistency_pnbd.R` — `num.periods
= 7.89`.

**NC-04** With covariate data 0 and model parameters forced equal, the
static-cov new-customer prediction equals the no-cov one. — `Rtest`.

**NC-05** The new-customer prediction is independent of covariate column
ordering (and, for dyncov, row ordering). — `Rtest` — two claims.

**NC-06** Different covariate data give different predictions: perturbing life
covariates, transaction covariates, or both give three mutually distinct
answers. — `Rtest` — asserted pairwise.

**NC-07** For dyncov, the prediction is independent of `first.transaction` when
the covariate data is effectively static. — `Rtest`.

**NC-08** The `dt.ABCD` table is formatted according to `first.transaction` and
the prediction end. — `Rtest`.

**NC-09** Works for `num.periods` less than 1, 2, and 3 — for no-cov, static
cov, and dyncov. — `Rtest:test_runability_newcustomer.R` — three claims.

**NC-10** Works with covariate data starting before `first.transaction`, ending
after `num.periods`, and drawn from a different period than the fitting data. —
`Rtest` — three claims.

**NC-11** `Cov.Date` and `first.transaction` accept `Date`, character,
`POSIXct`, and `POSIXlt`. — `Rtest`.

**NC-12** Works for spending models. — `Rtest`.

**NC-13** `num.periods` must be numeric and `>= 0`; covariate data must have the
right format and, for dyncov, the right dates; the `newcustomer` kind must match
the fitted model's kind; passing other parameters to `predict` alongside is an
error. — `Rtest:test_inputchecks_newcustomer.R` — eight claims.

# S12 — Uncertainty and bootstrapping

**B-01** `clv.bootstrapped.apply(object, num.boots, fn.boot.apply, fn.sample =
NULL, ...)` exists with that signature. — `Rdoc:clv.bootstrapped.apply.Rd`.

**B-02** Sampling *all* customers reproduces the original `clv.data` object
exactly — for no-cov, static cov, and dyncov. — `Rtest:test_correctness_bootstrapping.R`.

**B-03** The bootstrapped `clv.time` keeps all information including the holdout
period, even when the sample is a single zero-repeater. — `Rtest` — sampling id
`"102"` alone still gives estimation start `1997-01-01`.

**B-04** Sampling keeps holdout repeat transactions. — `Rtest`.

**B-05** Passing non-existent ids does not create them. — `Rtest`.

**B-06** Sampling reproduces the same cbs values for sampled customers,
**including for duplicated ids**. — `Rtest`.

**B-07** Sampling with replacement creates duplicate entries under *new* ids. — `Rtest`.

**B-08** Sampling with and without replacement selects the static covariates of
the same ids, and static covariates are sorted the same as the cbs. — `Rtest` —
two claims.

**B-09** Sampling selects the correct dynamic covariates. — `Rtest`.

**B-10** The bootstrap re-fit passes the correct specification arguments through
— separately asserted for no-cov models, static-cov models, dyncov models, and
spending models. — `Rtest` — four claims. **This is what catches a bootstrap
that silently drops `use.cor`, constraints, or lambdas.**

**B-11** Sampling all customers gives the same model estimate as the original
fit — no-cov, static cov, dyncov, and spending. — `Rtest`.

**B-12** Given a number of periods, bootstrapped predictions run to the same
prediction end even when the sampled transactions would imply a different
estimation end. — `Rtest`.

**B-13** Bootstrapped predictions have the correct format and satisfy
`lower CI < upper CI`. — `Rtest` — two claims.

**B-14** `predict(uncertainty = "boots")` works for no-cov, static-cov and
dyncov models, and in combination with `predict.spending`, `newdata`, and
`prediction.end`. — `Rtest:test_runability_bootstrapping.R`.

**B-15** `num.boots` must be a single positive integer; `level` a single numeric
in `[0, 1]`; `uncertainty` one of the allowed values; `fn.boot.apply` and
`fn.sample` must be functions. — `Rtest:test_inputchecks_bootstrapping.R` —
six claims.

# S13 — Inference generics

**I-01** `coef()` returns a named numeric vector, the same length as the
optimisation, whose names are in the same order as `vcov()` and as
`coef(summary())`, with no `NA`. — `Rtest:helper_s3_fitted.R` — five claims.
**The ordering agreement between the three is asserted explicitly.**

**I-02** `vcov()` returns a named numeric matrix whose names are in the same
order as `coef()` and `coef(summary())`, with no `NA`. — `Rtest` — four claims.

**I-03** `confint()` works with different `alpha`, with a character `parm`, and
with an integer `parm`, and returns `NA` for an unknown `parm`. — `Rtest` —
four claims.

**I-04** `summary()` has the documented structure and coefficient-table
structure, and prints. — `Rtest`, `Rdoc:summary.clv.fitted.Rd`.

**I-05** `hessian()` equals the optimiser's own hessian for every configuration:
no-cov (Pareto/NBD, BG/NBD, GGomNBD, Gamma-Gamma) and with correlation;
static-cov for the three transaction families × {default, constrained,
regularized, constrained + regularized}; and dyncov × {default, correlation,
constrained, regularized}. — `Rtest:test_correctness_hessian.R` — **29
configurations.**

**I-06** `hessian()` errors with "Cannot proceed" when a parameter is
non-finite. — `Rtest`.

**I-07** The internal LL accessor requires a *named* parameter vector, rejects
extra coefficients, and gives results independent of the order in which the
named parameters arrive. — `Rtest` — three claims. **Order-independence is the
one that catches a positional-indexing bug under constraints.**

**I-08** `nobs()` works on `clv.data` and on `clv.fitted`. —
`Rdoc:nobs.clv.data.Rd, nobs.clv.fitted.Rd`.

**I-09** `fitted()` works on `clv.fitted`. — `Rdoc:fitted.clv.fitted.Rd`.

**I-10** `lrtest()` runs for all models, when called as `lmtest::lrtest()`, and
when `lmtest` is attached. — `Rtest:test_runability_lrtest.R`,
`Rdoc:lrtest.Rd`.

**I-11** `logLik()` returns the log-likelihood with the right degrees of
freedom for `lrtest` to work. — implied by I-10.

# S14 — The formula interface

**FI-01** `latentAttrition(formula = , family = pnbd, data = <nocov>)` gives an
object equal to `pnbd(<nocov>)` (call and timing aside). —
`Rtest:test_correctness_latentattrition.R`.

**FI-02** `latentAttrition(~.|., family = pnbd, data = <staticcov>)` equals
`pnbd(<staticcov>)`. — `Rtest`.

**FI-03** `~Gender|Gender+Channel` selects `Gender` for life and
`(Gender, Channel)` for transactions. — `Rtest` — **the two sides of `|` are
life-then-transaction, in that order.**

**FI-04** `.` expands to all covariates on either side: `~.|.`, `~Gender|.`,
and `~.|.+I(Gender+1)` each select the documented sets. — `Rtest` — three
claims.

**FI-05** Naming the same covariate twice selects it once. — `Rtest`.

**FI-06** Transformations are applied and named after R's `make.names`:
`~I(Gender+1)|log(Gender+2)` yields columns `I.Gender...1.` and
`log.Gender...2.`, with values equal to `Gender+1` and `log(Gender+2)`. —
`Rtest` — **both the naming and the arithmetic are asserted.**

**FI-07** Interactions and exclusions work: `~Gender*Channel|.-Gender` gives
life `(Gender, Channel, Gender.Channel)` and transactions `(Channel)`. —
`Rtest` — asserted for static and again for dynamic covariates.

**FI-08** A formula applied to dynamic covariate data returns a dynamic-covariate
object, and one applied to static data returns a static object that is *not* a
dynamic one. — `Rtest`.

**FI-09** The formula interface copies the data — the resulting object shares no
storage with the input. — `Rtest` — three claims (`.`, single covs, transformed).

**FI-10** `spending(family = gg, data = ...)` equals `gg(...)`. —
`Rtest:test_correctness_spending.R`.

**FI-11** `remove.first.transaction` changes the result: every coefficient
differs, every cbs `x` differs, but the id set is unchanged and equals the
transaction data's ids. — `Rtest:helper_testthat_correctness_spending.R`.

**FI-12** A spending model's cbs `x` equals the Pareto/NBD cbs `x` when
`remove.first.transaction = TRUE` — asserted with and without a holdout. —
`Rtest` — **a cross-module invariant.**

**FI-13** `latentAttrition` fails on: non-`clv.data` data; a formula given when
none is needed; missing family; a family that is not an allowed method;
disallowed extra arguments for the family (with and without covariate data); no
formula; no second RHS; more than 2 RHS; a LHS present; RHS terms not in the
covariate data. — `Rtest:test_inputchecks_latentattrition.R` — 11 claims.

**FI-14** `spending` fails on: non-`clv.data` data; missing family; a family
that is not allowed; disallowed extra arguments for `gg`. — `Rtest` — four claims.

**FI-15** `latentAttrition` works with `optimx.args`, `verbose`, `start.params`,
correlation, regularization, single covariate selection, transformations named
inside `constraint()`, and `.` with exclusions. —
`Rtest:test_runability_latentattrition.R` — 12 claims. **`constraint()` naming a
transformed covariate is a distinct claim.**

# S15 — Input validation, generally

The R suite devotes 20 files and roughly 250 claims to rejecting bad input. The
recurring shape is worth stating once as a spec item in its own right, because
a port that validates nothing can still pass every numerical test above.

**V-01** For every start-parameter argument: fails if not a vector, not numeric,
unnamed, wrongly named, missing a single name, containing duplicates, containing
an unneeded parameter, containing `NA`, or `<= 0`. —
`Rtest:helper_testthat_inputchecks_nocov.R` — nine claims, applied to every family.

**V-02** For every covariate start-parameter argument: fails if not numeric, any
`NA`, any name not a covariate, a covariate missing, unnamed, one entry unnamed,
or duplicated. — `Rtest:helper_testthat_inputchecks_staticcov.R` — seven claims.

**V-03** `optimx.args` fails if not a list, `NULL`/`NA`, has unnamed elements, or
has top-level names that are not optimx arguments. — `Rtest` — four claims.

**V-04** Every user-facing function errors on unused arguments in `...` rather
than ignoring them. — `Rtest` — asserted in at least six files.

**V-05** For every single-logical argument: fails for `NULL`, `NA`, a
disallowed type, or a vector of length > 1. —
`Rtest:helper_inputchecks_single_logical.R` — four claims, applied throughout.

**V-06** `clvdata` validates: data present, non-`NA`, non-`NULL`; is a
data.frame; has rows and columns; no `NA` in any used column; `Price` numeric;
`Id`/`Date`/`Price` columns present; `date.format` a single valid character;
`time.unit` a single valid character; `estimation.split` numeric, character or
date, non-`NA`, in the right date format, not after the last transaction, not
within the last period, not before every customer's first transaction. —
`Rtest:test_inputchecks_clvdata_clvdata.R` — 42 claims.

**V-07** Arguments carry the documented defaults, and the ones that do not have
defaults raise when omitted (`data.transactions`, `date.format`, `time.unit`).
— `Rtest` — "Has no default argument" / "Has default argument" claims.

**V-08** `plot` validates `which`, `label` length/emptiness/duplicates,
`trans.bins`, `label.remaining`, `other.models` membership, `sample`
availability (cannot select holdout when there is none), `ids`, and
`annotate.ids`. — `Rtest:test_inputchecks_clvdata_plot.R,
test_inputchecks_plot_transactions.R, test_inputchecks_plot_spending.R` —
19 claims.

---

# Out of scope, deliberately

Items the sources contain that a Python port may reasonably decline. Each needs
a recorded decision, not silence — otherwise the audit cannot tell a gap from a
choice.

- **`ggplot2` rendering.** Every `plot()` claim about geoms, colours, labels and
  axis limits. The *data* behind the plot (`plot = FALSE`) is in scope; the
  rendering is not.
- **R S4/S3 mechanics.** `show`, `print` methods, `address()` identity checks as
  such — though the underlying "data is copied" semantics (D-13, C-06, S-07,
  FI-09) **are** in scope and need a Python-idiomatic equivalent.
- **`optimx`-specific behaviour.** "Works for all optimx optimization methods",
  `kkt1`/`kkt2` as optimx reports them — unless this port chose to reproduce
  the KKT flags, in which case F-09 is in scope.
- **`lmtest` integration.** I-10's second and third claims (dispatch from
  `lmtest::lrtest`, behaviour when the package is attached).
- **Timezone handling** beyond what pandas gives for free — T-04's
  Australia/Darwin case is a real claim, but its value depends on whether this
  port accepts tz-aware input at all.
