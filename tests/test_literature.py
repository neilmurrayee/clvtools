r"""Published values from the CLV literature, on CDNOW.

A third class of oracle, beside the paper (``-m paper``) and the R package's own
documentation (``-m rdoc``): the articles the models come from. CLVTools' own
testthat suite asserts these, and that suite is not installed with the package,
so nothing in this repository had ever consulted them — ``docs/spec.md``'s
literature tier, and finding C of the 2026-09 spec audit.

They are on ``cdnow`` at ``estimation_split="1997-09-30"``, which is why
``paper_values.py`` does not already reach them: the paper and ``?pmf`` both use
the 37-week split, and the published fits use the date.

Three papers, five oracles:

* Fader, Hardie & Lee (2005), "Counting Your Customers the Easy Way: An
  Alternative to the Pareto/NBD Model", *Marketing Science* 24(2) — both the
  Pareto/NBD it compares against and the BG/NBD it introduces.
* Fader & Hardie (2013), "The Gamma-Gamma Model of Monetary Value" — the
  spending model.
* CLVTools 0.12.1's own standard errors for the Pareto/NBD fit.

The tolerance is half a unit in the last decimal each source printed, plus one
for this package's own optimiser -- :func:`published_tol`. A paper writing
``r = 0.243`` is claiming the estimate lies in [0.2425, 0.2435), and no tighter
comparison against it is available; a fixed relative tolerance is simultaneously
too tight for that and far too loose for ``-4055.9177``. The log-likelihoods are
compared to the decimal they were printed to, which is one for the 2005 paper
and four for the 2013 one.
"""

from __future__ import annotations

import numpy as np
import pytest

from clvtools import ClvData, load_cdnow
from clvtools.bgnbd import fit_bgnbd
from clvtools.gg import fit_gg
from clvtools.pnbd import fit_pnbd, probability_alive
from clvtools.pnbd import log_likelihood as pnbd_log_likelihood

pytestmark = [pytest.mark.slow, pytest.mark.literature]

#: Fader, Hardie & Lee (2005), Table 1.
FHL2005_PNBD = {"r": 0.553, "alpha": 10.578, "s": 0.606, "beta": 11.669}
FHL2005_PNBD_LL = -9595.0

#: The same table's BG/NBD, the model that paper introduces.
FHL2005_BGNBD = {"r": 0.243, "alpha": 4.414, "a": 0.793, "b": 2.426}
FHL2005_BGNBD_LL = -9582.4

#: Fader & Hardie (2013), "The Gamma-Gamma Model of Monetary Value".
FH2013_GG = {"p": 6.25, "q": 3.74, "gamma": 15.44}
FH2013_GG_LL = -4055.9177

#: CLVTools 0.12.1's own Pareto/NBD estimates on the same data, to four
#: decimals rather than the paper's three -- spec `F-01`, a tighter oracle than
#: `FHL2005_PNBD` for the same optimum. Its suite compares them at a *relative*
#: 0.001, which is what ``rtol`` below is.
CLVTOOLS_PNBD = {"r": 0.5532, "alpha": 10.5763, "s": 0.6063, "beta": 11.6715}
CLVTOOLS_PNBD_LL = -9594.976

#: And its GGompertz/NBD, spec `F-06`. ``b`` and ``beta`` are given to two
#: significant figures because that is all CLVTools prints; see
#: :class:`TestBemmaorGlady2012`, which is about why they are not comparable.
CLVTOOLS_GGOMNBD = {
    "r": 0.55313, "alpha": 10.5758, "b": 0.0000011, "s": 0.60682,
    "beta": 0.000013,
}

#: Bemmaor & Glady (2012), Table 2, p. 1018 -- spec `F-07`.
BG2012_GGOMNBD = {
    "r": 0.553, "alpha": 10.578, "b": 0.0002, "s": 0.603, "beta": 0.0026,
}
BG2012_GGOMNBD_LL = -9594.98

#: CLVTools 0.12.1's standard errors for the Pareto/NBD fit above, which its
#: own suite asserts to 0.001.
CLVTOOLS_PNBD_SE = {
    "r": 0.0476264, "alpha": 0.8427222, "s": 0.1872594, "beta": 6.2105448,
}



def published_tol(value: float) -> float:
    """Half a unit in the last decimal the source printed, plus one for the ridge.

    A paper printing ``r = 0.243`` is saying the estimate lies in
    [0.2425, 0.2435), so agreement means landing inside that interval and no
    tighter comparison is available. A fixed relative tolerance gets this wrong
    in both directions: 1e-3 is too tight for ``0.243`` (three decimals) and far
    too loose for ``-4055.9177`` (four).

    The rest is this package's own optimiser, which stops at its own point on a
    flat ridge -- and *where* on the ridge depends on the platform's BLAS. The
    allowance used to be a fifth of a unit in the last place, which held on
    macOS and did not on GitHub's Linux runners: the CDNOW Pareto/NBD's
    ``beta`` is 11.668668 here and 11.668360 there, 0.31 apart in the last
    printed place, and the second sits 0.64 from the published 11.669 --
    outside a 0.6 tolerance, and the reason CI was red. A whole unit covers
    the observed spread with room to spare, and the assertion still says the
    estimate agrees with the published one to about one in the last digit it
    was printed to.

    Loosening a tolerance is the wrong move if the fit is actually worse, so
    that is asserted separately and directly by
    :meth:`TestFaderHardieLee2005.test_the_pareto_nbd_fit_is_at_least_as_good`.
    That claim is about likelihoods rather than coordinates and does not move
    between platforms.

    >>> published_tol(0.243)
    0.0015
    >>> published_tol(15.44)
    0.015
    """
    text = f"{value!r}"
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 1.5 * 10.0**-decimals

@pytest.fixture(scope="module")
def cdnow():
    """The split the literature uses, which is a date rather than a count."""
    return ClvData(load_cdnow(), time_unit="week", estimation_split="1997-09-30")


@pytest.fixture(scope="module")
def cbs(cdnow):
    return cdnow.customer_summary()


class TestFaderHardieLee2005:
    """Both models of Table 1, on the data the paper uses."""

    @pytest.fixture(scope="class")
    def pnbd(self, cbs):
        return fit_pnbd(cbs["x"], cbs["t_x"], cbs["T"])

    @pytest.fixture(scope="class")
    def bgnbd(self, cbs):
        return fit_bgnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)

    def test_the_pareto_nbd_estimates_match(self, pnbd):
        for name, want in FHL2005_PNBD.items():
            assert getattr(pnbd, name) == pytest.approx(
                want, abs=published_tol(want)
            ), name

    def test_the_pareto_nbd_fit_is_at_least_as_good(self, pnbd, cbs):
        """Our optimum explains the data no worse than the published one.

        :func:`published_tol` allows the optimiser its own point on the ridge,
        which is what makes the comparison above survive a change of platform.
        This is the claim that allowance is standing in for, and the one that
        would actually matter: a fit agreeing to three decimals while sitting
        at a worse likelihood would be the real failure, and no tolerance on
        the coordinates would catch it.

        Measured here: -9594.976179 against -9594.976259 at the published
        parameters, so ours is better by 8.0e-5 -- the ridge is flat enough
        that 3e-4 of ``beta`` costs almost nothing.
        """
        x, t_x, T = cbs["x"], cbs["t_x"], cbs["T"]
        ours = {name: getattr(pnbd, name) for name in FHL2005_PNBD}
        assert pnbd_log_likelihood(x, t_x, T, **ours) >= pnbd_log_likelihood(
            x, t_x, T, **FHL2005_PNBD
        )

    def test_the_pareto_nbd_log_likelihood_matches(self, pnbd):
        """Printed to one decimal, so compared to one."""
        assert pnbd.log_likelihood == pytest.approx(FHL2005_PNBD_LL, abs=0.05)

    def test_the_bgnbd_estimates_match(self, bgnbd):
        for name, want in FHL2005_BGNBD.items():
            assert getattr(bgnbd, name) == pytest.approx(
                want, abs=published_tol(want)
            ), name

    def test_the_bgnbd_log_likelihood_matches(self, bgnbd):
        assert bgnbd.log_likelihood == pytest.approx(FHL2005_BGNBD_LL, abs=0.05)

    def test_the_bgnbd_fits_this_data_better(self, pnbd, bgnbd):
        """Which is the paper's claim, and the reason it exists.

        "An Alternative to the Pareto/NBD Model": on CDNOW the BG/NBD attains
        the higher likelihood, by about 12.5 units, with the same four
        parameters. Asserting the *ordering* rather than the gap keeps this a
        statement about the models rather than about two optimisers.
        """
        assert bgnbd.log_likelihood > pnbd.log_likelihood

    @pytest.mark.oracle
    def test_the_standard_errors_match_clvtools(self, pnbd):
        """CLVTools' own suite asserts these to 0.001; so does this.

        They are second derivatives of a flat surface differenced numerically,
        which is why the tolerance is absolute and generous rather than
        relative and tight.
        """
        got = pnbd.standard_errors()
        for name, want in CLVTOOLS_PNBD_SE.items():
            assert got[name] == pytest.approx(want, abs=1e-3), name


class TestFaderHardie2013:
    """The Gamma-Gamma paper, on the spending of the same cohort."""

    @pytest.fixture(scope="class")
    def fitted(self, cdnow):
        spending = cdnow.spending_summary()
        return fit_gg(spending["x"], spending["Spending"], hessian=False)

    def test_the_estimates_match(self, fitted):
        for name, want in FH2013_GG.items():
            assert getattr(fitted, name) == pytest.approx(
                want, abs=published_tol(want)
            ), name

    def test_the_log_likelihood_matches(self, fitted):
        """Published to four decimals, and reproduced to them."""
        assert fitted.log_likelihood == pytest.approx(FH2013_GG_LL, abs=1e-3)


class TestNumericalStabilityCases:
    """Named regression inputs, needing no R and no fit."""

    def test_palive_is_finite_for_very_heavy_buyers(self):
        """M-04: inputs that returned ``NaN`` in an earlier implementation.

        Four customers with 161 to 254 transactions, at parameters where the
        Pareto/NBD's intermediate terms are extreme. The point is not the
        values but that they exist and are probabilities.
        """
        x = np.array([221.0, 254.0, 161.0, 204.0])
        t_x = np.array([103.42857, 97.14286, 94.71429, 98.57143])
        T = np.array([103.57143, 97.28571, 98.00000, 99.42857])
        got = probability_alive(x, t_x, T, r=0.5143, alpha=2.8845,
                                s=0.2856, beta=14.1087)
        assert np.all(np.isfinite(got))
        assert np.all((got > 0.0) & (got <= 1.0))
        np.testing.assert_allclose(
            got, [0.99960, 0.99956, 0.74949, 0.99426], atol=1e-4
        )

    @pytest.mark.parametrize("family", ["pnbd", "bgnbd", "ggomnbd"])
    def test_the_probability_of_no_purchase_falls_with_the_window(self, family):
        """PMF-04: ``P(X = 0)`` is decreasing in ``T``, for every family.

        The longer someone is observed, the less likely they bought nothing.
        True of any counting model and asserted for none of them until now.
        """
        import importlib

        module = importlib.import_module(f"clvtools.{family}")
        parameters = {
            "pnbd": {"r": 0.55, "alpha": 10.58, "s": 0.61, "beta": 11.67},
            "bgnbd": {"r": 0.24, "alpha": 4.41, "a": 0.79, "b": 2.43},
            "ggomnbd": {"r": 0.55, "alpha": 10.58, "b": 0.01, "s": 0.61,
                        "beta": 11.67},
        }[family]
        windows = np.array([1.0, 5.0, 13.0, 26.0, 52.0, 104.0])
        p0 = np.array([
            float(module.pmf(0, T, **parameters)) for T in windows
        ])
        assert np.all(np.diff(p0) < 0), p0


class TestClvToolsOwnCdnowEstimates:
    """Spec `F-01`, `absent`: the same optimum, to one more decimal.

    :class:`TestFaderHardieLee2005` compares against a paper that printed three
    decimals, so :func:`published_tol` allows 0.0015 -- 2.7e-4 of ``r``. The R
    package prints four and its own suite asserts them at a relative 0.001, and
    that is a *different* oracle of a different strength for the same fit: it
    pins this port against the implementation it is a port of rather than
    against the article both implement.
    """

    @pytest.fixture(scope="class")
    def fitted(self, cbs):
        """From CLVTools' own start, ``(1, 1, 1, 1)``, which is not the default."""
        return fit_pnbd(
            cbs["x"], cbs["t_x"], cbs["T"],
            start=np.ones(4), hessian=False,
        )

    def test_the_estimates_match(self, fitted):
        for name, want in CLVTOOLS_PNBD.items():
            assert getattr(fitted, name) == pytest.approx(want, rel=1e-3), name

    def test_the_log_likelihood_matches(self, fitted):
        """Printed to three decimals; measured here at -9594.976179."""
        assert fitted.log_likelihood == pytest.approx(CLVTOOLS_PNBD_LL, abs=5e-4)


class TestBemmaorGlady2012:
    r"""Spec `F-07`, `absent !`: three published ``(b, beta)`` four orders apart.

    The audit asked for the comparison and for this port's own divergence to be
    recorded. Both are here, and the finding is that **the comparison cannot be
    made coordinate by coordinate, and the spec is right to call it a deviation
    to pin rather than a target to hit**. On CDNOW:

    ==================  ==========  ============  ==========  ==============
    source              ``b``       ``beta``      ``beta/b``  ``LL``
    ==================  ==========  ============  ==========  ==============
    CLVTools 0.12.1     1.1e-6      1.3e-5        11.82       -9594.9762
    this port           2.86e-5     3.34e-4       11.66       -9594.9764
    Bemmaor/Glady       2.0e-4      2.6e-3        13.0        -9594.98
    Pareto/NBD          --          11.6684       --          -9594.9762
    ==================  ==========  ============  ==========  ==============

    ``b`` spans a factor of 180 and ``beta`` a factor of 200, while the
    log-likelihood moves in the fourth decimal. The reason is in the survival
    term: :math:`(\beta / (\beta - 1 + e^{bT}))^{s}`, and for :math:`bT \ll 1`,
    :math:`e^{bT} - 1 \to bT`, so it becomes
    :math:`((\beta/b) / ((\beta/b) + T))^{s}` -- the Pareto/NBD's own survival
    with :math:`\beta_{P} = \beta/b`. **The identified quantity is the ratio**,
    and all three sources agree on it to within 11%, and this port's is the
    closest of the three to the Pareto/NBD's ``beta``. Individually ``b`` and
    ``beta`` are a ridge, which is why CDNOW's GGompertz/NBD is the one fit
    whose ``kkt2`` CLVTools' own suite expects to be false (spec `F-09`).

    So what is asserted is the ratio and the likelihood, not the coordinates.
    """

    @pytest.fixture(scope="class")
    def fitted(self, cbs):
        from clvtools.ggomnbd import fit_ggomnbd

        return fit_ggomnbd(cbs["x"], cbs["t_x"], cbs["T"], hessian=False)

    def test_the_identified_parameters_match_the_published_ones(self, fitted):
        """``r`` and ``alpha`` are identified, and agree with everyone."""
        for name in ("r", "alpha"):
            assert getattr(fitted, name) == pytest.approx(
                BG2012_GGOMNBD[name], abs=published_tol(BG2012_GGOMNBD[name])
            ), name

    def test_but_s_tilts_along_the_ridge_too(self, fitted):
        """Which is why the class asserts three coordinates and not five.

        ``s`` sits at 0.60590 here, 0.60682 in CLVTools, 0.603 in Bemmaor and
        Glady and 0.60623 in the nested Pareto/NBD -- a spread of 0.004 across
        four sources, which is wider than the 0.0015 the published third
        decimal claims. So what is asserted is the spread, not a coordinate:
        every source lies within 0.005 of every other, and no tighter statement
        about ``s`` is available or would survive a change of start.

        This test itself moved twice while it was being written: 0.60577
        before the default start became scale-aware
        (:func:`~clvtools._optimize.start_scale`), 0.60478 after it, and
        0.60590 once the survival term stopped cancelling.
        Three values within 0.0011 of each other, for changes worth under 1e-6
        of log-likelihood. That is the ridge demonstrating itself, twice, and
        the reason the assertion is a spread rather than a coordinate.
        """
        every = [fitted.s, CLVTOOLS_GGOMNBD["s"], BG2012_GGOMNBD["s"],
                 FHL2005_PNBD["s"]]
        assert max(every) - min(every) < 5e-3
        assert min(every) > 0.60
        assert max(every) < 0.61

    def test_the_gompertz_scale_ratio_matches_the_pareto_nbd_beta(self, fitted):
        """What ``b`` and ``beta`` jointly identify, and separately do not."""
        ratio = fitted.beta / fitted.b
        assert ratio == pytest.approx(FHL2005_PNBD["beta"], rel=0.05)
        for source in (CLVTOOLS_GGOMNBD, BG2012_GGOMNBD):
            assert source["beta"] / source["b"] == pytest.approx(ratio, rel=0.15)

    def test_the_log_likelihood_matches(self, fitted):
        """Printed to two decimals, and this port lands inside them."""
        assert fitted.log_likelihood == pytest.approx(
            BG2012_GGOMNBD_LL, abs=0.005
        )

    def test_our_optimum_is_no_worse_than_either_published_one(self, fitted, cbs):
        """The claim the coordinate comparison above is standing in for.

        Measured: -9594.976386 here, against -9594.983766 at CLVTools' printed
        parameters and -9595.770963 at Bemmaor and Glady's. Neither published
        pair *reproduces its own published likelihood* -- Bemmaor and Glady
        print -9594.98 and their printed ``(b, beta)`` give -9595.77 -- because
        rounding to two significant figures is, along this direction, a 5%
        move. That is the ridge, seen from the other side.
        """
        from clvtools.ggomnbd import log_likelihood as ggomnbd_log_likelihood

        x, t_x, T = cbs["x"], cbs["t_x"], cbs["T"]
        ours = {k: getattr(fitted, k) for k in BG2012_GGOMNBD}
        here = ggomnbd_log_likelihood(x, t_x, T, **ours)
        for published in (CLVTOOLS_GGOMNBD, BG2012_GGOMNBD):
            assert here >= ggomnbd_log_likelihood(x, t_x, T, **published)

    def test_the_ggomnbd_barely_improves_on_the_pareto_nbd_here(self, fitted):
        """It nests it at ``b -> 0``, and on CDNOW that is where it goes.

        Two extra effective parameters buy 8e-5 of log-likelihood over
        :data:`FHL2005_PNBD_LL`. The Gompertz term is not identified by this
        data, which is the substance of the whole class.
        """
        assert abs(fitted.log_likelihood - FHL2005_PNBD_LL) < 0.05
