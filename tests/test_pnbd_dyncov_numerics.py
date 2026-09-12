r"""The time-varying Pareto/NBD's numerical fallbacks, where they take over.

``_hyp_alpha_ge_beta`` computes each :math:`F_2` term as a quotient of a
Gauss hypergeometric by :math:`\alpha^{r+s+x}`, and SciPy's ``hyp2f1`` returns
``nan`` rather than a value once :math:`r+s+x` is a couple of hundred with
:math:`z` near 1. Past that point the function switches to the asymptotic form,
which is what keeps a likelihood finite for a customer with hundreds of
transactions.

That branch was reached by a test and asked only whether its answer was
``nan``. Finite is not the same as right: corrupting the four ``gammaln`` calls
in its constant leaves the result perfectly finite and about 700 out in the
log, and that survived the whole suite.

What pins it is the same property that pins ``ggomnbd``'s overflow branch in
``test_ggomnbd_numerics.py`` -- an asymptotic form and the branch it replaces
compute the same quantity, so the function cannot have a kink where control
passes between them.

Found by mutation testing.
"""

from __future__ import annotations

import numpy as np


def test_the_fallback_agrees_with_the_branch_it_replaces():
    """Finite is not the same as right.

    The test above reaches the fallback and asks only that the answer is
    not ``nan``. Its ``log_c`` is four ``gammaln`` calls, and corrupting
    them -- ``gammaln(a+1) * gammaln(s)`` for the sum -- leaves the result
    perfectly finite and about 700 out in the log, which survived the whole
    suite.

    The fallback is the asymptotic form of the branch it stands in for, so
    the two have to join smoothly. On SciPy 1.18 at this ``z`` the direct
    branch survives ``x = 140`` and ``160`` and gives up by ``180``, so the
    grid below crosses the boundary: the log magnitude falls by about 414 per
    step of 20 and its second difference is 0.015. The corrupted constant
    would put ~700 into one of them.

    The crossover is asserted rather than assumed. Where ``hyp2f1`` stops
    converging is a property of SciPy's algorithm, not of IEEE-754 -- unlike
    ``ggomnbd``'s overflow branch, whose threshold is exact -- so a different
    SciPy build may well move it. If it moves outside this grid the smoothness
    check still passes, having compared the fallback only with itself; the
    straddle check below is what stops that quiet failure, and says to widen
    the grid rather than leaving a test that no longer tests anything.

    Found by mutation testing.
    """
    from clvtools.pnbd.dyncov import _hyp2f1, _hyp_alpha_ge_beta

    r, s = 1.5, 0.8
    grid = (140.0, 160.0, 180.0, 200.0, 220.0)

    # The same call, with the same `c = a + 1`, that the branch itself makes.
    z = np.float64(1.0 - 1.0 / 1e9)
    direct = [
        bool(np.isfinite(_hyp2f1(np.float64(r + s + x), s + 1.0, z))) for x in grid
    ]
    vacuous = (
        f"hyp2f1 finite at {direct} across x = {grid}, so the smoothness check "
        "below compares one branch with itself and would pass whatever the "
        "other computed. Widen the grid until both branches appear."
    )
    assert any(direct), f"the direct branch is never reached: {vacuous}"
    assert not all(direct), f"the fallback is never reached: {vacuous}"

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        magnitudes = np.array([
            float(np.ravel(_hyp_alpha_ge_beta(r, s, x, 1e9, 1.0, 1.1e9, 1.0)[0])[0])
            for x in grid
        ])
    assert np.all(np.isfinite(magnitudes))
    second = np.diff(np.diff(magnitudes))
    assert np.all(np.abs(second) < 1.0), (
        f"the fallback does not join the branch it replaces: second "
        f"differences {second}"
    )
