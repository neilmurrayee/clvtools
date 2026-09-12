r"""The time-varying Pareto/NBD's numerical fallbacks, where they take over.

``_hyp_alpha_ge_beta`` computes each :math:`F_2` term as a quotient of a
Gauss hypergeometric by :math:`\alpha^{r+s+x}`, and SciPy's ``hyp2f1`` stops
returning a finite value once :math:`r+s+x` is a couple of hundred with
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
    the two have to join smoothly. SciPy gives up somewhere near ``x =
    198`` at this ``z``, so a grid from 140 to 220 crosses the boundary:
    the log magnitude falls by about 414 per step of 20 and its second
    difference is 0.015. The corrupted constant would put ~700 into one of
    them.

    Found by mutation testing.
    """
    from clvtools.pnbd.dyncov import _hyp_alpha_ge_beta

    r, s = 1.5, 0.8
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        magnitudes = np.array([
            float(np.ravel(_hyp_alpha_ge_beta(r, s, x, 1e9, 1.0, 1.1e9, 1.0)[0])[0])
            for x in (140.0, 160.0, 180.0, 200.0, 220.0)
        ])
    assert np.all(np.isfinite(magnitudes))
    second = np.diff(np.diff(magnitudes))
    assert np.all(np.abs(second) < 1.0), (
        f"the fallback does not join the branch it replaces: second "
        f"differences {second}"
    )
