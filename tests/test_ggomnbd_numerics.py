r"""The GGom/NBD's ``CET`` in the regime where its arithmetic changes shape.

:func:`~clvtools.ggomnbd.conditional_expected_transactions` forms

.. math::
    \frac{(r+x)\,U}{(\alpha+T)\; b s \,(1 + b s P)}

and computes the denominator two ways. Where :math:`P` is representable it
divides directly; where :math:`\exp(\log P)` overflows, the ``1`` is negligible
beside :math:`b s P` and the whole expression is done in logs instead, which is
the difference between an answer and a ``NaN``.

That second branch had never been evaluated. It is reached only for a customer
with thousands of transactions, which no fixture contains and no fit produces,
so **twenty separate mutations of its one line survived the whole suite** --
including flipping the sign of its ``2 log(bs)`` term, which scales the result
by :math:`e^{58}`.

What pins it here is that the two branches compute the same quantity, so the
function has to be smooth where control passes from one to the other. Over an
evenly spaced grid of ``x`` that straddles the crossover, the second difference
of :math:`\log \mathrm{CET}` is about 0.003; the sign flip above would put
:math:`2|\log bs| \approx 29` into one of them. That is four orders of
magnitude of margin, and it needs no reference implementation -- only the fact
that a function cannot have a kink where its formula is merely rewritten.

Found by mutation testing.
"""

from __future__ import annotations

import numpy as np
import pytest

from clvtools.ggomnbd import conditional_expected_transactions

#: The apparel fit's parameters. ``b`` sits at the boundary the family always
#: finds on this data, which is what makes ``log(bs)`` large and negative and
#: so makes the mutated term so far out.
PARAMS = {"r": 1.449, "alpha": 10.0, "b": 8.1e-7, "s": 0.5613, "beta": 1e-4}

#: Evenly spaced, and chosen so the crossover falls inside: ``log P`` runs from
#: 632 at ``x = 4800`` to 712 at ``x = 5400`` against an overflow threshold of
#: ``log(DBL_MAX) = 709.78``. The last point is the first one in logs. Values
#: are around 1e-300 here -- small, but still normal doubles, which is why the
#: grid stops before ``x = 5600`` where they turn denormal and lose precision.
COUNTS = (4800.0, 5000.0, 5200.0, 5400.0)


@pytest.fixture(scope="module")
def log_cet():
    values = np.array([
        float(np.ravel(conditional_expected_transactions(
            x=x, t_x=90.0, T=104.0, t=52.0, **PARAMS
        ))[0])
        for x in COUNTS
    ])
    assert np.all(values > 0), f"the grid must stay representable, got {values}"
    return np.log(values)


class TestTheOverflowBranchJoinsTheOrdinaryOne:
    def test_the_curve_has_no_kink_where_the_branch_changes(self, log_cet):
        """Second differences near zero across the crossover."""
        second = np.diff(np.diff(log_cet))
        assert np.all(np.abs(second) < 0.05), (
            f"log CET is not smooth across the overflow crossover: second "
            f"differences {second}. A branch that disagrees with the one it "
            "replaces shows up here and nowhere else in the suite."
        )

    def test_and_it_is_still_decreasing(self, log_cet):
        """Direction as well as smoothness, so a sign slip cannot hide."""
        assert np.all(np.diff(log_cet) < 0)
