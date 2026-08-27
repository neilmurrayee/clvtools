"""A test-driven Python implementation of the CLVTools R package.

Follows Meierer, Bachmann, Naef, Schilter & Algesheimer, "Estimating Individual
Customer Lifetime Values with R: The CLVTools Package" (Journal of Statistical
Software, submission 5634), section by section.

Modules mirror the paper's structure:

``data``
    S6.1 - preparing and inspecting transaction data; the `clvdata()` analogue.
"""

from __future__ import annotations

from clvtools.data import (
    ClvData,
    load_apparel_dyn_cov,
    load_apparel_static_cov,
    load_apparel_trans,
    load_cdnow,
)

__all__ = [
    "ClvData",
    "load_apparel_dyn_cov",
    "load_apparel_static_cov",
    "load_apparel_trans",
    "load_cdnow",
]
