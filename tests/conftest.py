"""Shared fixtures.

``data/`` holds the datasets bundled with the R package CLVTools, exported to
CSV by ``tools/extract_data.R``. ``tests/fixtures/`` holds expectations
generated from that package by ``tools/oracle/generate_fixtures.R``; they are
committed, so the suite runs without R.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture_csv(name: str, **kwargs) -> pd.DataFrame:
    """An oracle fixture table. ``Id`` is always read as a string.

    CLVTools keys its customer tables by a character ``Id``; reading it as an
    integer here would silently reorder rows relative to the oracle.
    """
    return pd.read_csv(FIXTURES / f"{name}.csv", dtype={"Id": str}, **kwargs)


def fixture_json(name: str):
    """An oracle fixture of scalars or parameter vectors."""
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture(scope="session")
def apparel_trans() -> pd.DataFrame:
    """The apparel retailer transaction log used throughout the paper."""
    return pd.read_csv(
        DATA / "apparelTrans.csv", dtype={"Id": str}, parse_dates=["Date"]
    )


@pytest.fixture(scope="session")
def apparel_static_cov() -> pd.DataFrame:
    """Gender and acquisition Channel, one row per customer."""
    return pd.read_csv(DATA / "apparelStaticCov.csv", dtype={"Id": str})


@pytest.fixture(scope="session")
def cbs_estimation() -> pd.DataFrame:
    """Oracle ``(x, t_x, T)`` over the 104-week estimation period."""
    return fixture_csv("cbs_estimation")


@pytest.fixture(scope="session")
def cbs_full() -> pd.DataFrame:
    """Oracle ``(x, t_x, T)`` over the whole data, no holdout split."""
    return fixture_csv("cbs_full")
