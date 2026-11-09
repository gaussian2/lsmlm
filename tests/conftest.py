from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lsmlm.datasets import load_oecd15

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_JSON = ROOT / "tests" / "data" / "stata_reference.json"
STATA_LOG = ROOT / "reference" / "stata" / "outputs" / "lsmlm_oecd15_analysis.log"

# Observed agreement with Stata is about 1e-13; leave some headroom for
# different BLAS/LAPACK builds.
STATA_TOL = 1e-9


def load_reference() -> dict[str, Any]:
    return json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))


def stata_log_tables() -> list[str]:
    """Result tables printed by the interactive ``lsmlm y ...`` calls in the log."""
    lines = STATA_LOG.read_text(encoding="utf-8").splitlines()
    tables = []
    for i, line in enumerate(lines):
        if not re.match(r"^\.\s+lsmlm y", line):
            continue
        block = []
        for nxt in lines[i + 1 :]:
            if nxt.startswith(". "):
                break
            block.append(nxt)
        tables.append("\n".join(block).strip("\n"))
    return tables


@pytest.fixture(scope="session")
def oecd():
    return load_oecd15()


@pytest.fixture(scope="session")
def series(oecd):
    """Return ``(y, year)`` arrays for a country and year range."""

    def _series(country: str, start: int = 1870, end: int = 1994):
        g = oecd[(oecd["country"] == country) & oecd["year"].between(start, end)]
        return g["y"].to_numpy(), g["year"].to_numpy()

    return _series


@pytest.fixture(scope="session")
def france(series):
    return series("France")


@pytest.fixture
def random_walk():
    rng = np.random.default_rng(20260720)
    return np.cumsum(rng.normal(size=120))
