"""The Python port must reproduce the Stata command's stored results.

Reference values come from the ``.dta`` files written by the Stata run of
``lsmlm_oecd15_analysis.do`` (full double precision), converted to JSON by
``reference/stata/build_reference.py``.
"""

from __future__ import annotations

import math

import pytest

from conftest import STATA_TOL, load_reference
from lsmlm import lsmlm

REFERENCE = load_reference()
CASES = REFERENCE["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_matches_stata_stored_results(case, series):
    y, year = series(case["country"], case["start"], case["end"])
    result = lsmlm(y, year, print_results=False, **case["options"])

    assert case["expected"], "reference case has no expected values"
    for name, want in case["expected"].items():
        got = getattr(result, name)
        if isinstance(want, bool):
            assert got is want, name
        elif isinstance(want, int):
            assert got == want, name
        else:
            assert math.isclose(got, want, rel_tol=0.0, abs_tol=STATA_TOL), (name, got, want)


def test_reference_covers_every_table():
    sources = {c["source"] for c in CASES}
    assert sources == {
        "table1_two_break.dta",
        "table2_one_break.dta",
        "table3_subsamples.dta",
    }
    assert len(CASES) == 12


def test_stored_results_printed_in_log(france):
    """r() values that the log displays at 8 significant digits."""
    y, year = france
    r = lsmlm(y, year, breaks=2, model="break", maxlag=2, mindist=5, trim=0.10, print_results=False)
    assert f"{r.tau:.7f}" == "-7.2219081"
    assert f"{r.lambda1:.8f}" == "0.54471545"
    assert f"{r.lambda2:.7f}" == "0.6097561"
    assert f"{r.cv5:.7f}" == "-5.4179467"
    assert r.siglevel == "1%"
