"""The printed table must be identical, character for character, to Stata's."""

from __future__ import annotations

import pytest

from conftest import stata_log_tables
from lsmlm import lsmlm

LOG_TABLES = stata_log_tables()

# The five interactive calls of Section 3 of lsmlm_oecd15_analysis.do, in order.
SECTION3 = [
    ("no-break", 1870, {"breaks": 0, "maxlag": 2}),
    ("crash-1", 1870, {"breaks": 1, "model": "crash", "maxlag": 2, "trim": 0.10}),
    ("break-1", 1870, {"breaks": 1, "model": "break", "maxlag": 2, "trim": 0.10}),
    ("break-2", 1870, {"breaks": 2, "model": "break", "maxlag": 2, "mindist": 5, "trim": 0.10}),
    ("post-war", 1946, {"breaks": 1, "model": "break", "maxlag": 2, "trim": 0.10}),
]


def test_log_has_the_five_walkthrough_tables():
    assert len(LOG_TABLES) == len(SECTION3)


@pytest.mark.parametrize(
    ("index", "start", "options"),
    [(i, start, opts) for i, (_, start, opts) in enumerate(SECTION3)],
    ids=[name for name, _, _ in SECTION3],
)
def test_summary_equals_stata_log(index, start, options, series):
    y, year = series("France", start, 1994)
    result = lsmlm(y, year, print_results=False, **options)
    assert result.summary() == LOG_TABLES[index]
    assert str(result) == result.summary()


def test_print_results_writes_the_table(france, capsys):
    y, year = france
    result = lsmlm(y, year, breaks=0, maxlag=2)
    assert capsys.readouterr().out == "\n" + result.summary() + "\n"

    lsmlm(y, year, breaks=0, maxlag=2, print_results=False)
    assert capsys.readouterr().out == ""


def test_varname_and_decision(france):
    y, year = france
    result = lsmlm(y, year, breaks=0, maxlag=2, varname="log_rel_income", print_results=False)
    assert "  Series:        log_rel_income" in result.summary()
    assert result.decision == "Reject unit root at 1%"

    post = lsmlm(y[year >= 1946], year[year >= 1946], breaks=1, maxlag=2, print_results=False)
    assert post.decision == "Do not reject unit root"
    assert post.siglevel == "Not significant"
