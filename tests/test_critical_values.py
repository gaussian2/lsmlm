from __future__ import annotations

import numpy as np
import pytest

from lsmlm import CriticalValues, critical_values
from lsmlm.critical_values import (
    _CV_BREAK_ONE,
    _CV_BREAK_TWO,
    _CV_CRASH_ONE,
    _CV_CRASH_TWO,
    _CV_NO_BREAK,
    _two_break_cell,
)


def test_table_shapes_and_signs():
    assert _CV_NO_BREAK.shape == (4, 3)
    assert _CV_CRASH_ONE.shape == (4, 3)
    assert _CV_CRASH_TWO.shape == (4, 3)
    assert _CV_BREAK_ONE.shape == (36, 3)
    assert _CV_BREAK_TWO.shape == (48, 3)
    for table in (_CV_NO_BREAK, _CV_CRASH_ONE, _CV_CRASH_TWO, _CV_BREAK_ONE, _CV_BREAK_TWO):
        assert (table < 0).all()
        # 1% is more negative than 5%, which is more negative than 10%
        assert (np.diff(table, axis=1) > 0).all()


def test_tables_are_read_only():
    with pytest.raises(ValueError, match="read-only"):
        _CV_NO_BREAK[0, 0] = 0.0


def test_returns_named_tuple():
    cv = critical_values(0, "break", 50)
    assert isinstance(cv, CriticalValues)
    assert cv == (cv.cv1, cv.cv5, cv.cv10) == (-3.597, -3.031, -2.745)


@pytest.mark.parametrize("nobs", [10, 46, 99])
def test_below_smallest_sample_size_uses_first_row(nobs):
    assert critical_values(0, "break", nobs) == (-3.597, -3.031, -2.745)
    assert critical_values(1, "crash", nobs) == (-4.084, -3.487, -3.185)
    assert critical_values(2, "crash", nobs) == (-4.073, -3.563, -3.296)


@pytest.mark.parametrize("nobs", [1000, 5000])
def test_at_or_above_largest_sample_size_uses_last_row(nobs):
    assert critical_values(0, "crash", nobs) == (-3.566, -3.023, -2.748)
    assert critical_values(1, "crash", nobs) == (-3.798, -3.230, -2.925)


def test_sample_size_weights_follow_the_stata_code():
    """lsmlm_interp_T(): weight (nobs - T_lo)/(T_hi - T_lo) goes on the *lower* row.

    Pinned because it is the convention of the Stata command (and of the
    numbers in its log), not ordinary linear interpolation.
    """
    w_lo = (123 - 100) / (250 - 100)
    want = w_lo * _CV_NO_BREAK[0] + (1 - w_lo) * _CV_NO_BREAK[1]
    assert critical_values(0, "break", 123) == pytest.approx(tuple(want), abs=1e-15)
    # Stata log, Section 3(a): 1% -3.576, 5% -3.024, 10% -2.747
    assert [round(v, 3) for v in critical_values(0, "break", 123)] == [-3.576, -3.024, -2.747]
    # Stata log, Section 3(b): one break, crash
    assert [round(v, 3) for v in critical_values(1, "crash", 123)] == [-4.002, -3.402, -3.093]
    # Consequence of that convention: a sample size equal to a tabulated T
    # returns the row of the *next* tabulated T.
    assert critical_values(0, "break", 100) == tuple(_CV_NO_BREAK[1])
    assert critical_values(0, "break", 500) == tuple(_CV_NO_BREAK[3])


def test_model_is_ignored_without_breaks():
    assert critical_values(0, "crash", 123) == critical_values(0, "break", 123)


def test_crash_values_do_not_depend_on_lambda():
    assert critical_values(1, "crash", 80, 0.2) == critical_values(1, "crash", 80, 0.7)
    assert critical_values(2, "crash", 80, 0.2, 0.4) == critical_values(2, "crash", 80, 0.5, 0.9)


def test_one_break_grid_points_and_clamping():
    assert critical_values(1, "break", 50, 0.10) == tuple(_CV_BREAK_ONE[0])
    assert critical_values(1, "break", 50, 0.03) == tuple(_CV_BREAK_ONE[0])
    assert critical_values(1, "break", 50, 0.50) == tuple(_CV_BREAK_ONE[8])
    assert critical_values(1, "break", 1000, 0.50) == tuple(_CV_BREAK_ONE[35])


def test_one_break_interpolates_over_lambda():
    got = critical_values(1, "break", 50, 0.125)
    want = 0.5 * _CV_BREAK_ONE[0] + 0.5 * _CV_BREAK_ONE[1]
    assert got == pytest.approx(tuple(want), abs=1e-12)


@pytest.mark.parametrize("lam", [0.12, 0.26, 0.4, 0.49])
def test_one_break_is_symmetric_around_one_half(lam):
    assert critical_values(1, "break", 123, lam) == pytest.approx(
        critical_values(1, "break", 123, 1 - lam), abs=1e-12
    )


@pytest.mark.parametrize(
    ("lam1", "lam2", "cell"),
    [
        (0.2, 0.3, 0),
        (0.2, 0.8, 5),
        (0.3, 0.4, 6),
        (0.05, 0.1, 0),  # both snap to 0.2 -> forced apart
        (0.5, 0.5, 10),  # both snap to 0.5; tie resolved downward
        (0.4, 0.6, 11),  # last of the 12 tabulated pairs
        (0.7, 0.8, 0),  # reflected to (0.2, 0.3)
        (0.9, 0.95, 0),  # reflected to (0.05, 0.1)
    ],
)
def test_two_break_cell_selection(lam1, lam2, cell):
    assert _two_break_cell(lam1, lam2) == cell
    assert critical_values(2, "break", 50, lam1, lam2) == tuple(_CV_BREAK_TWO[cell])


def test_two_break_cells_stay_inside_the_table():
    grid = np.linspace(0.0, 1.0, 41)
    cells = {_two_break_cell(a, b) for a in grid for b in grid if a < b}
    assert cells <= set(range(12))


@pytest.mark.parametrize(("breaks", "model"), [(3, "break"), (-1, "crash"), (1, "trend"), (1, "")])
def test_invalid_arguments(breaks, model):
    with pytest.raises(ValueError, match="must be"):
        critical_values(breaks, model, 100)
