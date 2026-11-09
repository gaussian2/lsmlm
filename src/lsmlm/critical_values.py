# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Junsoo Lee
# Portions derived from third-party MIT-licensed code; see THIRD_PARTY_NOTICES.md.
"""Critical values for the Lee-Strazicich minimum LM unit root test.

The tables and the interpolation scheme are a faithful port of the Mata block
of the Stata command ``lsmlm.ado`` (v1.1.0, Ibrahim Ongoren), which takes them
from the ``leestra`` package by H. Ozan Eruygur (SSC s459688, MIT License).
Underlying sources: Schmidt & Phillips (1992) and Lee & Strazicich (2003, 2013).

Every function here reproduces the Stata arithmetic operation for operation,
including its conventions (see "Parity notes" in ``VALIDATION.md``), so that
the Python and Stata commands report identical critical values.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["CriticalValues", "critical_values"]

FloatArray = NDArray[np.float64]

#: Model flags used by the Stata command (``mflag``).
CRASH = 1
BREAK = 2


class CriticalValues(NamedTuple):
    """1%, 5% and 10% critical values of the test statistic."""

    cv1: float
    cv5: float
    cv10: float


def _table(rows: list[list[float]]) -> FloatArray:
    arr = np.array(rows, dtype=np.float64)
    arr.setflags(write=False)
    return arr


# Sample sizes at which every table is tabulated.
_SAMPLE_SIZES = _table([[100.0, 250.0, 500.0, 1000.0]])[0]

# Break fractions at which the trend-break tables are tabulated.
_LAMBDA_ONE_BREAK = _table([[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]])[0]
_LAMBDA_TWO_BREAK = _table([[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]])[0]
_N_PAIRS_TWO_BREAK = 12

# Columns are the 1%, 5%, 10% levels; rows are T = 100, 250, 500, 1000.
# No break (Schmidt-Phillips): lsmlm_cv0().
_CV_NO_BREAK = _table(
    [
        [-3.597, -3.031, -2.745],
        [-3.572, -3.023, -2.747],
        [-3.570, -3.021, -2.748],
        [-3.566, -3.023, -2.748],
    ]
)

# One break, crash model: lsmlm_cvc1().
_CV_CRASH_ONE = _table(
    [
        [-4.084, -3.487, -3.185],
        [-3.987, -3.387, -3.076],
        [-3.840, -3.277, -2.985],
        [-3.798, -3.230, -2.925],
    ]
)

# Two breaks, crash model: lsmlm_cvc2().
_CV_CRASH_TWO = _table(
    [
        [-4.073, -3.563, -3.296],
        [-4.101, -3.594, -3.345],
        [-4.261, -3.647, -3.287],
        [-3.828, -3.248, -3.005],
    ]
)

# One break, trend-break model: lsmlm_cvb1().
# 4 sample sizes x 9 break fractions = 36 rows.
_CV_BREAK_ONE = _table(
    [
        # T = 100
        [-4.630, -4.064, -3.787],
        [-4.704, -4.132, -3.843],
        [-4.769, -4.199, -3.906],
        [-4.820, -4.253, -3.963],
        [-4.857, -4.293, -4.008],
        [-4.891, -4.324, -4.042],
        [-4.899, -4.338, -4.058],
        [-4.910, -4.348, -4.071],
        [-4.915, -4.351, -4.073],
        # T = 250
        [-4.523, -3.974, -3.697],
        [-4.536, -3.982, -3.702],
        [-4.563, -4.010, -3.726],
        [-4.609, -4.060, -3.777],
        [-4.647, -4.098, -3.819],
        [-4.667, -4.125, -3.846],
        [-4.672, -4.130, -3.856],
        [-4.681, -4.144, -3.870],
        [-4.685, -4.145, -3.873],
        # T = 500
        [-4.502, -3.959, -3.679],
        [-4.489, -3.944, -3.664],
        [-4.496, -3.950, -3.671],
        [-4.534, -3.991, -3.714],
        [-4.570, -4.031, -3.756],
        [-4.587, -4.052, -3.777],
        [-4.598, -4.065, -3.794],
        [-4.612, -4.081, -3.809],
        [-4.602, -4.074, -3.799],
        # T = 1000
        [-4.466, -3.928, -3.651],
        [-4.455, -3.919, -3.640],
        [-4.469, -3.933, -3.651],
        [-4.501, -3.968, -3.687],
        [-4.523, -3.993, -3.713],
        [-4.546, -4.018, -3.740],
        [-4.562, -4.033, -3.756],
        [-4.576, -4.049, -3.773],
        [-4.593, -4.065, -3.790],
    ]
)

# Two breaks, trend-break model: lsmlm_cvb2().
# 4 sample sizes x 12 break-fraction pairs = 48 rows.
_CV_BREAK_TWO = _table(
    [
        # T = 100
        [-6.750, -6.108, -5.779],
        [-7.196, -6.312, -5.893],
        [-6.932, -6.175, -5.825],
        [-7.004, -6.185, -5.828],
        [-6.691, -6.152, -5.798],
        [-6.821, -5.917, -5.541],
        [-6.963, -6.201, -5.890],
        [-6.821, -6.166, -5.832],
        [-6.978, -6.288, -5.998],
        [-7.032, -6.375, -6.011],
        [-6.863, -6.268, -5.956],
        [-7.014, -6.446, -6.072],
        # T = 250
        [-5.667, -5.177, -4.921],
        [-5.974, -5.342, -5.004],
        [-5.869, -5.398, -5.112],
        [-6.064, -5.462, -5.192],
        [-5.957, -5.374, -5.104],
        [-5.783, -5.272, -4.943],
        [-5.934, -5.300, -4.988],
        [-5.771, -5.357, -5.032],
        [-5.958, -5.421, -5.179],
        [-5.751, -5.349, -5.063],
        [-5.971, -5.264, -5.018],
        [-6.035, -5.484, -5.161],
        # T = 500
        [-5.640, -4.855, -4.572],
        [-5.392, -4.939, -4.691],
        [-5.608, -4.947, -4.661],
        [-5.602, -5.059, -4.782],
        [-5.533, -4.989, -4.752],
        [-5.619, -5.038, -4.723],
        [-5.511, -4.898, -4.607],
        [-5.494, -5.035, -4.742],
        [-5.548, -5.043, -4.811],
        [-5.716, -5.149, -4.912],
        [-5.509, -4.963, -4.674],
        [-5.655, -4.967, -4.705],
        # T = 1000
        [-5.116, -4.539, -4.195],
        [-5.240, -4.578, -4.278],
        [-5.437, -4.917, -4.533],
        [-5.387, -4.884, -4.567],
        [-5.099, -4.716, -4.481],
        [-5.247, -4.776, -4.472],
        [-5.085, -4.576, -4.301],
        [-5.209, -4.697, -4.398],
        [-5.212, -4.734, -4.534],
        [-5.183, -4.773, -4.535],
        [-5.100, -4.699, -4.364],
        [-5.291, -4.803, -4.481],
    ]
)


def _interp_sample_size(table: FloatArray, nobs: float) -> FloatArray:
    """Combine the rows of a (4 x 3) table for a given sample size.

    Port of ``lsmlm_interp_T()``. Below T = 100 the first row is used and from
    T = 1000 upward the last row is used. In between, the Stata code weights
    the *lower* bracketing row by ``(nobs - T_lo) / (T_hi - T_lo)`` and the
    upper row by the complement; that convention is reproduced here unchanged
    so the two implementations agree (see ``VALIDATION.md``, "Parity notes").
    """
    sizes = _SAMPLE_SIZES
    last = len(sizes) - 1
    lo, hi, w_lo, w_hi = last, last, 1.0, 0.0
    for i in range(len(sizes)):
        if nobs < sizes[i]:
            if i == 0:
                lo, hi, w_lo, w_hi = 0, 0, 1.0, 0.0
            else:
                w_lo = float((nobs - sizes[i - 1]) / (sizes[i] - sizes[i - 1]))
                lo = i - 1
                w_hi = 1.0 - w_lo
                hi = i
            break
    result: FloatArray = w_lo * table[lo, :] + w_hi * table[hi, :]
    return result


def _cv_break_one(lam: float, nobs: float) -> FloatArray:
    """One break, trend-break model. Port of ``lsmlm_interp_cvb1()``.

    Linear interpolation over the break fraction (folded onto [0, 0.5]), then
    :func:`_interp_sample_size` over the sample size.
    """
    grid = _LAMBDA_ONE_BREAK
    n_grid = len(grid)
    if lam > 0.5:
        lam = 1.0 - lam

    i1, i2, w1, w2 = 0, 0, 1.0, 0.0
    if lam <= grid[0]:
        pass
    elif lam >= grid[-1]:
        i1 = i2 = n_grid - 1
    else:
        for i in range(1, n_grid):
            if lam < grid[i]:
                w2 = float((lam - grid[i - 1]) / (grid[i] - grid[i - 1]))
                w1 = 1.0 - w2
                i1, i2 = i - 1, i
                break

    by_size = np.empty((len(_SAMPLE_SIZES), 3), dtype=np.float64)
    for i in range(len(_SAMPLE_SIZES)):
        off = i * n_grid
        by_size[i, :] = w1 * _CV_BREAK_ONE[off + i1, :] + w2 * _CV_BREAK_ONE[off + i2, :]
    return _interp_sample_size(by_size, nobs)


def _two_break_cell(lam1: float, lam2: float) -> int:
    """Zero-based row (within one sample-size block) of the two-break table.

    Port of the cell selection in ``lsmlm_interp_cvb2()``: reflect the pair
    when ``lam1 + lam2 >= 1``, snap each fraction to the nearest grid point,
    force the two grid points apart, then apply the Mata index formula.
    """
    grid = _LAMBDA_TWO_BREAK
    n_grid = len(grid)

    a, b = lam1, lam2
    if a + b >= 1.0:
        a, b = 1.0 - lam2, 1.0 - lam1

    dist1, dist2, i1, i2 = 1.0, 1.0, 0, 0
    for i in range(n_grid):
        if abs(a - grid[i]) < dist1:
            i1 = i
            dist1 = abs(a - grid[i])
        if abs(b - grid[i]) < dist2:
            i2 = i
            dist2 = abs(b - grid[i])

    if i1 == i2:
        if i1 == 0:
            i2 = i1 + 1
        elif i1 == n_grid - 1:
            i1 = i2 - 1
        elif (a - grid[i1]) + (b - grid[i2]) > 0:
            i2 = i1 + 1
        else:
            i1 = i2 - 1

    # Mata is one-based: idx = (cols(bp) - i1) * (i1 - 1) + i2 - 1.
    m1, m2 = i1 + 1, i2 + 1
    idx = (n_grid - m1) * (m1 - 1) + m2 - 1
    idx = min(max(idx, 1), _N_PAIRS_TWO_BREAK)
    return idx - 1


def _cv_break_two(lam1: float, lam2: float, nobs: float) -> FloatArray:
    """Two breaks, trend-break model. Port of ``lsmlm_interp_cvb2()``."""
    cell = _two_break_cell(lam1, lam2)
    rows = [i * _N_PAIRS_TWO_BREAK + cell for i in range(len(_SAMPLE_SIZES))]
    return _interp_sample_size(_CV_BREAK_TWO[rows, :], nobs)


def critical_values(
    breaks: int,
    model: str,
    nobs: float,
    lambda1: float | None = None,
    lambda2: float | None = None,
) -> CriticalValues:
    """Return the 1%, 5% and 10% critical values used by :func:`lsmlm.lsmlm`.

    Parameters
    ----------
    breaks
        Number of breaks: 0, 1 or 2.
    model
        ``"break"`` (level and trend shift) or ``"crash"`` (level shift).
        Ignored when ``breaks == 0``.
    nobs
        Number of observations in the final test regression.
    lambda1, lambda2
        Break fractions. Only used by the trend-break model, where the
        critical values depend on the location of the break(s).

    Returns
    -------
    CriticalValues
        Named tuple ``(cv1, cv5, cv10)``.
    """
    if breaks not in (0, 1, 2):
        raise ValueError("breaks must be 0, 1, or 2")
    if model not in ("break", "crash"):
        raise ValueError("model must be 'break' or 'crash'")
    flag = CRASH if model == "crash" else BREAK
    lam1 = 0.0 if lambda1 is None else float(lambda1)
    lam2 = 0.0 if lambda2 is None else float(lambda2)

    if breaks == 0:
        cv = _interp_sample_size(_CV_NO_BREAK, nobs)
    elif flag == CRASH:
        cv = _interp_sample_size(_CV_CRASH_ONE if breaks == 1 else _CV_CRASH_TWO, nobs)
    elif breaks == 1:
        cv = _cv_break_one(lam1, nobs)
    else:
        cv = _cv_break_two(lam1, lam2, nobs)
    return CriticalValues(float(cv[0]), float(cv[1]), float(cv[2]))
