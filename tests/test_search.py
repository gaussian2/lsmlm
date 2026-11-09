"""Break search, lag selection and Stata edge-case semantics."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from lsmlm import LSMLMError, lsmlm

QUIET = {"print_results": False}


def _tau_at(y, positions, trend_break, maxlag):
    """Independent, deliberately naive LM statistic at fixed break positions.

    Written from the description in Lee & Strazicich (2003) with ``lstsq``;
    shares no code with the package.
    """
    n = len(y)
    t = np.arange(1, n + 1)
    dy = np.diff(y)  # dy[i] = y[i+1] - y[i], i.e. observation t = i + 2

    dz = [np.ones(n - 1)]
    for b in positions:
        dz.append((t[1:] == b + 1).astype(float))
    if trend_break:
        for b in positions:
            dz.append((t[1:] > b).astype(float))
    dz = np.column_stack(dz)
    delta, *_ = np.linalg.lstsq(dz, dy, rcond=None)

    # S(1) = 0 and D.S equal to the residual of D.y on DZ.
    s = np.concatenate([[0.0], np.cumsum(dy - dz @ delta)])
    ds = np.diff(s)

    for k in range(maxlag, -1, -1):
        rows = np.arange(k, n - 1)  # usable rows of dy
        cols = [dz[rows], s[rows][:, None]] + [ds[rows - lag][:, None] for lag in range(1, k + 1)]
        x = np.column_stack(cols)
        if len(rows) <= 20:
            continue
        beta, *_ = np.linalg.lstsq(x, dy[rows], rcond=None)
        resid = dy[rows] - x @ beta
        cov = resid @ resid / (len(rows) - x.shape[1]) * np.linalg.inv(x.T @ x)
        tstat = beta / np.sqrt(np.diag(cov))
        if k == 0 or abs(tstat[-1]) >= 1.645:
            return tstat[dz.shape[1]], k
    raise AssertionError("no admissible lag")


@pytest.fixture(scope="module")
def short_series():
    rng = np.random.default_rng(42)
    n = 48
    y = np.cumsum(rng.normal(size=n)) * 0.3 + rng.normal(size=n)
    y[20:] += 2.0
    y[33:] += 0.15 * np.arange(n - 33)
    return y


@pytest.mark.parametrize("model", ["crash", "break"])
@pytest.mark.parametrize("breaks", [0, 1, 2])
def test_statistic_is_the_minimum_over_the_grid(short_series, model, breaks):
    """Checks every model, including two-break crash, against a brute-force search."""
    y = short_series
    n, maxlag, trim, mindist = len(y), 2, 0.10, 3
    result = lsmlm(
        y, breaks=breaks, model=model, maxlag=maxlag, trim=trim, mindist=mindist, **QUIET
    )

    lo = max(math.ceil(trim * n), 3, maxlag + 2)
    hi = min(math.floor((1 - trim) * n), n - 3)
    if breaks == 0:
        grid = [()]
    elif breaks == 1:
        grid = [(b,) for b in range(lo, hi + 1)]
    else:
        grid = [
            (a, b) for a, b in itertools.product(range(lo, hi + 1), repeat=2) if b > a + mindist
        ]

    best = min(
        ((*_tau_at(y, pos, model == "break", maxlag), pos) for pos in grid),
        key=lambda item: item[0],
    )
    assert result.tau == pytest.approx(best[0], rel=1e-9)
    assert result.k == best[1]
    if breaks >= 1:
        assert result.tb1_index == best[2][0]
    if breaks == 2:
        assert result.tb2_index == best[2][1]
        assert result.tb2_index > result.tb1_index + mindist


def test_maxlag_zero_runs_without_augmentation(france):
    result = lsmlm(*france, breaks=1, maxlag=0, **QUIET)
    assert result.k == 0
    assert result.N == 124  # T - 1


def test_break_window_respects_trim(france):
    y, year = france
    for trim in (0.10, 0.25, 0.40):
        r = lsmlm(y, year, breaks=1, maxlag=1, trim=trim, **QUIET)
        assert math.ceil(trim * 125) <= r.tb1_index <= math.floor((1 - trim) * 125)


def test_lambda_uses_the_lag_adjusted_convention(france):
    r = lsmlm(*france, breaks=2, maxlag=2, mindist=5, **QUIET)
    lower = r.k + 2
    assert r.lambda1 == (r.tb1_index - lower) / (125 - lower + 1)
    assert r.lambda2 == (r.tb2_index - lower) / (125 - lower + 1)


def test_empty_break_window_is_an_error(random_walk):
    with pytest.raises(LSMLMError, match="empty break window"):
        lsmlm(random_walk[:24], breaks=1, maxlag=12, trim=0.45, **QUIET)
    # without breaks there is no window to be empty
    assert lsmlm(random_walk[:40], breaks=0, maxlag=12, trim=0.45, **QUIET).k <= 12


def test_constant_series_cannot_be_estimated():
    with pytest.raises(LSMLMError, match="no estimable configuration"):
        lsmlm(np.ones(60), breaks=1, **QUIET)


def test_too_few_regression_observations_cannot_be_estimated(random_walk):
    # T = 21 leaves at most 20 observations, and the regressions need more than 20
    with pytest.raises(LSMLMError, match="no estimable configuration"):
        lsmlm(random_walk[:21], breaks=0, maxlag=1, **QUIET)
    assert lsmlm(random_walk[:22], breaks=0, maxlag=1, **QUIET).N == 21


def test_adjacent_breaks_with_mindist_zero(france):
    """mindist=0 admits b2 = b1 + 1, where D1 - D2 = B1 exactly.

    Stata's regress omits the redundant dummy and carries on; so does the port.
    """
    y, year = france
    wide = lsmlm(y, year, breaks=2, maxlag=2, mindist=5, **QUIET)
    tight = lsmlm(y, year, breaks=2, maxlag=2, mindist=0, **QUIET)
    assert math.isfinite(tight.tau)
    # a larger search set can only lower the minimum
    assert tight.tau <= wide.tau


def _tiny_window_series(seed):
    # T = 24, trim 0.45, maxlag 1 -> the window is positions 11..13, so with
    # mindist = 0 the admissible pairs are (11, 12), (11, 13) and (12, 13).
    return np.cumsum(np.random.default_rng(seed).normal(size=24))


def test_omitted_dummy_reports_a_missing_t_ratio():
    """When the selected breaks are adjacent, the dropped dummy has no t-ratio."""
    r = lsmlm(
        _tiny_window_series(1), breaks=2, model="break", maxlag=1, trim=0.45, mindist=0, **QUIET
    )
    assert (r.tb1_index, r.tb2_index) == (11, 12)
    assert math.isfinite(r.tau)
    assert math.isfinite(r.tB1)
    assert math.isfinite(r.tB2)
    assert math.isfinite(r.tD1)
    assert math.isnan(r.tD2)  # D2 = D1 - B1 is the redundant regressor
    # Stata: max(abs(tB2), abs(.)) ignores the missing value
    assert r.break2_sig10 is (abs(r.tB2) >= 1.645)


def test_non_adjacent_breaks_keep_all_dummies():
    r = lsmlm(
        _tiny_window_series(3), breaks=2, model="break", maxlag=1, trim=0.45, mindist=0, **QUIET
    )
    assert (r.tb1_index, r.tb2_index) == (11, 13)
    assert all(math.isfinite(v) for v in (r.tB1, r.tB2, r.tD1, r.tD2))


def test_crash_model_has_no_collinearity_with_adjacent_breaks():
    r = lsmlm(
        _tiny_window_series(1), breaks=2, model="crash", maxlag=1, trim=0.45, mindist=0, **QUIET
    )
    assert math.isfinite(r.tB1)
    assert math.isfinite(r.tB2)
    assert r.tD1 is None
    assert r.tD2 is None
