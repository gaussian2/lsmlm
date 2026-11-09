# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Junsoo Lee
# Portions derived from third-party MIT-licensed code; see THIRD_PARTY_NOTICES.md.
"""Lee-Strazicich minimum LM unit root test with zero, one or two breaks.

Python port of the Stata command ``lsmlm`` (Ibrahim Ongoren, Pamukkale
University, v1.1.0, 20jul2026). The computation follows ``lsmlm.ado`` step by
step so that both implementations return the same numbers.

Method
------
1. For each candidate break date (or pair of dates), regress ``D.y`` on the
   differenced deterministic terms ``DZ`` to recover the shift magnitudes,
   then build the LM-detrended series ``S(t)`` (``S(1) = 0`` and ``D.S`` equal
   to the residual of ``D.y`` on ``DZ``).
2. Regress ``D.y`` on ``DZ``, ``S(t-1)`` and lagged ``D.S`` terms. The number
   of lags is chosen general-to-specific: start at ``maxlag`` and keep the
   highest lag whose ``|t| >= 1.645``.
3. Select the break date(s) that minimise the t-ratio on ``S(t-1)``.
4. Compare that minimum with interpolated critical values.

Deterministic terms, for break ``j`` at (one-based) position ``b``::

    impulse B_j = 1{t == b+1}    step D_j = 1{t > b}    trend T_j = (t-b) 1{t > b}

    crash model : DZ = {const, B_j}        S subtracts the level steps
    break model : DZ = {const, B_j, D_j}   S subtracts level steps and trends
    no break    : DZ = {const}
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._ols import ols
from .critical_values import critical_values
from .results import LSMLMError, LSMLMResult

__all__ = ["lsmlm"]

FloatArray = NDArray[np.float64]

#: Two-sided 10% normal critical value used for lag and break significance.
_T_CRIT = 1.645
#: A test regression must have more than this many observations.
_MIN_REGRESSION_OBS = 20
#: Minimum number of usable observations.
_MIN_OBS = 20


class _Candidate(NamedTuple):
    """Outcome of the lag search at one break configuration."""

    tau: float
    k: int
    nobs: int
    t_impulse: tuple[float, ...]
    t_step: tuple[float, ...]


def _below(tstat: float, threshold: float = _T_CRIT) -> bool:
    """``abs(t) < threshold`` with Stata's rule that a missing value is large."""
    return bool(abs(tstat) < threshold)  # nan compares False, i.e. "not below"


def _evaluate(
    y: FloatArray,
    dy: FloatArray,
    tt: FloatArray,
    positions: tuple[int, ...],
    trend_break: bool,
    maxlag: int,
) -> _Candidate | None:
    """Run both regression stages for one break configuration.

    ``positions`` holds the one-based break positions. Returns ``None`` when
    no lag order leaves enough observations.
    """
    nobs_total = y.shape[0]
    n_breaks = len(positions)
    n_det = 1 + n_breaks * (2 if trend_break else 1)

    # Columns of the full design: DZ | S(t-1) | D.S(t-1) ... D.S(t-maxlag).
    design = np.zeros((nobs_total, n_det + 1 + maxlag), dtype=np.float64)
    design[:, 0] = 1.0
    steps = []
    for j, b in enumerate(positions):
        step = np.zeros(nobs_total, dtype=np.float64)
        step[b:] = 1.0  # t > b
        steps.append(step)
        design[b, 1 + j] = 1.0  # t == b + 1
        if trend_break:
            design[:, 1 + n_breaks + j] = step

    # Stage 1: D.y on DZ over t = 2..T.
    stage1 = ols(design[1:, :n_det], dy[1:])
    drift = float(stage1.beta[0])
    psi = float(y[0]) - (drift * 1.0)
    detrended = y - psi - (drift * tt)
    for j in range(n_breaks):
        detrended = detrended - (float(stage1.beta[1 + j]) * steps[j])
    if trend_break:
        for j, b in enumerate(positions):
            slope = float(stage1.beta[1 + n_breaks + j])
            detrended = detrended - (slope * ((tt - b) * steps[j]))

    d_detrended = np.empty(nobs_total, dtype=np.float64)
    d_detrended[0] = np.nan
    d_detrended[1:] = detrended[1:] - detrended[:-1]
    design[0, n_det] = np.nan
    design[1:, n_det] = detrended[:-1]
    for lag in range(1, maxlag + 1):
        design[:lag, n_det + lag] = np.nan
        design[lag:, n_det + lag] = d_detrended[: nobs_total - lag]

    # Stage 2: general-to-specific lag selection.
    for k in range(maxlag, -1, -1):
        first = k + 1  # first row with S(t-1) and all k lags available
        if nobs_total - first <= _MIN_REGRESSION_OBS:
            continue
        fit = ols(design[first:, : n_det + 1 + k], dy[first:])
        if k > 0 and _below(float(fit.tstat[n_det + k])):
            continue
        t_det = fit.tstat
        return _Candidate(
            tau=float(t_det[n_det]),
            k=k,
            nobs=fit.nobs,
            t_impulse=tuple(float(t_det[1 + j]) for j in range(n_breaks)),
            t_step=tuple(float(t_det[1 + n_breaks + j]) for j in range(n_breaks))
            if trend_break
            else (),
        )
    return None


def _configurations(breaks: int, lo: int, hi: int, mindist: int) -> Iterator[tuple[int, ...]]:
    """Break configurations in the order the Stata command visits them."""
    if breaks == 0:
        yield ()
    elif breaks == 1:
        for b1 in range(lo, hi + 1):
            yield (b1,)
    else:
        for b1 in range(lo, hi + 1):
            for b2 in range(max(lo, b1 + mindist + 1), hi + 1):
                yield (b1, b2)


def _prepare_series(y: ArrayLike, t: ArrayLike | None) -> tuple[FloatArray, NDArray[Any]]:
    """Validate the inputs; drop missing values and sort by time."""
    try:
        y_arr = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LSMLMError("y must be numeric") from exc
    if y_arr.ndim != 1:
        raise LSMLMError("y must be one-dimensional (a single time series)")
    if np.isinf(y_arr).any():
        raise LSMLMError("y contains infinite values")

    if t is None:
        keep = ~np.isnan(y_arr)
        y_arr = y_arr[keep]
        return y_arr, np.arange(1, y_arr.shape[0] + 1)

    t_arr = np.asarray(t)
    if t_arr.dtype.kind == "O":
        try:
            t_arr = t_arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise LSMLMError("t must be numeric or datetime-like") from exc
    if t_arr.ndim != 1 or t_arr.shape[0] != y_arr.shape[0]:
        raise LSMLMError("t must be one-dimensional and the same length as y")
    if t_arr.dtype.kind in "mM":
        t_missing = np.isnat(t_arr)
    elif t_arr.dtype.kind == "f":
        t_missing = np.isnan(t_arr)
    elif t_arr.dtype.kind in "iub":
        t_missing = np.zeros(t_arr.shape[0], dtype=bool)
    else:
        raise LSMLMError("t must be numeric or datetime-like")

    keep = ~(np.isnan(y_arr) | t_missing)
    y_arr, t_arr = y_arr[keep], t_arr[keep]
    order = np.argsort(t_arr, kind="stable")
    y_arr, t_arr = y_arr[order], t_arr[order]
    if t_arr.shape[0] > 1 and bool((t_arr[1:] == t_arr[:-1]).any()):
        raise LSMLMError(
            "the time variable is not unique in the sample. lsmlm operates on a "
            "single time series: pass one panel unit at a time."
        )
    return y_arr, t_arr


def _as_scalar(value: Any) -> Any:
    """Convert a NumPy scalar to the matching Python scalar where one exists."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _dummies_significant(tstats: tuple[float, ...]) -> bool:
    """Stata's ``max(abs(t), ...) >= 1.645`` (``max`` skips missing values)."""
    finite = [abs(v) for v in tstats if not math.isnan(v)]
    if not finite:
        return True  # an all-missing maximum is missing, and missing >= 1.645
    return max(finite) >= _T_CRIT


def lsmlm(
    y: ArrayLike,
    t: ArrayLike | None = None,
    breaks: int = 2,
    model: str = "break",
    maxlag: int | None = None,
    trim: float = 0.10,
    mindist: int | None = None,
    varname: str = "y",
    print_results: bool = True,
) -> LSMLMResult:
    """Lee-Strazicich minimum LM unit root test with zero, one or two breaks.

    The null hypothesis is a unit root; rejecting it is evidence that the
    series is (trend-)stationary around the estimated break(s).

    Parameters
    ----------
    y
        The series to test (a single time series). Missing values are dropped.
    t
        Time variable matching ``y`` (for example years), numeric or
        datetime-like, with unique values. The data are sorted by ``t`` before
        testing. If omitted, ``y`` is taken to be in time order and observations
        are numbered ``1..T``.
    breaks
        Number of structural breaks: 0, 1 or 2. ``breaks=0`` is the
        Schmidt-Phillips no-break LM test.
    model
        ``"break"``: level and trend shift at each break (Lee-Strazicich
        Model C). ``"crash"``: level shift only (Model A). Ignored when
        ``breaks=0``.
    maxlag
        Largest number of augmenting lags considered by the general-to-specific
        search. ``None`` (or any negative number) selects
        ``floor(4 * (T / 100) ** 0.25)``, with a minimum of 1.
    trim
        Fraction of the sample excluded at each end of the break search; must
        lie strictly between 0 and 0.5.
    mindist
        Minimum number of observations between the two breaks (two-break case
        only): pairs must satisfy ``b2 > b1 + mindist``. ``None`` (or any
        negative number) selects 2.
    varname
        Label for the series, used only in the printed table.
    print_results
        Print the results table. Set to ``False`` when calling in a loop.

    Returns
    -------
    LSMLMResult
        Test statistic, chosen lag order, break dates, critical values and
        rejection decisions.

    Raises
    ------
    LSMLMError
        If an option is invalid, the sample is too short, or no break
        configuration can be estimated.

    Examples
    --------
    >>> from lsmlm import lsmlm
    >>> from lsmlm.datasets import load_oecd15
    >>> france = load_oecd15().query("country == 'France'")
    >>> res = lsmlm(france["y"], france["year"], breaks=2, maxlag=2, mindist=5,
    ...             print_results=False)
    >>> res.tb1, res.tb2, round(res.tau, 4)
    (1939, 1947, -7.2219)
    """
    # ---------- validate options ----------
    if model not in ("break", "crash"):
        raise LSMLMError("model must be 'break' or 'crash'")
    if isinstance(breaks, bool) or breaks not in (0, 1, 2):
        raise LSMLMError("breaks must be 0, 1, or 2")
    breaks = int(breaks)
    try:
        trim = float(trim)
    except (TypeError, ValueError) as exc:
        raise LSMLMError("trim must be in (0, 0.5)") from exc
    if not (0 < trim < 0.5):
        raise LSMLMError("trim must be in (0, 0.5)")
    maxlag_opt = _integer_option(maxlag, "maxlag")
    mindist_opt = _integer_option(mindist, "mindist")
    trend_break = model == "break"

    # ---------- prepare the series ----------
    y_arr, t_arr = _prepare_series(y, t)
    nobs = int(y_arr.shape[0])
    if nobs < _MIN_OBS:
        raise LSMLMError("not enough observations after dropping missing values (need 20+)")

    if maxlag_opt is None:
        maxlag_opt = max(1, math.floor(4 * (nobs / 100) ** 0.25))
    if mindist_opt is None:
        mindist_opt = 2

    tt = np.arange(1, nobs + 1, dtype=np.float64)
    dy = np.empty(nobs, dtype=np.float64)
    dy[0] = np.nan
    dy[1:] = y_arr[1:] - y_arr[:-1]

    # ---------- break search window (one-based positions) ----------
    lo = max(math.ceil(trim * nobs), 3)
    hi = min(math.floor((1 - trim) * nobs), nobs - 3)
    # Collinearity guard: keep the earliest candidate at least maxlag + 2 so a
    # level dummy cannot be constant across the lagged-regression sample.
    lo = max(lo, maxlag_opt + 2)
    if breaks > 0 and lo > hi:
        raise LSMLMError("sample too short for the chosen trim and maxlag: empty break window")

    # ---------- grid search ----------
    best: _Candidate | None = None
    best_positions: tuple[int, ...] = ()
    for positions in _configurations(breaks, lo, hi, mindist_opt):
        candidate = _evaluate(y_arr, dy, tt, positions, trend_break, maxlag_opt)
        if candidate is None:
            continue
        # Stata: "if missing(best_tau) | final_tau < best_tau".
        if best is None or math.isnan(best.tau) or candidate.tau < best.tau:
            best, best_positions = candidate, positions

    if best is None or math.isnan(best.tau):
        raise LSMLMError(
            "no estimable configuration was found; try a larger sample, "
            "a smaller maxlag, or a smaller trim."
        )

    # ---------- break fractions: leestra's lag-adjusted convention ----------
    # lambda = (position - (k + 2)) / (T - (k + 2) + 1), so that the lookup
    # lands on the same table cell as the reference implementation.
    lower_eff = best.k + 2
    lambdas = [(b - lower_eff) / (nobs - lower_eff + 1) for b in best_positions]
    lambda1 = lambdas[0] if breaks >= 1 else None
    lambda2 = lambdas[1] if breaks == 2 else None

    cv = critical_values(breaks, model, best.nobs, lambda1, lambda2)
    reject10 = best.tau <= cv.cv10
    reject5 = best.tau <= cv.cv5
    reject1 = best.tau <= cv.cv1
    siglevel = "Not significant"
    if reject10:
        siglevel = "10%"
    if reject5:
        siglevel = "5%"
    if reject1:
        siglevel = "1%"

    def break_fields(j: int) -> dict[str, Any]:
        if breaks <= j:
            return {}
        n = j + 1
        t_impulse = best.t_impulse[j]
        t_step = best.t_step[j] if trend_break else None
        dummies = (t_impulse,) if t_step is None else (t_impulse, t_step)
        return {
            f"tb{n}": _as_scalar(t_arr[best_positions[j] - 1]),
            f"tb{n}_index": best_positions[j],
            f"lambda{n}": lambdas[j],
            f"break{n}_sig10": _dummies_significant(dummies),
            f"tB{n}": t_impulse,
            f"tD{n}": t_step,
        }

    result = LSMLMResult(
        tau=best.tau,
        k=best.k,
        breaks=breaks,
        N=best.nobs,
        maxlag=maxlag_opt,
        trim=trim,
        cv1=cv.cv1,
        cv5=cv.cv5,
        cv10=cv.cv10,
        reject1=reject1,
        reject5=reject5,
        reject10=reject10,
        siglevel=siglevel,
        model=model,
        varname=str(varname),
        mindist=mindist_opt if breaks == 2 else None,
        tmin=_as_scalar(t_arr[0]),
        tmax=_as_scalar(t_arr[-1]),
        **break_fields(0),
        **break_fields(1),
    )
    if print_results:
        print()
        print(result.summary())
    return result


def _integer_option(value: int | None, name: str) -> int | None:
    """Normalise an integer option where ``None`` or a negative means "default"."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise LSMLMError(f"{name} must be an integer")
    return None if value < 0 else int(value)
