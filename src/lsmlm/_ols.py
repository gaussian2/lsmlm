# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Junsoo Lee
"""Small OLS routine that mirrors how Stata's ``regress`` treats collinearity.

``lsmlm.ado`` runs every regression through ``regress`` and reads ``_b[]`` and
``_se[]``. Two ``regress`` behaviours matter for matching it:

* a regressor that is an exact linear combination of the constant and the
  regressors listed before it is *omitted*: its coefficient is reported as 0
  and its standard error as 0, so its t-ratio ``_b/_se`` is missing;
* with no residual degrees of freedom the standard errors are missing.

Here a missing t-ratio is represented by ``nan``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# A column is treated as collinear when its component orthogonal to the
# preceding columns is below this fraction of its own length.
_COLLINEARITY_TOL = 1e-10
# The same idea on the squared scale, used to decide whether the fast
# normal-equations path is safe (deliberately conservative).
_PIVOT_TOL = 1e-10


class OLSFit(NamedTuple):
    """Coefficients and t-ratios, one entry per column of the design matrix."""

    beta: FloatArray
    tstat: FloatArray
    nobs: int


def _independent_columns(x: FloatArray) -> list[int]:
    """Return the indices of the columns kept after dropping collinear ones.

    A column is dropped when it is (numerically) a linear combination of the
    columns before it, which is the rule ``regress`` applies with the constant
    placed first.
    """
    keep = list(range(x.shape[1]))
    norms = np.sqrt(np.einsum("ij,ij->j", x, x))
    while keep:
        r = np.linalg.qr(x[:, keep], mode="r")
        diag = np.abs(np.diagonal(r))
        bad = np.flatnonzero(diag <= _COLLINEARITY_TOL * norms[keep][: len(diag)])
        if bad.size == 0:
            if len(diag) < len(keep):  # more columns than rows
                keep = keep[: len(diag)]
            break
        # Only the first deficient pivot is reliable in an unpivoted QR.
        del keep[int(bad[0])]
    return keep


def ols(x: FloatArray, y: FloatArray) -> OLSFit:
    """Least squares of ``y`` on the columns of ``x`` (constant included in ``x``)."""
    nobs, ncol = x.shape
    beta = np.zeros(ncol, dtype=np.float64)
    tstat = np.full(ncol, np.nan, dtype=np.float64)

    # Fast path for the usual full-rank case. The squared Cholesky pivots are
    # the residual sums of squares of each column on the preceding ones, so a
    # (relatively) tiny pivot flags collinearity and sends us to the QR path.
    xtx = x.T @ x
    full_rank = nobs >= ncol
    if full_rank:
        try:
            pivots = np.diagonal(np.linalg.cholesky(xtx)) ** 2
        except np.linalg.LinAlgError:
            full_rank = False
        else:
            full_rank = bool(np.all(pivots > _PIVOT_TOL * np.diagonal(xtx)))

    if full_rank:
        keep = list(range(ncol))
        xtx_inv = np.linalg.inv(xtx)
        coef = xtx_inv @ (x.T @ y)
        var_diag = np.diagonal(xtx_inv)
        design = x
    else:
        keep = _independent_columns(x)
        if not keep:
            return OLSFit(beta, tstat, nobs)
        design = x[:, keep]
        q, r = np.linalg.qr(design)
        coef = np.linalg.solve(r, q.T @ y)
        r_inv = np.linalg.inv(r)
        var_diag = np.einsum("ij,ij->i", r_inv, r_inv)
    beta[keep] = coef

    df_resid = nobs - len(keep)
    if df_resid > 0:
        resid = y - design @ coef
        sigma2 = float(resid @ resid) / df_resid
        with np.errstate(divide="ignore", invalid="ignore"):
            t = coef / np.sqrt(sigma2 * var_diag)
        # Stata: x/0 is missing.
        t[~np.isfinite(t)] = np.nan
        tstat[keep] = t
    return OLSFit(beta, tstat, nobs)
