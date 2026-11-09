#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Junsoo Lee
# Portions derived from third-party MIT-licensed code; see THIRD_PARTY_NOTICES.md.
"""Quick tour of the lsmlm package.

Python counterpart of the Stata walkthrough ``lsmlm_example.do``. Run with::

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from lsmlm import lsmlm


def simulated_series(seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    """Annual series 1880-2019: a drifting random walk with a trend break in 1945.

    Illustrative only; it just shows that the command runs and what it returns.
    """
    rng = np.random.default_rng(seed)
    year = 1880 + np.arange(140)
    y = np.cumsum(0.02 + rng.normal(size=year.size))
    y = y + 0.03 * (year - 1945) * (year > 1945)
    return y, year


def main() -> None:
    y, year = simulated_series()

    # ------------------------------------------------------------------
    # 1. A single time series (any variable, any time unit)
    # ------------------------------------------------------------------
    # One break, trend-break model (the default model):
    lsmlm(y, year, breaks=1)

    # Two breaks (the default number of breaks):
    lsmlm(y, year, breaks=2)

    # Restrict the time range (post-1945 sub-sample):
    post = year >= 1946
    lsmlm(y[post], year[post], breaks=1)

    # Annual-data settings (8 lags, at least 5 observations between breaks):
    lsmlm(y, year, breaks=2, maxlag=8, mindist=5, trim=0.10)

    # No break (Schmidt-Phillips baseline):
    lsmlm(y, year, breaks=0)

    # Crash model (level shifts only) instead of the trend-break model:
    lsmlm(y, year, breaks=1, model="crash")
    lsmlm(y, year, breaks=2, model="crash")

    # Use the returned results:
    r = lsmlm(y, year, breaks=2, print_results=False)
    print()
    print("tau =", r.tau)
    print("break 1 =", r.tb1, "  break 2 =", r.tb2)
    print("5% CV =", r.cv5)
    print("reject unit root at 10%?", r.reject10)

    # ------------------------------------------------------------------
    # 2. Looping over panel units (one series per call)
    # ------------------------------------------------------------------
    # lsmlm tests a single series, so loop over the units and collect
    # result.to_dict() into a table. Needs pandas: pip install "lsmlm[data]"
    try:
        import pandas as pd
    except ImportError:
        return
    from lsmlm.datasets import load_oecd15

    panel = load_oecd15()
    panel = panel[panel["country"].isin(["France", "United Kingdom", "United States"])]
    rows = []
    for country, g in panel.groupby("country"):
        res = lsmlm(g["y"], g["year"], breaks=2, maxlag=2, mindist=5, print_results=False)
        rows.append({"country": country, **res.to_dict()})
    table = pd.DataFrame(rows)
    print()
    print(
        table[["country", "N", "k", "tb1", "tb2", "tau", "cv5", "siglevel"]].to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"
        )
    )


if __name__ == "__main__":
    main()
