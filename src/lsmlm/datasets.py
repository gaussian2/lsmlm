# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Junsoo Lee
"""Example data shipped with the package."""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["load_oecd15"]


def load_oecd15() -> pd.DataFrame:
    """Load the OECD15 relative-income panel, 1870-1994.

    Log relative per-capita income for 15 OECD countries, the example dataset
    of the Stata package (``lsmlm_oecd15.dta``). It is built from the Maddison
    Project Database 2023 (Bolt and van Zanden 2024), which is licensed under
    CC BY 4.0.

    Returns
    -------
    pandas.DataFrame
        One row per country and year (15 x 125 = 1875 rows), sorted by
        ``cid`` and ``year``, with columns

        ``cid``
            country identifier (1-15)
        ``countrycode``
            ISO 3166-1 alpha-3 code
        ``country``
            country name
        ``year``
            1870-1994
        ``gdppc``
            real GDP per capita
        ``rel_gdppc``
            ``gdppc`` divided by the 15-country average for that year
        ``y``
            ``log(rel_gdppc)``, the series the test is applied to

    Notes
    -----
    Requires pandas (``pip install "lsmlm[data]"``).
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "load_oecd15() needs pandas. Install it with: pip install 'lsmlm[data]'"
        ) from exc

    source = resources.files("lsmlm").joinpath("data", "lsmlm_oecd15.csv")
    with source.open("r", encoding="utf-8") as handle:
        # round_trip: parse every value to the exact double stored in the .dta
        return pd.read_csv(handle, float_precision="round_trip")
