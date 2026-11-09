from __future__ import annotations

import numpy as np
import pytest

from conftest import load_reference
from lsmlm.datasets import load_oecd15


def test_shape_and_columns(oecd):
    assert oecd.shape == (1875, 7)
    assert list(oecd.columns) == [
        "cid",
        "countrycode",
        "country",
        "year",
        "gdppc",
        "rel_gdppc",
        "y",
    ]
    assert not oecd.isna().any().any()


def test_balanced_panel(oecd):
    counts = oecd.groupby("country")["year"].agg(["min", "max", "count", "nunique"])
    assert len(counts) == 15
    assert (counts["min"] == 1870).all()
    assert (counts["max"] == 1994).all()
    assert (counts["count"] == 125).all()
    assert (counts["nunique"] == 125).all()


def test_variable_definitions(oecd):
    np.testing.assert_allclose(oecd["y"], np.log(oecd["rel_gdppc"]), rtol=0, atol=1e-12)
    average = oecd.groupby("year")["gdppc"].transform("mean")
    np.testing.assert_allclose(oecd["rel_gdppc"], oecd["gdppc"] / average, rtol=1e-6)


def test_each_call_returns_a_fresh_copy(oecd):
    other = load_oecd15()
    assert other is not oecd
    other.loc[0, "y"] = 123.0
    assert load_oecd15().loc[0, "y"] != 123.0


@pytest.mark.parametrize("row", load_reference()["descriptives"], ids=lambda r: r["country"])
def test_descriptives_match_stata(oecd, row):
    g = oecd[oecd["country"] == row["country"]]
    assert len(g) == row["N"]
    assert g["gdppc"].mean() == pytest.approx(row["mean_gdppc"], rel=1e-12)
    assert g["rel_gdppc"].mean() == pytest.approx(row["mean_rel"], rel=1e-12)
    assert g["y"].mean() == pytest.approx(row["mean_y"], rel=1e-12)
    assert g["y"].std() == pytest.approx(row["sd_y"], rel=1e-12)
    assert g["y"].min() == row["min_y"]
    assert g["y"].max() == row["max_y"]
