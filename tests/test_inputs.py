"""Option validation and input handling."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from lsmlm import LSMLMError, LSMLMResult, lsmlm

QUIET = {"print_results": False}


def _same(a: LSMLMResult, b: LSMLMResult, ignore=("tb1", "tb2", "tmin", "tmax")) -> bool:
    da, db = a.to_dict(), b.to_dict()
    return all(da[k] == db[k] for k in da if k not in ignore)


# ---------------------------------------------------------------- options


@pytest.mark.parametrize("model", ["Break", "trend", "", None, 2])
def test_rejects_unknown_model(france, model):
    with pytest.raises(LSMLMError, match="model must be"):
        lsmlm(*france, model=model, **QUIET)


@pytest.mark.parametrize("breaks", [-1, 3, 1.5, "2", True, None])
def test_rejects_invalid_breaks(france, breaks):
    with pytest.raises(LSMLMError, match="breaks must be"):
        lsmlm(*france, breaks=breaks, **QUIET)


@pytest.mark.parametrize("trim", [0, 0.5, -0.1, 0.9, "x", None])
def test_rejects_trim_outside_open_interval(france, trim):
    with pytest.raises(LSMLMError, match="trim must be"):
        lsmlm(*france, trim=trim, **QUIET)


@pytest.mark.parametrize("name", ["maxlag", "mindist"])
@pytest.mark.parametrize("value", [2.0, "2", True])
def test_rejects_non_integer_lag_options(france, name, value):
    with pytest.raises(LSMLMError, match=f"{name} must be an integer"):
        lsmlm(*france, **{name: value}, **QUIET)


def test_error_type_is_a_value_error():
    assert issubclass(LSMLMError, ValueError)


# ---------------------------------------------------------------- defaults


def test_default_maxlag_is_data_driven(france):
    y, year = france
    # floor(4 * (125 / 100) ** 0.25) = 4
    assert lsmlm(y, year, breaks=0, **QUIET).maxlag == 4
    # floor(4 * (30 / 100) ** 0.25) = 2
    assert lsmlm(y[:30], year[:30], breaks=0, **QUIET).maxlag == 2


def test_negative_options_mean_default_as_in_stata(france):
    y, year = france
    base = lsmlm(y, year, breaks=0, **QUIET)
    assert lsmlm(y, year, breaks=0, maxlag=-1, **QUIET) == base
    assert lsmlm(y, year, breaks=0, maxlag=None, **QUIET) == base


def test_default_mindist_is_two(series):
    y, year = series("France", 1870, 1945)
    default = lsmlm(y, year, breaks=2, maxlag=1, **QUIET)
    assert default.mindist == 2
    assert lsmlm(y, year, breaks=2, maxlag=1, mindist=2, **QUIET) == default
    assert lsmlm(y, year, breaks=2, maxlag=1, mindist=-1, **QUIET) == default


def test_maxlag_accepts_numpy_integers(france):
    y, year = france
    assert lsmlm(y, year, breaks=0, maxlag=np.int64(2), **QUIET).maxlag == 2


def test_fields_that_do_not_apply_are_none(france):
    y, year = france
    none = lsmlm(y, year, breaks=0, maxlag=2, **QUIET)
    for name in (
        "tb1",
        "tb1_index",
        "lambda1",
        "break1_sig10",
        "tB1",
        "tD1",
        "tb2",
        "tb2_index",
        "lambda2",
        "break2_sig10",
        "tB2",
        "tD2",
        "mindist",
    ):
        assert getattr(none, name) is None, name

    crash = lsmlm(y, year, breaks=1, model="crash", maxlag=2, **QUIET)
    assert crash.tB1 is not None
    assert crash.tD1 is None
    assert crash.tb2 is None
    assert crash.mindist is None


# ---------------------------------------------------------------- the series


def test_accepts_lists_and_pandas(oecd, france):
    y, year = france
    base = lsmlm(y, year, breaks=1, maxlag=2, **QUIET)
    assert lsmlm(list(y), list(year), breaks=1, maxlag=2, **QUIET) == base
    g = oecd[oecd["country"] == "France"]
    assert lsmlm(g["y"], g["year"], breaks=1, maxlag=2, **QUIET) == base


def test_input_order_does_not_matter(france):
    y, year = france
    base = lsmlm(y, year, breaks=1, maxlag=2, **QUIET)
    perm = np.random.default_rng(0).permutation(len(y))
    assert lsmlm(y[perm], year[perm], breaks=1, maxlag=2, **QUIET) == base


def test_missing_values_are_dropped(france):
    y, year = france
    base = lsmlm(y, year, breaks=1, maxlag=2, **QUIET)
    y_nan = np.concatenate([y[:40], [np.nan], y[40:]])
    year_nan = np.concatenate([year[:40], [1700], year[40:]]).astype(float)
    assert _same(lsmlm(y_nan, year_nan, breaks=1, maxlag=2, **QUIET), base)

    # a missing time value drops the observation too (Stata: missing(timevar))
    y_extra = np.concatenate([y, [99.0]])
    year_extra = np.concatenate([year.astype(float), [np.nan]])
    assert _same(lsmlm(y_extra, year_extra, breaks=1, maxlag=2, **QUIET), base)


def test_without_time_variable_positions_are_reported(france):
    y, year = france
    with_time = lsmlm(y, year, breaks=1, maxlag=2, **QUIET)
    without = lsmlm(y, breaks=1, maxlag=2, **QUIET)
    assert _same(with_time, without)
    assert without.tb1 == without.tb1_index == with_time.tb1 - 1869
    assert (without.tmin, without.tmax) == (1, 125)


def test_break_dates_keep_the_type_of_the_time_variable(france):
    y, year = france
    as_int = lsmlm(y, year, breaks=1, maxlag=2, **QUIET)
    assert isinstance(as_int.tb1, int)
    assert as_int.tb1 == 1946

    as_float = lsmlm(y, year + 0.5, breaks=1, maxlag=2, **QUIET)
    assert isinstance(as_float.tb1, float)
    assert as_float.tb1 == 1946.5
    assert "  Break 1:       1946.5   (lambda1 = 0.602)" in as_float.summary()


def test_datetime_time_variable(france):
    y, year = france
    dates = pd.to_datetime([f"{v}-12-31" for v in year])
    result = lsmlm(y, dates, breaks=1, maxlag=2, **QUIET)
    assert pd.Timestamp(result.tb1) == pd.Timestamp("1946-12-31")
    assert result.tau == lsmlm(y, year, breaks=1, maxlag=2, **QUIET).tau
    assert "1946-12-31" in result.summary()


def test_object_dtype_time_variable(france):
    y, year = france
    as_object = np.array([int(v) for v in year], dtype=object)
    assert (
        lsmlm(y, as_object, breaks=0, maxlag=2, **QUIET).tau
        == lsmlm(y, year, breaks=0, maxlag=2, **QUIET).tau
    )


def test_duplicate_time_values_are_an_error(oecd):
    two = oecd[oecd["country"].isin(["France", "Italy"])]
    with pytest.raises(LSMLMError, match="not unique"):
        lsmlm(two["y"], two["year"], breaks=1, **QUIET)


@pytest.mark.parametrize(
    ("y", "t", "message"),
    [
        (np.zeros((30, 2)), None, "one-dimensional"),
        (np.arange(30.0), np.arange(29), "same length"),
        (np.arange(30.0), np.zeros((30, 1)), "same length"),
        (["a"] * 30, None, "y must be numeric"),
        (np.arange(30.0), ["a"] * 30, "numeric or datetime"),
        (np.arange(30.0), np.array(["a"] * 30, dtype=object), "numeric or datetime"),
        (np.r_[np.arange(29.0), np.inf], None, "infinite"),
    ],
)
def test_rejects_malformed_series(y, t, message):
    with pytest.raises(LSMLMError, match=message):
        lsmlm(y, t, breaks=0, **QUIET)


def test_needs_twenty_observations(random_walk):
    with pytest.raises(LSMLMError, match="not enough observations"):
        lsmlm(random_walk[:19], breaks=0, **QUIET)
    with pytest.raises(LSMLMError, match="not enough observations"):
        lsmlm(np.r_[random_walk[:19], [np.nan] * 5], breaks=0, **QUIET)


# ---------------------------------------------------------------- the result


def test_result_is_immutable_and_exportable(france):
    result = lsmlm(*france, breaks=1, maxlag=2, **QUIET)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.tau = 0.0  # type: ignore[misc]
    as_dict = result.to_dict()
    assert as_dict["tau"] == result.tau
    assert set(as_dict) == {f.name for f in dataclasses.fields(LSMLMResult)}
    assert pd.DataFrame([as_dict]).loc[0, "tb1"] == 1946


def test_rejection_flags_are_consistent(france):
    y, year = france
    r = lsmlm(y, year, breaks=1, maxlag=2, **QUIET)  # rejects at 5%, not at 1%
    assert (r.reject1, r.reject5, r.reject10) == (False, True, True)
    assert r.siglevel == "5%"
    assert r.reject5 == (r.tau <= r.cv5)
