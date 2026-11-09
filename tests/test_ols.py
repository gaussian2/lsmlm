from __future__ import annotations

import numpy as np
import pytest

from lsmlm._ols import ols


def _reference(x, y):
    """Textbook OLS t-ratios via lstsq and the pseudo-inverse."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    sigma2 = resid @ resid / (x.shape[0] - x.shape[1])
    se = np.sqrt(sigma2 * np.diag(np.linalg.pinv(x.T @ x)))
    return beta, beta / se


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(50), rng.normal(size=(50, 3))])
    y = x @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(size=50)
    return x, y


def test_matches_textbook_ols(data):
    x, y = data
    fit = ols(x, y)
    beta, t = _reference(x, y)
    np.testing.assert_allclose(fit.beta, beta, rtol=1e-12)
    np.testing.assert_allclose(fit.tstat, t, rtol=1e-10)
    assert fit.nobs == 50


def test_collinear_column_is_omitted_like_stata(data):
    """A column that duplicates earlier ones gets b = 0 and a missing t."""
    x, y = data
    x_dup = np.column_stack([x, x[:, 1] - x[:, 2]])
    fit = ols(x_dup, y)
    base = ols(x, y)
    assert fit.beta[-1] == 0.0
    assert np.isnan(fit.tstat[-1])
    np.testing.assert_allclose(fit.beta[:-1], base.beta, rtol=1e-10)
    np.testing.assert_allclose(fit.tstat[:-1], base.tstat, rtol=1e-10)


def test_the_later_of_two_collinear_columns_is_the_one_dropped(data):
    x, y = data
    x_dup = np.column_stack([x[:, :2], 2.0 * x[:, 1], x[:, 2:]])
    fit = ols(x_dup, y)
    assert fit.beta[2] == 0.0
    assert np.isnan(fit.tstat[2])
    assert np.isfinite(np.delete(fit.tstat, 2)).all()


def test_column_collinear_with_constant_is_dropped(data):
    x, y = data
    fit = ols(np.column_stack([x, np.full(50, 3.0)]), y)
    assert fit.beta[-1] == 0.0
    assert np.isnan(fit.tstat[-1])
    assert np.isfinite(fit.tstat[0])


def test_zero_column_is_dropped(data):
    x, y = data
    fit = ols(np.column_stack([x, np.zeros(50)]), y)
    assert np.isnan(fit.tstat[-1])
    assert np.isfinite(fit.tstat[:-1]).all()


def test_perfect_fit_gives_missing_t_ratios():
    x = np.column_stack([np.ones(10), np.arange(10.0)])
    fit = ols(x, np.zeros(10))
    np.testing.assert_array_equal(fit.beta, [0.0, 0.0])
    assert np.isnan(fit.tstat).all()


def test_no_residual_degrees_of_freedom_gives_missing_t_ratios():
    rng = np.random.default_rng(2)
    for nrow, ncol in [(4, 4), (3, 5)]:
        fit = ols(rng.normal(size=(nrow, ncol)), rng.normal(size=nrow))
        assert np.isnan(fit.tstat).all()
