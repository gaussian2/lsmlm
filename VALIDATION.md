# Validation against the Stata command

The reference is the Stata package's own run of `lsmlm_oecd15_analysis.do`
(`lsmlm` 1.1.0, QUICK mode: France, United Kingdom and United States;
`maxlag(2)`, `mindist(5)`, `trim(0.10)`; 22 Jul 2026). Its log and result
datasets are in `reference/stata/outputs/`.

No Stata installation was available while building this package, so the
comparison is against those saved outputs, not against fresh Stata runs.

## What is compared

| Check | Reference | Agreement |
| --- | --- | --- |
| Stored results of 12 runs (Tables 1-3): `tau`, `k`, `N`, break dates, `lambda`, all critical values, rejection flags, dummy t-ratios | `.dta` files, full double precision | largest difference 1.3e-13 |
| Printed results table for the five Section 3 runs (no break, one-break crash, one-break trend-break, two-break trend-break, post-war sub-sample) | log file | identical, character for character |
| Piecewise-trend coefficients and R-squared | `piecewise_trends.dta` | within 1e-9 |
| Descriptive statistics | `desc_stats.dta` | within 1e-12 (relative) |
| Example dataset | `lsmlm_oecd15.dta` | bundled CSV is bit-for-bit equal |

Reproduce with:

```bash
pip install -e ".[dev]"
pytest tests/test_stata_parity.py tests/test_summary_output.py tests/test_examples.py -v
```

## Results

Two breaks, trend-break model, 1870-1994 (Table 1):

| Country | N | k | Breaks | Stata tau | Python tau | 5% CV |
| --- | --- | --- | --- | --- | --- | --- |
| France | 123 | 1 | 1939, 1947 | -7.221908096419 | -7.221908096419 | -5.418 |
| United Kingdom | 123 | 1 | 1938, 1948 | -4.680104157967 | -4.680104157967 | -5.418 |
| United States | 122 | 2 | 1927, 1947 | -4.518985750277 | -4.518985750277 | -5.625 |

One break, trend-break model, 1870-1994 (Table 2):

| Country | N | k | Break | Stata tau | Python tau | 5% CV |
| --- | --- | --- | --- | --- | --- | --- |
| France | 123 | 1 | 1946 | -4.556426018970 | -4.556426018970 | -4.162 |
| United Kingdom | 123 | 1 | 1947 | -3.665124915446 | -3.665124915446 | -4.161 |
| United States | 122 | 2 | 1905 | -3.841262749165 | -3.841262749165 | -4.098 |

Sub-samples (Table 3):

| Country | Sample | N | k | Break(s) | Stata tau | Python tau | 10% CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| France | 1870-1945 | 74 | 1 | 1921, 1936 | -5.109493774708 | -5.109493774708 | -5.779 |
| United Kingdom | 1870-1945 | 74 | 1 | 1915, 1926 | -4.997053452071 | -4.997053452071 | -5.890 |
| United States | 1870-1945 | 73 | 2 | 1877, 1930 | -4.949839339790 | -4.949839339790 | -5.541 |
| France | 1946-1994 | 46 | 2 | 1976 | -3.184075991177 | -3.184075991177 | -4.061 |
| United Kingdom | 1946-1994 | 48 | 0 | 1978 | -2.739383676273 | -2.739383676273 | -4.043 |
| United States | 1946-1994 | 48 | 0 | 1978 | -3.107704119431 | -3.107704119431 | -4.043 |

No-break and one-break crash runs for France (log, displayed precision):

| Model | Stata tau | Python tau | Stata CVs (1/5/10%) | Python CVs |
| --- | --- | --- | --- | --- |
| No break | -3.5815 | -3.5815 | -3.576, -3.024, -2.747 | same |
| One break, crash | -4.3506 | -4.3506 | -4.002, -3.402, -3.093 | same |

## What the Stata reference does not cover

- **Two-break crash model.** The reference run has no such case. The port's
  two-break crash results are checked against an independent brute-force
  implementation in `tests/test_search.py`, not against Stata.
- **Samples of 250 observations or more**, and one-break break fractions
  outside 0.26-0.61. The critical-value code is a direct transcription of the
  Mata functions and its tables were compared entry by entry, but only the
  cases above exercise it against Stata output.
- **`maxlag` above 2.** The lag search itself is exercised (lags 0, 1 and 2
  are all selected somewhere in the reference).

Running `lsmlm_validate.do` or the full 15-country analysis in Stata and adding
the results to `reference/stata/` would close these gaps.

## Parity notes

These are places where the port deliberately reproduces a convention of the
Stata command so that the numbers agree.

1. **Sample-size weighting of critical values.** `lsmlm_interp_T()` weights
   the *lower* tabulated sample size by `(N - T_lo)/(T_hi - T_lo)` and the
   upper one by the complement. Ordinary linear interpolation would swap those
   weights. As coded, N = 123 gives a value 85% of the way from the T = 100
   row to the T = 250 row (and N = 100 returns the T = 250 row exactly, N = 250
   the T = 500 row). Where critical values shrink in magnitude with T this
   makes them less strict than linear interpolation for N just above a
   tabulated size. For the two-break trend-break model at N = 123 the 5% value
   is -5.418; with the weights swapped it would be -6.114. This port
   matches Stata. Whether the convention is intended, and whether it is
   inherited from `leestra`, should be checked upstream.
2. **Break fractions** use the lag-adjusted convention
   `lambda = (b - (k + 2)) / (T - (k + 2) + 1)`, not `b / T`.
3. **Samples under 100 observations** use the T = 100 critical values without
   extrapolation.
4. **Collinear regressors and missing values.** Stata's `regress` omits a
   collinear regressor (coefficient 0, t-ratio missing), and Stata treats a
   missing value as larger than any number. The port mirrors both: an omitted
   dummy reports `nan`; a missing statistic is never selected as the minimum;
   a missing t-ratio on the last lag counts as significant. These rules were
   implemented from reading `lsmlm.ado` and Stata's documented behaviour. They
   only matter for degenerate inputs (adjacent breaks with `mindist=0`,
   constant series, more regressors than observations) and have not been
   confirmed against a Stata run.
5. **Time gaps are ignored.** Observations are numbered consecutively after
   dropping missing values, as in the Stata command, so a series with gaps is
   treated as if it were contiguous.
