# lsmlm

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-gaussian2.github.io%2Flsmlm-informational)](https://gaussian2.github.io/lsmlm/)

Lee-Strazicich minimum LM unit root test with zero, one, or two endogenous
structural breaks, for Python.

Documentation: <https://gaussian2.github.io/lsmlm/>

This is a port of the Stata command `lsmlm` (v1.1.0) by Ibrahim Ongoren,
Pamukkale University. It follows the Stata code step by step and reproduces its
results: on every stored result of the Stata reference run the two agree to
about 1e-13, and the printed results table is identical character for
character. See [VALIDATION.md](VALIDATION.md).

The null hypothesis is a unit root. Rejecting it is evidence that the series is
(trend-)stationary around the estimated break(s). Two deterministic models are
available:

- `model="break"`: a level and trend shift at each break (Lee-Strazicich Model C);
- `model="crash"`: a level shift only at each break (Model A).

With `breaks=0` the function reports the Schmidt-Phillips no-break LM test.

## Install

Requires Python 3.10 or later and NumPy.

```bash
pip install "git+https://github.com/gaussian2/lsmlm.git"
```

With the optional extras:

```bash
# + pandas, for the bundled example dataset
pip install "lsmlm[data] @ git+https://github.com/gaussian2/lsmlm.git"
# + matplotlib and openpyxl, for the worked example
pip install "lsmlm[examples] @ git+https://github.com/gaussian2/lsmlm.git"
```

From a clone of the repository, `pip install .` (or `pip install ".[examples]"`)
does the same.

## Quick start

```python
from lsmlm import lsmlm
from lsmlm.datasets import load_oecd15

panel = load_oecd15()  # 15 OECD countries, 1870-1994
france = panel[panel["country"] == "France"]

result = lsmlm(france["y"], france["year"], breaks=2, maxlag=2, mindist=5)
```

```text
------------------------------------------------------------------
  Lee-Strazicich minimum LM unit root test
------------------------------------------------------------------
  Series:        y
  Model:         trend break (level + trend shift)
  Breaks:        2   (endogenously selected)
  Max lag:       2    Lags chosen (GTOS): 1
  Obs (regr.):   123
  Sample:        1870 to 1994
  Break 1:       1939   (lambda1 = 0.545)
  Break 2:       1947   (lambda2 = 0.610)
------------------------------------------------------------------
  Test statistic (t on S(t-1)):    -7.2219
  Critical values:   1%  -6.108    5%  -5.418    10%  -5.162
  Decision:      Reject unit root at 1%
------------------------------------------------------------------
  Break significance (|t| on dummies >= 1.645):
    Break 1:     significant
    Break 2:     significant
------------------------------------------------------------------
```

Everything the Stata command returns in `r()` is an attribute of the result:

```python
result.tau  # test statistic (t on S(t-1))
result.k  # augmenting lags chosen
result.tb1, result.tb2  # break dates, in the units of the time variable
result.cv1, result.cv5, result.cv10
result.reject1, result.reject5, result.reject10
result.siglevel  # "1%", "5%", "10%" or "Not significant"
result.to_dict()  # all of it, e.g. as a row for a DataFrame
print(result)  # the table above
```

### Options

| Argument | Stata option | Meaning | Default |
| --- | --- | --- | --- |
| `y` | *varname* | the series (missing values are dropped) | required |
| `t` | `timevar()` / `tsset` | time variable; the data are sorted by it | positions `1..T` |
| `breaks` | `breaks()` | number of breaks: 0, 1 or 2 | `2` |
| `model` | `model()` | `"break"` or `"crash"` | `"break"` |
| `maxlag` | `maxlag()` | largest augmenting lag for general-to-specific selection | `floor(4*(T/100)^0.25)`, at least 1 |
| `trim` | `trim()` | fraction trimmed at each end of the break search | `0.10` |
| `mindist` | `mindist()` | minimum observations between the two breaks | `2` |
| `varname` | | label used in the printed table | `"y"` |
| `print_results` | `noprint` | print the table | `True` |

Stata's `if`/`in` corresponds to subsetting the arrays before the call.

### Panels

`lsmlm` tests one series at a time, like the Stata command. For a panel, loop
over the units and pass `print_results=False`:

```python
import pandas as pd

rows = []
for country, g in panel.groupby("country"):
    r = lsmlm(g["y"], g["year"], breaks=2, maxlag=8, mindist=5, print_results=False)
    rows.append({"country": country, **r.to_dict()})
table = pd.DataFrame(rows)
```

Passing several units at once raises an error, because the time variable is
then not unique.

## Method

For every candidate break date (or pair of dates) in the trimmed sample:

1. regress `D.y` on the differenced deterministic terms to recover the shift
   magnitudes, and build the LM-detrended series `S(t)`;
2. regress `D.y` on those terms, `S(t-1)` and lagged `D.S`, choosing the number
   of lags general-to-specific (start at `maxlag`, keep the highest lag with
   `|t| >= 1.645`).

The break date(s) minimising the t-ratio on `S(t-1)` are selected, and that
minimum is the test statistic. Critical values are interpolated over the sample
size and, for the trend-break model, depend on the break fraction(s). The
tables and the interpolation scheme come from the `leestra` Stata package
(H. Ozan Eruygur, SSC s459688).

The search is a grid search, so run time grows with the sample length and, for
two breaks, roughly with its square. On a 125-year series a two-break test with
`maxlag=8` takes about two seconds.

## Examples

- [`examples/quickstart.py`](examples/quickstart.py): every model variant on a
  simulated series, then a panel loop.
- [`examples/oecd15_analysis.py`](examples/oecd15_analysis.py): the worked
  application shipped with the Stata package. It asks whether per-capita
  incomes are converging among OECD countries, and writes descriptive
  statistics, three results tables, piecewise-trend fits, one figure per
  country, two summary figures and an Excel workbook.

  ```bash
  python examples/oecd15_analysis.py           # 3 countries, maxlag 2 (about 5 s)
  python examples/oecd15_analysis.py --full    # 15 countries, maxlag 8 (under 1 min)
  ```

  The default run uses the settings of the outputs shipped with the Stata
  package and reproduces them.

## Development

```bash
pip install -e ".[dev]"
pytest                        # 136 tests, under a minute
pytest --cov=lsmlm            # with coverage
ruff check . && ruff format --check .
mypy
python -m build               # sdist and wheel in dist/
```

Layout:

```text
src/lsmlm/
  core.py              lsmlm(): input handling, break search, lag selection
  critical_values.py   critical-value tables and interpolation
  results.py           LSMLMResult and LSMLMError
  datasets.py          load_oecd15()
  _ols.py              OLS that treats collinearity the way Stata's regress does
  data/                bundled example dataset
tests/                 unit tests and Stata parity tests
examples/              runnable scripts
docs/                  the documentation site (GitHub Pages)
reference/stata/       the Stata command, its log and result files (not installed)
```

## References

- Lee, J., and M. C. Strazicich. 2003. Minimum Lagrange multiplier unit root
  test with two structural breaks. *Review of Economics and Statistics* 85(4):
  1082-1089.
- Lee, J., and M. C. Strazicich. 2013. Minimum LM unit root test with one
  structural break. *Economics Bulletin* 33(4): 2483-2492.
- Schmidt, P., and P. C. B. Phillips. 1992. LM tests for a unit root in the
  presence of deterministic trends. *Oxford Bulletin of Economics and
  Statistics* 54(3): 257-287.
- Strazicich, M. C., J. Lee, and E. Day. 2004. Are incomes converging among
  OECD countries? Time series evidence with two structural breaks. *Journal of
  Macroeconomics* 26: 131-145.

## Credits and licence

MIT License; see [LICENSE](LICENSE). Notices for the third-party material the
package includes or derives from are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- The Python package is by Junsoo Lee (University of Alabama).
- The Stata command `lsmlm`, which this package ports, was written by Ibrahim
  Ongoren (Pamukkale University).
- The critical-value tables and interpolation routines come from the `leestra`
  Stata package by H. Ozan Eruygur (SSC s459688).
- The example dataset is derived from the Maddison Project Database 2023, which
  is licensed under CC BY 4.0. When using it, please cite: Bolt, J., and
  J. L. van Zanden. 2024. Maddison style estimates of the evolution of the world
  economy: A new 2023 update. *Journal of Economic Surveys* 1-41.
