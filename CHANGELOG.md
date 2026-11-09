# Changelog

## 1.2.0

Repackaging of the first Python port (1.1.0) as an installable, tested
package. Results for valid inputs are unchanged: against the original port the
largest difference over 180 runs (all 15 countries in eight configurations,
plus 60 simulated series) is 2e-14.

### Added
- `src/` layout, `pyproject.toml` with optional extras, type hints and a
  `py.typed` marker, licence file.
- Test suite, including regression tests against the Stata command's stored
  results and printed output.
- `lsmlm.datasets.load_oecd15()`; the example dataset now ships inside the
  package.
- `LSMLMResult.to_dict()`, `str(result)`, and a public
  `lsmlm.critical_values()` function.
- `t` may be a datetime-like array. Break dates are returned in the type of
  the time variable (`1939`, not `1939.0`).
- `maxlag=None` and `mindist=None` select the defaults (`-1` still works).
- Command-line options for the worked example (`--full`, `--outdir`).
- Continuous integration workflow, lint and type-check configuration.

### Changed
- NumPy is the only required dependency; pandas, matplotlib and openpyxl are
  optional extras.
- Python 3.10 or later is required.
- `LSMLMResult` is immutable.
- The modules were renamed: `lsmlm.cv_tables` is now `lsmlm.critical_values`.
  `from lsmlm import lsmlm, LSMLMResult, LSMLMError` is unchanged.

### Fixed (cases where the first port departed from the Stata command)
- `breaks=2, model="break", mindist=0` crashed with `ZeroDivisionError`. When
  the two breaks are adjacent the dummies are exactly collinear; Stata omits
  the redundant one and continues, and the port now does the same.
- A time variable with repeated values (for example two panel units passed at
  once) was accepted and tested as if it were one series. It is now an error,
  as in Stata.
- Observations with a missing time value are dropped, as in Stata.
- Non-numeric, multi-dimensional or infinite input raises `LSMLMError` with a
  clear message.
- Worked example: `mean_gdppc` was computed in single precision and differed
  from Stata from the seventh digit; break-year labels overlapped the axis
  labels in the country figures.
