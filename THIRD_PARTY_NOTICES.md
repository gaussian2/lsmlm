# Third-party notices

The `lsmlm` Python package is released under the MIT License (see `LICENSE`).
It includes or is derived from the third-party material listed below. Each
item remains under its own licence, and the notices here must be kept with any
copy or substantial portion of the package.

## 1. lsmlm (Stata command)

- **Used for:** the test procedure in `src/lsmlm/core.py`, the results table in
  `src/lsmlm/results.py` and the worked example in `examples/`, which are
  ported from `lsmlm.ado` and its do-files (v1.1.0). Unmodified copies of the
  Stata sources and their output are kept in `reference/stata/`.
- **Source:** Statistical Software Components S459809
- **Licence:** MIT

```text
MIT License

Copyright (c) 2026 Ibrahim Ongoren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. leestra (Stata package)

- **Used for:** the critical-value tables and interpolation routines in
  `src/lsmlm/critical_values.py`, taken via `lsmlm.ado`.
- **Source:** Statistical Software Components S459688
  (`ssc describe leestra` shows the original notice)
- **Licence:** MIT

```text
Copyright (c) H. Ozan Eruygur (leestra, SSC s459688)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. Maddison Project Database 2023

- **Used for:** the example dataset `src/lsmlm/data/lsmlm_oecd15.csv`.
- **Changes made:** restricted to 15 OECD countries and 1870-1994; relative
  income (`rel_gdppc`) and its logarithm (`y`) were added; converted from
  Stata to CSV format.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  <https://creativecommons.org/licenses/by/4.0/>
- **Required attribution:** Bolt, Jutta and Jan Luiten van Zanden (2024),
  "Maddison style estimates of the evolution of the world economy: A new 2023
  update", Journal of Economic Surveys, 1-41.
