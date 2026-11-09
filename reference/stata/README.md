# Stata reference material

Kept so that the Python port can be audited against the implementation it
reproduces. Nothing in this folder is installed with the package.

| File | What it is |
| --- | --- |
| `lsmlm.ado` | The Stata command, v1.1.0 (20jul2026), by Ibrahim Ongoren. The Python code follows it step by step. |
| `lsmlm_license.txt` | The Stata package's licence (MIT) and third-party notices, kept unmodified alongside its files. |
| `lsmlm_oecd15_analysis.do` | The worked application; `examples/oecd15_analysis.py` is its Python counterpart. |
| `outputs/lsmlm_oecd15_analysis.log` | Log of the author's Stata run (QUICK mode, 22 Jul 2026). The tests compare the printed result tables against it character for character. |
| `outputs/*.dta` | Result datasets written by that run. They hold the `r()` results at full double precision. |
| `build_reference.py` | Converts the `.dta` files into `tests/data/stata_reference.json`, which the test suite reads. |

To add reference cases, run the Stata command with new options, save the
`r()` results to a `.dta` file, and extend `build_reference.py`.
