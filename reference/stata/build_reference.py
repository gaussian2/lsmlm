"""Rebuild tests/data/stata_reference.json from the Stata output datasets.

The Stata do-file ``lsmlm_oecd15_analysis.do`` (QUICK run: France, United
Kingdom, United States; maxlag 2, mindist 5, trim 0.10) saved its results in
the ``.dta`` files under ``outputs/``. Those files hold the r() results at
full double precision, which makes them a much sharper reference than the
rounded numbers in the log. This script converts them to JSON so the test
suite does not need pandas to read Stata files.

Usage (from the repository root)::

    python reference/stata/build_reference.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
TARGET = HERE.parents[1] / "tests" / "data" / "stata_reference.json"

# Settings of the QUICK run, taken from lsmlm_oecd15_analysis.do.
MAXLAG, MAXLAG_SUB, MINDIST, TRIM = 2, 2, 5, 0.10

# Stata column -> LSMLMResult attribute.
RENAME = {
    "rej1": "reject1",
    "rej5": "reject5",
    "rej10": "reject10",
    "b1sig": "break1_sig10",
    "b2sig": "break2_sig10",
}
SKIP = {"country", "decision", "sig", "sample", "startyr", "endyr"}


def expected(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in row.items():
        if name in SKIP:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        key = RENAME.get(str(name), str(name))
        if key.startswith(("reject", "break")):
            out[key] = bool(value)
        elif key in {"N", "k", "tb1", "tb2"}:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def main() -> None:
    cases: list[dict[str, Any]] = []

    for _, row in pd.read_stata(OUTPUTS / "table1_two_break.dta").iterrows():
        cases.append(
            {
                "id": f"table1-{row['country']}",
                "source": "table1_two_break.dta",
                "country": row["country"],
                "start": 1870,
                "end": 1994,
                "options": {
                    "breaks": 2,
                    "model": "break",
                    "maxlag": MAXLAG,
                    "mindist": MINDIST,
                    "trim": TRIM,
                },
                "expected": expected(row),
            }
        )

    for _, row in pd.read_stata(OUTPUTS / "table2_one_break.dta").iterrows():
        cases.append(
            {
                "id": f"table2-{row['country']}",
                "source": "table2_one_break.dta",
                "country": row["country"],
                "start": 1870,
                "end": 1994,
                "options": {"breaks": 1, "model": "break", "maxlag": MAXLAG, "trim": TRIM},
                "expected": expected(row),
            }
        )

    for _, row in pd.read_stata(OUTPUTS / "table3_subsamples.dta").iterrows():
        if row["sample"] == "full":
            continue  # identical to Table 1
        breaks = 1 if row["sample"] == "post-war" else 2
        cases.append(
            {
                "id": f"table3-{row['sample']}-{row['country']}",
                "source": "table3_subsamples.dta",
                "country": row["country"],
                "start": int(row["startyr"]),
                "end": int(row["endyr"]),
                "options": {
                    "breaks": breaks,
                    "model": "break",
                    "maxlag": MAXLAG_SUB,
                    "mindist": MINDIST,
                    "trim": TRIM,
                },
                "expected": expected(row),
            }
        )

    piecewise = [
        {k: (v if isinstance(v, str) else float(v)) for k, v in row.items()}
        for _, row in pd.read_stata(OUTPUTS / "piecewise_trends.dta").iterrows()
    ]
    descriptives = [
        {k: (v if isinstance(v, str) else float(v)) for k, v in row.items()}
        for _, row in pd.read_stata(OUTPUTS / "desc_stats.dta").iterrows()
    ]

    payload = {
        "stata_command": "lsmlm 1.1.0 20jul2026",
        "stata_run": "lsmlm_oecd15_analysis.do, QUICK = 1, 22 Jul 2026",
        "cases": cases,
        "piecewise_trends": piecewise,
        "descriptives": descriptives,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {TARGET} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
