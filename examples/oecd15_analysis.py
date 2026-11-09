#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Junsoo Lee
# Portions derived from third-party MIT-licensed code; see THIRD_PARTY_NOTICES.md.
"""Worked application: are incomes converging among OECD countries.

Python counterpart of the Stata do-file ``lsmlm_oecd15_analysis.do``. It applies
the Lee-Strazicich minimum LM unit root test to the example dataset shipped
with the package: log relative per-capita income for 15 OECD countries,
1870-1994 (the question studied by Strazicich, Lee and Day 2004).

Economic question
-----------------
``y = log(country GDP per capita / OECD15 average GDP per capita)``.
If ``y`` is stationary (unit root rejected), the country's relative income
reverts to a fixed level: convergence. If ``y`` has a unit root, shocks to
relative income are permanent: no convergence. Allowing for structural breaks
matters, because ignoring a real break biases standard tests toward failing to
reject the unit root.

What this script produces (in ``--outdir``, default ``./outputs``)
----------------------------------------------------------------
* ``desc_stats.csv``          descriptive statistics
* ``table1_two_break.csv``    two-break LM test, full sample
* ``table2_one_break.csv``    one-break LM test, full sample
* ``table3_subsamples.csv``   LM test on sub-samples
* ``piecewise_trends.csv``    piecewise trend models at the Table 1 breaks
* ``figs/``                   one figure per country and two summary figures
* ``lsmlm_oecd15_results.xlsx``  every table in one workbook

Usage
-----
::

    pip install "lsmlm[examples]"
    python examples/oecd15_analysis.py            # quick run: 3 countries, maxlag 2
    python examples/oecd15_analysis.py --full     # 15 countries, maxlag 8

The quick run uses the settings of the outputs shipped with the Stata package
(France, United Kingdom, United States) and reproduces them exactly.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lsmlm import LSMLMError, lsmlm
from lsmlm.datasets import load_oecd15

QUICK_COUNTRIES = ["France", "United Kingdom", "United States"]

C_DATA = "#1A5276"  # deep blue
C_FIT = "#B03A2E"  # muted red
C_GRID = "#D9D9D9"
C_SUB = "#595959"


@dataclass(frozen=True)
class Settings:
    """Econometric settings for annual data."""

    maxlag: int = 8  # maximum augmenting lags, full sample
    maxlag_sub: int = 4  # shorter sub-samples need fewer lags
    mindist: int = 5  # minimum number of years between the two breaks
    trim: float = 0.10  # fraction trimmed at each end of the break search


QUICK = Settings(maxlag=2, maxlag_sub=2)
FULL = Settings()


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def significance(table: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [table["rej1"] == 1, table["rej5"] == 1, table["rej10"] == 1],
            ["1%", "5%", "10%"],
            default="ns",
        ),
        index=table.index,
    )


def decision(table: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(table["rej10"] == 1, "convergence", "no convergence"), index=table.index
    )


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "country": country,
            "N": len(g),
            "mean_gdppc": g["gdppc"].mean(),
            "mean_rel": g["rel_gdppc"].mean(),
            "mean_y": g["y"].mean(),
            "sd_y": g["y"].std(),
            "min_y": g["y"].min(),
            "max_y": g["y"].max(),
        }
        for country, g in df.groupby("country", sort=False)
    ]
    return pd.DataFrame(rows)


def single_country_walkthrough(df: pd.DataFrame, s: Settings, country: str = "France") -> None:
    """Run every model variant on one country so each output is visible."""
    g = df[df["country"] == country]
    y, year = g["y"].to_numpy(), g["year"].to_numpy()

    print("\n>>> (a) No break: Schmidt-Phillips LM test")
    lsmlm(y, year, breaks=0, maxlag=s.maxlag)

    print("\n>>> (b) One break, crash model (level shift)")
    lsmlm(y, year, breaks=1, model="crash", maxlag=s.maxlag, trim=s.trim)

    print("\n>>> (c) One break, trend-break model (level + trend shift)")
    lsmlm(y, year, breaks=1, model="break", maxlag=s.maxlag, trim=s.trim)

    print("\n>>> (d) Two breaks, trend-break model  [the main specification]")
    r = lsmlm(y, year, breaks=2, model="break", maxlag=s.maxlag, mindist=s.mindist, trim=s.trim)

    print("\n>>> Stored results from the last run:")
    print(f"   tau       = {r.tau:.8g}")
    print(f"   lags (k)  = {r.k}")
    print(f"   break 1   = {r.tb1}   lambda1 = {r.lambda1:.8g}")
    print(f"   break 2   = {r.tb2}   lambda2 = {r.lambda2:.8g}")
    print(f"   5% CV     = {r.cv5:.8g}")
    print(f"   decision  = {r.siglevel}")

    print("\n>>> (e) Post-war sub-sample only")
    post = year >= 1946
    lsmlm(y[post], year[post], breaks=1, model="break", maxlag=s.maxlag_sub, trim=s.trim)


def two_break_table(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """TABLE 1: two-break LM test, full sample."""
    rows = []
    for country, g in df.groupby("country", sort=False):
        print(f"  running: {country}", end="", flush=True)
        try:
            r = lsmlm(
                g["y"],
                g["year"],
                breaks=2,
                model="break",
                maxlag=s.maxlag,
                mindist=s.mindist,
                trim=s.trim,
                print_results=False,
            )
        except LSMLMError as exc:
            print(f"  FAILED ({exc})")
            rows.append({"country": country})
            continue
        print("  done")
        rows.append(
            {
                "country": country,
                "N": r.N,
                "k": r.k,
                "tb1": r.tb1,
                "tb2": r.tb2,
                "lambda1": r.lambda1,
                "lambda2": r.lambda2,
                "tau": r.tau,
                "cv1": r.cv1,
                "cv5": r.cv5,
                "cv10": r.cv10,
                "rej1": int(r.reject1),
                "rej5": int(r.reject5),
                "rej10": int(r.reject10),
                "b1sig": int(bool(r.break1_sig10)),
                "b2sig": int(bool(r.break2_sig10)),
                "tB1": r.tB1,
                "tD1": r.tD1,
                "tB2": r.tB2,
                "tD2": r.tD2,
            }
        )
    table = pd.DataFrame(rows)
    table["decision"] = decision(table)
    table["sig"] = significance(table)
    return table


def one_break_table(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """TABLE 2: one-break LM test, full sample.

    A robustness check, and the preferred specification when the second break
    is not statistically relevant in Table 1.
    """
    rows = []
    for country, g in df.groupby("country", sort=False):
        print(f"  running: {country}", end="", flush=True)
        try:
            r = lsmlm(
                g["y"],
                g["year"],
                breaks=1,
                model="break",
                maxlag=s.maxlag,
                trim=s.trim,
                print_results=False,
            )
        except LSMLMError as exc:
            print(f"  FAILED ({exc})")
            rows.append({"country": country})
            continue
        print("  done")
        rows.append(
            {
                "country": country,
                "N": r.N,
                "k": r.k,
                "tb1": r.tb1,
                "lambda1": r.lambda1,
                "tau": r.tau,
                "cv1": r.cv1,
                "cv5": r.cv5,
                "cv10": r.cv10,
                "rej1": int(r.reject1),
                "rej5": int(r.reject5),
                "rej10": int(r.reject10),
                "b1sig": int(bool(r.break1_sig10)),
                "tB1": r.tB1,
                "tD1": r.tD1,
            }
        )
    table = pd.DataFrame(rows)
    table["decision"] = decision(table)
    table["sig"] = significance(table)
    return table


def subsample_table(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """TABLE 3: the test on sub-samples.

    The world wars dominate the early part of the sample; splitting it shows
    whether the convergence evidence is driven by the war period.
    """
    # label, first year, last year, number of breaks, maxlag
    subsamples = [
        ("full", 1870, 1994, 2, s.maxlag),
        ("pre-war", 1870, 1945, 2, s.maxlag_sub),
        ("post-war", 1946, 1994, 1, s.maxlag_sub),
    ]
    rows = []
    for label, first, last, n_breaks, maxlag in subsamples:
        print(f"\n  --- sub-sample: {label} ({first}-{last}), breaks({n_breaks}), maxlag({maxlag})")
        for country, g in df.groupby("country", sort=False):
            g = g[g["year"].between(first, last)]
            row = {"country": country, "sample": label, "startyr": first, "endyr": last}
            try:
                r = lsmlm(
                    g["y"],
                    g["year"],
                    breaks=n_breaks,
                    model="break",
                    maxlag=maxlag,
                    mindist=s.mindist,
                    trim=s.trim,
                    print_results=False,
                )
            except LSMLMError as exc:
                print(f"     {country}: FAILED ({exc})")
                rows.append(row)
                continue
            rows.append(
                {
                    **row,
                    "N": r.N,
                    "k": r.k,
                    "tb1": r.tb1,
                    "tb2": r.tb2,
                    "tau": r.tau,
                    "cv1": r.cv1,
                    "cv5": r.cv5,
                    "cv10": r.cv10,
                    "rej10": int(r.reject10),
                }
            )
    table = pd.DataFrame(rows)
    table["decision"] = decision(table)
    return table


def piecewise_design(year: np.ndarray, breaks: Sequence[float]) -> np.ndarray:
    """Regressors of ``y = a + b t + sum_j (d_j D_j + g_j T_j)``."""
    cols = [np.ones(len(year)), year - year[0] + 1.0]
    for b in breaks:
        after = year > b
        cols += [after.astype(float), (year - b) * after]
    return np.column_stack(cols)


def piecewise_fit(y: np.ndarray, year: np.ndarray, breaks: Sequence[float]):
    x = piecewise_design(year, breaks)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    r2 = 1.0 - float(((y - fitted) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return beta, r2, fitted


def piecewise_table(df: pd.DataFrame, table1: pd.DataFrame) -> pd.DataFrame:
    """Deterministic trend with level and slope shifts at the Table 1 breaks."""
    rows = []
    for _, t1 in table1.iterrows():
        country = t1["country"]
        if pd.isna(t1.get("tb1")) or pd.isna(t1.get("tb2")):
            print(f"  {country}: no break dates in Table 1, skipped")
            continue
        g = df[df["country"] == country]
        y, year = g["y"].to_numpy(float), g["year"].to_numpy(float)
        beta, r2, _ = piecewise_fit(y, year, [t1["tb1"], t1["tb2"]])
        rows.append(
            {
                "country": country,
                "tb1": int(t1["tb1"]),
                "tb2": int(t1["tb2"]),
                "b_t": beta[1],
                "b_D1": beta[2],
                "b_T1": beta[3],
                "b_D2": beta[4],
                "b_T2": beta[5],
                "r2": r2,
                "N": len(y),
            }
        )
        print(f"  {country}: R2 = {r2:5.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _style_axes(ax, grid_axis: str) -> None:
    ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")


def _titles(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, fontsize=14, loc="left", color="black", fontweight="bold", pad=24)
    ax.text(
        0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=C_SUB, ha="left", va="bottom"
    )


def country_figure(g: pd.DataFrame, t1: pd.Series, path: Path) -> None:
    import matplotlib.pyplot as plt

    y, year = g["y"].to_numpy(float), g["year"].to_numpy(float)
    breaks = [float(t1[c]) for c in ("tb1", "tb2") if not pd.isna(t1.get(c))]
    _, _, fitted = piecewise_fit(y, year, breaks)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(year, y, color=C_DATA, linewidth=2.0, label="Observed")
    ax.plot(
        year,
        fitted,
        color=C_FIT,
        linewidth=1.8,
        linestyle="--",
        label="Piecewise trend at estimated breaks",
    )
    ax.axhline(0, color="#B0B0B0", linewidth=0.8)
    for i, b in enumerate(breaks):
        ax.axvline(b, color=C_FIT, linewidth=1.0, linestyle=(0, (4, 2)))
        # label the break next to its line; first one to the left, second to the right
        left = i == 0 and len(breaks) > 1
        ax.annotate(
            f"{b:.0f}",
            xy=(b, 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(-4 if left else 4, 0),
            textcoords="offset points",
            ha="right" if left else "left",
            va="bottom",
            fontsize=8,
            color=C_FIT,
        )

    ax.set_xlim(year.min() - 2, year.max() + 2)
    ax.set_xticks(range(1870, 1991, 20))
    _style_axes(ax, "y")
    ax.set_ylabel("Log relative per-capita income", fontsize=9)
    _titles(
        ax,
        t1["country"],
        f"Two-break LM: τ = {t1['tau']:.3f}, 5% CV = {t1['cv5']:.3f}  -  {t1['decision']}",
    )
    label = " and ".join(f"{b:.0f}" for b in breaks)
    fig.text(
        0.01,
        0.01,
        f"Vertical lines: estimated breaks ({label}). "
        "Horizontal line: OECD15 average (y = 0).\nData: Maddison Project Database 2023.",
        fontsize=7,
        color="#8C8C8C",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def summary_figures(table1: pd.DataFrame, figs: Path) -> None:
    import matplotlib.pyplot as plt

    height = max(4.0, 1.6 + 0.32 * len(table1))

    t = table1.dropna(subset=["tau"]).sort_values("tau", ascending=False).reset_index(drop=True)
    if len(t) >= 2:
        pos = np.arange(1, len(t) + 1)
        fig, ax = plt.subplots(figsize=(8, height))
        ax.hlines(pos, t["tau"], t["cv5"], color="#BFBFBF", linewidth=2)
        ax.scatter(t["tau"], pos, s=70, color=C_DATA, zorder=3, label="τ (test statistic)")
        ax.scatter(
            t["cv5"],
            pos,
            s=90,
            marker="|",
            color=C_FIT,
            linewidth=2.5,
            zorder=3,
            label="5% critical value",
        )
        ax.set_yticks(pos)
        ax.set_yticklabels(t["country"], fontsize=9)
        ax.set_ylim(len(t) + 0.6, 0.4)
        _style_axes(ax, "x")
        ax.set_xlabel("Test statistic", fontsize=9)
        _titles(
            ax,
            "Two-break LM test results",
            "Unit root is rejected when τ lies to the left of the 5% critical value",
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.85 / height),
            ncol=2,
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(figs / "fig_summary_test_results.png", dpi=300)
        plt.close(fig)
        print("  saved: figs/fig_summary_test_results.png")

    t = table1.dropna(subset=["tb1"]).sort_values("tb1").reset_index(drop=True)
    if len(t) >= 2:
        pos = np.arange(1, len(t) + 1)
        fig, ax = plt.subplots(figsize=(8, height))
        ax.hlines(pos, t["tb1"], t["tb2"], color="#BFBFBF", linewidth=2)
        ax.scatter(t["tb1"], pos, s=70, color=C_DATA, zorder=3, label="First break")
        ax.scatter(t["tb2"], pos, s=70, color=C_FIT, marker="D", zorder=3, label="Second break")
        ax.set_yticks(pos)
        ax.set_yticklabels(t["country"], fontsize=9)
        ax.set_ylim(len(t) + 0.6, 0.4)
        ax.set_xlim(1868, 1996)
        ax.set_xticks(range(1870, 1991, 20))
        _style_axes(ax, "x")
        _titles(
            ax,
            "Timing of the estimated structural breaks",
            "Each line spans a country's two estimated break dates",
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.45 / height),
            ncol=2,
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(figs / "fig_summary_break_timing.png", dpi=300)
        plt.close(fig)
        print("  saved: figs/fig_summary_break_timing.png")


def make_figures(df: pd.DataFrame, table1: pd.DataFrame, figs: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"}
    ):
        for _, t1 in table1.iterrows():
            country = t1["country"]
            if pd.isna(t1.get("tb1")):
                print(f"  {country}: no break dates, figure skipped")
                continue
            name = f"fig_{country.replace(' ', '_')}.png"
            country_figure(df[df["country"] == country], t1, figs / name)
            print(f"  saved: figs/{name}")
        summary_figures(table1, figs)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(outdir: Path, full: bool = False, figures: bool = True, excel: bool = True) -> dict:
    """Run the whole analysis and return the tables as DataFrames."""
    settings = FULL if full else QUICK
    outdir.mkdir(parents=True, exist_ok=True)

    banner("DATA: OECD15 relative per-capita income, 1870-1994")
    df = load_oecd15()
    if not full:
        print("\n*** QUICK MODE: 3 countries, maxlag(2). Use --full for the full run. ***")
        df = df[df["country"].isin(QUICK_COUNTRIES)]
    print(f"\nCountries in this run: {df['country'].nunique()}")

    banner("SECTION 2: Descriptive statistics")
    desc = descriptive_statistics(df)
    print(desc.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    banner("SECTION 3: Single-country walkthrough (France)")
    single_country_walkthrough(df, settings)

    banner("SECTION 4: TABLE 1 - two-break LM test, 1870-1994")
    table1 = two_break_table(df, settings)
    print("\nTABLE 1: Two-break minimum LM unit root test, 1870-1994")
    print(
        table1[["country", "N", "k", "tb1", "tb2", "tau", "cv5", "sig", "decision"]].to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"
        )
    )
    print(
        f"\n  Countries rejecting the unit root at 10%: "
        f"{int((table1['rej10'] == 1).sum())} of {len(table1)}"
    )

    banner("SECTION 5: TABLE 2 - one-break LM test, 1870-1994")
    table2 = one_break_table(df, settings)
    print("\nTABLE 2: One-break minimum LM unit root test, 1870-1994")
    print(
        table2[["country", "N", "k", "tb1", "tau", "cv5", "sig", "decision"]].to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"
        )
    )

    banner("SECTION 6: TABLE 3 - sub-sample analysis")
    table3 = subsample_table(df, settings)
    print("\nTABLE 3: LM test across sub-samples")
    shown = table3.sort_values(["country", "sample"]).copy()
    shown["tb2"] = shown["tb2"].map(lambda v: "." if pd.isna(v) else f"{v:.0f}")
    print(
        shown[["country", "sample", "N", "k", "tb1", "tb2", "tau", "cv10", "decision"]].to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"
        )
    )
    print("\n  Rejection counts by sub-sample (rejected at 10% / tested):")
    for label, sub in table3.groupby("sample", sort=True):
        print(f"    {label:<10s}  {int((sub['rej10'] == 1).sum())} / {sub['rej10'].notna().sum()}")

    banner("SECTION 7: Piecewise trend models at the Table 1 breaks")
    piecewise = piecewise_table(df, table1)
    print("\nPiecewise trend models (breaks taken from Table 1)")
    print(piecewise.to_string(index=False, float_format=lambda v: f"{v:9.5f}"))

    tables = {
        "desc_stats": desc,
        "table1_two_break": table1,
        "table2_one_break": table2,
        "table3_subsamples": table3,
        "piecewise_trends": piecewise,
    }
    for name, table in tables.items():
        table.to_csv(outdir / f"{name}.csv", index=False)

    if figures:
        banner("SECTION 8: Figures")
        make_figures(df, table1, outdir / "figs")
        print(f"\n  All figures written to: {outdir / 'figs'}")

    if excel:
        banner("SECTION 9: Excel export")
        sheets = [
            "Descriptives",
            "Table1_TwoBreak",
            "Table2_OneBreak",
            "Table3_Subsamples",
            "PiecewiseTrends",
        ]
        workbook = outdir / "lsmlm_oecd15_results.xlsx"
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            for sheet, table in zip(sheets, tables.values(), strict=True):
                table.to_excel(writer, sheet_name=sheet, index=False)
        print(f"  saved: {workbook}")

    banner("FINISHED")
    print(f"  Output folder: {outdir}")
    if not full:
        print("\n  NOTE: this was a QUICK run (3 countries, maxlag 2).")
        print("        Use --full for the full 15-country analysis.")
    return tables


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--full",
        action="store_true",
        help="all 15 countries with maxlag 8 (default: quick 3-country run)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs"),
        help="folder for tables and figures (default: ./outputs)",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip the figures")
    parser.add_argument("--no-excel", action="store_true", help="skip the Excel workbook")
    args = parser.parse_args(argv)
    run(args.outdir, full=args.full, figures=not args.no_figures, excel=not args.no_excel)


if __name__ == "__main__":
    main()
