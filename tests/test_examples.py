"""End-to-end: the worked example must reproduce the Stata package's output files."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

from conftest import STATA_TOL, load_reference

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
REFERENCE = load_reference()

pytestmark = pytest.mark.slow


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while it is being executed
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    pytest.importorskip("matplotlib")
    pytest.importorskip("openpyxl")
    outdir = tmp_path_factory.mktemp("oecd15")
    tables = _load("oecd15_analysis").run(outdir)
    return outdir, tables


def test_output_files_are_written(analysis):
    outdir, tables = analysis
    names = {p.relative_to(outdir).as_posix() for p in outdir.rglob("*") if p.is_file()}
    assert names == {
        "desc_stats.csv",
        "table1_two_break.csv",
        "table2_one_break.csv",
        "table3_subsamples.csv",
        "piecewise_trends.csv",
        "lsmlm_oecd15_results.xlsx",
        "figs/fig_France.png",
        "figs/fig_United_Kingdom.png",
        "figs/fig_United_States.png",
        "figs/fig_summary_test_results.png",
        "figs/fig_summary_break_timing.png",
    }
    assert all((outdir / n).stat().st_size > 0 for n in names)
    assert set(tables) == {
        "desc_stats",
        "table1_two_break",
        "table2_one_break",
        "table3_subsamples",
        "piecewise_trends",
    }


@pytest.mark.parametrize("row", REFERENCE["piecewise_trends"], ids=lambda r: r["country"])
def test_piecewise_trends_match_stata(analysis, row):
    table = analysis[1]["piecewise_trends"].set_index("country")
    got = table.loc[row["country"]]
    for name, want in row.items():
        if name != "country":
            assert got[name] == pytest.approx(want, rel=0, abs=STATA_TOL), name


def test_tables_match_stata(analysis):
    _, tables = analysis
    t1 = tables["table1_two_break"].set_index("country")
    t2 = tables["table2_one_break"].set_index("country")
    t3 = tables["table3_subsamples"].set_index(["sample", "country"])
    checked = 0
    for case in REFERENCE["cases"]:
        source, country = case["source"], case["country"]
        if source == "table1_two_break.dta":
            got = t1.loc[country]
        elif source == "table2_one_break.dta":
            got = t2.loc[country]
        else:
            sample = case["id"].split("-", 1)[1].rsplit("-" + country, 1)[0]
            got = t3.loc[(sample, country)]
        for name in ("N", "k", "tb1", "tb2", "tau", "cv1", "cv5", "cv10"):
            if name in case["expected"]:
                assert got[name] == pytest.approx(case["expected"][name], rel=0, abs=STATA_TOL)
                checked += 1
    assert checked == 90

    assert list(t1["decision"]) == ["convergence", "no convergence", "no convergence"]
    assert list(t1["sig"]) == ["1%", "ns", "ns"]
    assert list(t2["sig"]) == ["5%", "ns", "10%"]


def test_quickstart_runs(capsys):
    runpy.run_path(str(EXAMPLES / "quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert out.count("Lee-Strazicich minimum LM unit root test") == 7
    assert "reject unit root at 10%?" in out
