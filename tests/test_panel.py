"""The panel on a made-up world: the transform, the coverage and gate G1 (b), the target, and the look-ahead test
of gate G1 (c), which must pass on code that uses nothing after the end of month t-1 and must catch a
characteristic planted with a value from month t. No licensed data is read."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import world  # noqa: E402
from mlrp import characteristics as K  # noqa: E402
from mlrp import constants as C  # noqa: E402
from mlrp import panel as P  # noqa: E402


@pytest.fixture(scope="module")
def w():
    return world.make()


@pytest.fixture(scope="module")
def full(w):
    return P.build(w["ulong"], w)


def test_the_rank_transform():
    months = pd.Series([pd.Period("2000-01", "M")] * 5 + [pd.Period("2000-02", "M")] * 2)
    values = pd.Series([3.0, 1.0, 2.0, 2.0, np.nan, 7.0, np.nan])
    mapped = P.rank_transform(values, months)
    # January: ranks 4, 1, 2.5, 2.5 of four, so (r - 1) / 3 mapped to -1 to 1.
    assert mapped.iloc[:4].tolist() == pytest.approx([1.0, -1.0, 0.0, 0.0])
    assert np.isnan(mapped.iloc[4]) and mapped.iloc[5] == 0.0 and np.isnan(mapped.iloc[6])


def test_the_transform_fills_with_the_median_and_counts(full):
    frame = full.assign(target=0.0, date_prev=pd.NaT)
    out, filled = P.transform(frame)
    for c in C.CHARACTERISTICS:
        assert filled[c] == int(frame[c].isna().sum())
        assert out[c].between(C.RANK_LOW, C.RANK_HIGH).all() and out[c].notna().all()
        assert (out.loc[frame[c].isna(), c] == C.FILL_VALUE).all()


def test_the_target_is_the_total_return_less_the_bill_rate():
    returns = pd.DataFrame({"month": pd.PeriodIndex(["2000-01", "2000-02"], freq="M"), "permno": [1, 1],
                            "ret_total": [0.05, np.nan]})
    rf = pd.Series([0.004, 0.005], index=pd.PeriodIndex(["2000-01", "2000-02"], freq="M"))
    out = P.target(returns, rf)
    assert out["target"].iloc[0] == pytest.approx(0.046) and np.isnan(out["target"].iloc[1])


def test_coverage_and_gate_g1b():
    months = pd.PeriodIndex(["1991-01"] * 10 + ["1991-02"] * 10, freq="M")
    frame = pd.DataFrame({c: np.ones(20) for c in C.CHARACTERISTICS})
    frame["month"] = months
    frame.loc[0, "momentum"] = np.nan                       # 0.9 in January: at the level, met
    coverage = P.coverage_by_month(frame)
    assert coverage.loc[pd.Period("1991-01", "M"), "momentum"] == pytest.approx(0.9)
    assert P.g1b(coverage)["g1b_met"]
    frame.loc[[10, 11, 12], "book_to_market"] = np.nan      # 0.7 in February, below 0.8
    verdict = P.g1b(P.coverage_by_month(frame))
    assert not verdict["g1b_met"]
    assert verdict["by_characteristic"]["book_to_market"] == {"level": 0.8, "lowest_share": 0.7,
                                                              "lowest_month": "1991-02", "months_below_level": 1}


def test_the_lookahead_months_are_drawn_from_the_seed():
    months = P.lookahead_months()
    assert len(months) == C.LOOKAHEAD_MONTHS == len(set(months))
    assert list(months) == sorted(months) and months.isin(P.panel_months()).all()
    assert list(P.lookahead_months()) == list(months)


def test_the_cut_removes_every_row_after_the_day(w):
    day = pd.Timestamp("1991-08-30")
    cut = P.cut_inputs(w, day)
    assert cut["monthly"]["date"].max() <= day and cut["daily"].index.max() <= day
    assert cut["market_excess"].index.max() <= day and (cut["delistings"]["dlstdt"] <= day).all()
    assert cut["funda"]["datadate"].max() <= pd.Timestamp("1991-02-28")


def test_the_build_on_cut_inputs_equals_the_full_build(w, full):
    for month in ("1991-02", "1991-07", "1992-01", "1992-06", "1992-12"):
        rows = w["ulong"][w["ulong"]["month"] == pd.Period(month, "M")]
        result = P.compare_month(full[full["month"] == pd.Period(month, "M")], P.build_cut_month(rows, w))
        assert result["equal"] and result["presence_differs"] == 0 and result["largest_difference"] == 0.0, month


def test_the_lookahead_test_catches_a_value_from_month_t(w, full):
    month = pd.Period("1992-06", "M")
    rows = w["ulong"][w["ulong"]["month"] == month]
    leaky = full[full["month"] == month].copy()
    m = w["monthly"]
    this_month = m[pd.PeriodIndex(m["date"], freq="M") == month].set_index("permno")["ret"]
    leaky["reversal"] = leaky["permno"].map(this_month).values        # the return of month t in place of t-1
    result = P.compare_month(leaky, P.build_cut_month(rows, w))
    assert not result["equal"] and result["largest_difference"] > C.LOOKAHEAD_TOLERANCE


def test_assemble_keeps_every_member_month_and_counts_the_missing(w, full):
    m = K.monthly_characteristics(w["ulong"], w["monthly"], w["delistings"])
    d = K.daily_characteristics(w["ulong"], w["daily"], w["market_excess"], w["risk_free"])
    a, _, _ = K.accounting_characteristics(w["ulong"], w["links"], w["funda"])
    targets = w["ulong"][["month", "permno"]].assign(target=0.01)
    targets.loc[targets.index[:3], "target"] = np.nan
    frame = P.assemble(w["ulong"], m, d, a, targets)
    assert len(frame) == len(w["ulong"])
    assert list(frame.columns) == ["month", "permno", "date_prev", *C.CHARACTERISTICS, "target"]
    pd.testing.assert_frame_equal(frame[["month", "permno", *C.CHARACTERISTICS]], full)
    years = P.missing_by_year(frame)
    assert years["member_months"].sum() == len(frame) and years["member_months_without_target"].sum() == 3


def markdown(name):
    """The text of a notebook's markdown cells, as committed."""
    nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_the_figures_the_panel_notebooks_quote_before_their_run():
    # Notebooks 01 to 05 quote these in "What I expect to see" and in their terms, each from the constants.
    months = (pd.Period(C.G1A_LAST_MONTH, "M") - pd.Period(C.G1A_FIRST_MONTH, "M")).n + 1
    quoted = {
        "01_extraction.ipynb": [C.MANIFEST_SHA256[:8], f"{C.MANIFEST_FILES} files", f"{C.MANIFEST_ROWS:,} rows",
                                f"{C.TABLE_COUNT}", f"{C.MANIFEST_ENTRIES} files", f"{C.MANIFEST_ENTRIES_ROWS:,} rows",
                                "28 September 2026"],
        "02_universe_and_returns.ipynb": [f"{months:,} months", f"{C.G1A_MEMBER_MONTHS:,} member-months",
                                          f"{C.PANEL_MEMBER_MONTHS:,} member-months",
                                          f"{C.PANEL_MEMBER_MONTHS_NO_RETURN} have no return", f"{C.TEST_MEMBER_MONTHS:,}"],
        "03_market_characteristics.ipynb": [f"at least {C.DAILY_MIN_DAYS} of them", f"the {C.BETA_WINDOW_DAYS} trading days",
                                            f"at least {C.BETA_MIN_DAYS} days", f"{C.PANEL_MEMBER_MONTHS:,} member-months"],
        "04_accounting_characteristics.ipynb": [f"at most {C.ACCOUNTING_MAX_AGE_MONTHS} months old",
                                                f"{round(100 * C.G1B_COVERAGE_MIN_BOOK_EQUITY)} per cent",
                                                f"{round(100 * C.G1B_COVERAGE_MIN)} per cent"],
        "05_panel.ipynb": [f"{C.PANEL_MEMBER_MONTHS:,} member-months", f"{C.PANEL_MEMBER_MONTHS_NO_RETURN} have no target",
                           f"{C.TEST_MEMBER_MONTHS:,}", f"{C.LOOKAHEAD_MONTHS} of the panel's {C.PANEL_MONTHS} months",
                           f"{C.LOOKAHEAD_TOLERANCE:.0e}", f"{C.SEED}", f"at least {round(100 * C.G1B_COVERAGE_MIN)} per cent",
                           f"at least {round(100 * C.G1B_COVERAGE_MIN_BOOK_EQUITY)} per cent"],
    }
    for name, figures in quoted.items():
        text = markdown(name)
        missing = [f for f in figures if f not in text]
        assert not missing, f"{name} does not quote {missing}"
    assert C.MANIFEST_SHA256.startswith("23e96d18") and C.TABLE_COUNT == len(C.TABLES) == 8


def test_the_counts_notebook_04_quotes_on_the_member_months_without_a_company():
    # Notebook 04 quotes these in what was learned, from the summary its run wrote.
    words = {3: "three", 5: "five"}
    out = json.loads((ROOT / C.RESULTS_DIR / "accounting_summary.json").read_text())["member_months_without_a_company"]
    by, years, types = out["member_months_by_member"], out["years_by_member"], out["member_months_reached_by_link_type"]
    singles = sum(1 for n in by if n == 1)
    assert by[0] == C.PANEL_MONTHS and len(by) == out["members"] and sum(by) == out["member_months"]
    assert out["member_months"] - out["member_months_reached_by_another_link_type"] == singles
    quoted = [f"The {out['member_months']:,} member-months without a company are {words[out['members']]} members",
              f"one in all {by[0]} months, one in {by[1]} months from {years[1][0]} and {words[singles]} in one month each",
              f"reach {out['member_months_reached_by_another_link_type']:,} of them, LS all {types['LS']:,} and LX {types['LX']:,} of those"]
    text = markdown("04_accounting_characteristics.ipynb")
    missing = [q for q in quoted if q not in text]
    assert not missing, f"notebook 04 does not quote {missing}"
