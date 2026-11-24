"""The twelve characteristics on a made-up world, each against a value worked out here by a second route: the
monthly ones from the monthly file, the daily ones with numpy's own functions on the month's or the window's
days, and the accounting ones from the record the timing rule selects. No licensed data is read."""
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


@pytest.fixture(scope="module")
def w():
    return world.make()


def row_of(frame, month, permno):
    hit = frame[(frame["month"] == pd.Period(month, "M")) & (frame["permno"] == permno)]
    assert len(hit) == 1
    return hit.iloc[0]


def monthly_row(w, month, permno):
    m = w["monthly"]
    return m[(m["permno"] == permno) & (pd.PeriodIndex(m["date"], freq="M") == pd.Period(month, "M"))].iloc[0]


def test_the_monthly_characteristics(w):
    out = K.monthly_characteristics(w["ulong"], w["monthly"], w["delistings"])
    assert len(out) == len(w["ulong"]) and list(out.columns) == ["month", "permno", *K.MONTHLY]
    r = row_of(out, "1992-05", 10006)
    before = monthly_row(w, "1992-04", 10006)
    assert r["size"] == pytest.approx(np.log(abs(before["prc"]) * before["shrout"]), rel=1e-12)
    assert r["reversal"] == before["ret"]
    assert r["dollar_volume"] == pytest.approx(np.log(abs(before["prc"]) * before["vol"] * 100), rel=1e-12)
    rets = [monthly_row(w, str(pd.Period("1992-05", "M") - k), 10006)["ret"]
            for k in range(C.MOMENTUM_LAST_LAG, C.MOMENTUM_FIRST_LAG + 1)]
    assert r["momentum"] == pytest.approx(np.prod([1 + x for x in rets]) - 1, rel=1e-10)
    assert np.isnan(row_of(out, "1991-03", 10003)["dollar_volume"])   # volume -99 in February 1991
    assert np.isnan(row_of(out, "1991-04", 10004)["reversal"])         # no return in March 1991


def test_the_daily_characteristics(w):
    out = K.daily_characteristics(w["ulong"], w["daily"], w["market_excess"], w["risk_free"])
    assert len(out) == len(w["ulong"]) and list(out.columns) == ["month", "permno", *K.DAILY]
    r = row_of(out, "1992-03", 10007)
    days = w["daily"].index
    feb = w["daily"].loc[(days >= "1992-02-01") & (days <= "1992-02-29"), 10007].dropna()
    assert r["volatility"] == pytest.approx(feb.std(ddof=1), rel=1e-10)
    assert r["max_daily_return"] == feb.max()
    end = days.searchsorted(pd.Timestamp("1992-02-28"), side="right")
    window = days[end - C.BETA_WINDOW_DAYS:end]
    y = (w["daily"].loc[window, 10007] - w["risk_free"].loc[window]).dropna()
    x = w["market_excess"].loc[y.index]
    slope, intercept = np.polyfit(x.values, y.values, 1)
    residual = y.values - intercept - slope * x.values
    assert r["beta"] == pytest.approx(slope, rel=1e-9)
    assert r["idiosyncratic_volatility"] == pytest.approx(np.sqrt((residual ** 2).sum() / (len(y) - 2)), rel=1e-9)
    few = row_of(out, "1991-04", 10005)   # ten returns in March 1991, below the minimum of 15
    assert np.isnan(few["volatility"]) and np.isnan(few["max_daily_return"])
    assert not np.isnan(few["beta"])


def test_the_beta_needs_its_minimum_of_days(w):
    daily = w["daily"].copy()
    daily.loc[:"1991-12-31", 10008] = np.nan
    out = K.daily_characteristics(w["ulong"], daily, w["market_excess"], w["risk_free"])
    assert np.isnan(row_of(out, "1992-03", 10008)["beta"])        # about 40 days with a return
    assert not np.isnan(row_of(out, "1992-12", 10008)["beta"])    # about 230


def test_nothing_after_the_end_of_month_t_minus_1_moves_a_daily_characteristic(w):
    daily = w["daily"].copy()
    daily.loc["1992-03-02":, :] = 0.5
    a = K.daily_characteristics(w["ulong"], w["daily"], w["market_excess"], w["risk_free"])
    b = K.daily_characteristics(w["ulong"], daily, w["market_excess"], w["risk_free"])
    march = a["month"] == pd.Period("1992-03", "M")
    pd.testing.assert_frame_equal(a[march], b[march])


def test_the_annual_records_follow_the_book_equity_rules():
    f = pd.DataFrame({"gvkey": ["a", "b", "c", "d"], "datadate": pd.to_datetime(["2000-12-31"] * 4),
                      "fyear": [2000.0] * 4,
                      "seq": [100.0, np.nan, np.nan, 10.0], "ceq": [1.0, 80.0, np.nan, 1.0],
                      "pstk": [5.0, 4.0, 3.0, 50.0], "at": [500.0, 500.0, 300.0, 500.0], "lt": [1.0, 1.0, 120.0, 1.0],
                      "txditc": [7.0, np.nan, 2.0, 0.0], "pstkrv": [np.nan, 6.0, np.nan, np.nan],
                      "pstkl": [8.0, 9.0, np.nan, np.nan], "prcc_f": [10.0] * 4, "csho": [10.0] * 4,
                      "ib": [-5.0, 3.0, 4.0, 1.0], "revt": [200.0, 200.0, 200.0, np.nan],
                      "cogs": [50.0, np.nan, np.nan, 10.0], "xsga": [np.nan, np.nan, 20.0, 1.0],
                      "xint": [10.0, np.nan, np.nan, 1.0]})
    records, counts = K.annual_records(f)
    r = records.set_index("gvkey")
    assert r.loc["a", "book_equity"] == 100 + 7 - 8          # `seq`, `txditc`, `pstkl` when `pstkrv` is missing
    assert r.loc["b", "book_equity"] == 80 + 4 + 0 - 6       # `ceq` plus `pstk`, `txditc` missing as zero, `pstkrv`
    assert r.loc["c", "book_equity"] == 300 - 120 + 2 - 3    # `at` minus `lt`, `pstk` as the last preferred item
    assert np.isnan(r.loc["d", "book_equity"])               # 10 + 0 - 50 is not positive
    assert r.loc["a", "operating_profit"] == 200 - 50 - 10   # the missing cost counts as zero
    assert np.isnan(r.loc["b", "operating_profit"])          # no cost present
    assert np.isnan(r.loc["d", "operating_profit"])          # no revenue
    assert r.loc["a", "market_value"] == 100
    assert counts["book_equity_not_positive_or_missing"] == 1 and counts["records"] == 4


def test_the_accounting_timing_and_the_second_share_class(w):
    out, multiple, counts = K.accounting_characteristics(w["ulong"], w["links"], w["funda"])
    assert len(out) == len(w["ulong"]) and multiple == 0
    records, _ = K.annual_records(w["funda"])
    gvkey = w["gvkeys"][10006]
    june, july = row_of(out, "1992-06", 10006), row_of(out, "1992-07", 10006)
    assert june["age_months"] == 17 and july["age_months"] == 6   # December 1990, then December 1991
    used = records[(records["gvkey"] == gvkey) & (records["fyear"] == 1991)].iloc[0]
    prior = records[(records["gvkey"] == gvkey) & (records["fyear"] == 1990)].iloc[0]
    assert july["book_to_market"] == pytest.approx(used["book_equity"] / used["market_value"], rel=1e-12)
    assert july["earnings_to_price"] == pytest.approx(used["earnings"] / used["market_value"], rel=1e-12)
    assert july["operating_profitability"] == pytest.approx(used["operating_profit"] / used["book_equity"], rel=1e-12)
    assert july["asset_growth"] == pytest.approx(used["total_assets"] / prior["total_assets"] - 1, rel=1e-12)
    twin, first = row_of(out, "1992-07", 10012), row_of(out, "1992-07", 10001)
    assert twin["gvkey"] == first["gvkey"] and twin["book_to_market"] == first["book_to_market"]


def test_a_record_older_than_the_maximum_age_is_not_used(w):
    funda = w["funda"][w["funda"]["fyear"] <= 1989]
    out, _, _ = K.accounting_characteristics(w["ulong"], w["links"], funda)
    # December 1989 is 24 months before December 1991, the month before January 1992, and 25 before February.
    assert row_of(out, "1992-01", 10006)["age_months"] == 24
    assert np.isnan(row_of(out, "1992-02", 10006)["book_to_market"])


def test_the_member_months_without_a_company_and_the_links_of_other_types(w):
    links = w["links"].copy()
    # 10003 loses its LC link and gains an LS link valid from March 1991; 10004's link ends in June 1992.
    links.loc[links["lpermno"] == 10003, "linktype"] = "LS"
    links.loc[links["lpermno"] == 10003, "linkdt"] = pd.Timestamp("1991-03-01")
    links.loc[links["lpermno"] == 10004, "linkenddt"] = pd.Timestamp("1992-06-15")
    out = K.without_a_company(w["ulong"], links)
    accounting, _, _ = K.accounting_characteristics(w["ulong"], links, w["funda"])
    assert out["member_months"] == int(accounting["gvkey"].isna().sum()) == 24 + 6
    assert out["members"] == 2
    assert out["member_months_by_member"] == [24, 6] and out["years_by_member"] == [[1991, 1992], [1992, 1992]]
    # The LS link is valid on the last trading day of the month before from April 1991: 21 of 10003's 24 months.
    assert out["member_months_reached_by_another_link_type"] == 21
    assert out["member_months_reached_by_link_type"] == {"LS": 21}
    assert set(out) == {"member_months", "members", "member_months_by_member", "years_by_member",
                        "member_months_reached_by_another_link_type", "member_months_reached_by_link_type"}


def test_the_link_types_of_the_accounting_characteristics_are_the_copied_sector_join_s():
    from mlrp.source import constants as SC
    assert C.ACCOUNTING_LINK_TYPES == SC.LINK_TYPES
