"""
The twelve characteristics of every member-month, each measured with what is known on the last trading day of
month t-1, the day the copied universe records as `date_prev`.

Three groups, one function each, so that notebook 05's look-ahead test can call the same functions on inputs cut
at that day and compare the two builds:

- monthly_characteristics(): size, momentum, reversal and dollar volume, from CRSP's monthly stock file.
- daily_characteristics(): volatility, the largest daily return, beta and idiosyncratic volatility, from CRSP's
  daily stock file and French's daily market excess return.
- accounting_characteristics(): book-to-market, earnings-to-price, operating profitability and asset growth, from
  Compustat's annual fundamentals through the CRSP-Compustat link.

Every function takes the member-months as the copied universe gives them (month, permno, date_prev) and returns
one row per member-month, with a characteristic missing where its rule says so. without_a_company() counts the
member-months that the links leave without a company, for notebook 04. Nothing here reads or writes a
file: the notebooks read the extraction through src/mlrp/data.py and keep the result in local/, which git ignores.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import constants as C
from .source import returns as R
from .source import sectors as S
from .source import sorts as T

MONTHLY = ("size", "momentum", "reversal", "dollar_volume")
DAILY = ("volatility", "max_daily_return", "beta", "idiosyncratic_volatility")
ACCOUNTING = ("book_to_market", "earnings_to_price", "operating_profitability", "asset_growth")

# CRSP's monthly share volume is in hundreds of shares (CRSP's guide, p. 91).
SHARES_PER_VOLUME_UNIT = 100


# ---------------------------------------------------------------------------
# From the monthly stock file
# ---------------------------------------------------------------------------

def monthly_characteristics(ulong: pd.DataFrame, monthly: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    """Size, momentum, reversal and dollar volume of every member-month in `ulong`.

    - size: the log of the start-of-month capitalisation, the copied returns module's `cap`: the absolute price
      times shares outstanding at the end of month t-1, missing where that is missing or not positive.
    - momentum: the return from month t-12 to month t-2, by the copied momentum function, missing where the
      price at t-13 or the return at t-2 is missing.
    - reversal: the return of month t-1 from the monthly file.
    - dollar_volume: the log of the absolute price at the end of month t-1 times the month's share volume in
      shares, missing where the volume is missing or not positive.
    """
    panel = R.monthly_panel(ulong, monthly, delistings)
    out = ulong[["month", "permno"]].merge(panel[["month", "permno", "cap"]], on=["month", "permno"], how="left")
    out["size"] = np.log(out["cap"])
    wide = T.monthly_wide(monthly, columns=("prc", "ret", "vol"))
    # The matrices run to the last month of `ulong` as well, so that a build on a monthly file cut at the end of
    # month t-1 still has a row for month t, empty, where the copied momentum function looks month t up.
    last = max(wide["ret"].index.max(), pd.PeriodIndex(ulong["month"], freq="M").max())
    months = pd.period_range(wide["ret"].index.min(), last, freq="M")
    wide = {k: w.reindex(months) for k, w in wide.items()}
    # The copied function's first three columns are the month, the PERMNO and the compounded return.
    mom = T.momentum(ulong, wide, first=str(ulong["month"].min())).iloc[:, :3]
    mom = mom.set_axis(["month", "permno", "momentum"], axis=1)
    out = out.merge(mom, on=["month", "permno"], how="left")
    previous = pd.PeriodIndex(out["month"], freq="M") - 1
    row = wide["ret"].index.get_indexer(previous)
    col = wide["ret"].columns.get_indexer(out["permno"])
    found = (row >= 0) & (col >= 0)
    r, c = np.where(found, row, 0), np.where(found, col, 0)
    out["reversal"] = np.where(found, wide["ret"].to_numpy()[r, c], np.nan)
    price = np.where(found, wide["prc"].to_numpy()[r, c], np.nan)
    volume = np.where(found, wide["vol"].to_numpy()[r, c], np.nan)
    dollars = price * volume * SHARES_PER_VOLUME_UNIT
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dollar_volume"] = np.where(dollars > 0, np.log(np.where(dollars > 0, dollars, 1.0)), np.nan)
    return out[["month", "permno", *MONTHLY]]


# ---------------------------------------------------------------------------
# From the daily stock file
# ---------------------------------------------------------------------------

def daily_characteristics(ulong: pd.DataFrame, daily: pd.DataFrame, market_excess: pd.Series,
                          risk_free: pd.Series) -> pd.DataFrame:
    """Volatility, the largest daily return, beta and idiosyncratic volatility of every member-month in `ulong`.

    `daily` holds the daily returns, trading days by PERMNO, as the copied layer.daily_matrix returns them, its
    index CRSP's calendar of trading days. `market_excess` and `risk_free` are French's daily market excess
    return and risk-free rate, in decimals, by date.

    - volatility: the standard deviation (divisor n - 1) of the daily returns in month t-1, with at least
      constants.DAILY_MIN_DAYS of them.
    - max_daily_return: the largest daily return in month t-1, with the same minimum.
    - beta: the slope of the daily excess return (return less the risk-free rate) on the market's daily excess
      return, with an intercept, over the constants.BETA_WINDOW_DAYS trading days to the end of month t-1,
      with at least constants.BETA_MIN_DAYS days on which both are present.
    - idiosyncratic_volatility: the standard deviation of that regression's residuals, the square root of their
      sum of squares over n - 2, since the regression estimates two numbers.

    Each month is computed from its own window alone, so a build on inputs cut at the end of month t-1 gives
    the same numbers for month t.
    """
    ulong = ulong.reset_index(drop=True)
    calendar = pd.DatetimeIndex(daily.index)
    values = daily.to_numpy(dtype=float)
    columns = daily.columns.get_indexer(ulong["permno"])
    market = market_excess.reindex(calendar).to_numpy(dtype=float)
    rf = risk_free.reindex(calendar).to_numpy(dtype=float)
    day_month = pd.PeriodIndex(calendar, freq="M")
    parts = []
    for month, group in ulong.groupby("month", sort=True):
        end = calendar.searchsorted(pd.Timestamp(group["date_prev"].iloc[0]), side="right")
        cols = columns[group.index.to_numpy()]
        present_col = cols >= 0
        safe = np.where(present_col, cols, 0)
        out = pd.DataFrame({"month": group["month"].values, "permno": group["permno"].values})
        # Month t-1: the trading days of the month that ends on date_prev.
        inside = np.flatnonzero(day_month[:end] == (pd.Period(month, "M") - 1))
        block = values[inside][:, safe]
        block[:, ~present_col] = np.nan
        n_month = np.sum(~np.isnan(block), axis=0)
        enough = n_month >= C.DAILY_MIN_DAYS
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            # A stock with fewer than two returns in the month gives numpy a warning and a missing value, and
            # the minimum of DAILY_MIN_DAYS sets it missing in any case.
            warnings.simplefilter("ignore", RuntimeWarning)
            vol = np.nanstd(block, axis=0, ddof=1) if len(inside) else np.full(len(safe), np.nan)
            top = np.nanmax(np.where(np.isnan(block), -np.inf, block), axis=0) if len(inside) else np.full(len(safe), np.nan)
        out["volatility"] = np.where(enough, vol, np.nan)
        out["max_daily_return"] = np.where(enough, top, np.nan)
        # The regression window: the last BETA_WINDOW_DAYS trading days up to date_prev.
        start = max(0, end - C.BETA_WINDOW_DAYS)
        y = values[start:end][:, safe] - rf[start:end, None]
        y[:, ~present_col] = np.nan
        x = np.broadcast_to(market[start:end, None], y.shape)
        use = ~np.isnan(y) & ~np.isnan(x)
        n = use.sum(axis=0)
        xs, ys = np.where(use, x, 0.0), np.where(use, y, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            x_mean = xs.sum(axis=0) / n
            y_mean = ys.sum(axis=0) / n
            dx = np.where(use, x - x_mean, 0.0)
            dy = np.where(use, y - y_mean, 0.0)
            beta = (dx * dy).sum(axis=0) / (dx * dx).sum(axis=0)
            residual = np.where(use, dy - beta * dx, 0.0)
            idio = np.sqrt((residual * residual).sum(axis=0) / (n - 2))
        ok = n >= C.BETA_MIN_DAYS
        out["beta"] = np.where(ok, beta, np.nan)
        out["idiosyncratic_volatility"] = np.where(ok, idio, np.nan)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)[["month", "permno", *DAILY]]


# ---------------------------------------------------------------------------
# From Compustat's annual fundamentals
# ---------------------------------------------------------------------------

def _stockholders_equity(frame: pd.DataFrame) -> pd.Series:
    """Stockholders' equity by the first rule whose items are all present: `seq`, else `ceq` plus `pstk`, else
    `at` minus `lt` (constants.BOOK_EQUITY_STOCKHOLDERS names the three groups in this order)."""
    (equity,), (common, preferred), (assets, liabilities) = C.BOOK_EQUITY_STOCKHOLDERS
    out = frame[equity].copy()
    second = out.isna() & frame[common].notna() & frame[preferred].notna()
    out[second] = frame.loc[second, common] + frame.loc[second, preferred]
    third = out.isna() & frame[assets].notna() & frame[liabilities].notna()
    out[third] = frame.loc[third, assets] - frame.loc[third, liabilities]
    return out


def _preferred_stock(frame: pd.DataFrame) -> pd.Series:
    """Preferred stock by the first item present among `pstkrv`, `pstkl` and `pstk`, zero when none is."""
    out = pd.Series(np.nan, index=frame.index)
    for item in C.BOOK_EQUITY_PREFERRED:
        out = out.fillna(frame[item])
    return out.fillna(0.0)


def annual_records(funda: pd.DataFrame) -> tuple:
    """One record per company and fiscal year end, with book equity, the market value at the fiscal year end,
    earnings, operating profit and total assets now and a fiscal year before. Returns the records and the
    counts of what the rules decided.

    - Book equity: stockholders' equity (`seq`, else `ceq` plus `pstk`, else `at` minus `lt`), plus deferred
      taxes and investment tax credit (`txditc`, zero when missing), minus preferred stock (`pstkrv`, else
      `pstkl`, else `pstk`, zero when missing). Not positive is missing.
    - Market value at the fiscal year end: `prcc_f` times `csho`, missing when not positive.
    - Operating profit: revenue less cost of goods sold, selling, general and administrative expenses and
      interest expense, where revenue and at least one of the three costs are present. A missing cost among the
      others counts as zero.
    - Total assets a fiscal year before: `at` of the same company's record for fiscal year fyear - 1.
    """
    f = funda.dropna(subset=["gvkey", "datadate"]).copy()
    duplicates = int(f.duplicated(["gvkey", "datadate"]).sum())
    f = f.sort_values(["gvkey", "datadate", "fyear"]).drop_duplicates(["gvkey", "datadate"], keep="last")
    book = _stockholders_equity(f) + f[C.BOOK_EQUITY_DEFERRED_TAXES].fillna(0.0) - _preferred_stock(f)
    market = f[C.MARKET_VALUE_AT_FISCAL_YEAR_END[0]] * f[C.MARKET_VALUE_AT_FISCAL_YEAR_END[1]]
    costs = f[list(C.OPERATING_COSTS)]
    operating = np.where(f[C.REVENUE].notna() & costs.notna().any(axis=1),
                         f[C.REVENUE] - costs.fillna(0.0).sum(axis=1), np.nan)
    records = pd.DataFrame({
        "gvkey": f["gvkey"].values, "datadate": pd.to_datetime(f["datadate"]).values, "fyear": f["fyear"].values,
        "book_equity": np.where(book > 0, book, np.nan),
        "market_value": np.where(market > 0, market, np.nan),
        "earnings": f[C.EARNINGS].values, "operating_profit": operating, "total_assets": f[C.TOTAL_ASSETS].values,
    })
    before = records[["gvkey", "fyear", "total_assets"]].dropna(subset=["fyear"]).copy()
    before["fyear"] = before["fyear"] + 1
    before = before.drop_duplicates(["gvkey", "fyear"], keep="last").rename(columns={"total_assets": "total_assets_before"})
    records = records.merge(before, on=["gvkey", "fyear"], how="left")
    counts = {"rows": int(len(funda)), "rows_without_gvkey_or_date": int(len(funda) - len(funda.dropna(subset=["gvkey", "datadate"]))),
              "duplicate_company_dates": duplicates, "records": int(len(records)),
              "book_equity_not_positive_or_missing": int(records["book_equity"].isna().sum()),
              "market_value_missing": int(records["market_value"].isna().sum()),
              "records_without_fiscal_year": int(records["fyear"].isna().sum())}
    return records, counts


def accounting_characteristics(ulong: pd.DataFrame, raw_links: pd.DataFrame, funda: pd.DataFrame) -> tuple:
    """Book-to-market, earnings-to-price, operating profitability and asset growth of every member-month.

    The company of a member-month is the GVKEY of the link valid on date_prev, by the copied sectors module with
    the sector join's links (the two link types, any primary flag), so that a second share class reaches its
    company. Its record is the one with the latest fiscal year end whose age on date_prev, in months, is from
    constants.ACCOUNTING_MIN_AGE_MONTHS to constants.ACCOUNTING_MAX_AGE_MONTHS: a fiscal year ending in December
    is first used for month t = July of the next year, when month t-1, June, is six months on.

    Returns the frame (month, permno, gvkey, age_months and the four characteristics), the count of member-months
    with more than one valid link, and the counts of annual_records().
    """
    links = S.sector_links(raw_links)
    linked, multiple = S.link_gvkeys(ulong[["month", "permno", "date_prev"]], links)
    records, counts = annual_records(funda)
    records["record_month"] = pd.PeriodIndex(records["datadate"], freq="M").asi8
    left = linked.copy()
    left["last_month"] = (pd.PeriodIndex(left["month"], freq="M") - 1).asi8
    left["cutoff"] = left["last_month"] - C.ACCOUNTING_MIN_AGE_MONTHS
    with_key = left[left["gvkey"].notna()].sort_values("cutoff")
    right = records.dropna(subset=["gvkey"]).sort_values("record_month")
    matched = pd.merge_asof(with_key, right, left_on="cutoff", right_on="record_month", by="gvkey", direction="backward")
    matched["age_months"] = matched["last_month"] - matched["record_month"]
    too_old = matched["age_months"] > C.ACCOUNTING_MAX_AGE_MONTHS
    matched.loc[too_old, ["book_equity", "market_value", "earnings", "operating_profit", "total_assets",
                          "total_assets_before", "record_month"]] = np.nan
    matched.loc[too_old, "age_months"] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        matched["book_to_market"] = matched["book_equity"] / matched["market_value"]
        matched["earnings_to_price"] = matched["earnings"] / matched["market_value"]
        matched["operating_profitability"] = matched["operating_profit"] / matched["book_equity"]
        before = matched["total_assets_before"]
        matched["asset_growth"] = np.where(before > 0, matched["total_assets"] / before - 1, np.nan)
    out = left[["month", "permno", "gvkey"]].merge(
        matched[["month", "permno", "age_months", *ACCOUNTING]], on=["month", "permno"], how="left")
    out = out.sort_values(["month", "permno"], ignore_index=True)
    return out[["month", "permno", "gvkey", "age_months", *ACCOUNTING]], multiple, counts


def without_a_company(ulong: pd.DataFrame, raw_links: pd.DataFrame) -> dict:
    """Counts of the member-months that the study's links leave without a company, and of those that a link of
    another type would reach. The study's links are those of accounting_characteristics(). A link of another
    type, any type outside constants.ACCOUNTING_LINK_TYPES and any primary flag, reaches a member-month when it
    names the member's PERMNO and is valid on date_prev, both ends included. Counts and years only, with no
    identifier: the member-months and the members, each member's member-months and its first and last year,
    largest first, and the member-months reached, in all and by link type.

    Added on 24 November 2026.
    """
    linked, _ = S.link_gvkeys(ulong[["month", "permno", "date_prev"]], S.sector_links(raw_links))
    missing = linked.loc[linked["gvkey"].isna(), ["month", "permno", "date_prev"]].copy()
    missing["year"] = pd.PeriodIndex(missing["month"], freq="M").year
    by_member = (missing.groupby("permno")
                 .agg(member_months=("month", "size"), first_year=("year", "min"), last_year=("year", "max"))
                 .sort_values(["member_months", "first_year"], ascending=[False, True]))
    other = raw_links[raw_links["lpermno"].notna() & ~raw_links["linktype"].isin(C.ACCOUNTING_LINK_TYPES)].copy()
    other["permno"] = other["lpermno"].astype("int64")
    other["linkenddt"] = other["linkenddt"].fillna(S.OPEN_END)
    m = missing.merge(other[["permno", "linktype", "linkdt", "linkenddt"]], on="permno", how="inner")
    m = m[(m["linkdt"] <= m["date_prev"]) & (m["date_prev"] <= m["linkenddt"])]
    by_type = m.drop_duplicates(["permno", "month", "linktype"])["linktype"].value_counts().sort_index()
    return {"member_months": int(len(missing)), "members": int(len(by_member)),
            "member_months_by_member": [int(n) for n in by_member["member_months"]],
            "years_by_member": [[int(a), int(b)] for a, b in zip(by_member["first_year"], by_member["last_year"])],
            "member_months_reached_by_another_link_type": int(len(m.drop_duplicates(["permno", "month"]))),
            "member_months_reached_by_link_type": {str(k): int(v) for k, v in by_type.items()}}
