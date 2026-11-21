"""
The panel: one row per member of the universe and month t, from January 1991 to December 2024, with the twelve
characteristics measured at the end of month t-1, the target of month t, and the checks of gate G1 (b) and (c).

- assemble() joins the three groups of characteristics and the target onto the member-months.
- coverage_by_month() and g1b() measure and judge the share of each month's members with each characteristic.
- transform() ranks each characteristic within its month and maps the ranks to the interval from -1 to 1, ties
  at their average rank, then sets a missing value to 0, the month's median.
- lookahead_months(), cut_inputs(), build_cut_month() and compare_month() run the look-ahead test of gate G1 (c).

Nothing here reads or writes a file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import characteristics as K
from . import constants as C

KEYS = ["month", "permno"]


def panel_months() -> pd.PeriodIndex:
    """The panel's months, January 1991 to December 2024."""
    return pd.period_range(C.PANEL_FIRST_MONTH, C.PANEL_LAST_MONTH, freq="M")


def target(returns_panel: pd.DataFrame, risk_free_monthly: pd.Series) -> pd.DataFrame:
    """The target of every member-month: the total return of month t with the copied delisting rule applied,
    `ret_total`, less French's one-month bill rate of month t. Missing where the member has no return and no
    delisting in the month."""
    out = returns_panel[["month", "permno", "ret_total"]].copy()
    rf = risk_free_monthly.copy()
    rf.index = pd.PeriodIndex(rf.index, freq="M")
    out["risk_free"] = rf.reindex(pd.PeriodIndex(out["month"], freq="M")).values
    out["target"] = out["ret_total"] - out["risk_free"]
    return out


def assemble(ulong: pd.DataFrame, monthly: pd.DataFrame, daily: pd.DataFrame, accounting: pd.DataFrame,
             targets: pd.DataFrame) -> pd.DataFrame:
    """Every member-month of the panel's months with its twelve characteristics, before the transform, and its
    target. A member-month without a target stays in the frame, so that coverage counts every member."""
    months = panel_months()
    base = ulong[pd.PeriodIndex(ulong["month"], freq="M").isin(months)][["month", "permno", "date_prev"]]
    out = (base.merge(monthly, on=KEYS, how="left")
               .merge(daily, on=KEYS, how="left")
               .merge(accounting[[*KEYS, *K.ACCOUNTING]], on=KEYS, how="left")
               .merge(targets[[*KEYS, "target"]], on=KEYS, how="left"))
    return out.sort_values(KEYS, ignore_index=True)[["month", "permno", "date_prev", *C.CHARACTERISTICS, "target"]]


def coverage_by_month(frame: pd.DataFrame) -> pd.DataFrame:
    """For each month, the share of its members with each characteristic present, before the fill."""
    present = frame[list(C.CHARACTERISTICS)].notna()
    present["month"] = frame["month"].values
    return present.groupby("month").mean()


def g1b(coverage: pd.DataFrame) -> dict:
    """Gate G1 (b): each characteristic present for at least constants.G1B_COVERAGE_MIN of the month's members in
    every month from G1B_FIRST_MONTH to G1B_LAST_MONTH, and the two that need a positive book equity for at
    least G1B_COVERAGE_MIN_BOOK_EQUITY. Returns, by characteristic, its level, its lowest monthly share, the
    month of that share and the months below the level, and the verdict."""
    window = coverage[(coverage.index >= pd.Period(C.G1B_FIRST_MONTH, "M"))
                      & (coverage.index <= pd.Period(C.G1B_LAST_MONTH, "M"))]
    rows = {}
    for c in C.CHARACTERISTICS:
        level = C.G1B_COVERAGE_MIN_BOOK_EQUITY if c in C.G1B_BOOK_EQUITY_CHARACTERISTICS else C.G1B_COVERAGE_MIN
        s = window[c]
        rows[c] = {"level": level, "lowest_share": round(float(s.min()), 6), "lowest_month": str(s.idxmin()),
                   "months_below_level": int((s < level).sum())}
    return {"months": int(len(window)), "by_characteristic": rows,
            "g1b_met": all(r["months_below_level"] == 0 for r in rows.values())}


def missing_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    """By year: the member-months, those without a target, and the share of member-months missing each
    characteristic before the fill."""
    years = pd.PeriodIndex(frame["month"], freq="M").year
    missing = frame[list(C.CHARACTERISTICS)].isna()
    out = missing.groupby(years).mean().round(6)
    out.insert(0, "member_months_without_target", frame["target"].isna().groupby(years).sum().astype("int64"))
    out.insert(0, "member_months", frame.groupby(years).size())
    out.index.name = "year"
    return out.reset_index()


def rank_transform(values: pd.Series, months: pd.Series) -> pd.Series:
    """One characteristic ranked within each month across the members for which it is present, ties at their
    average rank, and mapped to the interval from constants.RANK_LOW to RANK_HIGH: rank r of n becomes
    RANK_LOW + (RANK_HIGH - RANK_LOW) (r - 1) / (n - 1), and a month with one value gives it the midpoint.
    A missing value stays missing here."""
    groups = values.groupby(months.values)
    rank = groups.rank(method=C.RANK_TIES)
    n = groups.transform("count")
    span = C.RANK_HIGH - C.RANK_LOW
    mapped = C.RANK_LOW + span * (rank - 1) / (n - 1)
    return mapped.where(n > 1, C.RANK_LOW + span / 2).where(values.notna())


def transform(frame: pd.DataFrame) -> tuple:
    """The panel with every characteristic ranked and mapped within its month, then each missing value set to
    constants.FILL_VALUE, which is the month's median on that scale. Returns the frame and the count of filled
    cells by characteristic."""
    out = frame.copy()
    filled = {}
    for c in C.CHARACTERISTICS:
        mapped = rank_transform(frame[c], frame["month"])
        filled[c] = int(mapped.isna().sum())
        out[c] = mapped.fillna(C.FILL_VALUE)
    return out, filled


def lookahead_months(months: pd.PeriodIndex = None) -> pd.PeriodIndex:
    """The months of the look-ahead test: constants.LOOKAHEAD_MONTHS of the panel's months, drawn without
    replacement from the seed's stream for this test, in calendar order."""
    months = panel_months() if months is None else months
    rng = np.random.default_rng([C.SEED, C.SEED_STREAMS["lookahead_months"]])
    picked = np.sort(rng.choice(len(months), size=C.LOOKAHEAD_MONTHS, replace=False))
    return months[picked]


def cut_inputs(inputs: dict, day: pd.Timestamp) -> dict:
    """The inputs of the characteristics as they stood on `day`, the last trading day of month t-1.

    Every row dated later is removed: from the monthly file and the delistings by their dates, from the daily
    returns and French's daily series by their index, and from the links by their start. A link's end after
    `day` was not known on `day`, so it becomes missing, which the copied links read as still in force. Every
    annual record less than constants.ACCOUNTING_MIN_AGE_MONTHS months old on `day` is removed.
    """
    day = pd.Timestamp(day)
    last = pd.Period(day, "M")
    links = inputs["links"][inputs["links"]["linkdt"] <= day].copy()
    links.loc[links["linkenddt"] > day, "linkenddt"] = pd.NaT
    funda = inputs["funda"]
    old_enough = pd.PeriodIndex(funda["datadate"], freq="M") <= last - C.ACCOUNTING_MIN_AGE_MONTHS
    return {
        "monthly": inputs["monthly"][inputs["monthly"]["date"] <= day],
        "delistings": inputs["delistings"][inputs["delistings"]["dlstdt"] <= day],
        "daily": inputs["daily"].loc[:day],
        "market_excess": inputs["market_excess"].loc[:day],
        "risk_free": inputs["risk_free"].loc[:day],
        "links": links,
        "funda": funda[old_enough],
    }


def build(rows: pd.DataFrame, inputs: dict) -> pd.DataFrame:
    """The twelve characteristics of the member-months in `rows`, from `inputs`, by the three functions of
    src/mlrp/characteristics.py."""
    rows = rows[["month", "permno", "date_prev"]].reset_index(drop=True)
    monthly = K.monthly_characteristics(rows, inputs["monthly"], inputs["delistings"])
    daily = K.daily_characteristics(rows, inputs["daily"], inputs["market_excess"], inputs["risk_free"])
    accounting, _, _ = K.accounting_characteristics(rows, inputs["links"], inputs["funda"])
    out = rows[KEYS].merge(monthly, on=KEYS, how="left").merge(daily, on=KEYS, how="left").merge(
        accounting[[*KEYS, *K.ACCOUNTING]], on=KEYS, how="left")
    return out[[*KEYS, *C.CHARACTERISTICS]].sort_values(KEYS, ignore_index=True)


def build_cut_month(rows: pd.DataFrame, inputs: dict) -> pd.DataFrame:
    """One month's characteristics built again from the inputs cut at that month's date_prev."""
    day = pd.Timestamp(rows["date_prev"].iloc[0])
    return build(rows, cut_inputs(inputs, day))


def compare_month(full: pd.DataFrame, cut_build: pd.DataFrame, tolerance: float = C.LOOKAHEAD_TOLERANCE) -> dict:
    """The full build's rows of one month against the rows built again from the cut inputs: the cells compared,
    the cells where one build has a value and the other has none, the largest absolute difference where both
    have one, and whether every cell agrees to `tolerance`."""
    a = full.sort_values(KEYS, ignore_index=True)
    b = cut_build.sort_values(KEYS, ignore_index=True)
    if not a[KEYS].equals(b[KEYS]):
        return {"cells": 0, "presence_differs": None, "largest_difference": None, "equal": False}
    x, y = a[list(C.CHARACTERISTICS)].to_numpy(float), b[list(C.CHARACTERISTICS)].to_numpy(float)
    presence = int((np.isnan(x) != np.isnan(y)).sum())
    both = ~np.isnan(x) & ~np.isnan(y)
    largest = float(np.abs(x[both] - y[both]).max()) if both.any() else 0.0
    return {"cells": int(x.size), "presence_differs": presence, "largest_difference": largest,
            "equal": presence == 0 and largest <= tolerance}
