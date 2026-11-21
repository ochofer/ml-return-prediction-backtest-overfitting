"""A made-up world in the shape of the extraction, for the tests of the characteristics and the panel: twelve
stocks of ten companies over 1989 to 1992, with a monthly file, delistings, daily returns, French's daily series,
links and annual fundamentals. Every number is drawn from a fixed seed. No licensed data is read."""
import numpy as np
import pandas as pd

PERMNOS = list(range(10001, 10013))


def calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("1989-01-02", "1992-12-31")


def month_ends(days: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(days, index=pd.PeriodIndex(days, freq="M"))
    return s.groupby(level=0).max()


def make(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    days = calendar()
    ends = month_ends(days)
    months = ends.index
    # The monthly file: one row per stock and month-end.
    rows = []
    for p in PERMNOS:
        for m, d in ends.items():
            price = float(rng.uniform(10, 100)) * (-1 if rng.random() < 0.05 else 1)
            rows.append({"permno": p, "date": d, "prc": price, "ret": float(rng.normal(0.01, 0.08)),
                         "retx": float(rng.normal(0.008, 0.08)), "shrout": float(rng.uniform(1e4, 1e5)),
                         "vol": float(rng.uniform(1e3, 1e5))})
    monthly = pd.DataFrame(rows)
    monthly.loc[(monthly["permno"] == 10003) & (monthly["date"] == ends[pd.Period("1991-02", "M")]), "vol"] = -99.0
    monthly.loc[(monthly["permno"] == 10004) & (monthly["date"] == ends[pd.Period("1991-03", "M")]), "ret"] = np.nan
    delistings = pd.DataFrame({"permno": [10011, 10001], "dlstdt": [pd.Timestamp("1992-06-15"), pd.NaT],
                               "dlstcd": [233.0, 100.0], "dlret": [0.05, np.nan], "dlretx": [0.05, np.nan]})
    # Daily returns, with gaps, and one stock with few returns in March 1991.
    daily = pd.DataFrame(rng.normal(0.0004, 0.02, size=(len(days), len(PERMNOS))), index=days, columns=PERMNOS)
    daily = daily.mask(rng.random(daily.shape) < 0.03)
    march = (days >= "1991-03-01") & (days <= "1991-03-31")
    daily.loc[march, 10005] = np.nan
    daily.loc[days[march][:10], 10005] = 0.01 * np.arange(10)
    market = pd.Series(rng.normal(0.0003, 0.01, len(days)), index=days)
    risk_free = pd.Series(0.0001, index=days)
    # The links: one company per stock, and 10012 a second share class of 10001's company, flag J.
    gvkeys = {p: f"{p - 9000:06d}" for p in PERMNOS}
    gvkeys[10012] = gvkeys[10001]
    links = pd.DataFrame({"gvkey": [gvkeys[p] for p in PERMNOS], "lpermno": [float(p) for p in PERMNOS],
                          "lpermco": [float(p) for p in PERMNOS], "linktype": "LC",
                          "linkprim": ["J" if p == 10012 else "P" for p in PERMNOS],
                          "liid": "01", "linkdt": pd.Timestamp("1980-01-01"), "linkenddt": pd.NaT})
    # Annual fundamentals, fiscal years ending in December.
    frows = []
    for g in sorted(set(gvkeys.values())):
        for year in range(1986, 1993):
            frows.append({"gvkey": g, "datadate": pd.Timestamp(f"{year}-12-31"), "fyear": float(year),
                          "seq": float(rng.uniform(100, 1000)), "ceq": float(rng.uniform(100, 1000)),
                          "pstk": float(rng.uniform(0, 50)), "at": float(rng.uniform(1000, 5000)),
                          "lt": float(rng.uniform(100, 900)), "txditc": float(rng.uniform(0, 30)),
                          "pstkrv": float(rng.uniform(0, 40)), "pstkl": float(rng.uniform(0, 40)),
                          "prcc_f": float(rng.uniform(10, 100)), "csho": float(rng.uniform(10, 100)),
                          "ib": float(rng.normal(50, 40)), "revt": float(rng.uniform(500, 2000)),
                          "cogs": float(rng.uniform(100, 400)), "xsga": float(rng.uniform(50, 200)),
                          "xint": float(rng.uniform(0, 50))})
    funda = pd.DataFrame(frows)
    # Members: every stock in 1991 and 1992, except 10002 in 1991-01 to 1991-03.
    members = [(m, p) for m in months if m >= pd.Period("1991-01", "M") for p in PERMNOS
               if not (p == 10002 and m <= pd.Period("1991-03", "M"))]
    ulong = pd.DataFrame(members, columns=["month", "permno"])
    ulong["month"] = pd.PeriodIndex(ulong["month"], freq="M")
    ulong["date_prev"] = ends.reindex(ulong["month"] - 1).values
    return {"monthly": monthly, "delistings": delistings, "daily": daily, "market_excess": market,
            "risk_free": risk_free, "links": links, "funda": funda, "ulong": ulong, "ends": ends, "gvkeys": gvkeys}
