"""
Gate G3: the known-answer simulation of the overfitting statistics, on made-up returns only.

Each repetition draws a matrix of independent standard normal monthly returns, with as many
months and trials as the study's test window and trials (constants section 12). With zero
means no trial has a true effect, so the statistics have a known answer: the best Sharpe
ratio should match the expected best of Bailey and López de Prado (2014), the deflated
Sharpe ratio should sit near one half and rarely clear its level, and the probability of
backtest overfitting should sit near one half. With one trial given a true Sharpe ratio of
1.0 a year, that trial should be the best and the probability of backtest overfitting should
fall. The four clauses of gate G3 test these answers. No licensed data are read.

The draws come from the seed's stream for this simulation, with one sub-stream per clause,
so that each clause's draws are the same whatever the number of repetitions of the others.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import platform

import numpy as np
import pandas as pd
import scipy

from . import constants as C
from . import overfitting as O

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "simulation"
CLAUSE_STREAMS = {"a_b": 0, "c": 1, "d": 2}


def rng_for(clause: str) -> np.random.Generator:
    """The random generator of one clause: the study's seed, this simulation's stream and the clause's sub-stream."""
    return np.random.default_rng([C.SEED, C.SEED_STREAMS["known_answer"], CLAUSE_STREAMS[clause]])


def made_up_returns(rng: np.random.Generator, months: int, trials: int, planted_monthly_sharpe: float = 0.0) -> np.ndarray:
    """A months-by-trials matrix of independent standard normal returns. The first trial's
    mean is `planted_monthly_sharpe`, which is then its true monthly Sharpe ratio, since
    the standard deviation is one."""
    r = rng.standard_normal((months, trials))
    r[:, 0] += planted_monthly_sharpe
    return r


def planted_monthly_sharpe() -> float:
    """The planted trial's true Sharpe ratio a month: 1.0 a year over the square root of 12."""
    return C.G3_D_ANNUAL_SHARPE / math.sqrt(C.MONTHS_PER_YEAR)


def clauses_a_b(repetitions: int = C.G3_REPETITIONS_A_B, months: int = C.G3_MONTHS, trials: int = C.G3_TRIALS):
    """Zero means. Per repetition: the highest Sharpe ratio of the trials and the deflated
    Sharpe ratio of the trial that has it."""
    rng = rng_for("a_b")
    rows = []
    for i in range(repetitions):
        out = O.deflated_sharpe_ratio(made_up_returns(rng, months, trials))
        rows.append((i, out["best_sharpe"], out["deflated_sharpe_ratio"]))
    return pd.DataFrame(rows, columns=["repetition", "best_sharpe", "deflated_sharpe_ratio"])


def clause_c(repetitions: int = C.G3_REPETITIONS_C_D, months: int = C.G3_MONTHS, trials: int = C.G3_TRIALS,
             blocks: int = C.G3_BLOCKS):
    """Zero means. Per repetition: the probability of backtest overfitting."""
    rng = rng_for("c")
    rows = []
    for i in range(repetitions):
        out = O.probability_of_backtest_overfitting(made_up_returns(rng, months, trials), blocks)
        rows.append((i, out["pbo"]))
    return pd.DataFrame(rows, columns=["repetition", "pbo"])


def clause_d(repetitions: int = C.G3_REPETITIONS_C_D, months: int = C.G3_MONTHS, trials: int = C.G3_TRIALS,
             blocks: int = C.G3_BLOCKS):
    """One trial with a true Sharpe ratio of 1.0 a year. Per repetition: whether it has the
    highest Sharpe ratio over the whole window, and the probability of backtest overfitting."""
    rng = rng_for("d")
    planted = planted_monthly_sharpe()
    rows = []
    for i in range(repetitions):
        r = made_up_returns(rng, months, trials, planted)
        best = int(np.argmax(O.sharpe_ratio(r)))
        out = O.probability_of_backtest_overfitting(r, blocks)
        rows.append((i, best == 0, out["pbo"]))
    return pd.DataFrame(rows, columns=["repetition", "planted_is_best", "pbo"])


def verdicts(ab: pd.DataFrame, c: pd.DataFrame, d: pd.DataFrame, months: int = C.G3_MONTHS,
             trials: int = C.G3_TRIALS) -> dict:
    """The four clauses of gate G3, each with its figures, its level and whether it is met."""
    expected = O.expected_max_sharpe(trials, 1 / (months - 1))
    mean_best = float(ab["best_sharpe"].mean())
    ratio = mean_best / expected
    share_above = float((ab["deflated_sharpe_ratio"] > C.DSR_LEVEL).mean())
    median_dsr = float(ab["deflated_sharpe_ratio"].median())
    mean_pbo_c = float(c["pbo"].mean())
    share_best = float(d["planted_is_best"].mean())
    mean_pbo_d = float(d["pbo"].mean())
    lo_b, hi_b = C.G3_B_MEDIAN_RANGE
    lo_c, hi_c = C.G3_C_PBO_RANGE
    out = {
        "a": {"repetitions": int(len(ab)), "mean_best_sharpe": mean_best, "expected_best_sharpe": expected,
              "ratio": ratio, "relative_tolerance": C.G3_A_TOLERANCE,
              "met": bool(abs(ratio - 1) <= C.G3_A_TOLERANCE)},
        "b": {"repetitions": int(len(ab)), "share_above_level": share_above, "level": C.DSR_LEVEL,
              "share_max": C.G3_B_SHARE_ABOVE_LEVEL_MAX, "median": median_dsr, "median_range": [lo_b, hi_b],
              "met": bool(share_above <= C.G3_B_SHARE_ABOVE_LEVEL_MAX and lo_b <= median_dsr <= hi_b)},
        "c": {"repetitions": int(len(c)), "mean_pbo": mean_pbo_c, "sd_pbo": float(c["pbo"].std(ddof=1)),
              "range": [lo_c, hi_c], "met": bool(lo_c <= mean_pbo_c <= hi_c)},
        "d": {"repetitions": int(len(d)), "planted_annual_sharpe": C.G3_D_ANNUAL_SHARPE,
              "share_planted_best": share_best, "share_min": C.G3_D_BEST_SHARE_MIN,
              "mean_pbo": mean_pbo_d, "pbo_max": C.G3_D_PBO_MAX,
              "met": bool(share_best >= C.G3_D_BEST_SHARE_MIN and mean_pbo_d < C.G3_D_PBO_MAX)},
    }
    out["all_met"] = all(out[k]["met"] for k in "abcd")
    return out


def settings() -> dict:
    """The settings the run used, read from the constants."""
    return {"months": C.G3_MONTHS, "trials": C.G3_TRIALS, "blocks": C.G3_BLOCKS,
            "repetitions_a_b": C.G3_REPETITIONS_A_B, "repetitions_c_d": C.G3_REPETITIONS_C_D,
            "seed": C.SEED, "stream": C.SEED_STREAMS["known_answer"], "clause_streams": CLAUSE_STREAMS}


def write_results(ab: pd.DataFrame, c: pd.DataFrame, d: pd.DataFrame, summary: dict, written_by: str) -> list:
    """Write the run's tables and its record into results/simulation/. The tables and the
    summary depend on the seed alone. The provenance record holds the versions and the time."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    ab.to_csv(RESULTS_DIR / "g3_repetitions_a_b.csv", index=False, float_format="%.12g")
    c.to_csv(RESULTS_DIR / "g3_repetitions_c.csv", index=False, float_format="%.12g")
    d.to_csv(RESULTS_DIR / "g3_repetitions_d.csv", index=False, float_format="%.12g")
    paths += [RESULTS_DIR / n for n in ("g3_repetitions_a_b.csv", "g3_repetitions_c.csv", "g3_repetitions_d.csv")]
    with open(RESULTS_DIR / "g3_summary.json", "w", encoding="utf-8") as f:
        json.dump({"settings": settings(), "clauses": summary}, f, indent=1)
        f.write("\n")
    record = {
        "written_by": written_by,
        "written_at": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "data": "made-up returns only, no licensed data",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "platform": platform.platform()},
    }
    with open(RESULTS_DIR / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    return paths + [RESULTS_DIR / "g3_summary.json", RESULTS_DIR / "provenance.json"]
