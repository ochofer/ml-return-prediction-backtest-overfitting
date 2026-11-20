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

The draws come from the seed's stream for this simulation, with one sub-stream per clause
(constants section 14), so that each clause's draws are the same whatever the number of
repetitions of the others.

Beside the clauses, and read by no level, the module computes the exact expected best Sharpe
ratio of trials with no effect, against which the formula of clause (a) is an approximation,
and the splits of clause (d): how often the trial with the true effect is picked on one half
of the months, and which splits make up the probability of backtest overfitting.
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
from scipy import integrate, stats

from . import constants as C
from . import overfitting as O

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "simulation"


def rng_for(clause: str) -> np.random.Generator:
    """The random generator of one clause: the study's seed, this simulation's stream and the clause's sub-stream."""
    return np.random.default_rng([C.SEED, C.SEED_STREAMS["known_answer"], C.KNOWN_ANSWER_SUB_STREAMS[clause]])


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


def expected_max_standard_normal(trials: int) -> float:
    """The expected highest of `trials` independent standard normal numbers, by numerical
    integration: the integral of x times the density of the highest, N f(x) F(x)^(N-1), where
    f and F are the standard normal density and distribution function."""
    def integrand(x):
        return x * trials * stats.norm.pdf(x) * stats.norm.cdf(x) ** (trials - 1)
    value, _ = integrate.quad(integrand, -np.inf, np.inf)
    return float(value)


def sharpe_sd_no_effect(months: int) -> float:
    """The standard deviation of one trial's monthly Sharpe ratio over `months` independent
    normal months with a true mean of zero. The square root of the number of months times the
    Sharpe ratio is the t-statistic of the mean, which follows Student's t distribution with
    months - 1 degrees of freedom. Its variance is (T - 1) / (T - 3), so the Sharpe ratio's
    standard deviation is the square root of (T - 1) / ((T - 3) T)."""
    return math.sqrt(float(stats.t(months - 1).var()) / months)


def exact_expected_best_sharpe(months: int = C.G3_MONTHS, trials: int = C.G3_TRIALS) -> float:
    """The expected highest of `trials` independent monthly Sharpe ratios with no effect over
    `months` normal months, by numerical integration over the Student's t distribution of the
    square root of the months times each Sharpe ratio. This is the exact value that the formula
    of Bailey and López de Prado (2014) approximates."""
    t = stats.t(months - 1)

    def integrand(x):
        return x * trials * t.pdf(x) * t.cdf(x) ** (trials - 1)
    value, _ = integrate.quad(integrand, -np.inf, np.inf)
    return float(value) / math.sqrt(months)


def beside_clause_a(ab: pd.DataFrame, months: int = C.G3_MONTHS, trials: int = C.G3_TRIALS) -> dict:
    """The simulation's mean best Sharpe ratio against the exact expected value, with the mean's
    standard error, the standard deviation of the best Sharpe ratios over the square root of the
    number of repetitions, and the distance between the two in standard errors. No level reads it."""
    exact = exact_expected_best_sharpe(months, trials)
    mean = float(ab["best_sharpe"].mean())
    se = float(ab["best_sharpe"].std(ddof=1)) / math.sqrt(len(ab))
    return {"repetitions": int(len(ab)), "mean_best_sharpe": mean, "standard_error": se,
            "exact_expected_best_sharpe": exact, "distance_in_standard_errors": (mean - exact) / se,
            "mean_over_exact": mean / exact,
            "formula_expected_best_sharpe": O.expected_max_sharpe(trials, 1 / (months - 1)),
            "formula_over_exact": O.expected_max_sharpe(trials, 1 / (months - 1)) / exact,
            "expected_max_standard_normal": expected_max_standard_normal(trials),
            "formula_expected_max_standard_normal": O.expected_max_sharpe(trials, 1.0),
            "sharpe_sd_no_effect": sharpe_sd_no_effect(months),
            "formula_sharpe_sd": math.sqrt(1 / (months - 1))}


def clause_d_splits(repetitions: int = C.G3_REPETITIONS_C_D, months: int = C.G3_MONTHS,
                    trials: int = C.G3_TRIALS, blocks: int = C.G3_BLOCKS) -> pd.DataFrame:
    """The draws of clause (d) again, split by split. Per repetition, counts out of the splits:
    the splits in which the planted trial is the in-sample best, the splits in which another
    trial is the in-sample best and ranks in the lower half out-of-sample, and the splits in
    which the planted trial is the in-sample best and ranks there. The last two together are
    the splits that the probability of backtest overfitting counts."""
    rng = rng_for("d")
    planted = planted_monthly_sharpe()
    rows = []
    for i in range(repetitions):
        r = made_up_returns(rng, months, trials, planted)
        out = O.probability_of_backtest_overfitting(r, blocks)
        picked = out["best_in_sample"] == 0
        lower = out["logits"] <= 0
        rows.append((i, out["splits"], int(picked.sum()), int((~picked & lower).sum()), int((picked & lower).sum())))
    return pd.DataFrame(rows, columns=["repetition", "splits", "planted_picked", "other_picked_lower", "planted_picked_lower"])


def beside_clause_d(splits: pd.DataFrame) -> dict:
    """The share of the splits in which the planted trial is the in-sample best, as its mean and
    its median over the repetitions, and the mean probability of backtest overfitting in two
    parts: from the splits in which another trial is picked, and from those in which the planted
    trial is picked. Of the splits in which another trial is picked, the share that ends in the
    lower half out-of-sample is pooled over the repetitions. No level reads it."""
    n = splits["splits"]
    share_planted = splits["planted_picked"] / n
    other_picked = (n - splits["planted_picked"]).sum()
    return {"repetitions": int(len(splits)), "splits": int(n.iloc[0]),
            "share_planted_picked_mean": float(share_planted.mean()),
            "share_planted_picked_median": float(share_planted.median()),
            "share_other_picked_mean": float(1 - share_planted.mean()),
            "share_lower_when_other_picked": float(splits["other_picked_lower"].sum() / other_picked) if other_picked else None,
            "pbo_from_other_picked": float((splits["other_picked_lower"] / n).mean()),
            "pbo_from_planted_picked": float((splits["planted_picked_lower"] / n).mean()),
            "mean_pbo": float(((splits["other_picked_lower"] + splits["planted_picked_lower"]) / n).mean())}


def settings() -> dict:
    """The settings the run used, read from the constants."""
    return {"months": C.G3_MONTHS, "trials": C.G3_TRIALS, "blocks": C.G3_BLOCKS,
            "repetitions_a_b": C.G3_REPETITIONS_A_B, "repetitions_c_d": C.G3_REPETITIONS_C_D,
            "seed": C.SEED, "stream": C.SEED_STREAMS["known_answer"], "clause_streams": C.KNOWN_ANSWER_SUB_STREAMS}


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


def _twelve_digits(figures: dict) -> dict:
    """The figures with every number rounded to 12 significant digits, as the tables are written,
    so that a run on another machine writes the same bytes."""
    return {k: float(f"{v:.12g}") if isinstance(v, float) else v for k, v in figures.items()}


def write_figures_beside(a: dict, d: dict, written_by: str, directory: pathlib.Path = RESULTS_DIR) -> pathlib.Path:
    """Write the figures beside clauses (a) and (d) to a file of their own in results/simulation/.
    They depend on the seed alone, and no level reads them."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "g3_figures_beside.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"written_by": written_by, "settings": settings(), "beside_clause_a": _twelve_digits(a),
                   "beside_clause_d": _twelve_digits(d)}, f, indent=1)
        f.write("\n")
    return path
