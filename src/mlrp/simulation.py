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

Three sets of figures beside the clauses were added on 24 November 2026, after the
simulation's results were known, and no level reads them: the deflated Sharpe ratio of the
best trial on clause (d)'s own draws, the detectable effects (how both statistics respond to
one trial with a true Sharpe ratio from 0 to 1.5 a year), and the share beside clause (d)'s
bound over further repetitions. The last two draw from sub-streams of their own (constants
section 14).
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


def rng_beside(name: str, *position: int) -> np.random.Generator:
    """The random generator of a set of figures beside the clauses: the study's seed, this simulation's
    stream, the set's sub-stream and, for the detectable effects, the position of the row."""
    return np.random.default_rng([C.SEED, C.SEED_STREAMS["known_answer"], C.BESIDE_G3_SUB_STREAMS[name], *position])


def made_up_returns(rng: np.random.Generator, months: int, trials: int, planted_monthly_sharpe: float = 0.0) -> np.ndarray:
    """A months-by-trials matrix of independent standard normal returns. The first trial's
    mean is `planted_monthly_sharpe`, which is then its true monthly Sharpe ratio, since
    the standard deviation is one."""
    r = rng.standard_normal((months, trials))
    r[:, 0] += planted_monthly_sharpe
    return r


def planted_monthly_sharpe(annual: float = C.G3_D_ANNUAL_SHARPE) -> float:
    """The planted trial's true Sharpe ratio a month: its Sharpe ratio a year, 1.0 in clause (d), over the
    square root of 12."""
    return annual / math.sqrt(C.MONTHS_PER_YEAR)


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
                    trials: int = C.G3_TRIALS, blocks: int = C.G3_BLOCKS, rng: np.random.Generator = None) -> pd.DataFrame:
    """The draws of clause (d) again, split by split. Per repetition, counts out of the splits:
    the splits in which the planted trial is the in-sample best, the splits in which another
    trial is the in-sample best and ranks in the lower half out-of-sample, and the splits in
    which the planted trial is the in-sample best and ranks there. The last two together are
    the splits that the probability of backtest overfitting counts. With `rng` given, the
    repetitions are new draws of clause (d)'s setting from that generator."""
    rng = rng_for("d") if rng is None else rng
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
    lower half out-of-sample is pooled over the repetitions, with its standard error over the repetitions and
    the bound it cannot exceed on average. The bound: the two halves of the months are independent, so the
    out-of-sample rank of a picked trial without an effect is uniform among the trials - 1 trials without one.
    It ends in the lower half only if at most trials / 2 - 1 of them are below it, which has a probability of
    (trials / 2) / (trials - 1), 16/31 for 32 trials, and less when the trial with the effect is below it as
    well. No level reads any of it."""
    n = splits["splits"]
    share_planted = splits["planted_picked"] / n
    other = n - splits["planted_picked"]
    other_picked = other.sum()
    lower = splits["other_picked_lower"]
    share_lower = lower.sum() / other_picked if other_picked else np.nan
    # The standard error of a ratio of two sums over independent repetitions: the standard deviation of
    # lower - share x other, over the square root of the repetitions, divided by the mean of other.
    se_lower = (float(np.std(lower - share_lower * other, ddof=1)) / np.sqrt(len(splits)) / float(other.mean())
                if other_picked and len(splits) > 1 else np.nan)
    return {"repetitions": int(len(splits)), "splits": int(n.iloc[0]),
            "share_planted_picked_mean": float(share_planted.mean()),
            "share_planted_picked_median": float(share_planted.median()),
            "share_other_picked_mean": float(1 - share_planted.mean()),
            "share_lower_when_other_picked": float(share_lower) if other_picked else None,
            "share_lower_when_other_picked_se": float(se_lower) if other_picked else None,
            "share_lower_bound_without_effect": (C.G3_TRIALS // 2) / (C.G3_TRIALS - 1),
            "pbo_from_other_picked": float((splits["other_picked_lower"] / n).mean()),
            "pbo_from_planted_picked": float((splits["planted_picked_lower"] / n).mean()),
            "mean_pbo": float(((splits["other_picked_lower"] + splits["planted_picked_lower"]) / n).mean())}


def clause_d_deflated(repetitions: int = C.G3_REPETITIONS_C_D, months: int = C.G3_MONTHS,
                      trials: int = C.G3_TRIALS) -> pd.DataFrame:
    """The draws of clause (d) again. Per repetition: whether the planted trial has the highest Sharpe
    ratio, and the deflated Sharpe ratio of the trial that has it, with its threshold, the expected best
    Sharpe ratio at the variance of the 32 trials' Sharpe ratios. Beside them, added on 25 November 2026,
    the same deflated Sharpe ratio and threshold with the variance taken from the 31 trials without an
    effect alone, the number of trials still 32."""
    rng = rng_for("d")
    planted = planted_monthly_sharpe()
    rows = []
    for i in range(repetitions):
        r = made_up_returns(rng, months, trials, planted)
        out = O.deflated_sharpe_ratio(r)
        variance_31 = float(np.var(O.sharpe_ratio(r)[1:], ddof=1))
        rows.append((i, out["best_trial"] == 0, out["deflated_sharpe_ratio"], out["threshold"],
                     O.deflated_sharpe_ratio_from_moments(out["best_sharpe"], out["n_obs"], out["skewness"],
                                                          out["kurtosis"], trials, variance_31),
                     O.expected_max_sharpe(trials, variance_31)))
    return pd.DataFrame(rows, columns=["repetition", "planted_is_best", "deflated_sharpe_ratio", "threshold",
                                       "deflated_sharpe_ratio_variance_of_31", "threshold_variance_of_31"])


def beside_clause_d_deflated(dd: pd.DataFrame, months: int = C.G3_MONTHS, trials: int = C.G3_TRIALS) -> dict:
    """The deflated Sharpe ratio on clause (d)'s repetitions: the share above the level, its median and the
    mean threshold, beside the threshold with no effect at the variance of 1 / (T - 1). With the columns of
    clause_d_deflated() added on 25 November 2026, the same three figures with the variance of the 31 trials
    without an effect alone. No level reads them."""
    out = {"repetitions": int(len(dd)), "level": C.DSR_LEVEL,
           "share_above_level": float((dd["deflated_sharpe_ratio"] > C.DSR_LEVEL).mean()),
           "median": float(dd["deflated_sharpe_ratio"].median()),
           "mean_threshold": float(dd["threshold"].mean()),
           "threshold_without_effect": float(O.expected_max_sharpe(trials, 1 / (months - 1)))}
    if "deflated_sharpe_ratio_variance_of_31" in dd:
        out["with_the_variance_of_31_without_effect_added_on_25_november_2026"] = {
            "share_above_level": float((dd["deflated_sharpe_ratio_variance_of_31"] > C.DSR_LEVEL).mean()),
            "median": float(dd["deflated_sharpe_ratio_variance_of_31"].median()),
            "mean_threshold": float(dd["threshold_variance_of_31"].mean())}
    return out


def detectable_effects(annual_sharpes=C.G3_DETECTABLE_ANNUAL_SHARPES, repetitions: int = C.G3_DETECTABLE_REPETITIONS,
                       months: int = C.G3_MONTHS, trials: int = C.G3_TRIALS, blocks: int = C.G3_BLOCKS) -> pd.DataFrame:
    """One row per true Sharpe ratio a year of the first trial, with the other trials at zero. Each row's
    repetitions come from its own position on the sub-stream of the detectable effects. Per row: the share
    of repetitions in which the first trial has the highest Sharpe ratio, the share in which the best trial's
    deflated Sharpe ratio is above the level and its median, the mean probability of backtest overfitting and
    the share below 0.25, and the shares in which expectations E8 and E9 would be met. No level reads it."""
    rows = []
    for position, annual in enumerate(annual_sharpes):
        rng = rng_beside("detectable_effects", position)
        planted = planted_monthly_sharpe(annual)
        best, dsr, pbo = [], [], []
        for _ in range(repetitions):
            r = made_up_returns(rng, months, trials, planted)
            out = O.deflated_sharpe_ratio(r)
            best.append(out["best_trial"] == 0)
            dsr.append(out["deflated_sharpe_ratio"])
            pbo.append(O.probability_of_backtest_overfitting(r, blocks)["pbo"])
        best, dsr, pbo = np.array(best), np.array(dsr), np.array(pbo)
        rows.append({"annual_sharpe": float(annual), "monthly_sharpe": planted, "repetitions": int(repetitions),
                     "share_planted_best": float(best.mean()),
                     "share_dsr_above_0.95": float((dsr > C.DSR_LEVEL).mean()),
                     "median_dsr": float(np.median(dsr)),
                     "mean_pbo": float(pbo.mean()),
                     "share_pbo_below_0.25": float((pbo < C.E9_PBO_MIN).mean()),
                     "share_e8_met": float((dsr < C.DSR_LEVEL).mean()),
                     "share_e9_met": float((pbo >= C.E9_PBO_MIN).mean())})
    return pd.DataFrame(rows)


def bound_share(repetitions: int = C.G3_BOUND_REPETITIONS) -> dict:
    """The share beside clause (d)'s bound over further repetitions of clause (d)'s setting, drawn from the
    sub-stream of the bound: of the splits in which another trial is picked, the share that ends in the
    lower half out-of-sample, with its standard error, the bound and the distance from it in standard
    errors. No level reads it."""
    out = beside_clause_d(clause_d_splits(repetitions, rng=rng_beside("bound")))
    share, se, bound = out["share_lower_when_other_picked"], out["share_lower_when_other_picked_se"], out["share_lower_bound_without_effect"]
    return {"repetitions": out["repetitions"], "share_lower_when_other_picked": share, "standard_error": se,
            "bound": bound, "distance_from_bound_in_standard_errors": (share - bound) / se,
            "mean_pbo": out["mean_pbo"]}


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
    so that a run on another machine writes the same bytes, in nested figures too."""
    return {k: float(f"{v:.12g}") if isinstance(v, float) else _twelve_digits(v) if isinstance(v, dict) else v
            for k, v in figures.items()}


def settings_beside() -> dict:
    """The settings of the figures beside the clauses added on 24 November 2026, read from the constants."""
    return {"sub_streams": C.BESIDE_G3_SUB_STREAMS, "detectable_annual_sharpes": list(C.G3_DETECTABLE_ANNUAL_SHARPES),
            "detectable_repetitions": C.G3_DETECTABLE_REPETITIONS, "bound_repetitions": C.G3_BOUND_REPETITIONS}


def write_figures_beside(a: dict, d: dict, written_by: str, directory: pathlib.Path = RESULTS_DIR,
                         d_deflated: dict = None, bound: dict = None) -> pathlib.Path:
    """Write the figures beside clauses (a) and (d) to a file of their own in results/simulation/,
    with the deflated Sharpe ratio on clause (d)'s repetitions and the bound's share over further
    repetitions when they are given. They depend on the seed alone, and no level reads them."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "g3_figures_beside.json"
    record = {"written_by": written_by, "settings": settings(), "beside_clause_a": _twelve_digits(a),
              "beside_clause_d": _twelve_digits(d)}
    if d_deflated is not None or bound is not None:
        record["settings_added_on_24_november_2026"] = settings_beside()
    if d_deflated is not None:
        record["deflated_sharpe_ratio_on_clause_d"] = _twelve_digits(d_deflated)
    if bound is not None:
        record["bound_further_repetitions"] = _twelve_digits(bound)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    return path


def write_detectable_effects(table: pd.DataFrame, directory: pathlib.Path = RESULTS_DIR) -> pathlib.Path:
    """Write the detectable effects to results/simulation/g3_detectable_effects.csv. They depend on the seed
    alone, and no level reads them."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "g3_detectable_effects.csv"
    table.to_csv(path, index=False, float_format="%.12g")
    return path
