"""
Builds the study's notebooks from the cells written below, so that each notebook's text lives in one
place and every notebook is committed without outputs. Run from the repository's root:

    python3 build_notebooks.py

Each notebook opens with what it does, why it comes at this point, what it judges and what to expect,
and closes with what was learned and what could be wrong.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"

SETUP = '''import pathlib
import sys

# The package lives in src/ beside this folder. The notebook finds the repository's root from where it runs.
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "src" / "mlrp").is_dir())
sys.path.insert(0, str(ROOT / "src"))'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(number, cells):
    # Each cell carries an id, as nbformat 4.5 requires, made of the notebook's number and the cell's position,
    # so that a build writes the same bytes every time.
    cells = [dict(c, id=f"{number}-{i:02d}") for i, c in enumerate(cells)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


NB12 = [
    md('''
# 12 The known-answer simulation: gate G3

**What this notebook does.** It tests the code of the study's two overfitting statistics on made-up returns whose answer is known. Each repetition makes up 228 months of returns for 32 trials, with a random generator, from the standard normal distribution. With a mean of zero in every column, no trial has a true effect, and the statistics have to say so. A last set of repetitions gives one trial a true effect, and measures how the two statistics respond.

**Why it comes at this point.** Gate G3 must pass before either statistic is computed on the study's own returns. It reads no data of any kind, so it runs as soon as its settings are in the repository. The notebook needs no data root.

**What it judges.** Gate G3, in four clauses, (a) to (d).

**What to expect.** The numbers under "What I expect to see", each with the reason behind it. The whole run takes about two minutes, most of it in the further repetitions of step 6.
'''),
    md('''
## Terms used in this notebook

| Term | Meaning |
|---|---|
| trial | one column of monthly returns over the test window. In the study a trial is the long-short portfolio of one configuration, net of costs. Here it is made up. There are 32 |
| repetition | one new table of made-up returns, 228 months by 32 trials |
| seed | the number that starts the random generator, so that every run makes up the same returns |
| draw | one number the random generator produces |
| Sharpe ratio | the mean monthly return over the standard deviation of the monthly returns. In this notebook it is monthly and never annualised |
| standard error | the standard deviation of an estimate across repetitions: how far an estimate typically lands from its true value |
| expected best Sharpe ratio | the mean of the highest of N Sharpe ratios when no trial has a true effect, from the formula of Bailey and López de Prado (2014) |
| Student's t distribution | the distribution of a sample's mean over its estimated standard error, when the sample is normal with a true mean of zero. It has one setting, the degrees of freedom: the sample's length less one. With many degrees of freedom it is close to the standard normal distribution, with slightly more extreme values |
| numerical integration | computing an integral from the values of its function at many points, here with `scipy.integrate.quad` |
| kurtosis | the mean of the fourth power of the returns' distance from their mean, over the square of their variance. It is 3 for normal returns |
| deflated Sharpe ratio | the confidence, from 0 to 1, that the best trial's true Sharpe ratio is above the expected best Sharpe ratio of N trials with no true effect (Bailey and López de Prado 2014) |
| in-sample | the half of the months on which the best trial is picked |
| out-of-sample | the other half, on which the picked trial is ranked against the others |
| probability of backtest overfitting | the share of the ways of splitting the months into two halves in which the trial that is best in-sample ranks in the lower half out-of-sample (Bailey, Borwein, López de Prado and Zhu 2017) |
| logit | the number log(w / (1 - w)) of a relative rank w between 0 and 1. It is at or below zero when w is at or below one half |
| verdict | met or not met, written in the same words either way |
| provenance | the record of what a result was built from: the settings, the seed and the versions of the software |
'''),
    md('''
## What I expect to see

The levels were set on 18 November 2026 and stand in `src/mlrp/constants.py`, section 12, with the seed in section 14.

1. **Clause (a).** Over 4,000 repetitions with zero means, the mean of the best of the 32 Sharpe ratios lies within 5 per cent of the expected best Sharpe ratio. At 228 months and 32 trials the formula gives 0.1394 a month (step 2 computes it). The formula approximates the highest of 32 normal numbers, so the simulation's mean should sit near it and need not equal it.
2. **Clause (b).** In the same 4,000 repetitions, the best trial's deflated Sharpe ratio is above 0.95 in at most 5 per cent of them, and its median lies between 0.40 and 0.60. With no true effect the best Sharpe ratio scatters around the expected best, so the deflated Sharpe ratio scatters around one half and rarely reaches 0.95.
3. **Clause (c).** Over 1,000 new repetitions with zero means, the mean probability of backtest overfitting lies between 0.45 and 0.55. With no true effect, the trial that is best in-sample is an ordinary trial out-of-sample, so it lands in the lower half of the ranks in half of the splits.
4. **Clause (d).** With the first trial given a true Sharpe ratio of 1.0 a year, which is 1 / √12 = 0.289 a month, that trial has the best Sharpe ratio over the 228 months in at least 90 per cent of 1,000 repetitions, and the mean probability of backtest overfitting is below 0.25. Over 228 months a trial's Sharpe ratio has a standard error near 1 / √227 = 0.066 a month, so 0.289 stands more than four standard errors above zero and well above 0.139, the expected best of the other 31 trials, which have no effect. The words of the first part of (d) were written on 20 November 2026, after the run of 19 November. Its level stands as committed before the run.

A clause not met stops the study before notebook 13. The statistic's code is then corrected, each change recorded, the same seed run again, and no level changes.
'''),
    code(SETUP + '''

import math

import numpy as np
import pandas as pd

from mlrp import constants as C
from mlrp import overfitting as O
from mlrp import simulation as S

print("months", C.G3_MONTHS, "| trials", C.G3_TRIALS, "| blocks", C.G3_BLOCKS,
      "| repetitions", C.G3_REPETITIONS_A_B, "and", C.G3_REPETITIONS_C_D, "| seed", C.SEED)'''),
    md('''
## 1. One repetition

The random generator starts from the study's seed, on the stream kept for this simulation, with one sub-stream for each clause. `numpy.random.default_rng` creates the generator, and its method `standard_normal` fills a table of the given shape with draws from the standard normal distribution: mean zero, standard deviation one. The cell makes the first repetition of clauses (a) and (b) and shows its first five months for its first five trials, then the Sharpe ratio of every trial. The true standard deviation is one, and over 228 months each trial's sample standard deviation is close to one, so a trial's Sharpe ratio is close to its mean return.
'''),
    code('''
rng = S.rng_for("a_b")
r = S.made_up_returns(rng, C.G3_MONTHS, C.G3_TRIALS)
print("shape:", r.shape)
display(pd.DataFrame(r[:5, :5]).round(3))

sharpe = O.sharpe_ratio(r)
print("Sharpe ratios of the first eight trials:", sharpe[:8].round(3))
print("best:", round(float(sharpe.max()), 4), "for trial", int(sharpe.argmax()))'''),
    md('''
## 2. The expected best Sharpe ratio

With no true effect, a trial's monthly Sharpe ratio over T months has a variance close to 1 / (T - 1), so its standard deviation is close to 1 / √227 = 0.066 at T = 228. The highest of N such numbers is larger than any one of them on average. Bailey and López de Prado (2014), equation (2), give its expected value as

`SR0 = √V × ((1 - g) × Zinv(1 - 1/N) + g × Zinv(1 - 1/(N e)))`

where V is the variance of the trials' Sharpe ratios, g = 0.5772 is the Euler-Mascheroni constant, e is Euler's number and Zinv is the inverse of the standard normal distribution. The cell computes it for N = 32 and V = 1 / 227, the number clause (a) compares with. It then computes the deflated Sharpe ratio of the repetition of step 1. That statistic asks how sure one can be that the best trial's true Sharpe ratio is above SR0, given the sample's length, its skewness and its kurtosis:

`DSR = Z((SR - SR0) √(T - 1) / √(1 - skewness × SR + (kurtosis - 1) / 4 × SR²))`

where Z is the standard normal distribution and SR the best trial's Sharpe ratio. In the study V is the variance of the 32 trials' own Sharpe ratios, as here.
'''),
    code('''
expected = O.expected_max_sharpe(C.G3_TRIALS, 1 / (C.G3_MONTHS - 1))
print("expected best Sharpe ratio, 32 trials, 228 months:", round(expected, 4))

one = O.deflated_sharpe_ratio(r)
print("this repetition: best Sharpe ratio", round(one["best_sharpe"], 4),
      "| threshold", round(one["threshold"], 4), "| deflated Sharpe ratio", round(one["deflated_sharpe_ratio"], 4))'''),
    md('''
## 3. Clauses (a) and (b): 4,000 repetitions with zero means

Each repetition keeps two numbers: the best Sharpe ratio and the deflated Sharpe ratio of the trial that has it. `describe()` is a pandas call that lists a column's count, mean, standard deviation, smallest value, quartiles and largest value.
'''),
    code('''
ab = S.clauses_a_b()
ab[["best_sharpe", "deflated_sharpe_ratio"]].describe().round(4)'''),
    md('''
**The exact expected best Sharpe ratio.** The figures of this part of the step were added on 20 November 2026, after the run of 19 November, and no level reads them.

The formula of step 2 is an approximation. Its multiplier, the part in brackets, approximates the expected highest of 32 standard normal numbers: it gives 2.0999, where numerical integration gives 2.0697. The exact expected best Sharpe ratio of 32 trials with no effect follows from the distribution of one trial's Sharpe ratio:

- √T times a trial's Sharpe ratio is the t-statistic of its mean return, which follows Student's t distribution with T - 1 = 227 degrees of freedom.
- That distribution has a variance of (T - 1) / (T - 3), so one Sharpe ratio has a standard deviation of √((T - 1) / ((T - 3) T)) = 0.0665 at T = 228, where the formula uses √(1 / 227) = 0.0664.
- The expected highest of 32 independent Sharpe ratios of that distribution is an integral, which `scipy.integrate.quad` computes: 0.1380 a month, where the formula gives 0.1394.

The cell prints these with the simulation's mean best Sharpe ratio, its standard error (the standard deviation of the 4,000 best Sharpe ratios over √4,000) and the distance between the mean and the exact value in standard errors.
'''),
    code('''
a_beside = S.beside_clause_a(ab)
rows = [
    ("expected highest of 32 standard normal numbers", a_beside["expected_max_standard_normal"], a_beside["formula_expected_max_standard_normal"]),
    ("standard deviation of one Sharpe ratio, no effect", a_beside["sharpe_sd_no_effect"], a_beside["formula_sharpe_sd"]),
    ("expected best Sharpe ratio", a_beside["exact_expected_best_sharpe"], a_beside["formula_expected_best_sharpe"]),
]
display(pd.DataFrame(rows, columns=["figure", "exact", "formula"]).round(4))
print("simulation: mean best Sharpe ratio", round(a_beside["mean_best_sharpe"], 5),
      "| standard error", round(a_beside["standard_error"], 5),
      "| distance from the exact value", round(a_beside["distance_in_standard_errors"], 2), "standard errors")'''),
    md('''
## 4. The probability of backtest overfitting, on one repetition

The 228 months are cut into 12 blocks of 19 months in their order. Every choice of 6 of the 12 blocks is one split: those 6 blocks are the in-sample half and the other 6 the out-of-sample half, which gives 924 splits. In each split the trial with the highest in-sample Sharpe ratio is picked, and its out-of-sample Sharpe ratio is ranked among the 32, from 1 for the lowest to 32 for the highest. The rank over 33 is its relative rank w, and the logit log(w / (1 - w)) is at or below zero when the picked trial lands in the lower half. The probability of backtest overfitting is the share of the 924 splits with a logit at or below zero. The cell runs it on the repetition of step 1.
'''),
    code('''
pbo_one = O.probability_of_backtest_overfitting(r, C.G3_BLOCKS)
print("splits:", pbo_one["splits"], "| probability of backtest overfitting:", round(pbo_one["pbo"], 4))
pd.Series(pbo_one["logits"], name="logit").describe().round(3)'''),
    md('''
## 5. Clause (c): 1,000 repetitions with zero means

One estimate of the probability varies widely from one repetition to the next, which is why the clause reads the mean over 1,000 repetitions and the notebook shows the spread beside it.
'''),
    code('''
c = S.clause_c()
c["pbo"].describe().round(4)'''),
    md('''
## 6. Clause (d): one trial with a true effect

The first trial's returns get a mean of 0.289 a month, a true Sharpe ratio of 1.0 a year. Each repetition keeps whether that trial has the highest Sharpe ratio over the 228 months, and the probability of backtest overfitting.
'''),
    code('''
d = S.clause_d()
print("planted trial best in", round(100 * d["planted_is_best"].mean(), 1), "per cent of", len(d), "repetitions")
d["pbo"].describe().round(4)'''),
    md('''
**The deflated Sharpe ratio on clause (d)'s repetitions.** Added on 24 November 2026, after the simulation's results were known, and no level reads it.

Clause (d) tests the probability of backtest overfitting, and not the deflated Sharpe ratio. On its repetitions the best trial's deflated Sharpe ratio is above 0.95 in 0.542 of them, with a median of 0.957: the variance across trials includes the planted trial's Sharpe ratio, which raises the threshold from 0.1394 to 0.1753 a month on average.

Added on 25 November 2026, after the simulation's results were known, and no level reads it: with the variance of the 31 other trials alone the share is 0.720, so the larger part of the shortfall from one comes from the sample's 228 months.

The cell draws clause (d)'s 1,000 repetitions again, from its own sub-stream, and computes the deflated Sharpe ratio of the best trial in each, with its threshold, once with the variance of the 32 trials' Sharpe ratios and once with the variance of the 31 trials without an effect.
'''),
    code('''
dd = S.clause_d_deflated()
assert dd["planted_is_best"].equals(d["planted_is_best"])
d_deflated = S.beside_clause_d_deflated(dd)
rows = [
    ("share of repetitions with the best trial's deflated Sharpe ratio above 0.95", d_deflated["share_above_level"]),
    ("median deflated Sharpe ratio of the best trial", d_deflated["median"]),
    ("mean threshold, a month", d_deflated["mean_threshold"]),
    ("threshold with no effect at a variance of 1 / 227, a month", d_deflated["threshold_without_effect"]),
    ("share above 0.95 with the variance of the 31 trials without an effect",
     d_deflated["with_the_variance_of_31_without_effect_added_on_25_november_2026"]["share_above_level"]),
    ("mean threshold with that variance, a month",
     d_deflated["with_the_variance_of_31_without_effect_added_on_25_november_2026"]["mean_threshold"]),
]
display(pd.DataFrame(rows, columns=["figure", "value"]).round(4))'''),
    md('''
**The splits of clause (d).** The figures of this part of the step were added on 20 November 2026, after the run of 19 November, and no level reads them.

Clause (d) counts repetitions: in each one, the planted trial has or has not the highest Sharpe ratio over the 228 months. The probability of backtest overfitting counts splits, and in each split the best trial is picked on one half of the months, 6 blocks of 19, which is 114 months. For each repetition the cell counts three kinds of split:

- the splits in which the planted trial is the in-sample best
- the splits in which another trial is the in-sample best and ranks in the lower half out-of-sample
- the splits in which the planted trial is the in-sample best and ranks in the lower half out-of-sample

The last two kinds together are the splits that the probability of backtest overfitting counts, so its mean over the repetitions is the sum of two parts. A half of 114 months picks the planted trial less often than the whole window of 228 months does, and the splits in which another trial is picked are where a probability above zero comes from.
'''),
    code('''
splits = S.clause_d_splits()
d_beside = S.beside_clause_d(splits)
rows = [
    ("share of splits with the planted trial picked, mean over repetitions", d_beside["share_planted_picked_mean"]),
    ("share of splits with the planted trial picked, median over repetitions", d_beside["share_planted_picked_median"]),
    ("share of splits with another trial picked, mean over repetitions", d_beside["share_other_picked_mean"]),
    ("of the splits with another trial picked, the share in the lower half out-of-sample", d_beside["share_lower_when_other_picked"]),
    ("that share's standard error over the repetitions", d_beside["share_lower_when_other_picked_se"]),
    ("the bound on that share for a trial without an effect, 16/31", d_beside["share_lower_bound_without_effect"]),
    ("probability of backtest overfitting from the splits with another trial picked", d_beside["pbo_from_other_picked"]),
    ("probability of backtest overfitting from the splits with the planted trial picked", d_beside["pbo_from_planted_picked"]),
    ("mean probability of backtest overfitting, the two parts together", d_beside["mean_pbo"]),
]
display(pd.DataFrame(rows, columns=["figure", "value"]).round(4))'''),
    md('''
**The bound beside the share.** Added on 21 November 2026, after the simulation's results were known, with no verdict: of the splits in which another trial is picked, the share that ends in the lower half out-of-sample cannot exceed 16/31 = 0.516 on average, because the two halves of the months are independent, so the out-of-sample rank of a picked trial without an effect is uniform among the 31 trials without one, and it ends in the lower half only if at most 15 of them are below it. The standard error in the table is computed from the counts of each repetition, as the spread of a ratio of two sums over 1,000 independent repetitions.
'''),
    md('''
**The share over further repetitions.** Added on 24 November 2026, after the simulation's results were known, and no level reads it.

On these 1,000 repetitions the share is 0.543, 2.4 standard errors above the bound. Over 20,000 further repetitions it is 0.515, with a standard error of 0.0023.

The cell draws the further repetitions of clause (d)'s setting from the sub-stream of the bound (constants section 14). It takes about a minute.
'''),
    code('''
bound = S.bound_share()
rows = [
    ("share on clause (d)'s repetitions", d_beside["share_lower_when_other_picked"], d_beside["share_lower_when_other_picked_se"], d_beside["repetitions"]),
    ("share on further repetitions", bound["share_lower_when_other_picked"], bound["standard_error"], bound["repetitions"]),
]
display(pd.DataFrame(rows, columns=["figure", "share", "standard error", "repetitions"]).round(4))
print("bound 16/31:", round(bound["bound"], 4), "| distance of the further share from it:",
      round(bound["distance_from_bound_in_standard_errors"], 1), "standard errors")'''),
    md('''
**The detectable effects, beside clauses (b) and (d).** Added on 24 November 2026, after the simulation's results were known, and no level reads it.

Clause (b) has no trial with an effect, and clause (d) one trial of 1.0 a year. The table fills in between and above: in each row the first trial has the true Sharpe ratio a year of its row, 0, 0.25, 0.5, 0.75, 1.0 or 1.5, and the other 31 trials have none, at 228 months, 32 trials and 12 blocks, over 1,000 repetitions. Each row draws from its own position on the sub-stream of the detectable effects (constants section 14). For each row the table gives the share of repetitions in which the first trial is the best, the share in which the best trial's deflated Sharpe ratio is above 0.95 and its median, and the mean probability of backtest overfitting and the share of repetitions in which it is below 0.25.

Expectations E8 and E9 (constants section 13) read the same two statistics on the study's own 32 trials: E8 says the best trial's deflated Sharpe ratio is below 0.95, and E9 that the probability of backtest overfitting is at least 0.25. The last two columns give the share of repetitions in which each would be met. The cell takes about half a minute.
'''),
    code('''
effects = S.detectable_effects()
display(effects.drop(columns="repetitions").round(3))'''),
    md('''
With no effect in any of 32 trials, E8 would be met in 1.00 of the repetitions and E9 in 0.91. With a true Sharpe ratio of 0.5 a year in one of them, half the effect gate G3 plants, they would be met in 0.98 and 0.70. So a verdict of met on E8 hardly separates no effect from an effect of that size.
'''),
    md('''
## 7. The verdicts

The four clauses against their levels. The cell writes the three tables of repetitions, the summary and the provenance record to `results/simulation/`, and the figures of steps 3 and 6 that no level reads to two files of their own there, `g3_figures_beside.json` and `g3_detectable_effects.csv`.
'''),
    code('''
v = S.verdicts(ab, c, d)
word = {True: "met", False: "not met"}
rows = [
    ("(a)", "mean best Sharpe ratio over the expected best", round(v["a"]["ratio"], 4), "between 0.95 and 1.05", word[v["a"]["met"]]),
    ("(b)", "share of deflated Sharpe ratios above 0.95", round(v["b"]["share_above_level"], 4), "at most 0.05", ""),
    ("(b)", "median deflated Sharpe ratio", round(v["b"]["median"], 4), "between 0.40 and 0.60", word[v["b"]["met"]]),
    ("(c)", "mean probability of backtest overfitting, no effect", round(v["c"]["mean_pbo"], 4), "between 0.45 and 0.55", word[v["c"]["met"]]),
    ("(d)", "share of repetitions with the planted trial best", round(v["d"]["share_planted_best"], 4), "at least 0.90", ""),
    ("(d)", "mean probability of backtest overfitting, one effect", round(v["d"]["mean_pbo"], 4), "below 0.25", word[v["d"]["met"]]),
]
display(pd.DataFrame(rows, columns=["clause", "figure", "value", "level", "verdict"]))
print("expected best Sharpe ratio:", round(v["a"]["expected_best_sharpe"], 4), "| mean best:", round(v["a"]["mean_best_sharpe"], 4))
print("Gate G3:", word[v["all_met"]])

for path in S.write_results(ab, c, d, v, written_by="notebooks/12_known_answer.ipynb"):
    print("written:", path.relative_to(ROOT))
path = S.write_figures_beside(a_beside, d_beside, written_by="notebooks/12_known_answer.ipynb",
                              d_deflated=d_deflated, bound=bound)
print("written:", path.relative_to(ROOT))
path = S.write_detectable_effects(effects)
print("written:", path.relative_to(ROOT))'''),
    md('''
## What was learned, and what could be wrong

What the gate shows when its four clauses are met:

- The code of the expected best Sharpe ratio, of the deflated Sharpe ratio and of the probability of backtest overfitting returns the known answers on returns with no true effect, and when one trial has a true effect of 1.0 a year the mean probability of backtest overfitting falls from 0.498 to 0.100.
- The deflated Sharpe ratio of the best of 32 trials with no effect sits near one half: picking the best of 32 gives a Sharpe ratio that looks better than any single trial, and the statistic takes that out.

What could be wrong, and where each point is taken up:

- The returns here are independent and normal. The study's long-short returns have more extreme months than the normal distribution allows. The deflated Sharpe ratio takes the best trial's own skewness and kurtosis into account, and notebook 13 shows both beside it.
- The statistic counts the 32 trials as independent. The configurations of one model move together, so the study's trials are fewer in effect. Counting 32 makes the bar higher, and notebook 13 shows the paper's adjustment for correlated trials beside it, with no verdict.
- One estimate of the probability of backtest overfitting is noisy, as the spread in step 5 shows. Notebook 13 shows the study's figure beside the two distributions this notebook wrote, at the same months, trials and blocks.
'''),
]


# The first code cell of notebooks 01 to 05: find the repository, import the modules, open the extraction.
DATA_SETUP = '''import hashlib
import json
import pathlib
import sys
import time

# The package lives in src/ beside this folder. The notebook finds the repository's root from where it runs.
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "src" / "mlrp").is_dir())
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from mlrp import constants as C
from mlrp import data as D
from mlrp.source import extraction as X
from mlrp.source import layer as L
from mlrp.source import returns as R
from mlrp.source import universe as U

# Stops with one sentence when the data root is not set, when the manifest is not the one the constants fix,
# or when one of the study's eight tables is not the one they record.
root, manifest = D.open_extraction()
print("manifest SHA-256", C.MANIFEST_SHA256[:12], "| files", manifest["counts"]["files"],
      "| rows", f"{manifest['counts']['rows']:,}", "| entries of the eight tables", C.MANIFEST_ENTRIES_SHA256[:12])'''

# Stops a notebook after notebook 02 when gate G1 (a) is not met, as the gate's rule says.
G1A_GUARD = '''g1a = D.read_result("g1a.json")
if not g1a["g1a_met"]:
    raise SystemExit("Gate G1 (a) is not met, so nothing after notebook 02 runs.")
print("Gate G1 (a): met, in notebook 02")'''

# ---------------------------------------------------------------------------
# 01: the extraction
# ---------------------------------------------------------------------------

NB01 = [
    md('''
# 01 The extraction: what the study reads, checked before anything is read

**What this notebook does.** It checks that the CRSP and Compustat files on this machine are the extraction the study is built on, lists the eight tables the study reads with their rows, fetch dates and hashes, checks that every column the later notebooks read is there, and checks the code copied from two other studies. It reads no row of any table. Everything it prints comes from the extraction's manifest, from each file's own metadata and from the copied files.

**Why it comes first.** Every number in the later notebooks is a statement about these files. If a file had changed since it was fetched, every later number would describe another extraction. The checks take about a minute and run before anything else.

**What it judges.** No gate and no expectation. A file that differs from the manifest stops the study here.

**The data.** CRSP and Compustat, from Wharton Research Data Services, accessed through Queen Mary University of London, extracted on 28 September 2026. The extraction stays on the machine that holds it and is never committed.
'''),
    md('''
## Terms used in this notebook

| Term | Meaning |
|---|---|
| extraction | the copy of CRSP and Compustat tables taken on 28 September 2026, kept in the folder named by the environment variable `MLRP_DATA_ROOT` on the machine that holds it |
| manifest | the extraction's own record. For every data file it lists the query that produced it, its rows and columns, its size, its SHA-256 hash and the time it was fetched |
| SHA-256 hash | a fingerprint of 64 characters computed from a file's bytes. A change of one byte changes the hash |
| digest of the entries | the SHA-256 of the manifest's records for the study's eight tables alone: each file's name, rows and hash. A pull that adds other tables to the extraction changes the manifest's hash and leaves this digest as it is |
| table | one of the extraction's datasets, such as CRSP's monthly stock file. A table is stored in one file or in one file per year |
| copied code | seven modules of the study Stock-Level Tracking and one of the study Volatility-managed portfolios, copied unchanged into `src/mlrp/source/` |
| Sharpe ratio | a portfolio's mean return over the standard deviation of its return |
| provenance record | `results/20260928/provenance.json`: the extraction, its two checks, the fetch dates, the commits the copied code comes from and the versions of the tools every later result was built with |
'''),
    md('''
## What I expect to see, and where it comes from

1. The manifest's SHA-256 begins `23e96d18`, and it lists 310 files and 112,161,796 rows: the values in `src/mlrp/constants.py`, read from the manifest on 18 November 2026.
2. The study reads eight tables in 95 files: CRSP's list of index members, its monthly index file, its monthly and daily stock files, its delistings, its names, its link to Compustat, and Compustat's annual fundamentals. Together they hold 73,595,093 rows, all fetched on 28 September 2026. The count of files is `MANIFEST_ENTRIES` in the constants, and the count of rows `MANIFEST_ENTRIES_ROWS`, the sum of the manifest's counts for the eight tables, read on 21 November 2026.
3. Every file's SHA-256 on disk equals the manifest's, and every file's row count in its own footer equals the manifest's.
4. Every column the later notebooks read is in every file of its table.
5. The SHA-256 of each copied file equals the value in the constants.

Running time: about a minute, most of it spent computing the hash of the daily stock file, 2.2 GB in 37 yearly files.
'''),
    code(DATA_SETUP),
    md('''
## 1. The tables the study reads

The setup cell has already run two checks: the manifest's SHA-256 against the constants, and the digest of the eight tables' entries. Each row below is one table: the files it is stored in, its rows and columns, and the date it was fetched, in Amsterdam time.
'''),
    code('''
records = D.files_used(manifest)
files = pd.DataFrame(records)
by_table = (files.groupby(["key", "table"], as_index=False)
                 .agg(files=("file", "size"), rows=("rows", "sum"), columns=("columns", "first"), fetch_date=("fetch_date", "max")))
print(by_table.to_string(index=False))
print()
print("tables", len(by_table), "| files", len(files), "| rows", f"{files['rows'].sum():,}")
'''),
    md('''
## 2. The files on disk against the manifest

Two checks for each file, each independent of the other. The SHA-256 of the file on disk is computed again and compared with the manifest's. The row count in the file's footer, which the parquet format writes when the file is saved, is compared with the count in the manifest. A file that fails either check stops the study here.
'''),
    code('''
t0 = time.time()
checks = pd.DataFrame(X.verify_files(root, records))
print(f"checked {len(checks)} files in {time.time() - t0:.0f} s")
print("present", int(checks["exists"].sum()), "| hash equal to the manifest's", int(checks["sha256_matches"].sum()),
      "| row count equal to the manifest's", int(checks["rows_match"].sum()))
failed = checks[~(checks["exists"] & checks["sha256_matches"] & checks["rows_match"])]
if len(failed):
    print(failed.to_string(index=False))
    raise SystemExit("At least one file differs from the manifest, so the study stops here.")
'''),
    md('''
## 3. The columns the later notebooks read

Each later notebook reads only the columns it needs, listed in `src/mlrp/data.py` as `COLUMNS`: the copied lists for the seven CRSP tables, the monthly share volume added for dollar volume, and the company key, the fiscal year end, the fiscal year and fifteen items of Compustat's annual fundamentals. The schema of every file is checked here, once.
'''),
    code('''
lacking = D.missing_columns(root, records)
print("files lacking a column a later notebook reads:", len(lacking))
if lacking:
    print(lacking)
    raise SystemExit("A column a later notebook reads is missing, so the study stops here.")
print({key: len(cols) for key, cols in D.COLUMNS.items()})
'''),
    md('''
## 4. The copied code

The universe, the returns and the reading of the extraction come from Stock-Level Tracking's data layer, copied unchanged from its commit named in the constants, and the interval for the difference of two Sharpe ratios comes from Volatility-managed portfolios. The SHA-256 of each copy is computed here and compared with the constants. The tests compare each copy with its source file as well.
'''),
    code('''
source = ROOT / C.COPIED_INTO
copies = [(name, path, digest) for name, (path, digest) in C.COPIED_FILES.items()]
copies.append((pathlib.PurePosixPath(C.BOOTSTRAP_FILE).name, C.BOOTSTRAP_FILE, C.BOOTSTRAP_SHA256))
table = pd.DataFrame([{"file": name, "from": path, "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()[:16],
                       "equal_to_the_constants": hashlib.sha256((source / name).read_bytes()).hexdigest() == digest}
                      for name, path, digest in copies])
print(table.to_string(index=False))
print("copied from", C.SOURCE_REPOSITORY, C.SOURCE_COMMIT[:7], "and", C.BOOTSTRAP_SOURCE_REPOSITORY, C.BOOTSTRAP_SOURCE_COMMIT)
if not table["equal_to_the_constants"].all():
    raise SystemExit("A copied file differs from the constants, so the study stops here.")
'''),
    md('''
## 5. What is committed

Two files go into `results/20260928/`. `extraction_files.csv` has one row per file used: its key, table, path, the query that produced it, its rows, columns and size, its SHA-256 and its fetch date. It is the manifest's metadata and holds no value from inside any file. A reader with access to Wharton Research Data Services can run the same queries and compare the hashes. `provenance.json` is the provenance record that every later result names.
'''),
    code('''
committed = files[["key", "table", "file", "query", "rows", "columns", "bytes", "sha256", "fetch_date"]]
p1 = D.write_results("extraction_files.csv", frame=committed)
record = D.provenance(manifest, records, written_by="notebooks/01_extraction.ipynb")
p2 = D.write_results("provenance.json", record=record)
print("written:", p1.relative_to(ROOT), "and", p2.relative_to(ROOT))
summary = pd.DataFrame([
    ("manifest SHA-256 and the entries' digest", "equal to the constants", "equal (the setup cell stops otherwise)"),
    ("tables, files, rows", f"{C.TABLE_COUNT}, {C.MANIFEST_ENTRIES}, {C.MANIFEST_ENTRIES_ROWS:,}", f"{len(by_table)}, {len(files)}, {files['rows'].sum():,}"),
    ("files equal to the manifest", "95 of 95", f"{int((checks['sha256_matches'] & checks['rows_match']).sum())} of {len(checks)}"),
    ("files lacking a column", "0", str(len(lacking))),
    ("copies equal to the constants", "8 of 8", f"{int(table['equal_to_the_constants'].sum())} of {len(table)}"),
], columns=["check", "expected", "result"])
print(summary.to_string(index=False))
'''),
    md('''
## What was learned, and what could be wrong

- The table above sets each expectation of this notebook beside its result.
- A hash shows that a file has not changed since it was fetched. It does not show that the file holds what the vendor intended. Notebook 02 tests the universe and the returns against Stock-Level Tracking's counts, which were themselves tested against CRSP's own counts and returns.
- CRSP corrects its history with each release of its data, so the same queries run later can return other files, with other hashes, for reasons that are not errors.
- The row counts in the manifest and in the footers are both counts of rows as stored. Neither says anything about a value inside a row. Notebooks 03 and 04 count the missing values of each characteristic by year.
'''),
]

# ---------------------------------------------------------------------------
# 02: the universe and its returns
# ---------------------------------------------------------------------------

NB02 = [
    md('''
# 02 The universe and its returns: gate G1 (a)

**What this notebook does.** It builds the universe, the S&P 500's members month by month, and their monthly returns with delistings applied, with the code copied from Stock-Level Tracking. It then checks that the copy gives Stock-Level Tracking's committed counts to the last digit, and counts the member-months of the panel's window.

**Why it comes here.** The copied data layer was built and tested in Stock-Level Tracking, against CRSP's own count of members and its own index returns. Reproducing that study's committed tables here shows that the copy, run on this machine's extraction through this study's wrapper, is the same layer. Everything after this notebook rests on it.

**What it judges.** Gate G1 (a). If it is not met, nothing after this notebook runs.

**The data.** CRSP, from Wharton Research Data Services, accessed through Queen Mary University of London, extracted on 28 September 2026.
'''),
    md('''
## Terms used in this notebook

| Term | Meaning |
|---|---|
| universe | the stocks in CRSP's list of S&P 500 members on the last trading day of the month before. A stock is in the universe in month t when one of its membership spells covers the last trading day of month t-1 |
| member-month | one stock in the universe in one month |
| membership spell | one interval in CRSP's list, from the day a stock entered the index to the day it left, both days included |
| PERMNO | CRSP's permanent number for a security. It does not change when the company changes its name or its ticker |
| CRSP's count | the number of securities in the index on a day, which CRSP stores on each row of its monthly index file |
| count check | the universe's count in each month beside CRSP's count on the same day, summarised by year: the mean difference and the months more than 10 names apart |
| delisting | the end of a stock's listing on its exchange. CRSP records its date, a code for its reason and, where it can, a delisting return |
| delisting rule | Stock-Level Tracking's rule for a delisting, after Shumway (1997): CRSP's delisting return where it gives one, minus 30 per cent for a code from 400 to 599 where it gives none, and zero for any other code. In a month with both a return and a delisting return, the two are compounded |
| total return | the month's return with dividends and with the delisting rule applied |
| start-of-month capitalisation | the absolute price times shares outstanding at the end of month t-1, in thousands of dollars |
| reference table | one of the three tables Stock-Level Tracking committed at the commit the copied code is taken from, read from GitHub by that commit's hash |
| gate G1 (a) | the copied layer gives the reference tables' member-months, count check by year and delisting counts by year to the last digit |
| panel's window | the months January 1991 to December 2024, the months of the study's panel. The test window, January 2006 to December 2024, is its last 228 months |
'''),
    md('''
## What I expect to see, and where it comes from

1. CRSP's list holds 2,064 spells of 1,936 PERMNOs, and the universe runs 1,188 months, January 1926 to December 2024, with 440,724 member-months: Stock-Level Tracking's `universe_summary.json`, at the commit in the constants.
2. The count check by year, 99 rows, is the same file byte for byte as Stock-Level Tracking's `universe_count_check_by_year.csv`.
3. The delisting counts by year, 99 rows, are the same file byte for byte as Stock-Level Tracking's `delistings_by_year.csv`.
4. Over the panel's window: 204,520 member-months, of which 100 have no return and no delisting, and 114,521 over the test window. These are the constants of section 3, read from the same reference table on 18 November 2026. They are compared here and judged by no gate.

Expectations 1 to 3 together are gate G1 (a). Running time: about a minute.
'''),
    code(DATA_SETUP),
    md('''
## 1. CRSP's list and the universe

The list is read whole: PERMNO, start and end. Five counts say whether it is well formed: spells, PERMNOs, PERMNOs with more than one spell, spells that overlap an earlier spell of the same PERMNO, and spells that end before they start. The universe is then built month by month from the list and from CRSP's month-end dates in its monthly index file.
'''),
    code('''
t0 = time.time()
universe = L.load_universe(root)
spells, matrix, ulong = universe["spells"], universe["matrix"], universe["ulong"]
checks = U.spell_checks(spells)
print(json.dumps(checks, indent=1))
print("date values present that did not parse:", universe["unparsed"] or "none")
print(f"months {len(matrix)}, from {matrix.index[0]} to {matrix.index[-1]} | member-months {len(ulong):,} | {time.time() - t0:.0f} s")
'''),
    md('''
## 2. The count check against the reference table

The universe's count in month t is set beside CRSP's count on its row for month t-1, which is the count on the same day, the last trading day of month t-1. The table by year is written with the same writer as Stock-Level Tracking's and compared with the reference table byte for byte, and by SHA-256.
'''),
    code('''
def against_reference(name, frame):
    """Write one table and compare its bytes with the reference table of the same name."""
    path = D.write_results(name, frame=frame)
    ours = path.read_bytes()
    theirs, where = D.reference_table("results/20260928/" + name)
    return {"table": name, "rows": int(len(frame)), "sha256": hashlib.sha256(ours).hexdigest()[:16],
            "reference_sha256": hashlib.sha256(theirs).hexdigest()[:16], "equal": ours == theirs, "reference_read_from": where}

counts = U.counts_against_crsp(matrix, universe["index_file"])
count_by_year = U.count_check_by_year(counts)
count_check = against_reference("universe_count_check_by_year.csv", count_by_year)
print(json.dumps(count_check, indent=1))
'''),
    md('''
## 3. The monthly returns and the delisting counts

The monthly stock file is read for the universe's stocks, with the columns this study reads, and the delistings are read whole. The copied `monthly_panel` gives every member-month its total return with the delisting rule applied, its start-of-month capitalisation and how its delisting, if any, was treated. The counts by year are compared with the second reference table.
'''),
    code('''
t0 = time.time()
monthly, bad_monthly = D.read_table(root, "crsp_msf")
monthly = monthly[monthly["permno"].isin(matrix.columns)].reset_index(drop=True)
delistings, bad_delistings = D.read_table(root, "crsp_msedelist")
returns_panel = R.monthly_panel(ulong, monthly, delistings)
print(f"monthly rows for the universe's stocks {len(monthly):,} | member-months with a row in the panel {len(returns_panel):,} | {time.time() - t0:.0f} s")
print("date values present that did not parse:", {**bad_monthly, **bad_delistings} or "none")
delisting_check = against_reference("delistings_by_year.csv", R.delisting_by_year(returns_panel))
print(json.dumps(delisting_check, indent=1))
'''),
    md('''
## 4. Gate G1 (a)

The first part of the gate is the member-months, their first and last month and their number of months, against the reference summary and the constants. The other two parts are the comparisons of sections 2 and 3. The verdict is written to `g1a.json`, and notebooks 03 to 05 stop if it is not met.
'''),
    code('''
reference_summary = json.loads(D.reference_table("results/20260928/universe_summary.json")[0])
members = {"member_months": int(len(ulong)), "first_month": str(matrix.index[0]), "last_month": str(matrix.index[-1]),
           "months": int(len(matrix))}
members_equal = (members["member_months"] == reference_summary["member_months"] == C.G1A_MEMBER_MONTHS
                 and members["first_month"] == reference_summary["first_month"] == C.G1A_FIRST_MONTH
                 and members["last_month"] == reference_summary["last_month"] == C.G1A_LAST_MONTH
                 and members["months"] == reference_summary["months"])
g1a = {"member_months": {**members, "reference": {k: reference_summary[k] for k in members}, "equal": members_equal},
       "count_check_by_year": count_check, "delistings_by_year": delisting_check,
       "g1a_met": bool(members_equal and count_check["equal"] and delisting_check["equal"])}
D.write_results("g1a.json", record=g1a)
print(json.dumps(members))
print("Gate G1 (a):", "met" if g1a["g1a_met"] else "not met", f"| member-months {members['member_months']:,} against {C.G1A_MEMBER_MONTHS:,}"
      f" | count check equal: {count_check['equal']} | delisting counts equal: {delisting_check['equal']}")
'''),
    md('''
## 5. The panel's window

The panel keeps the member-months from January 1991 to December 2024. A member-month with no return and no delisting has no target and is counted. The counts are compared with the constants and judged by no gate.
'''),
    code('''
months = pd.PeriodIndex(returns_panel["month"], freq="M")
window = returns_panel[(months >= pd.Period(C.PANEL_FIRST_MONTH, "M")) & (months <= pd.Period(C.PANEL_LAST_MONTH, "M"))]
test = window[pd.PeriodIndex(window["month"], freq="M") >= pd.Period(C.TEST_FIRST_MONTH, "M")]
counts_window = {"panel_months": int(window["month"].nunique()), "panel_member_months": int(len(window)),
                 "member_months_without_return": int(window["return_missing"].sum()),
                 "test_months": int(test["month"].nunique()), "test_member_months": int(len(test))}
expected = {"panel_months": C.PANEL_MONTHS, "panel_member_months": C.PANEL_MEMBER_MONTHS,
            "member_months_without_return": C.PANEL_MEMBER_MONTHS_NO_RETURN, "test_months": C.TEST_MONTHS,
            "test_member_months": C.TEST_MEMBER_MONTHS}
print(pd.DataFrame({"result": counts_window, "constants": expected}).to_string())
'''),
    md('''
## 6. What is committed

Four files go into `results/20260928/`, each an aggregate with no identifier of a security: the count check by year and the delisting counts by year, both equal to the reference tables when the gate is met, the gate's record `g1a.json`, and a summary of the counts printed here.
'''),
    code('''
summary = {"written_by": "notebooks/02_universe_and_returns.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
           "list": checks, **members, "monthly_rows_for_the_universe": int(len(monthly)), "panel_window": counts_window,
           "date_values_not_parsed": {**universe["unparsed"], **bad_monthly, **bad_delistings}}
p = D.write_results("universe_and_returns_summary.json", record=summary)
print("written: results/20260928/universe_count_check_by_year.csv, delistings_by_year.csv, g1a.json and", p.name)
results = pd.DataFrame([
    ("member-months, January 1926 to December 2024", f"{C.G1A_MEMBER_MONTHS:,}", f"{members['member_months']:,}", "met" if members_equal else "not met"),
    ("count check by year, byte for byte", "equal", "equal" if count_check["equal"] else "different", "met" if count_check["equal"] else "not met"),
    ("delisting counts by year, byte for byte", "equal", "equal" if delisting_check["equal"] else "different", "met" if delisting_check["equal"] else "not met"),
    ("member-months of the panel's window", f"{C.PANEL_MEMBER_MONTHS:,}", f"{counts_window['panel_member_months']:,}", "no verdict"),
    ("without a return or a delisting", str(C.PANEL_MEMBER_MONTHS_NO_RETURN), str(counts_window["member_months_without_return"]), "no verdict"),
    ("member-months of the test window", f"{C.TEST_MEMBER_MONTHS:,}", f"{counts_window['test_member_months']:,}", "no verdict"),
], columns=["expectation", "expected", "result", "verdict"])
print(results.to_string(index=False))
if not g1a["g1a_met"]:
    raise SystemExit("Gate G1 (a) is not met, so nothing after this notebook runs.")
'''),
    md('''
## What was learned, and what could be wrong

- The table above sets each expectation beside its result. Gate G1 (a) is met only if all three of its parts are.
- Equal counts show that the copy is Stock-Level Tracking's layer on this extraction. They do not test the layer again: that study set the universe's count beside CRSP's count, and the universe's return beside the index's own return, in its notebooks 02 and 04.
- A count that agrees does not show that the right stocks are in the universe, only that their number is right. Stock-Level Tracking's comparison of returns is the finer test, and it stands for this study through the copy.
- The universe reads membership on the last trading day of the month before, so a stock that joins mid-month enters the universe the month after. Every characteristic in notebooks 03 and 04 is measured on that same day, so the panel uses nothing dated after the last trading day of the month before. Notebooks 04 and 05 give what this does not rule out: restated accounts and revised data.
'''),
]

# ---------------------------------------------------------------------------
# 03: the market characteristics
# ---------------------------------------------------------------------------

NB03 = [
    md('''
# 03 The market characteristics: eight measures from prices, returns and trading

**What this notebook does.** For every member-month of the panel's window it measures eight characteristics from CRSP's monthly and daily stock files and from Kenneth French's daily market return: size, momentum, reversal, dollar volume, volatility, the largest daily return, the stock's sensitivity to the market's moves, called beta, and the volatility of its return that the market does not explain, called idiosyncratic volatility. Each is measured at the end of month t-1, so that it uses only what was known when month t began. It counts by year how many member-months have each one.

**Why it comes here.** The models of milestones 2 and 3 forecast next month's return from twelve characteristics. Eight of them come from prices and trading, and the accounting four come in notebook 04. Gate G1 (b), judged in notebook 05, asks each to be present for at least 90 per cent of each month's members, and this notebook shows where each stands by year.

**What it judges.** No gate and no expectation with a level. Notebook 05 judges coverage on these columns and runs one more test on them, the look-ahead test: a check that no characteristic used a number dated after the day it is measured on.

**The data.** CRSP, from Wharton Research Data Services, accessed through Queen Mary University of London, extracted on 28 September 2026, and Kenneth French's data library, downloaded at run time.
'''),
    md('''
## Terms used in this notebook

| Term | Meaning |
|---|---|
| characteristic | a number measured for one stock in one month, used to forecast its return in the next month |
| month t, month t-1 | month t is the month whose return is forecast. Month t-1 is the month before. Every characteristic is measured on the last trading day of month t-1 |
| trading day | a day on which the daily stock file has rows |
| size | the log of the start-of-month capitalisation: the absolute price times shares outstanding at the end of month t-1, in thousands of dollars (CRSP's guide, pp. 70 and 79) |
| momentum | the stock's return compounded over the months t-12 to t-2. Month t-1 is left out. It is missing when the price at t-13 or the return at t-2 is missing |
| reversal | the stock's return in month t-1 |
| share volume | the number of shares traded in a month. CRSP's monthly file records it in hundreds of shares (CRSP's guide, p. 91) |
| dollar volume | the log of the price at the end of month t-1 times the month's share volume in shares |
| daily return | the stock's return on one trading day, with dividends |
| volatility | the standard deviation of the daily returns in month t-1, with at least 15 of them |
| largest daily return | the highest daily return in month t-1, with at least 15 daily returns |
| risk-free rate | French's daily return on one-month Treasury bills |
| excess return | a return less the risk-free rate of the same day |
| market excess return | French's daily return on the market of all CRSP stocks, each weighted by its market value, less the risk-free rate |
| regression | a fit of a straight line, y = a + b x, by least squares: the a and b that make the sum of squared distances from the points to the line smallest |
| beta | the slope b of the stock's daily excess return on the market excess return, over the 252 trading days to the end of month t-1, with at least 120 days on which both are present |
| residual | the distance from one point to the fitted line, y - a - b x |
| idiosyncratic volatility | the standard deviation of the same regression's residuals: the square root of their sum of squares over n - 2, where n is the number of days, because the fit estimates two numbers |
| coverage | the share of the member-months in which a characteristic is present |
| verdict | met or not met: what a gate's level, or an expectation's, says of a result. An expectation with no verdict is set beside its result and judged by nothing |
| provenance record | where a result's inputs came from: for French's files, each file's address, the time it was read and its SHA-256 |
'''),
    md('''
## What I expect to see, and where it comes from

1. Size is present in all but 10 of the panel's 204,520 member-months: Stock-Level Tracking's `delistings_by_year.csv` counts 10 member-months without a start-of-month capitalisation from 1991 to 2024.
2. Volatility, the largest daily return, beta and idiosyncratic volatility are present for more than 98 per cent of member-months in every year. In Stock-Level Tracking's `daily_coverage_by_year.csv` at least 99.64 per cent of member-days have a daily return in every year from 1990, the lowest in 2000. A member needs 15 returns in a month and 120 in 252 days, and a stock that entered the index from a recent listing can fall short of 120.
3. Momentum needs 13 months of history, so it is missing for a stock listed less than 13 months before. The S&P 500 adds few such stocks, so momentum is expected above 97 per cent in every year, a figure not read from the data.
4. The mean of beta across each year's member-months lies between 0.7 and 1.3 in every year. The members make up most of the market's value, so their betas weighted by value average close to one. The mean here gives each member the same weight, so it can stray further, most likely around 2000, when the largest technology stocks moved the market and many other members did not move with them. With no verdict.
5. The median volatility across members is highest in 2008 or 2009. The volatility of a member-month in year y is measured in the month before, so the falls of late 2008 and early 2009 reach both years. With no verdict.

Gate G1 (b) on these columns is judged in notebook 05. Running time: about five minutes, most of it spent reading the daily stock file a year at a time.
'''),
    code(DATA_SETUP),
    code(G1A_GUARD),
    md('''
## 1. The member-months of the panel's window

The universe is built again, as in notebook 02, and cut to the panel's window, January 1991 to December 2024. Each member-month carries `date_prev`, the last trading day of month t-1, the day every characteristic is measured on.
'''),
    code('''
universe = L.load_universe(root)
ulong = universe["ulong"]
inside = pd.PeriodIndex(ulong["month"], freq="M")
members = ulong[(inside >= pd.Period(C.PANEL_FIRST_MONTH, "M")) & (inside <= pd.Period(C.PANEL_LAST_MONTH, "M"))].reset_index(drop=True)
permnos = np.sort(members["permno"].unique())
print(f"member-months {len(members):,} | months {members['month'].nunique()} | stocks {len(permnos):,}")
'''),
    md('''
## 2. The four characteristics from the monthly file

The monthly file is read for the panel's stocks over every month it holds, since momentum in January 1991 reaches back to December 1989. `characteristics.monthly_characteristics` gives size, momentum, reversal and dollar volume for every member-month. Momentum is the copied function of Stock-Level Tracking, which follows Kenneth French's definition.
'''),
    code('''
t0 = time.time()
monthly, bad_monthly = D.read_table(root, "crsp_msf")
monthly = monthly[monthly["permno"].isin(permnos)].reset_index(drop=True)
delistings, bad_delistings = D.read_table(root, "crsp_msedelist")
from mlrp import characteristics as K
monthly_chars = K.monthly_characteristics(members, monthly, delistings)
print(f"monthly rows {len(monthly):,} | {time.time() - t0:.0f} s")
print("date values present that did not parse:", {**bad_monthly, **bad_delistings} or "none")
print(monthly_chars[list(K.MONTHLY)].notna().mean().round(4).to_string())
'''),
    md('''
## 3. French's daily market return and risk-free rate

French's daily data hold the market excess return and the risk-free rate, in per cent, which the copied loader turns into decimals. The files are downloaded once into `data/french/`, which git ignores, and their record, with each file's SHA-256 and the date in its header, goes into `provenance_french.json`.
'''),
    code('''
factors, risk_free, french_records = D.french_daily()
market_excess = factors["Mkt-RF"]
print("French's daily series from", market_excess.index.min().date(), "to", market_excess.index.max().date(), "|", len(market_excess), "days")
for r in french_records:
    print(r["filename"], "| SHA-256", r["zip_sha256"][:16], "|", r["vintage_line"])
'''),
    md('''
## 4. The daily returns

The daily stock file is read a year at a time, from 1990, for the panel's stocks, and held in memory as one table of trading days by stock. It is never written. The calendar of trading days is the file's own, and French's series are aligned to it. A trading day with no market return in French's file would leave that day out of every regression, and those days are counted.
'''),
    code('''
t0 = time.time()
years = list(range(int(C.DAILY_FILE_START[:4]), int(C.PANEL_LAST_MONTH[:4]) + 1))
daily, calendar = L.daily_matrix(root, permnos, years, progress=True)
print(f"trading days {len(calendar):,}, {calendar[0].date()} to {calendar[-1].date()} | stocks {daily.shape[1]:,} | {time.time() - t0:.0f} s")
print("trading days without French's market return:", int(market_excess.reindex(calendar).isna().sum()))
'''),
    md('''
## 5. The four characteristics from the daily file

`characteristics.daily_characteristics` measures volatility and the largest daily return on the trading days of month t-1, and beta and idiosyncratic volatility on the 252 trading days to its last day. Each month is computed from its own days alone.

**A worked example of the regression.** Take three days on which the market's excess return was 1, 0 and -1 per cent and a stock's was 2, 0.5 and -1 per cent. The line that fits best has slope b = 1.5 and intercept a = 0.5 per cent: the fitted values 2, 0.5 and -1 equal the stock's, so the residuals are 0, 0 and 0, and the idiosyncratic volatility is 0. A stock with the same slope and residuals of +1, -2 and +1 per cent has a sum of squares of 6, so its idiosyncratic volatility is the square root of 6 / (3 - 2), 2.45 per cent a day. The study fits the same line over up to 252 days.
'''),
    code('''
t0 = time.time()
daily_chars = K.daily_characteristics(members, daily, market_excess, risk_free)
print(f"{time.time() - t0:.0f} s")
print(daily_chars[list(K.DAILY)].notna().mean().round(4).to_string())
'''),
    md('''
## 6. Coverage and medians by year

Two tables by year, both aggregates over each year's member-months: the share with each characteristic present, and the median of each across members. The mean of beta is printed beside them for expectation 4.
'''),
    code('''
market = members[["month", "permno"]].merge(monthly_chars, on=["month", "permno"]).merge(daily_chars, on=["month", "permno"])
names = list(K.MONTHLY) + list(K.DAILY)
year = pd.PeriodIndex(market["month"], freq="M").year
coverage = market[names].notna().groupby(year).mean().round(6)
coverage.insert(0, "member_months", market.groupby(year).size())
coverage.index.name = "year"
medians = market[names].groupby(year).median().round(6)
medians.index.name = "year"
beta_mean = market["beta"].groupby(year).mean().round(3)
pd.set_option("display.width", 200)
print(coverage.to_string())
print()
print(medians.to_string())
print()
print("mean beta by year, lowest", beta_mean.min(), "in", int(beta_mean.idxmin()), "| highest", beta_mean.max(), "in", int(beta_mean.idxmax()))
print("the three years of the highest median volatility:", [int(y) for y in medians["volatility"].sort_values(ascending=False).index[:3]])
'''),
    md('''
## 7. What is kept and what is committed

The eight characteristics of every member-month are kept in `local/market_characteristics.parquet` for notebook 05. The folder is ignored by git and stays on the machine that holds the extraction. Three files go into `results/20260928/`: the coverage by year, the medians by year and a summary. French's provenance record goes beside them.
'''),
    code('''
D.write_local("market_characteristics.parquet", market.assign(month=market["month"].astype(str)))
written = [D.write_results("market_coverage_by_year.csv", frame=coverage.reset_index()),
           D.write_results("market_medians_by_year.csv", frame=medians.reset_index()),
           D.write_results("market_summary.json", record={
               "written_by": "notebooks/03_market_characteristics.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
               "member_months": int(len(market)), "stocks": int(len(permnos)), "monthly_rows": int(len(monthly)),
               "trading_days": int(len(calendar)), "first_trading_day": str(calendar[0].date()), "last_trading_day": str(calendar[-1].date()),
               "trading_days_without_market_return": int(market_excess.reindex(calendar).isna().sum()),
               "present": {c: int(market[c].notna().sum()) for c in names},
               "mean_beta_by_year": {int(k): float(v) for k, v in beta_mean.items()},
               "date_values_not_parsed": {**bad_monthly, **bad_delistings}}),
           D.write_french_provenance(french_records)]
for path in written:
    print("written:", path.relative_to(ROOT))
results = pd.DataFrame([
    ("1. size missing in member-months", "10", str(int(market["size"].isna().sum()))),
    ("2. lowest yearly coverage of the four daily ones", "above 0.98", f"{coverage[list(K.DAILY)].min().min():.4f}"),
    ("3. lowest yearly coverage of momentum", "above 0.97", f"{coverage['momentum'].min():.4f}"),
    ("4. mean beta by year", "0.7 to 1.3", f"{beta_mean.min():.3f} to {beta_mean.max():.3f}"),
    ("5. year of the highest median volatility", "2008 or 2009", str(int(medians["volatility"].idxmax()))),
], columns=["expectation", "expected", "result"])
print(results.to_string(index=False))
'''),
    md('''
## What was learned, and what could be wrong

- The table above sets each expectation beside its result. None has a verdict here: coverage is judged by gate G1 (b) in notebook 05.
- Beta and idiosyncratic volatility are measured against French's market, the return of all CRSP stocks weighted by their market value, and over a year of daily returns. Ang, Hodrick, Xing and Zhang (2006) measure idiosyncratic volatility within one month, against the market and two portfolios that stand for size and value. This study's version is the one in the constants.
- The monthly file's volume enters only dollar volume. Anderson and Dyl (2007) state that trading volume has historically been overstated for NASDAQ stocks relative to NYSE stocks. Dollar volume is ranked within each month, so an overstatement that moves every NASDAQ stock alike moves their ranks against NYSE stocks. The panel does not correct for it.
- Each characteristic is measured from days up to the last trading day of month t-1. Notebook 05 builds 24 months a second time, from inputs cut at that day, and compares the two builds, which is the test that nothing later was used.
'''),
]

# ---------------------------------------------------------------------------
# 04: the accounting characteristics
# ---------------------------------------------------------------------------

NB04 = [
    md('''
# 04 The accounting characteristics: four measures from the annual accounts

**What this notebook does.** For every member-month of the panel's window it finds the stock's company in Compustat through CRSP's link, picks the company's latest annual record that is at least six and at most 24 months old on the last trading day of month t-1, and measures four characteristics from it: book-to-market, earnings-to-price, operating profitability and asset growth. It counts by year how many member-months have each one, and why the others do not.

**Why it comes here.** The accounts of a fiscal year are published some weeks after it ends. A record used before it was published would give the models information no investor had. The six-month rule, from Gu, Kelly and Xiu (2020, p. 2248), leaves time for the accounts to appear.

**What it judges.** No gate and no expectation with a level. Gate G1 (b), judged in notebook 05, asks book-to-market and operating profitability to be present for at least 80 per cent of each month's members, and the other two for at least 90 per cent.

**The data.** CRSP and Compustat, from Wharton Research Data Services, accessed through Queen Mary University of London, extracted on 28 September 2026.
'''),
    md('''
## Terms used in this notebook

| Term | Meaning |
|---|---|
| Compustat | the database of company accounts from S&P Global. Its annual fundamentals hold one record per company and fiscal year |
| GVKEY | Compustat's permanent number for a company |
| link | CRSP's table of which GVKEY belongs to which PERMNO, and over which dates. The study uses links of the types LC and LU with any primary flag, so that a second share class reaches its company |
| primary flag | CRSP's mark of which share class of a company is its main one: P or C for the main class, J or N for another |
| fiscal year end | the last day of a company's accounting year, Compustat's `datadate` |
| annual record | one company's accounts for one fiscal year |
| age of a record | the months from the month of its fiscal year end to month t-1. A record of December 2023 is 6 months old in June 2024 |
| book equity | stockholders' equity, plus deferred taxes and investment tax credit, minus preferred stock. Stockholders' equity is `seq`, else common equity, the item named `ceq`, plus `pstk`, else total assets minus total liabilities. Preferred stock is the redemption value `pstkrv`, else the liquidating value `pstkl`, else the par value `pstk`, else zero. Book equity that is not positive is missing |
| market value at the fiscal year end | the price at the fiscal year end times shares outstanding, `prcc_f` times `csho` |
| book-to-market | book equity, as Kenneth French's data library defines it, over the market value at the same fiscal year end, where the library divides by the market value at the end of December |
| earnings-to-price | income before extraordinary items, `ib`, over the same market value. Negative values are kept (Basu 1983) |
| operating profitability | revenue less cost of goods sold, selling, general and administrative expenses and interest expense, over book equity (Fama and French 2015). Revenue and at least one of the three costs must be present |
| asset growth | total assets over the same company's total assets a fiscal year before, less one (Cooper, Gulen and Schill 2008) |
'''),
    md('''
## What I expect to see, and where it comes from

1. A GVKEY for at least 99.6 per cent of member-months in every year: Stock-Level Tracking's `sector_coverage_by_year.csv`, through the same links, finds one for at least 99.60 per cent of member-months in every year from 1990, the lowest in 2023.
2. A record of the right age for more than 97 per cent of the member-months with a GVKEY in every year. S&P 500 companies file accounts every year, so a record should be missing mainly for a company whose first fiscal year in Compustat ended less than six months before, such as a recent spin-off. A figure not read from the data.
3. A record without book-to-market for less than 10 per cent of member-months in every year. Most of these should have a book equity that is not positive, a share that grew after 2010 as large buybacks took the equity of some members below zero. The rest lack the price or the shares at the fiscal year end. A figure not read from the data.
4. Book-to-market and operating profitability present for more than 85 per cent of member-months in every year, earnings-to-price and asset growth for more than 95 per cent. Gate G1 (b) asks 80 and 90 per cent in every month.
5. In about half of the member-months the record used is 6 to 11 months old, and in the other half 12 to 17: a company with a December year end is read at 6 months in July and at 11 in the following June, and the record moves on each July.

Running time: about a minute.
'''),
    code(DATA_SETUP),
    code(G1A_GUARD),
    md('''
## 1. The member-months and the links

The universe is built again and cut to the panel's window. The link is read whole and cut to the two link types, with any primary flag. Each member-month takes the GVKEY of the link valid on the last trading day of month t-1, by the copied `link_gvkeys`.
'''),
    code('''
from mlrp import characteristics as K
universe = L.load_universe(root)
ulong = universe["ulong"]
inside = pd.PeriodIndex(ulong["month"], freq="M")
members = ulong[(inside >= pd.Period(C.PANEL_FIRST_MONTH, "M")) & (inside <= pd.Period(C.PANEL_LAST_MONTH, "M"))].reset_index(drop=True)
raw_links, bad_links = D.read_table(root, "crsp_ccm_lnkhist")
print(f"member-months {len(members):,} | link rows {len(raw_links):,}")
'''),
    md('''
## 2. The annual records

The annual fundamentals are read with the company key, the fiscal year end, the fiscal year and the fifteen items the four characteristics need. `characteristics.annual_records` keeps one record per company and fiscal year end, computes book equity, the market value at the fiscal year end and operating profit, and attaches the total assets of the fiscal year before.
'''),
    code('''
t0 = time.time()
funda, bad_funda = D.read_table(root, "comp_funda")
records, record_counts = K.annual_records(funda)
print(f"rows {len(funda):,} | {time.time() - t0:.0f} s")
print(json.dumps(record_counts, indent=1))
print("date values present that did not parse:", {**bad_links, **bad_funda} or "none")
'''),
    md('''
## 3. The four characteristics

`characteristics.accounting_characteristics` picks, for each member-month with a GVKEY, the record with the latest fiscal year end that is at least 6 months old on the last trading day of month t-1, and drops it when it is more than 24 months old.

**A worked example of the timing.** A company with a fiscal year ending on 31 December 2022 publishes its accounts in February or March 2023. For month t = June 2023, month t-1 is May 2023, 5 months after the year end, so the study still uses the record of December 2021, 17 months old. For month t = July 2023, month t-1 is June 2023, 6 months on, and the record of December 2022 is used. For a December year end that is the timing of Fama and French (1992), who match the accounts of a year with returns from July of the next.
'''),
    code('''
t0 = time.time()
accounting, more_than_one_link, _ = K.accounting_characteristics(members, raw_links, funda)
print(f"{time.time() - t0:.0f} s | member-months with more than one valid link: {more_than_one_link}")
print(accounting[list(K.ACCOUNTING)].notna().mean().round(4).to_string())
'''),
    md('''
## 4. Coverage by year, and why a characteristic is missing

One table by year, an aggregate over each year's member-months: the share with a GVKEY, the share with a record of the right age, the share whose record gives no book-to-market (a book equity that is not positive, or no price or shares at the fiscal year end), the share with each characteristic, and the member-months that share their company with another member in the same month, the second share classes.
'''),
    code('''
a = accounting
year = pd.PeriodIndex(a["month"], freq="M").year
has_record = a["age_months"].notna()
shared = a[a["gvkey"].notna()].duplicated(["month", "gvkey"], keep=False)
table = pd.DataFrame({
    "member_months": a.groupby(year).size(),
    "share_with_gvkey": a["gvkey"].notna().groupby(year).mean(),
    "share_with_record": has_record.groupby(year).mean(),
    "share_with_record_without_book_to_market": (has_record & a["book_to_market"].isna()).groupby(year).mean(),
    **{f"share_with_{c}": a[c].notna().groupby(year).mean() for c in K.ACCOUNTING},
    "member_months_sharing_a_company": shared.reindex(a.index, fill_value=False).groupby(year).sum().astype("int64"),
}).round(6)
table.index.name = "year"
pd.set_option("display.width", 200)
print(table.to_string())
age = a["age_months"].dropna()
age_bands = pd.cut(age, bins=[5, 11, 17, 24], labels=["6 to 11", "12 to 17", "18 to 24"]).value_counts(normalize=True).sort_index().round(4)
print()
print("age of the record used, share of member-months with a record:")
print(age_bands.to_string())
'''),
    md('''
## 5. The member-months without a company

The table above counts the member-months the links leave without a GVKEY. `characteristics.without_a_company` counts how many members they are, each member's member-months with its first and last year, and how many of them a link of another type would reach: a link in CRSP's table that names the member's PERMNO and is valid on the last trading day of month t-1, of a type other than LC and LU, with any primary flag. It prints counts and years only. Added on 24 November 2026.
'''),
    code('''
without = K.without_a_company(members, raw_links)
print(json.dumps(without, indent=1))
'''),
    md('''
## 6. What is kept and what is committed

The four characteristics of every member-month, with its GVKEY and the age of its record, are kept in `local/accounting_characteristics.parquet` for notebook 05. Two files go into `results/20260928/`: the table by year and a summary with the counts of the annual records, the ages and the member-months without a company.
'''),
    code('''
D.write_local("accounting_characteristics.parquet", a.assign(month=a["month"].astype(str)))
written = [D.write_results("accounting_coverage_by_year.csv", frame=table.reset_index()),
           D.write_results("accounting_summary.json", record={
               "written_by": "notebooks/04_accounting_characteristics.ipynb", "manifest_sha256": C.MANIFEST_SHA256,
               "member_months": int(len(a)), "link_rows": int(len(raw_links)), "annual_records": record_counts,
               "member_months_with_more_than_one_valid_link": int(more_than_one_link),
               "age_of_the_record_used": {str(k): float(v) for k, v in age_bands.items()},
               "present": {c: int(a[c].notna().sum()) for c in K.ACCOUNTING},
               "date_values_not_parsed": {**bad_links, **bad_funda},
               "member_months_without_a_company": without})]
for path in written:
    print("written:", path.relative_to(ROOT))
results = pd.DataFrame([
    ("1. lowest yearly share with a GVKEY", "at least 0.996", f"{table['share_with_gvkey'].min():.4f}"),
    ("2. lowest yearly share with a record, of those with a GVKEY", "above 0.97", f"{(table['share_with_record'] / table['share_with_gvkey']).min():.4f}"),
    ("3. highest yearly share with a record and no book-to-market", "below 0.10", f"{table['share_with_record_without_book_to_market'].max():.4f}"),
    ("4. lowest yearly coverage of book-to-market and operating profitability", "above 0.85",
     f"{table[['share_with_book_to_market', 'share_with_operating_profitability']].min().min():.4f}"),
    ("4. lowest yearly coverage of earnings-to-price and asset growth", "above 0.95",
     f"{table[['share_with_earnings_to_price', 'share_with_asset_growth']].min().min():.4f}"),
    ("5. records 6 to 11 and 12 to 17 months old", "about half each", f"{age_bands.iloc[0]:.2f} and {age_bands.iloc[1]:.2f}"),
], columns=["expectation", "expected", "result"])
print(results.to_string(index=False))
'''),
    md('''
## What was learned, and what could be wrong

- The table above sets each expectation beside its result. None has a verdict here: coverage is judged by gate G1 (b) in notebook 05.
- Compustat's records are the values as they stand in the extraction of 28 September 2026. A company that restated its accounts shows the restated figures, which were not known on the day the study uses them. The six-month rule protects against using a year's accounts before they were published. It does not undo restatements, and no data held here could.
- Two share classes of one company carry the same four values, since they have the same accounts. Their count by year is in the table.
- The 450 member-months without a company are five members, one in all 408 months, one in 39 months from 2021 and three in one month each, and links of two types the study does not use reach 447 of them, LS all 447 and LX 39 of those, leaving three.
- A book equity that is not positive leaves book-to-market and operating profitability missing. Notebook 05 sets those cells to the month's median after ranking, as for any missing value, and counts them.
'''),
]

# ---------------------------------------------------------------------------
# 05: the panel
# ---------------------------------------------------------------------------

NB05 = [
    md('''
# 05 The panel: the target, the transform, and gate G1 (b) and (c)

**What this notebook does.** It joins the twelve characteristics of notebooks 03 and 04 with the target, the return each member earned in month t above one-month Treasury bills, into the panel the models will read. It judges coverage, gate G1 (b). It ranks each characteristic within its month onto the interval from -1 to 1 and sets a missing value to the month's median. And it builds 24 months again from inputs cut at the last trading day of month t-1, to show that no characteristic used anything later, gate G1 (c).

**Why it comes here.** A model can only be as honest as its inputs. A characteristic missing for a large share of members would leave the models little to learn from, and one that used a later number would make every forecast look better than any investor could have made it, without showing.

**What it judges.** Gate G1 (b) and (c). With part (a) from notebook 02, all three must be met before milestone 2 fits any model.

**The data.** CRSP and Compustat, from Wharton Research Data Services, accessed through Queen Mary University of London, extracted on 28 September 2026, and Kenneth French's data library, downloaded at run time.
'''),
    md('''
## Terms used in this notebook

| Term | Meaning |
|---|---|
| panel | one row per member and month t, January 1991 to December 2024, with the twelve characteristics measured at the end of month t-1 and the target of month t |
| risk-free rate | the return on one-month Treasury bills, which French's library gives month by month as the one-month bill rate |
| excess return | a return less the risk-free rate of the same month |
| target | the member's excess return in month t: its total return, with the delisting rule applied, less French's one-month bill rate of month t |
| one-month bill rate | French's monthly return on one-month Treasury bills, the risk-free rate of the month |
| coverage | the share of a month's members with a characteristic present, before any value is filled in |
| rank | a value's position when the month's values are sorted, from 1 for the smallest. Tied values share the average of the positions they occupy |
| rank transform | each characteristic ranked within its month and mapped to the interval from -1 to 1: rank r of n present values becomes -1 + 2 (r - 1) / (n - 1) |
| fill | a missing value set to 0 after the rank transform. On that scale 0 is the month's median |
| seed | the number that starts the random generator, 20261119 in the constants, so that every number it gives can be made again |
| look-ahead test | 24 of the panel's 408 months, drawn by the study's seed, built again from inputs from which every row dated after the last trading day of month t-1 is removed, and compared with the full build |
| tolerance | the largest difference allowed between the two builds, 1e-12 |
| verdict | met or not met: what a gate's level says of a result |
| gate G1 | (a) the copied layer gives Stock-Level Tracking's counts, judged in notebook 02. (b) Each characteristic is present for at least 90 per cent of each month's members, book-to-market and operating profitability for at least 80 per cent, in every month from January 1991 to December 2024. (c) The look-ahead test passes on its 24 months |
'''),
    md('''
## What I expect to see, and where it comes from

1. 204,520 member-months, of which 100 have no target, and 114,521 in the test window: the counts of notebook 02 and the constants.
2. Gate G1 (b) is met: every characteristic at or above its level in all 408 months. The lowest coverage is expected in book-to-market and operating profitability, which need a positive book equity, and in the early 1990s, when Compustat holds fewer years of history for each company. Notebooks 03 and 04 give the yearly figures.
3. Gate G1 (c) is met with a largest difference of exactly 0. The second build runs the same three functions as the full build, on the same numbers up to the cut, so it gives the same results to the last bit. A difference above 1e-12, or a value present in one build and missing in the other, would mean a characteristic used a row from after the cut.
4. After the transform, each characteristic lies between -1 and 1 in every member-month, the count of filled cells equals the count of missing values before the fill, and each month's values average 0. Ranks with ties at their average always sum to n (n + 1) / 2, which the map turns into a mean of 0, and a filled cell adds a 0.

Running time: about five minutes, most of it spent reading the daily file again for the look-ahead test.
'''),
    code(DATA_SETUP),
    code(G1A_GUARD),
    md('''
## 1. The target

The returns of notebook 02 are built again with the copied `monthly_panel`. French's monthly file gives the one-month bill rate, and `panel.target` subtracts it from the total return.
'''),
    code('''
from mlrp import characteristics as K
from mlrp import panel as P
universe = L.load_universe(root)
ulong = universe["ulong"]
monthly, _ = D.read_table(root, "crsp_msf")
monthly = monthly[monthly["permno"].isin(universe["matrix"].columns)].reset_index(drop=True)
delistings, _ = D.read_table(root, "crsp_msedelist")
returns_panel = R.monthly_panel(ulong, monthly, delistings)
_, bill_rate, french_monthly_records = D.french_monthly()
targets = P.target(returns_panel, bill_rate)
print("bill rate from", bill_rate.index.min(), "to", bill_rate.index.max())
'''),
    md('''
## 2. The panel, before the transform

The characteristics kept by notebooks 03 and 04 are read from `local/` and joined with the target onto every member-month of the window.
'''),
    code('''
def from_local(name):
    frame = D.read_local(name)
    return frame.assign(month=pd.PeriodIndex(frame["month"], freq="M"))

market = from_local("market_characteristics.parquet")
accounting = from_local("accounting_characteristics.parquet")
monthly_part = market[["month", "permno", *K.MONTHLY]]
daily_part = market[["month", "permno", *K.DAILY]]
panel = P.assemble(ulong, monthly_part, daily_part, accounting, targets)
test = panel[pd.PeriodIndex(panel["month"], freq="M") >= pd.Period(C.TEST_FIRST_MONTH, "M")]
counts = {"member_months": int(len(panel)), "without_target": int(panel["target"].isna().sum()),
          "months": int(panel["month"].nunique()), "test_member_months": int(len(test))}
print(pd.DataFrame({"result": counts, "expected": {"member_months": C.PANEL_MEMBER_MONTHS, "without_target": C.PANEL_MEMBER_MONTHS_NO_RETURN,
                                                   "months": C.PANEL_MONTHS, "test_member_months": C.TEST_MEMBER_MONTHS}}).to_string())
'''),
    md('''
## 3. Coverage and gate G1 (b)

The share of each month's members with each characteristic present, before the fill, in each of the 408 months. The gate asks every characteristic to be at or above its level in every month. The table gives each characteristic's lowest month.
'''),
    code('''
coverage = P.coverage_by_month(panel)
g1b = P.g1b(coverage)
print(pd.DataFrame(g1b["by_characteristic"]).T.to_string())
print("Gate G1 (b):", "met" if g1b["g1b_met"] else "not met", f"over {g1b['months']} months")
missing = P.missing_by_year(panel)
'''),
    md('''
## 4. The rank transform and the fill

Within each month, each characteristic is ranked across the members for which it is present and mapped to the interval from -1 to 1. A missing value is then set to 0, the month's median on that scale.

**A worked example.** In one month five members have momentum of 0.30, -0.10, 0.05, 0.05 and a missing value. The four present values ranked from the smallest are -0.10 (rank 1), the two values of 0.05 (ranks 2 and 3, so 2.5 each) and 0.30 (rank 4). With n = 4, rank r becomes -1 + 2 (r - 1) / 3: -1, 0, 0 and 1. The missing value becomes 0. So the largest momentum of the month becomes 1 and the smallest -1, whatever their size, unless two members tie there, as two share classes of one company can on an accounting characteristic. A model sees only the order of the members within the month.
'''),
    code('''
transformed, filled = P.transform(panel)
names = list(C.CHARACTERISTICS)
checks = {
    "within -1 and 1": bool(transformed[names].apply(lambda s: s.between(C.RANK_LOW, C.RANK_HIGH)).all().all()),
    "filled cells equal the missing values": all(filled[c] == int(panel[c].isna().sum()) for c in names),
    # 1e-12: far above the rounding error of a mean of about 500 values between -1 and 1.
    "each month averages 0": bool((transformed.groupby("month")[names].mean().abs() <= 1e-12).all().all()),
}
print(json.dumps(checks, indent=1))
print("filled cells by characteristic:", filled)
print("largest absolute mean of a month:", float(transformed.groupby("month")[names].mean().abs().max().max()))
'''),
    md('''
## 5. The look-ahead test, gate G1 (c)

For each of the 24 months drawn by the seed, the inputs are cut at the last trading day of month t-1. Every row dated later is removed from the monthly file, the delistings, the daily returns, French's daily series and the links, a link's end date after that day is set to missing, and every annual record less than six months old on that day is removed. The twelve characteristics of the month's members are then built again by the same three functions and compared with the full build, cell by cell.
'''),
    code('''
t0 = time.time()
months = P.lookahead_months()
days = ulong.drop_duplicates("month").set_index("month")["date_prev"]
needed = sorted({y for m in months for y in (days[m].year - 1, days[m].year)} & set(range(int(C.DAILY_FILE_START[:4]), int(C.PANEL_LAST_MONTH[:4]) + 1)))
rows_needed = ulong[ulong["month"].isin(months)]
daily, _ = L.daily_matrix(root, rows_needed["permno"].unique(), needed)
factors, risk_free, _ = D.french_daily()
raw_links, _ = D.read_table(root, "crsp_ccm_lnkhist")
funda, _ = D.read_table(root, "comp_funda")
inputs = {"monthly": monthly, "delistings": delistings, "daily": daily, "market_excess": factors["Mkt-RF"],
          "risk_free": risk_free, "links": raw_links, "funda": funda}
built = panel[["month", "permno", *names]]
lookahead = []
for m in months:
    rows = rows_needed[rows_needed["month"] == m]
    result = P.compare_month(built[built["month"] == m], P.build_cut_month(rows, inputs))
    lookahead.append({"month": str(m), "members": int(len(rows)), **result})
lookahead = pd.DataFrame(lookahead)
g1c_met = bool(lookahead["equal"].all())
print(lookahead.to_string(index=False))
print(f"{time.time() - t0:.0f} s | Gate G1 (c):", "met" if g1c_met else "not met",
      f"| largest difference {lookahead['largest_difference'].max()} | cells present in one build only {int(lookahead['presence_differs'].sum())}")
'''),
    md('''
## 6. Gate G1

The three parts together. If any is not met, nothing after this notebook runs and the counts by characteristic and year go to the study's design for a ruling.
'''),
    code('''
g1 = {"g1a_met": bool(g1a["g1a_met"]), "g1b_met": bool(g1b["g1b_met"]), "g1c_met": g1c_met}
g1["g1_met"] = all(g1.values())
print("Gate G1:", "met" if g1["g1_met"] else "not met", json.dumps(g1))
'''),
    md('''
## 7. What is kept and what is committed

The panel is kept in `local/` twice, before and after the transform, for milestone 2. Five files go into `results/20260928/`, each an aggregate with no identifier of a security: the missing shares by characteristic and year, gate G1 (b) with each characteristic's lowest month, the look-ahead test's 24 rows, gate G1 and a summary.
'''),
    code('''
D.write_local("panel.parquet", panel.assign(month=panel["month"].astype(str)))
D.write_local("panel_transformed.parquet", transformed.assign(month=transformed["month"].astype(str)))
written = [D.write_results("panel_missing_by_year.csv", frame=missing),
           D.write_results("g1b_coverage.json", record=g1b),
           D.write_results("g1c_lookahead.csv", frame=lookahead),
           D.write_results("g1.json", record=g1),
           D.write_results("panel_summary.json", record={
               "written_by": "notebooks/05_panel.ipynb", "manifest_sha256": C.MANIFEST_SHA256, **counts,
               "filled_cells": filled, "transform_checks": checks}),
           D.write_french_provenance(french_monthly_records)]
for path in written:
    print("written:", path.relative_to(ROOT))
results = pd.DataFrame([
    ("1. member-months, without a target, in the test window", f"{C.PANEL_MEMBER_MONTHS:,}, {C.PANEL_MEMBER_MONTHS_NO_RETURN}, {C.TEST_MEMBER_MONTHS:,}",
     f"{counts['member_months']:,}, {counts['without_target']}, {counts['test_member_months']:,}", "no verdict"),
    ("2. gate G1 (b), every characteristic at its level in every month", "met", "met" if g1b["g1b_met"] else "not met", "met" if g1b["g1b_met"] else "not met"),
    ("3. gate G1 (c), 24 months equal to 1e-12", "met, largest difference 0", f"largest difference {lookahead['largest_difference'].max()}", "met" if g1c_met else "not met"),
    ("4. the transform's three checks", "all true", str(all(checks.values())), "no verdict"),
    ("gate G1, all three parts", "met", "met" if g1["g1_met"] else "not met", "met" if g1["g1_met"] else "not met"),
], columns=["expectation", "expected", "result", "verdict"])
print(results.to_string(index=False))
if not g1["g1_met"]:
    raise SystemExit("Gate G1 is not met, so nothing after this notebook runs.")
'''),
    md('''
## What was learned, and what could be wrong

- The table above sets each expectation beside its result and gives the gate's verdict.
- The look-ahead test shows that the code uses nothing dated after the last trading day of month t-1 in the 24 months it builds a second time. It cannot show that the data themselves were known on that day: CRSP and Compustat correct their history, and the extraction holds the corrected values. That limit is the data's, and no test on one extraction can lift it.
- The fill puts a member with a missing value at the month's median. A model can then learn nothing from that characteristic for that member, which is the purpose, but a characteristic missing for a quarter of the members in some month would weaken what any model learns from it in that month. The missing shares by year are committed for that reason.
- The rank transform keeps the order of the members within a month and drops the size of the differences. A model fitted on it forecasts from where a stock stands among the month's members, which is the treatment of Gu, Kelly and Xiu (2020) that the study follows.
'''),
]


def main():
    NOTEBOOKS.mkdir(exist_ok=True)
    for number, name, cells in (("01", "01_extraction.ipynb", NB01),
                                ("02", "02_universe_and_returns.ipynb", NB02),
                                ("03", "03_market_characteristics.ipynb", NB03),
                                ("04", "04_accounting_characteristics.ipynb", NB04),
                                ("05", "05_panel.ipynb", NB05),
                                ("12", "12_known_answer.ipynb", NB12)):
        with open(NOTEBOOKS / name, "w", encoding="utf-8") as f:
            json.dump(notebook(number, cells), f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("built", name)


if __name__ == "__main__":
    main()
