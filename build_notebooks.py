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

**What this notebook does.** It tests the code of the study's two overfitting statistics on made-up returns whose answer is known. Each repetition makes up 228 months of returns for 32 trials, with a random generator, from the standard normal distribution. With a mean of zero in every column, no trial has a true effect, and the statistics have to say so. A last set of repetitions gives one trial a true effect, and the statistics have to find it.

**Why it comes at this point.** Gate G3 must pass before either statistic is computed on the study's own returns. It reads no data of any kind, so it runs as soon as its settings are in the repository. The notebook needs no data root.

**What it judges.** Gate G3, in four clauses, (a) to (d).

**What to expect.** The numbers under "What I expect to see", each with the reason behind it. The whole run takes about ten seconds.
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
4. **Clause (d).** With the first trial given a true Sharpe ratio of 1.0 a year, which is 1 / √12 = 0.289 a month, that trial has the best Sharpe ratio over the 228 months in at least 90 per cent of 1,000 repetitions, and the mean probability of backtest overfitting is below 0.25. Over 228 months a trial's Sharpe ratio has a standard error near 1 / √227 = 0.066 a month, so 0.289 stands more than four standard errors above zero and well above 0.139, the expected best of the other 31 trials, which have no effect.

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
    ("probability of backtest overfitting from the splits with another trial picked", d_beside["pbo_from_other_picked"]),
    ("probability of backtest overfitting from the splits with the planted trial picked", d_beside["pbo_from_planted_picked"]),
    ("mean probability of backtest overfitting, the two parts together", d_beside["mean_pbo"]),
]
display(pd.DataFrame(rows, columns=["figure", "value"]).round(4))'''),
    md('''
## 7. The verdicts

The four clauses against their levels. The cell writes the three tables of repetitions, the summary and the provenance record to `results/simulation/`, and the figures of steps 3 and 6 that no level reads to a file of their own there, `g3_figures_beside.json`.
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
path = S.write_figures_beside(a_beside, d_beside, written_by="notebooks/12_known_answer.ipynb")
print("written:", path.relative_to(ROOT))'''),
    md('''
## What was learned, and what could be wrong

What the gate shows when its four clauses are met:

- The code of the expected best Sharpe ratio, of the deflated Sharpe ratio and of the probability of backtest overfitting returns the known answers on returns with no true effect, and finds a true effect of 1.0 a year when one trial has it.
- The deflated Sharpe ratio of the best of 32 trials with no effect sits near one half: picking the best of 32 gives a Sharpe ratio that looks better than any single trial, and the statistic takes that out.

What could be wrong, and where each point is taken up:

- The returns here are independent and normal. The study's long-short returns have more extreme months than the normal distribution allows. The deflated Sharpe ratio takes the best trial's own skewness and kurtosis into account, and notebook 13 shows both beside it.
- The statistic counts the 32 trials as independent. The configurations of one model move together, so the study's trials are fewer in effect. Counting 32 makes the bar higher, and notebook 13 shows the paper's adjustment for correlated trials beside it, with no verdict.
- One estimate of the probability of backtest overfitting is noisy, as the spread in step 5 shows. Notebook 13 shows the study's figure beside the two distributions this notebook wrote, at the same months, trials and blocks.
'''),
]


def main():
    NOTEBOOKS.mkdir(exist_ok=True)
    for number, name, cells in (("12", "12_known_answer.ipynb", NB12),):
        with open(NOTEBOOKS / name, "w", encoding="utf-8") as f:
            json.dump(notebook(number, cells), f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("built", name)


if __name__ == "__main__":
    main()
