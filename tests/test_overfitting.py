"""The overfitting statistics return the published numbers of their papers' worked example, and the
probability of backtest overfitting agrees with a plain loop over the splits on made-up data."""
import itertools
import math
import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlrp import constants as C  # noqa: E402
from mlrp import overfitting as O  # noqa: E402


# The worked example of Bailey and López de Prado (2014), pages 9 and 10: 100 trials whose
# annualised Sharpe ratios have a variance of 1/2, five years of daily returns (1,250 days, 250 a
# year), and a best trial with an annualised Sharpe ratio of 2.5, skewness -3 and kurtosis 10.
DAYS, YEARS = 250, 5
SR = 2.5 / math.sqrt(DAYS)
VARIANCE = 0.5 / DAYS


def test_the_expected_best_sharpe_ratio_of_the_worked_example():
    assert round(O.expected_max_sharpe(100, VARIANCE), 4) == 0.1132


def test_the_deflated_sharpe_ratio_of_the_worked_example():
    n_obs = DAYS * YEARS
    assert round(O.deflated_sharpe_ratio_from_moments(SR, n_obs, -3, 10, 100, VARIANCE), 4) == 0.9004
    # With 46 trials the same Sharpe ratio would clear the 95 per cent level, and with normal
    # returns it would clear it up to 88 trials.
    assert round(O.deflated_sharpe_ratio_from_moments(SR, n_obs, -3, 10, 46, VARIANCE), 4) == 0.9505
    assert round(O.deflated_sharpe_ratio_from_moments(SR, n_obs, 0, 3, 88, VARIANCE), 4) == 0.9505


def test_the_expected_best_rises_with_the_number_of_trials():
    values = [O.expected_max_sharpe(n, 1.0) for n in (2, 10, 32, 100, 1000)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_the_sharpe_ratio_of_a_known_series():
    r = np.array([0.01, 0.03, -0.02, 0.02])
    assert O.sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1))


def test_the_deflated_sharpe_ratio_picks_the_best_column_and_its_moments():
    rng = np.random.default_rng([C.SEED, C.SEED_STREAMS["made_up_data"]])
    r = rng.normal(0.0, 1.0, size=(120, 8))
    r[:, 5] += 0.3
    out = O.deflated_sharpe_ratio(r)
    srs = O.sharpe_ratio(r)
    assert out["best_trial"] == int(np.argmax(srs)) == 5
    assert out["trial_variance"] == pytest.approx(np.var(srs, ddof=1))
    assert out["kurtosis"] == pytest.approx(stats.kurtosis(r[:, 5], fisher=False))
    expected = O.deflated_sharpe_ratio_from_moments(srs[5], 120, out["skewness"], out["kurtosis"], 8, out["trial_variance"])
    assert out["deflated_sharpe_ratio"] == pytest.approx(expected)


def test_the_splits_are_every_half_of_the_blocks():
    split = O.cscv_splits(C.PBO_BLOCKS)
    assert split.shape == (C.PBO_SPLITS, C.PBO_BLOCKS)
    assert (split.sum(axis=1) == C.PBO_BLOCKS // 2).all()
    assert len({tuple(row) for row in split}) == C.PBO_SPLITS


def loop_pbo(r, n_blocks):
    """The probability of backtest overfitting written as a plain loop over the splits, for the test."""
    n_obs, n_trials = r.shape
    size = n_obs // n_blocks
    logits = []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        rows_in = np.concatenate([np.arange(b * size, (b + 1) * size) for b in combo])
        rows_out = np.setdiff1d(np.arange(n_obs), rows_in)
        sr_in = O.sharpe_ratio(r[rows_in])
        sr_out = O.sharpe_ratio(r[rows_out])
        best = int(np.argmax(sr_in))
        rank = stats.rankdata(sr_out)[best]
        w = rank / (n_trials + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits


def test_the_probability_of_backtest_overfitting_equals_a_plain_loop():
    rng = np.random.default_rng([C.SEED, C.SEED_STREAMS["made_up_data"]])
    r = rng.normal(0.0, 1.0, size=(48, 10))
    out = O.probability_of_backtest_overfitting(r, 8)
    pbo, logits = loop_pbo(r, 8)
    assert out["splits"] == math.comb(8, 4) == len(logits)
    assert out["pbo"] == pbo
    assert np.allclose(out["logits"], logits)


def test_a_trial_best_everywhere_gives_zero_and_a_trial_that_reverses_gives_one():
    rng = np.random.default_rng([C.SEED, C.SEED_STREAMS["made_up_data"]])
    r = rng.normal(0.0, 1.0, size=(48, 6))
    r[:, 0] += 5.0
    assert O.probability_of_backtest_overfitting(r, 4)["pbo"] == 0.0
    # Trial 1 earns in the first two blocks and loses in the last two, and the others are noise
    # around zero: whenever it is best in sample, it is the worst out of sample.
    r = rng.normal(0.0, 0.1, size=(48, 6))
    r[:24, 1] += 3.0
    r[24:, 1] -= 3.0
    out = O.probability_of_backtest_overfitting(r, 4)
    chosen = out["best_in_sample"] == 1
    assert chosen.any() and (out["logits"][chosen] < 0).all()
