"""
The overfitting statistics of section 11 of the constants.

Three quantities, each from the paper that defines it:

- The expected best Sharpe ratio of N independent trials with no true effect, and the
  deflated Sharpe ratio, from Bailey and López de Prado (2014), equation (2). The deflated
  Sharpe ratio is the probabilistic Sharpe ratio of the best trial with the expected best
  Sharpe ratio as its threshold: the confidence, from 0 to 1, that the best trial's true
  Sharpe ratio is above what the best of N trials with no true effect would show.
- The probability of backtest overfitting, by the combinatorially symmetric
  cross-validation of Bailey, Borwein, López de Prado and Zhu (2017), Algorithm 2.3 and
  section 3.1: the share of the splits of the months into two halves in which the trial
  that is best in one half ranks in the lower half of the other.

Sharpe ratios here are per period (monthly in this study) and are never annualised inside
this module. A Sharpe ratio is the mean over the sample standard deviation, with n - 1 in
the denominator of the variance. Skewness and kurtosis are the sample moments, and the
kurtosis is the plain fourth moment, equal to 3 for normal returns.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns, axis=0):
    """The per-period Sharpe ratio of each column (axis=0) of `returns`: the mean over the
    sample standard deviation."""
    r = np.asarray(returns, dtype=float)
    return r.mean(axis=axis) / r.std(axis=axis, ddof=1)


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """The expected best of `n_trials` independent Sharpe ratios with mean zero and the
    given variance, the threshold of equation (2) of Bailey and López de Prado (2014):
    sqrt(V) ((1 - g) Z^-1[1 - 1/N] + g Z^-1[1 - 1/(N e)]), where g is the
    Euler-Mascheroni constant and Z^-1 the inverse of the standard normal distribution."""
    if n_trials < 2:
        raise ValueError("the expected best Sharpe ratio needs at least two trials")
    g = EULER_MASCHERONI
    z = (1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(variance) * z


def probabilistic_sharpe_ratio(sharpe: float, threshold: float, n_obs: int,
                               skewness: float, kurtosis: float) -> float:
    """The probability that the true Sharpe ratio is above `threshold`, given an estimated
    per-period Sharpe ratio from `n_obs` returns with the given skewness and kurtosis:
    Z[(SR - SR0) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2)]."""
    spread = 1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe ** 2
    return float(stats.norm.cdf((sharpe - threshold) * math.sqrt(n_obs - 1) / math.sqrt(spread)))


def deflated_sharpe_ratio_from_moments(sharpe: float, n_obs: int, skewness: float, kurtosis: float,
                                       n_trials: int, trial_variance: float) -> float:
    """The deflated Sharpe ratio of a trial with the given per-period Sharpe ratio, sample
    length, skewness and kurtosis, selected as the best of `n_trials` whose per-period
    Sharpe ratios have the variance `trial_variance`."""
    threshold = expected_max_sharpe(n_trials, trial_variance)
    return probabilistic_sharpe_ratio(sharpe, threshold, n_obs, skewness, kurtosis)


def deflated_sharpe_ratio(trial_returns, n_trials: int | None = None) -> dict:
    """The deflated Sharpe ratio of the best column of `trial_returns` (periods by trials).

    The number of trials defaults to the number of columns, and the variance across trials
    is the sample variance of the columns' Sharpe ratios. The best trial's own skewness and
    kurtosis enter the statistic."""
    r = np.asarray(trial_returns, dtype=float)
    n_obs, n_cols = r.shape
    n = n_cols if n_trials is None else n_trials
    srs = sharpe_ratio(r)
    best = int(np.argmax(srs))
    variance = float(np.var(srs, ddof=1))
    skewness = float(stats.skew(r[:, best]))
    kurtosis = float(stats.kurtosis(r[:, best], fisher=False))
    threshold = expected_max_sharpe(n, variance)
    value = probabilistic_sharpe_ratio(float(srs[best]), threshold, n_obs, skewness, kurtosis)
    return {"deflated_sharpe_ratio": value, "best_trial": best, "best_sharpe": float(srs[best]),
            "threshold": threshold, "trial_variance": variance, "skewness": skewness,
            "kurtosis": kurtosis, "n_trials": n, "n_obs": n_obs}


def cscv_splits(n_blocks: int) -> np.ndarray:
    """Every choice of half of `n_blocks` blocks as the in-sample half, one row per split,
    as a 0-1 matrix of splits by blocks (Algorithm 2.3, third step)."""
    if n_blocks % 2:
        raise ValueError("the number of blocks must be even")
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    indicator = np.zeros((len(combos), n_blocks))
    for i, c in enumerate(combos):
        indicator[i, list(c)] = 1.0
    return indicator


def probability_of_backtest_overfitting(trial_returns, n_blocks: int) -> dict:
    """The probability of backtest overfitting of the periods-by-trials matrix
    `trial_returns`, with the periods cut into `n_blocks` contiguous blocks of equal length.

    For each split, the trial with the highest Sharpe ratio on the in-sample half is
    found, and its rank among the trials on the other half gives the relative rank
    w = rank / (N + 1), with rank 1 the lowest, and the logit log(w / (1 - w)). The
    statistic is the share of splits with a logit at or below zero, that is, with the
    in-sample best trial in the lower half out of sample."""
    r = np.asarray(trial_returns, dtype=float)
    n_obs, n_trials = r.shape
    if n_obs % n_blocks:
        raise ValueError("the number of periods must be a multiple of the number of blocks")
    blocks = r.reshape(n_blocks, n_obs // n_blocks, n_trials)
    block_sum = blocks.sum(axis=1)
    block_sq = (blocks ** 2).sum(axis=1)
    split = cscv_splits(n_blocks)
    half = (n_obs // n_blocks) * (n_blocks // 2)

    def half_sharpe(s1, s2):
        mean = s1 / half
        var = (s2 - half * mean ** 2) / (half - 1)
        return mean / np.sqrt(var)

    in_sum, in_sq = split @ block_sum, split @ block_sq
    sr_in = half_sharpe(in_sum, in_sq)
    sr_out = half_sharpe(block_sum.sum(axis=0) - in_sum, block_sq.sum(axis=0) - in_sq)
    best = np.argmax(sr_in, axis=1)
    best_out = sr_out[np.arange(len(best)), best]
    rank = 1 + (sr_out < best_out[:, None]).sum(axis=1)
    w = rank / (n_trials + 1)
    logits = np.log(w / (1 - w))
    return {"pbo": float(np.mean(logits <= 0)), "logits": logits, "splits": int(len(split)),
            "best_in_sample": best}
