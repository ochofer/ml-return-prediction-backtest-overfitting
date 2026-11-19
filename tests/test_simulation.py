"""The known-answer simulation draws what it says, from the seed alone, with one stream per clause, and
its verdicts read the levels of the constants. These tests use six repetitions and judge nothing."""
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlrp import constants as C  # noqa: E402
from mlrp import overfitting as O  # noqa: E402
from mlrp import simulation as S  # noqa: E402

FEW = 6


def test_the_made_up_returns_have_the_shape_and_the_planted_mean():
    rng = np.random.default_rng([C.SEED, C.SEED_STREAMS["made_up_data"]])
    r = S.made_up_returns(rng, 2400, 4, 0.5)
    assert r.shape == (2400, 4)
    assert abs(r[:, 0].mean() - 0.5) < 0.1 and abs(r[:, 1:].mean()) < 0.1


def test_the_planted_sharpe_ratio_is_one_a_year():
    assert S.planted_monthly_sharpe() * math.sqrt(C.MONTHS_PER_YEAR) == C.G3_D_ANNUAL_SHARPE


def test_the_draws_depend_on_the_seed_alone_and_each_clause_has_its_own_stream():
    first = S.clauses_a_b(FEW)
    again = S.clauses_a_b(FEW)
    assert first.equals(again)
    longer = S.clauses_a_b(FEW + 3)
    assert longer.head(FEW).equals(first)
    assert not np.allclose(S.made_up_returns(S.rng_for("c"), 10, 3), S.made_up_returns(S.rng_for("d"), 10, 3))


def test_each_clause_returns_one_row_per_repetition():
    ab, c, d = S.clauses_a_b(FEW), S.clause_c(FEW), S.clause_d(FEW)
    assert len(ab) == len(c) == len(d) == FEW
    assert ab["deflated_sharpe_ratio"].between(0, 1).all() and c["pbo"].between(0, 1).all()
    assert d["planted_is_best"].dtype == bool


def test_the_verdicts_read_the_levels_of_the_constants():
    out = S.verdicts(S.clauses_a_b(FEW), S.clause_c(FEW), S.clause_d(FEW))
    assert set(out) == {"a", "b", "c", "d", "all_met"}
    assert out["a"]["expected_best_sharpe"] == O.expected_max_sharpe(C.G3_TRIALS, 1 / (C.G3_MONTHS - 1))
    assert out["b"]["level"] == C.DSR_LEVEL and out["d"]["pbo_max"] == C.G3_D_PBO_MAX
    assert out["all_met"] == all(out[k]["met"] for k in "abcd")


def test_the_figures_the_notebook_quotes_before_its_run():
    # Notebook 12 quotes these in "What I expect to see" and in step 2.
    assert round(O.expected_max_sharpe(C.G3_TRIALS, 1 / (C.G3_MONTHS - 1)), 4) == 0.1394
    assert round(S.planted_monthly_sharpe(), 3) == 0.289
    assert round(1 / math.sqrt(C.G3_MONTHS - 1), 3) == 0.066
    assert round(O.expected_max_sharpe(C.G3_TRIALS - 1, 1 / (C.G3_MONTHS - 1)), 3) == 0.139
    assert C.PBO_SPLITS == 924 and C.G3_MONTHS // C.G3_BLOCKS == 19
