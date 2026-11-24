"""The known-answer simulation draws what it says, from the seed alone, with one stream per clause, and
its verdicts read the levels of the constants, and the figures beside clauses (a) and (d) are computed as
their docstrings say. These tests use six repetitions and judge nothing, except the last, which reads the
files of notebook 12's run and checks the figures its text quotes."""
import json
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


def test_the_figures_the_notebook_quotes_beside_clauses_a_and_d():
    # Notebook 12 quotes these in steps 3 and 6. None of them depends on the run.
    assert round(O.expected_max_sharpe(C.G3_TRIALS, 1.0), 4) == 2.0999
    assert round(S.expected_max_standard_normal(C.G3_TRIALS), 4) == 2.0697
    assert round(S.sharpe_sd_no_effect(C.G3_MONTHS), 4) == 0.0665
    assert round(math.sqrt(1 / (C.G3_MONTHS - 1)), 4) == 0.0664
    assert round(S.exact_expected_best_sharpe(C.G3_MONTHS, C.G3_TRIALS), 4) == 0.1380
    assert round(O.expected_max_sharpe(C.G3_TRIALS, 1 / (C.G3_MONTHS - 1)), 4) == 0.1394
    assert (C.G3_BLOCKS // 2) * (C.G3_MONTHS // C.G3_BLOCKS) == 114


def test_the_sub_streams_are_set_in_the_constants():
    assert not hasattr(S, "CLAUSE_STREAMS")
    assert S.settings()["clause_streams"] is C.KNOWN_ANSWER_SUB_STREAMS
    for clause, sub in C.KNOWN_ANSWER_SUB_STREAMS.items():
        direct = np.random.default_rng([C.SEED, C.SEED_STREAMS["known_answer"], sub]).standard_normal(5)
        assert np.array_equal(S.rng_for(clause).standard_normal(5), direct)


def test_the_expected_highest_standard_normal_number_has_its_known_values():
    # The highest of one number has mean zero, of two 1 / sqrt(pi), of three 3 / (2 sqrt(pi)).
    assert abs(S.expected_max_standard_normal(1)) < 1e-10
    assert math.isclose(S.expected_max_standard_normal(2), 1 / math.sqrt(math.pi), rel_tol=1e-9)
    assert math.isclose(S.expected_max_standard_normal(3), 3 / (2 * math.sqrt(math.pi)), rel_tol=1e-9)


def test_the_standard_deviation_of_a_sharpe_ratio_with_no_effect():
    for t in (10, 60, 228):
        assert math.isclose(S.sharpe_sd_no_effect(t), math.sqrt((t - 1) / ((t - 3) * t)), rel_tol=1e-12)
    rng = np.random.default_rng([C.SEED, C.SEED_STREAMS["made_up_data"]])
    sharpe = O.sharpe_ratio(rng.standard_normal((24, 20000)))
    assert abs(sharpe.std(ddof=1) / S.sharpe_sd_no_effect(24) - 1) < 0.02


def test_the_exact_expected_best_sharpe_ratio_agrees_with_draws_and_with_the_normal_limit():
    assert abs(S.exact_expected_best_sharpe(228, 1)) < 1e-10
    # Over many months the Sharpe ratio is close to normal, so the exact value nears the normal one.
    big = 1_000_000
    assert math.isclose(S.exact_expected_best_sharpe(big, 32), S.expected_max_standard_normal(32) / math.sqrt(big), rel_tol=1e-4)
    # Against the highest of 32 draws of Student's t over 20,000 rows, within four standard errors.
    rng = np.random.default_rng([C.SEED, C.SEED_STREAMS["made_up_data"]])
    highest = rng.standard_t(C.G3_MONTHS - 1, size=(20000, C.G3_TRIALS)).max(axis=1) / math.sqrt(C.G3_MONTHS)
    se = highest.std(ddof=1) / math.sqrt(len(highest))
    assert abs(highest.mean() - S.exact_expected_best_sharpe()) < 4 * se


def test_the_figures_beside_clause_a():
    ab = S.clauses_a_b(FEW)
    out = S.beside_clause_a(ab)
    assert out["mean_best_sharpe"] == ab["best_sharpe"].mean()
    assert math.isclose(out["standard_error"], ab["best_sharpe"].std(ddof=1) / math.sqrt(FEW))
    assert math.isclose(out["distance_in_standard_errors"],
                        (out["mean_best_sharpe"] - out["exact_expected_best_sharpe"]) / out["standard_error"])
    assert out["formula_expected_best_sharpe"] == S.verdicts(ab, S.clause_c(FEW), S.clause_d(FEW))["a"]["expected_best_sharpe"]


def test_the_splits_of_clause_d_add_up_to_its_probability():
    splits, d = S.clause_d_splits(FEW), S.clause_d(FEW)
    assert (splits["splits"] == C.PBO_SPLITS).all()
    assert ((splits["other_picked_lower"] + splits["planted_picked_lower"]) / splits["splits"]).equals(d["pbo"])
    assert (splits["planted_picked_lower"] <= splits["planted_picked"]).all()
    assert (splits["other_picked_lower"] <= splits["splits"] - splits["planted_picked"]).all()
    out = S.beside_clause_d(splits)
    assert math.isclose(out["pbo_from_other_picked"] + out["pbo_from_planted_picked"], d["pbo"].mean())
    assert math.isclose(out["mean_pbo"], d["pbo"].mean())
    assert math.isclose(out["share_planted_picked_mean"] + out["share_other_picked_mean"], 1)
    assert out["share_lower_bound_without_effect"] == 16 / 31
    other = splits["splits"] - splits["planted_picked"]
    ratio = splits["other_picked_lower"].sum() / other.sum()
    by_hand = np.std(splits["other_picked_lower"] - ratio * other, ddof=1) / math.sqrt(FEW) / other.mean()
    assert math.isclose(out["share_lower_when_other_picked"], ratio)
    assert math.isclose(out["share_lower_when_other_picked_se"], by_hand)


def test_the_figures_beside_are_written_to_a_file_of_their_own(tmp_path):
    a = S.beside_clause_a(S.clauses_a_b(FEW))
    d = S.beside_clause_d(S.clause_d_splits(FEW))
    path = S.write_figures_beside(a, d, written_by="test", directory=tmp_path)
    assert path == tmp_path / "g3_figures_beside.json" and list(tmp_path.iterdir()) == [path]
    saved = json.load(open(path, encoding="utf-8"))
    assert saved["settings"] == json.loads(json.dumps(S.settings()))
    assert math.isclose(saved["beside_clause_d"]["mean_pbo"], d["mean_pbo"], rel_tol=1e-11)
    assert set(saved) == {"written_by", "settings", "beside_clause_a", "beside_clause_d"}
    dd = S.beside_clause_d_deflated(S.clause_d_deflated(FEW))
    bound = S.bound_share(FEW)
    S.write_figures_beside(a, d, written_by="test", directory=tmp_path, d_deflated=dd, bound=bound)
    saved = json.load(open(path, encoding="utf-8"))
    assert saved["settings_added_on_24_november_2026"] == json.loads(json.dumps(S.settings_beside()))
    assert math.isclose(saved["deflated_sharpe_ratio_on_clause_d"]["median"], dd["median"], rel_tol=1e-11)
    assert math.isclose(saved["bound_further_repetitions"]["standard_error"], bound["standard_error"], rel_tol=1e-11)
    table = S.detectable_effects((0.0, 1.0), FEW)
    written = S.write_detectable_effects(table, directory=tmp_path)
    assert written == tmp_path / "g3_detectable_effects.csv" and len(written.read_text().splitlines()) == 3


# The three sets of figures beside the clauses added on 24 November 2026.

def test_the_beside_sub_streams_are_set_in_the_constants_and_differ_from_the_clauses():
    assert not set(C.BESIDE_G3_SUB_STREAMS.values()) & set(C.KNOWN_ANSWER_SUB_STREAMS.values())
    for name, sub in C.BESIDE_G3_SUB_STREAMS.items():
        direct = np.random.default_rng([C.SEED, C.SEED_STREAMS["known_answer"], sub, 2]).standard_normal(5)
        assert np.array_equal(S.rng_beside(name, 2).standard_normal(5), direct)


def test_the_deflated_sharpe_ratio_on_clause_d_uses_clause_d_draws():
    dd, d = S.clause_d_deflated(FEW), S.clause_d(FEW)
    assert dd["planted_is_best"].equals(d["planted_is_best"])
    assert dd["deflated_sharpe_ratio"].between(0, 1).all()
    out = S.beside_clause_d_deflated(dd)
    assert out["share_above_level"] == (dd["deflated_sharpe_ratio"] > C.DSR_LEVEL).mean()
    assert out["median"] == dd["deflated_sharpe_ratio"].median() and out["mean_threshold"] == dd["threshold"].mean()
    assert out["threshold_without_effect"] == O.expected_max_sharpe(C.G3_TRIALS, 1 / (C.G3_MONTHS - 1))


def test_the_detectable_effects_draw_each_row_from_its_own_position():
    both = S.detectable_effects((0.0, 1.0), FEW)
    alone = S.detectable_effects((0.0,), FEW)
    assert list(both["annual_sharpe"]) == [0.0, 1.0] and (both["repetitions"] == FEW).all()
    assert both.iloc[[0]].reset_index(drop=True).equals(alone)
    for column in ("share_planted_best", "share_dsr_above_0.95", "median_dsr", "mean_pbo", "share_pbo_below_0.25"):
        assert both[column].between(0, 1).all()
    assert np.allclose(both["share_e8_met"] + both["share_dsr_above_0.95"], 1)
    assert np.allclose(both["share_e9_met"] + both["share_pbo_below_0.25"], 1)
    assert both["monthly_sharpe"].iloc[1] == S.planted_monthly_sharpe()


def test_the_bound_share_draws_further_repetitions_of_clause_d():
    out = S.bound_share(FEW)
    assert out["repetitions"] == FEW and out["bound"] == 16 / 31
    again = S.beside_clause_d(S.clause_d_splits(FEW, rng=S.rng_beside("bound")))
    assert out["share_lower_when_other_picked"] == again["share_lower_when_other_picked"]
    assert not S.clause_d_splits(FEW, rng=S.rng_beside("bound")).equals(S.clause_d_splits(FEW))
    assert math.isclose(out["distance_from_bound_in_standard_errors"],
                        (out["share_lower_when_other_picked"] - out["bound"]) / out["standard_error"])


def test_the_figures_notebook_12_quotes_from_its_run():
    # Notebook 12 quotes these in steps 6 and in what was learned, from the files its run wrote.
    import pandas as pd
    folder = ROOT / "results" / "simulation"
    summary = json.loads((folder / "g3_summary.json").read_text(encoding="utf-8"))["clauses"]
    beside = json.loads((folder / "g3_figures_beside.json").read_text(encoding="utf-8"))
    effects = pd.read_csv(folder / "g3_detectable_effects.csv").set_index("annual_sharpe")
    nb = json.loads((ROOT / "notebooks" / "12_known_answer.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    dd, bound, d = beside["deflated_sharpe_ratio_on_clause_d"], beside["bound_further_repetitions"], beside["beside_clause_d"]
    quoted = [
        f"falls from {summary['c']['mean_pbo']:.3f} to {summary['d']['mean_pbo']:.3f}",
        f"above 0.95 in {dd['share_above_level']:.3f} of them, with a median of {dd['median']:.3f}",
        f"from {dd['threshold_without_effect']:.4f} to {dd['mean_threshold']:.4f} a month on average",
        f"the share is {d['share_lower_when_other_picked']:.3f}, "
        f"{(d['share_lower_when_other_picked'] - d['share_lower_bound_without_effect']) / d['share_lower_when_other_picked_se']:.1f} standard errors above the bound",
        f"Over {bound['repetitions']:,} further repetitions it is {bound['share_lower_when_other_picked']:.3f}, "
        f"with a standard error of {bound['standard_error']:.4f}",
        f"E8 would be met in {effects.loc[0.5, 'share_e8_met']:.3f} and E9 in {effects.loc[0.5, 'share_e9_met']:.3f} of the repetitions",
    ]
    missing = [q for q in quoted if q not in text]
    assert not missing, f"notebook 12 does not quote {missing}"
    assert bound["repetitions"] >= 20_000 and list(effects.index) == list(C.G3_DETECTABLE_ANNUAL_SHARPES)
