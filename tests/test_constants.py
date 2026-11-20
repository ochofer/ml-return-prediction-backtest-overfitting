"""The settings agree with each other: every count that two settings both imply is the same in both, and every
setting still to be set is None or has the length its count fixes."""
import datetime as dt
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlrp import constants as C  # noqa: E402


def months_between(first, last):
    a = dt.date.fromisoformat(first + "-01")
    b = dt.date.fromisoformat(last + "-01")
    return (b.year - a.year) * 12 + (b.month - a.month) + 1


def test_the_32_configurations_and_the_32_trials():
    assert sum(C.CONFIGURATION_COUNTS.values()) == C.CONFIGURATIONS == 32
    assert C.TRIALS == C.DSR_TRIAL_COUNT == C.FIXED_CONFIGURATIONS == C.G3_TRIALS == C.CONFIGURATIONS
    assert tuple(C.CONFIGURATION_COUNTS) == C.MODELS
    assert C.TUNED_MODELS == len(C.MODELS) == 6


def test_each_family_has_the_count_its_grid_implies():
    counts = C.CONFIGURATION_COUNTS
    assert counts["ols3"] == counts["ols12"] == 1
    assert len(C.ENET_LASSO_SHARES) * C.ENET_PENALTY_COUNT == counts["elastic_net"]
    assert C.RF_DEPTH_COUNT * C.RF_CHARACTERISTICS_PER_SPLIT_COUNT == counts["random_forest"]
    assert len(C.GBT_DEPTHS) * C.GBT_LEARNING_RATE_COUNT * C.GBT_TREE_COUNT_COUNT == counts["boosted_trees"]
    assert C.NN_WEIGHT_PENALTY_COUNT * C.NN_PASSES_COUNT == counts["network"]


def test_values_still_to_be_set_are_none_or_have_their_count():
    pairs = [
        (C.ENET_PENALTY_VALUES, C.ENET_PENALTY_COUNT),
        (C.RF_DEPTH_VALUES, C.RF_DEPTH_COUNT),
        (C.RF_CHARACTERISTICS_PER_SPLIT_VALUES, C.RF_CHARACTERISTICS_PER_SPLIT_COUNT),
        (C.GBT_LEARNING_RATE_VALUES, C.GBT_LEARNING_RATE_COUNT),
        (C.GBT_TREE_COUNT_VALUES, C.GBT_TREE_COUNT_COUNT),
        (C.NN_WEIGHT_PENALTY_VALUES, C.NN_WEIGHT_PENALTY_COUNT),
        (C.NN_PASSES_VALUES, C.NN_PASSES_COUNT),
    ]
    for values, count in pairs:
        assert values is None or (len(values) == count and len(set(values)) == count)


def test_the_twelve_characteristics():
    assert len(C.CHARACTERISTICS) == len(set(C.CHARACTERISTICS)) == C.CHARACTERISTIC_COUNT == 12
    assert set(C.OLS3_CHARACTERISTICS) <= set(C.CHARACTERISTICS) and len(C.OLS3_CHARACTERISTICS) == 3
    grouped = [c for group in C.E5_GROUPS.values() for c in group]
    assert len(grouped) == len(set(grouped)) == 8 and set(grouped) <= set(C.CHARACTERISTICS)
    assert math.isclose(C.E5_SHARE_BY_COUNT, len(grouped) / C.CHARACTERISTIC_COUNT)
    assert set(C.G1B_BOOK_EQUITY_CHARACTERISTICS) <= set(C.CHARACTERISTICS)


def test_the_panel_and_the_split():
    assert months_between(C.PANEL_FIRST_MONTH, C.PANEL_LAST_MONTH) == C.PANEL_MONTHS == 408
    assert months_between(C.TEST_FIRST_MONTH, C.TEST_LAST_MONTH) == C.TEST_MONTHS == 228
    assert C.TEST_YEARS == tuple(range(int(C.TEST_FIRST_MONTH[:4]), int(C.TEST_LAST_MONTH[:4]) + 1))
    assert len(C.TEST_YEARS) == C.REFITS == 19 and C.TEST_MONTHS == 12 * C.REFITS
    first, last = C.TEST_YEARS[0], C.TEST_YEARS[-1]
    assert 12 * (first - C.VALIDATION_YEARS - C.TRAIN_FIRST_YEAR) == C.FIRST_TRAINING_MONTHS == 120
    assert 12 * (last - C.VALIDATION_YEARS - C.TRAIN_FIRST_YEAR) == C.LAST_TRAINING_MONTHS == 336
    assert int(C.PANEL_FIRST_MONTH[:4]) == C.TRAIN_FIRST_YEAR
    assert C.G1B_FIRST_MONTH == C.PANEL_FIRST_MONTH and C.G1B_LAST_MONTH == C.PANEL_LAST_MONTH
    assert C.PANEL_LAST_MONTH == C.CRSP_END[:7] == C.G1A_LAST_MONTH
    assert C.TEST_MEMBER_MONTHS < C.PANEL_MEMBER_MONTHS < C.G1A_MEMBER_MONTHS
    assert 0 < C.LOOKAHEAD_MONTHS <= C.PANEL_MONTHS and 0 < C.LOOKAHEAD_TOLERANCE < 1e-9
    assert C.REFIT_ON_TRAINING_AND_VALIDATION is False


def test_the_windows_and_the_timing():
    assert C.MOMENTUM_FIRST_LAG > C.MOMENTUM_LAST_LAG > 1
    assert 0 < C.DAILY_MIN_DAYS and 0 < C.BETA_MIN_DAYS <= C.BETA_WINDOW_DAYS
    assert 0 < C.ACCOUNTING_MIN_AGE_MONTHS < C.ACCOUNTING_MAX_AGE_MONTHS
    assert C.RANK_LOW < C.FILL_VALUE < C.RANK_HIGH and C.FILL_VALUE == (C.RANK_LOW + C.RANK_HIGH) / 2


def test_the_blocks_divide_the_test_window():
    assert C.PBO_BLOCKS * C.PBO_BLOCK_MONTHS == C.TEST_MONTHS
    assert math.comb(C.PBO_BLOCKS, C.PBO_BLOCKS // 2) == C.PBO_SPLITS == 924
    assert C.G3_BLOCKS == C.PBO_BLOCKS and C.G3_MONTHS == C.TEST_MONTHS
    assert C.BOOTSTRAP_BLOCK_MONTHS in C.SIZE_CHECK_BOOTSTRAP_BLOCKS
    for block in C.SIZE_CHECK_BOOTSTRAP_BLOCKS:
        assert C.TEST_MONTHS % block == 0
    assert C.CV_EMBARGO_MONTHS == C.MOMENTUM_FIRST_LAG


def test_the_models_named_by_the_measures_and_the_expectations():
    assert C.DM_BENCHMARK == C.MODELS[0]
    assert set(C.E2_NONLINEAR_MODELS) <= set(C.MODELS) and C.E7_MODELS == C.E2_NONLINEAR_MODELS
    assert C.E6_MODELS == tuple(m for m in C.MODELS if m != C.DM_BENCHMARK)
    assert set(C.CV_MODELS) <= set(C.MODELS)
    assert C.HOLM_FAMILIES == {"diebold_mariano": len(C.MODELS) - 1, "rank_correlation": len(C.MODELS),
                               "sharpe_difference": len(C.E6_MODELS)}


def test_the_levels_are_ordered():
    for low, high in (C.G2_OLS3_R2_RANGE, C.G3_B_MEDIAN_RANGE, C.G3_C_PBO_RANGE):
        assert low < high
    assert 0 < C.G1B_COVERAGE_MIN_BOOK_EQUITY < C.G1B_COVERAGE_MIN < 1
    assert 0 < C.DSR_LEVEL < 1 and 0 < C.INTERVAL_LEVEL < 1 and 0 < C.SIZE_CHECK_POWER < 1
    assert C.G3_C_PBO_RANGE[0] < 0.5 < C.G3_C_PBO_RANGE[1] and C.G3_D_PBO_MAX < C.G3_C_PBO_RANGE[0]
    assert C.G3_REPETITIONS_A_B > C.G3_REPETITIONS_C_D > 0
    assert C.COST_BP < C.COST_BP_SENSITIVITY


def test_the_record_of_the_data():
    assert re.fullmatch(r"[0-9a-f]{64}", C.MANIFEST_SHA256)
    dt.date.fromisoformat(C.EXTRACTION_DATE)
    dt.date.fromisoformat(C.CRSP_END)
    dt.date.fromisoformat(C.DAILY_FILE_START)
    assert len(C.TABLES) == C.TABLE_COUNT == 8
    assert all(re.fullmatch(r"(crsp|comp)\.[a-z0-9_]+", v) for v in C.TABLES.values())
    assert C.DATA_ROOT_ENV == "MLRP_DATA_ROOT"


def test_the_copies_are_named_with_their_commits_and_hashes():
    assert len(C.COPIED_MODULES) == len(set(C.COPIED_MODULES)) == 7
    assert "constants.py" not in C.COPIED_MODULES
    assert re.fullmatch(r"[0-9a-f]{40}", C.SOURCE_COMMIT)
    assert set(C.COPIED_FILES) == set(C.COPIED_MODULES)
    for name, (path, digest) in C.COPIED_FILES.items():
        assert path == "src/slt/" + name and re.fullmatch(r"[0-9a-f]{64}", digest)
    assert re.fullmatch(r"[0-9a-f]{64}", C.SOURCE_CONSTANTS_SHA256)
    assert re.fullmatch(r"[0-9a-f]{64}", C.BOOTSTRAP_SHA256)
    assert re.fullmatch(r"[0-9a-f]{7,40}", C.BOOTSTRAP_SOURCE_COMMIT)
    assert len(C.G1A_REFERENCE_TABLES) == 3


def test_the_digest_of_the_manifest_entries():
    assert re.fullmatch(r"[0-9a-f]{64}", C.MANIFEST_ENTRIES_SHA256)
    assert len(C.TABLES) <= C.MANIFEST_ENTRIES <= C.MANIFEST_FILES


def test_the_seed_and_its_streams():
    assert isinstance(C.SEED, int) and C.SEED > 0
    assert len(set(C.SEED_STREAMS.values())) == len(C.SEED_STREAMS)
