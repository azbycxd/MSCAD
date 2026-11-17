from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts" / "thesis" / "02_collaboration" / "run_swat2015_v3r_cross_only.py"
    spec = importlib.util.spec_from_file_location("run_swat2015_v3r_cross_only", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def values():
    return np.arange(1200 * 51, dtype=np.float32).reshape(1200, 51)


def test_01_cross_only_excludes_every_target_lag():
    m = module(); columns = m.cross_only_columns(7)
    assert not np.any(columns % 51 == 7)


def test_02_initial_feature_count_is_250():
    m = module(); assert len(m.cross_only_columns(7)) == m.INITIAL_FEATURE_COUNT == 250


def test_03_predictor_count_is_fixed_to_five():
    assert module().TOP_PREDICTOR_COUNT == 5


def test_04_selected_feature_count_is_25():
    m = module(); assert len(m.selected_columns([1, 2, 3, 4, 5])) == m.SELECTED_FEATURE_COUNT == 25


def test_05_ridge_alpha_is_fixed():
    assert module().ALPHA == 1.0


def test_06_five_lags_are_frozen():
    assert module().LAGS.tolist() == [1, 5, 10, 20, 30]


def test_07_train_fit_stride_is_four():
    assert module().FIT_STRIDE == 4


def test_08_target_qualification_formula_is_fixed():
    m = module()
    assert m.stable_target(.30, .20, .20, 1.0)
    assert not m.stable_target(.299, .20, .20, 1.0)
    assert not m.stable_target(.30, .199, .20, 1.0)


def test_09_validation_halves_are_ordered_and_disjoint():
    calibration, evaluation = module().split_validation_times(99000)
    assert calibration[-1] < evaluation[0]


def test_10_r2_targets_are_strictly_r1_eligible_rows():
    import csv
    output = ROOT / "outputs" / "thesis_v3r_cross_only"
    r1 = list(csv.DictReader((output / "r1_target_metrics.csv").open(encoding="utf-8")))
    r2 = list(csv.DictReader((output / "r2_target_metrics.csv").open(encoding="utf-8")))
    frozen = {row["target"] for row in r1 if row["cross_only_stable_target"].lower() == "true"}
    assert {row["target"] for row in r2} == frozen


def test_11_corruption_preserves_target():
    m = module(); data = values(); times = np.asarray([600]); donors = np.asarray([200])
    clean, corrupt = m.hybrid_windows(data, times, donors, np.asarray([1, 2, 3, 4, 5]), 7)
    assert np.array_equal(clean[:, :, 7], corrupt[:, :, 7])


def test_12_corruption_changes_only_top5_predictors():
    m = module(); data = values(); times = np.asarray([600]); donors = np.asarray([200]); predictors = np.asarray([1, 2, 3, 4, 5])
    clean, corrupt = m.hybrid_windows(data, times, donors, predictors, 7)
    untouched = [index for index in range(51) if index not in predictors]
    assert np.array_equal(clean[:, :, untouched], corrupt[:, :, untouched])


def test_13_cyclic_donor_is_deterministic_and_separated():
    m = module(); universe = np.arange(100, 1100); samples = np.asarray([500, 600])
    first = m.cyclic_donors(universe, samples); second = m.cyclic_donors(universe, samples)
    assert np.array_equal(first, second)
    assert np.all(np.abs(first - samples) >= 300)


def test_14_no_attack_label_or_cache_read():
    source = (ROOT / "scripts" / "thesis" / "02_collaboration" / "run_swat2015_v3r_cross_only.py").read_text().casefold()
    assert "load_attack_cache" not in source
    assert "attack_v0" not in source
    assert "attack_labels" not in source


def test_15_cross_only_score_is_finite():
    import csv
    score = module().residual_score(np.asarray([0.0, 1.0]), 0.5, 1.0)
    rows = list(csv.DictReader((ROOT / "outputs" / "thesis_v3r_cross_only" / "r2_target_metrics.csv").open(encoding="utf-8")))
    numeric = [float(row[key]) for row in rows for key in ("v3_auroc", "v3_aupr", "a0_auroc", "a0_aupr", "corruption_recall", "normal_fpr")]
    assert np.isfinite(score).all() and np.isfinite(numeric).all()
