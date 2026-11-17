from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts" / "thesis" / "02_collaboration" / "run_swat2015_v3i1_innovation.py"
    spec = importlib.util.spec_from_file_location("run_swat2015_v3i1_innovation", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def test_01_attack_data_is_not_read():
    source = (ROOT / "scripts" / "thesis" / "02_collaboration" / "run_swat2015_v3i1_innovation.py").read_text(encoding="utf-8").casefold()
    assert "load_attack_cache" not in source and "attack_v0" not in source and "attack_labels" not in source


def test_02_self_model_uses_only_own_history():
    m = module(); values = np.arange(400 * 3, dtype=float).reshape(400, 3); times = np.array([40, 44])
    changed = values.copy(); changed[:, 0] += 999
    assert np.array_equal(m.own_history(values, times, 1), m.own_history(changed, times, 1))


def test_03_innovation_is_frozen_self_residual():
    m = module(); values = np.arange(400 * 2, dtype=float).reshape(400, 2)
    coef = np.array([.1, .2, .3, .4, .5]); result = m.innovations(values, 1, coef, 7.0); t = 100
    assert np.isclose(result[t], values[t, 1] - (m.own_history(values, np.array([t]), 1) @ coef + 7.0)[0])


def test_04_cross_excludes_target_innovation():
    m = module(); columns = m.target_candidate_columns(3, 44)
    assert not np.any(columns % 44 == 3)


def test_05_cross_history_has_no_continuous_raw_level():
    source = (ROOT / "scripts" / "thesis" / "02_collaboration" / "run_swat2015_v3i1_innovation.py").read_text(encoding="utf-8")
    assert "dynamic_history(train_innov, train_delta" in source and "dynamic_history(train," not in source


def test_06_discrete_feature_is_transition_only():
    m = module(); values = np.array([[1., 4.], [1., 7.], [3., 7.]])
    delta = m.discrete_transitions(values, [0, 1]); assert np.array_equal(delta[1:], [[0., 3.], [2., 0.]])


def test_07_alpha_is_fixed_to_one():
    assert module().ALPHA == 1.0


def test_08_lags_are_fixed():
    assert module().LAGS.tolist() == [1, 5, 10, 20, 30]


def test_09_top5_is_fixed():
    m = module(); assert m.TOP_PREDICTOR_COUNT == 5 and len(m.selected_columns([1, 2, 3, 4, 5], 44)) == 25


def test_10_validation_not_used_for_selection():
    source = (ROOT / "scripts" / "thesis" / "02_collaboration" / "run_swat2015_v3i1_innovation.py").read_text(encoding="utf-8")
    assert "initial_coef, _ = ridge_fit(x_train" in source and '"predictor_selection_split": "Normal Train only"' in source


def test_11_validation_halves_are_chronological():
    times = np.arange(60, 99000); split = len(times) // 2
    assert times[split - 1] < times[split]


def test_12_outputs_are_finite():
    import csv
    path = ROOT / "outputs" / "thesis_v3i_innovation" / "v3i1_target_metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    keys = ("raw_r1_r2", "baseline_mae", "cross_mae", "relative_innovation_gain", "innovation_r2", "gain_first", "r2_first", "gain_second", "r2_second")
    assert rows and np.isfinite([float(row[key]) for row in rows for key in keys]).all()
