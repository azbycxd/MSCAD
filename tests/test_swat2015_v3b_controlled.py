from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from data.swat2015_official import OfficialSWaTError


ROOT = Path(__file__).resolve().parents[1]
V3A = ROOT / "outputs" / "thesis_midterm_s1" / "v3a_incremental_gate_float64"


def module():
    path = ROOT / "scripts" / "thesis" / "02_collaboration" / "run_swat2015_v3b_controlled.py"
    spec = importlib.util.spec_from_file_location("run_swat2015_v3b_controlled", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def frozen_rows():
    rows = list(csv.DictReader((V3A / "target_metrics.csv").open(encoding="utf-8")))
    return [row for row in rows if row["incremental_collaboration_target"].lower() == "true"]


def synthetic_normal():
    return np.arange(1000 * 51, dtype=np.float32).reshape(1000, 51)


def test_01_target_count_is_fixed_to_four():
    assert len(frozen_rows()) == 4


def test_02_targets_come_directly_from_v3a_without_reselection():
    assert {row["target"] for row in frozen_rows()} == {"FIT201", "DPIT301", "FIT301", "FIT601"}
    assert all(float(row["relative_gain"]) >= 0.10 for row in frozen_rows())


def test_03_target_history_is_bitwise_identical_after_corruption():
    m = module(); values = synthetic_normal(); times = np.asarray([500, 600]); donors = times - 60
    clean, corrupt = m.build_hybrid_windows(values, times, donors, 7)
    assert np.array_equal(clean[:, :, 7], corrupt[:, :, 7])


def test_04_other_context_is_copied_from_real_normal_rows():
    m = module(); values = synthetic_normal(); times = np.asarray([500]); donors = np.asarray([440])
    _, corrupt = m.build_hybrid_windows(values, times, donors, 7)
    assert np.array_equal(corrupt[0, :, 8], values[377:441, 8])


def test_05_temporal_shift_is_frozen_to_60_seconds():
    assert module().SHIFT_SECONDS == 60


def test_06_matched_donor_separation_is_at_least_300_seconds():
    m = module(); values = synthetic_normal(); samples = np.asarray([500]); pool = np.asarray([100, 200, 800, 900])
    donor = m.select_matched_donors(values, samples, pool, 7)
    assert abs(int(donor[0]) - 500) >= 300


def test_07_matched_donor_selection_is_deterministic():
    m = module(); values = synthetic_normal(); samples = np.asarray([500]); pool = np.asarray([100, 200, 800, 900])
    assert np.array_equal(m.select_matched_donors(values, samples, pool, 7), m.select_matched_donors(values, samples, pool, 7))


def test_08_script_has_no_attack_cache_or_label_read():
    source = (ROOT / "scripts" / "thesis" / "02_collaboration" / "run_swat2015_v3b_controlled.py").read_text()
    assert "load_attack_cache" not in source
    assert "attack_v0" not in source.casefold()
    assert "attack_labels" not in source.casefold()


def test_09_self_input_and_residual_are_unchanged():
    m = module(); values = synthetic_normal(); times = np.asarray([500, 600]); donors = times - 60
    full_coef = np.zeros((51, 255)); full_intercept = np.zeros(51)
    self_coef = np.ones((51, 5)); self_intercept = np.zeros(51)
    clean = m.predict_errors(values, times, 7, full_coef, full_intercept, self_coef, self_intercept)
    corrupt = m.predict_errors(values, times, 7, full_coef, full_intercept, self_coef, self_intercept, donors)
    assert np.array_equal(clean[0], corrupt[0])


def test_10_calibration_and_evaluation_times_do_not_overlap():
    calibration, evaluation = module().split_validation_times(99000)
    assert calibration[-1] < evaluation[0]


def test_11_collaboration_score_formula_is_exact():
    m = module(); gain = np.asarray([3.0, 1.0, -1.0])
    assert m.collaboration_score(gain, 1.0, 2.0).tolist() == [0.0, 0.0, 1.0]


def test_12_q99_accepts_only_calibration_normal():
    m = module(); score = np.arange(100, dtype=float)
    assert m.calibration_q99(score, source_split="calibration_normal") == pytest.approx(np.quantile(score, .99))
    with pytest.raises(OfficialSWaTError):
        m.calibration_q99(score, source_split="evaluation_normal")


def test_13_a0_checkpoint_read_does_not_modify_it():
    m = module(); path = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "best.pt"
    before = m.sha256(path); path.open("rb").read(1); assert m.sha256(path) == before


def test_14_core_scores_reject_or_avoid_nan_inf():
    score = module().collaboration_score(np.asarray([0.0, 1.0]), 0.5, 1.0)
    assert np.isfinite(score).all()
