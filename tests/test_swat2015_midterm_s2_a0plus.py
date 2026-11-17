from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from data.swat2015_official import OfficialSWaTError, WINDOW, build_split_plan


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "thesis" / "01_baseline" / "score_swat2015_midterm_a0plus.py"
    spec = importlib.util.spec_from_file_location("score_swat2015_midterm_a0plus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def residual_fixture() -> np.ndarray:
    base = np.arange(204, dtype=np.float64).reshape(4, 51) / 100.0
    return base


def test_01_calibration_accepts_only_train_residual():
    module = load_module()
    module.fit_channel_calibration(residual_fixture(), source_split="normal_train")
    with pytest.raises(OfficialSWaTError):
        module.fit_channel_calibration(residual_fixture(), source_split="normal_validation")


def test_02_calibration_has_51_medians_and_scales():
    module = load_module()
    median, mad, scale = module.fit_channel_calibration(residual_fixture(), source_split="normal_train")
    assert median.shape == mad.shape == scale.shape == (51,)


def test_03_scale_is_never_below_fixed_floor():
    module = load_module()
    constant = np.ones((5, 51), dtype=np.float64)
    _, _, scale = module.fit_channel_calibration(constant, source_split="normal_train")
    assert np.all(scale >= 1e-6)


def test_04_calibrated_z_is_finite_and_nonnegative():
    module = load_module()
    median, _, scale = module.fit_channel_calibration(residual_fixture(), source_split="normal_train")
    z = module.calibrate_residual(residual_fixture(), median, scale)
    assert np.isfinite(z).all()
    assert np.all(z >= 0)


def test_05_k_is_fixed_to_ceil_ten_percent_of_51():
    module = load_module()
    assert module.TOP_K == 6
    assert module.TOP_K == int(np.ceil(51 * 0.10))


def test_06_top6_score_is_mean_of_six_largest_values():
    module = load_module()
    z = np.arange(51, dtype=np.float64)[None, :]
    assert module.top6_score(z)[0] == pytest.approx(np.mean([45, 46, 47, 48, 49, 50]))


def test_07_threshold_accepts_only_validation_scores():
    module = load_module()
    scores = np.arange(100, dtype=np.float64)
    assert module.validation_q99(scores, source_split="normal_validation") == pytest.approx(np.quantile(scores, 0.99))
    with pytest.raises(OfficialSWaTError):
        module.validation_q99(scores, source_split="attack_test")


def test_08_test_endpoints_must_be_identical_to_a0():
    module = load_module()
    timestamp = np.asarray([1, 2, 3], dtype=np.int64)
    labels = np.asarray([0, 1, 0], dtype=np.int8)
    module.assert_same_a0_endpoints(timestamp, labels, timestamp.copy(), labels.copy())
    with pytest.raises(OfficialSWaTError):
        module.assert_same_a0_endpoints(timestamp, labels, timestamp + 1, labels)


def test_09_frozen_test_plan_does_not_cross_gap():
    import json

    manifest = json.loads(
        (ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "segment_manifest.json").read_text(encoding="utf-8")
    )
    left_count = int(manifest["test_segments"][0]["row_count"])
    right_count = int(manifest["test_segments"][1]["row_count"])
    left_start = int(np.datetime64(manifest["test_segments"][0]["start"], "s").astype(np.int64))
    right_start = int(np.datetime64(manifest["test_segments"][1]["start"], "s").astype(np.int64))
    timestamps = np.concatenate(
        [np.arange(left_start, left_start + left_count), np.arange(right_start, right_start + right_count)]
    ).astype(np.int64)
    plan = build_split_plan(timestamps, manifest)
    assert plan.test_window_starts_by_segment[0][-1] + WINDOW == left_count
    assert plan.test_window_starts_by_segment[1][0] == left_count


def test_10_config_forbids_all_test_driven_tuning():
    import json

    config = json.loads((ROOT / "configs" / "thesis" / "swat2015_midterm_s2_a0plus.yaml").read_text())
    assert config["test_driven_tuning"] is False
    assert config["retraining"] is False
    assert config["point_adjustment"] is False
    assert config["primary"]["k"] == 6
    assert config["threshold_quantile"] == 0.99
