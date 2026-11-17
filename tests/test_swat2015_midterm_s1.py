from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from data.swat2015_official import (
    EXACT_LABEL_MAP,
    EXPECTED_FEATURES,
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    VALIDATION_RAW_ROWS,
    WINDOW,
    OfficialSWaTError,
    build_split_plan,
    canonicalize_point_label,
    endpoint_labels,
    fit_scaler_on_declared_train,
    make_window_starts,
    reject_legacy_path,
)


ROOT = Path(__file__).resolve().parents[1]


def load_v3a_module():
    path = ROOT / "scripts" / "thesis" / "02_collaboration" / "run_swat2015_v3a_gate.py"
    spec = importlib.util.spec_from_file_location("run_swat2015_v3a_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_loader_rejects_legacy_fallback_paths():
    with pytest.raises(OfficialSWaTError):
        reject_legacy_path(Path("dataset/Swat/merged.csv"))


def test_frozen_input_has_51_features():
    assert EXPECTED_FEATURES == 51


def test_frozen_train_has_396000_raw_rows():
    assert TRAIN_RAW_ROWS == 396_000


def test_frozen_validation_has_99000_raw_rows():
    assert VALIDATION_RAW_ROWS == 99_000
    assert TRAIN_RAW_ROWS + VALIDATION_RAW_ROWS == EXPECTED_NORMAL_ROWS


def test_frozen_window_is_64():
    assert WINDOW == 64


def test_train_and_validation_windows_do_not_cross_split():
    train = make_window_starts(0, TRAIN_RAW_ROWS)
    validation = make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS)
    assert train[-1] + WINDOW == TRAIN_RAW_ROWS
    assert validation[0] == TRAIN_RAW_ROWS


def test_test_windows_do_not_cross_hard_gap():
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
    assert len(plan.test_window_starts_by_segment) == 2
    assert plan.test_window_starts_by_segment[0][-1] + WINDOW == left_count
    assert plan.test_window_starts_by_segment[1][0] == left_count


def test_scaler_fits_only_declared_train_prefix():
    normal = np.vstack([np.zeros((8, 2), dtype=np.float32), np.full((2, 2), 100.0, dtype=np.float32)])
    scaler, metadata = fit_scaler_on_declared_train(
        normal, train_raw_rows=8, expected_total_rows=10, expected_features=2
    )
    assert np.all(scaler.mean_ == 0.0)
    assert metadata["fit_row_count"] == 8
    assert metadata["validation_rows_seen"] == 0
    assert metadata["attack_rows_seen"] == 0


def test_endpoint_label_alignment_uses_last_window_row():
    labels = np.zeros(80, dtype=np.int8)
    labels[63] = 1
    labels[64] = 1
    assert endpoint_labels(labels, np.asarray([0, 1])).tolist() == [1, 1]


def test_a_space_ttack_exact_mapping_is_attack():
    assert canonicalize_point_label("A ttack") == 1
    assert EXACT_LABEL_MAP == {"Normal": 0, "Attack": 1, "A ttack": 1}


def test_unknown_label_is_hard_error_without_stripping():
    with pytest.raises(OfficialSWaTError):
        canonicalize_point_label(" Attack")


def test_self_model_columns_use_only_target_history():
    module = load_v3a_module()
    assert module.self_history_columns(7, 51, 5).tolist() == [7, 58, 109, 160, 211]


def test_full_history_contains_only_past_rows_not_current_input():
    module = load_v3a_module()
    values = np.arange(40, dtype=np.float32).reshape(20, 2)
    history, current = module.build_full_history(values, [1, 3])
    assert history[0].tolist() == values[[2, 0]].reshape(-1).tolist()
    assert current[0].tolist() == values[3].tolist()
    assert current[0, 0] not in history[0, ::2]


def test_full_history_has_255_features():
    module = load_v3a_module()
    history, _ = module.build_full_history(np.zeros((40, 51), dtype=np.float32), [1, 5, 10, 20, 30])
    assert history.shape[1] == 255


def test_relative_gain_formula():
    module = load_v3a_module()
    gain = module.relative_gain(np.asarray([2.0, 0.0]), np.asarray([1.5, 0.0]), 1e-8)
    assert gain[0] == pytest.approx(0.25)
    assert gain[1] == pytest.approx(0.0)
