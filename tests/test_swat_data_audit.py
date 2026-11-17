from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage4_3d_audit_swat_data_pipeline.py"
SPEC = importlib.util.spec_from_file_location("stage4_3d_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def frame(rows):
    return pd.DataFrame(rows, columns=[" Timestamp", "F1", " Normal/Attack"])


def test_chunked_prefix_match_and_first_mismatch(tmp_path):
    normal = tmp_path / "normal.csv"
    merged = tmp_path / "merged.csv"
    rows = [[f"01/01/2020 00:00:0{i}", i, " Normal "] for i in range(5)]
    frame(rows).to_csv(normal, index=False)
    frame(rows + [["01/01/2020 00:00:05", 9, "Attack"]]).to_csv(merged, index=False)
    result = audit.prefix_audit(normal, merged, chunksize=2)
    assert result["exact_prefix_match"] is True
    changed = rows.copy()
    changed[3] = [changed[3][0], 99, changed[3][2]]
    frame(changed).to_csv(merged, index=False)
    result = audit.prefix_audit(normal, merged, chunksize=2)
    assert result["exact_prefix_match"] is False
    assert result["first_mismatch_index"] == 3


def test_label_strip_case_and_numeric_mapping():
    assert [audit.normalize_label(v) for v in [" Normal ", "NORMAL", "0", 0.0]] == ["Normal"] * 4
    assert [audit.normalize_label(v) for v in [" Attack ", "ATTACK", "1", 1.0]] == ["Attack"] * 4
    assert audit.loader_label(" Normal ") == 0
    assert audit.loader_label("Attack") == 1


def test_segment_extraction():
    labels = ["normal", " Normal ", "ATTACK", "attack", "Normal"]
    times = [str(i) for i in range(5)]
    segments = audit.extract_segments(labels, times, "toy.csv")
    assert [(s["label"], s["start_index"], s["end_index"]) for s in segments] == [
        ("Normal", 0, 1), ("Attack", 2, 3), ("Normal", 4, 4)
    ]
    assert segments[1]["previous_label"] == "Normal"
    assert segments[1]["next_label"] == "Normal"


def test_duplicate_hash_count():
    hashes = np.array([2, 2, 2, 4, 5, 5], dtype=np.uint64)
    assert audit.duplicate_count(hashes) == 3


def test_timestamp_monotonicity():
    assert audit.timestamp_monotonic(["01/01/2020 00:00:00", "01/01/2020 00:00:01"])
    assert not audit.timestamp_monotonic(["01/01/2020 00:00:01", "01/01/2020 00:00:00"])


def test_source_index_continuity():
    source = frame([
        ["01/01/2020 00:00:00", 0, "Normal"],
        ["01/01/2020 00:00:01", 1, "Attack"],
        ["01/01/2020 00:00:02", 2, "Normal"],
        ["01/01/2020 00:00:03", 3, "Attack"],
    ])
    detail, summary = audit.temporal_continuity(source, " Normal/Attack", " Timestamp")
    assert detail.iloc[0]["source_index_gap"] == 2
    assert bool(detail.iloc[0]["consecutive_in_source"]) is False
    assert summary["source_index_nonconsecutive_pairs"] == 1


def test_audit_helpers_are_read_only(tmp_path):
    path = tmp_path / "sample.csv"
    frame([["01/01/2020 00:00:00", 1, "Normal"]]).to_csv(path, index=False)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    audit.profile_csv(path, chunksize=1)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def test_canonical_pre_scaler_values_ignore_csv_float_formatting():
    left = frame([["01/01/2020 00:00:00", "2", "Normal"]])
    right = frame([["01/01/2020 00:00:00", "2.0", " Normal "]])
    result = audit.compare_frames(left, right, " Normal/Attack", " Timestamp")
    assert bool(result["row_equal"].all())
    assert result["feature_equal_count"] == result["feature_value_count"]


def test_script_has_no_model_imports():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "model.load" not in text


def test_script_has_no_training_operations():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "optimizer" not in text
    assert "backward(" not in text
