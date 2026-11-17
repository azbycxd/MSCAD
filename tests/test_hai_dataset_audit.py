from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hai_step1_audit_dataset.py"
SPEC = importlib.util.spec_from_file_location("hai_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_numeric_labels():
    assert audit.normalize_label(0) == "Normal"
    assert audit.normalize_label(1.0) == "Attack"


def test_string_labels():
    assert audit.normalize_label(" normal ") == "Normal"
    assert audit.normalize_label("ATTACK") == "Attack"


def test_normal_attack_normal_detection():
    summary, _ = audit.sequence_stats([0, 0, 1, 1, 0])
    assert summary["contains_normal_attack_normal"]
    assert summary["attack_segment_count"] == 1


def test_attack_only_is_not_complete_test():
    summary, _ = audit.sequence_stats([1, 1, 1])
    assert not summary["contains_normal_attack_normal"]
    assert summary["normal_count"] == 0


def test_timestamp_decrease_and_gap_detection():
    stats = audit.timestamp_stats(["2021-01-01 00:00:00", "2021-01-01 00:00:01", "2021-01-01 00:00:05", "2021-01-01 00:00:04"])
    assert stats["timestamp_decrease_count"] == 1
    assert stats["non_standard_interval_count"] == 2
    assert not stats["timestamp_monotonic_non_decreasing"]


def test_schema_consistency():
    base = {"readable": True, "split": "train", "file_name": "train1.csv", "feature_columns": ["a", "b"],
            "raw_column_names": '["timestamp", "a", "b", "Attack"]',
            "stripped_column_names": '["timestamp", "a", "b", "Attack"]', "feature_count": 2,
            "timestamp_column": "timestamp", "label_column": "Attack"}
    other = dict(base, file_name="test1.csv", split="test")
    _, _, info = audit.schema_consistency([base, other])
    assert info["train_test_feature_schema_consistent"]
    assert info["feature_count"] == 2


def test_sampled_window_overlap():
    values = np.arange(300, dtype=np.uint64)
    assert audit.sampled_window_hashes(values, 64) == audit.sampled_window_hashes(values.copy(), 64)
    assert audit.sampled_window_hashes(values, 128).isdisjoint(audit.sampled_window_hashes(values + 10_000, 128))


def test_boundaries_are_never_joined():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"can_form_windows_with_next_file": False' in text


def test_audit_file_is_read_only(tmp_path):
    path = tmp_path / "train1.csv"
    pd.DataFrame({"timestamp": ["2021-01-01 00:00:00"], "x": [1.0], "Attack": [0]}).to_csv(path, index=False)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = audit.audit_file(path, "train")
    assert result["readable"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_no_model_or_training_code():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "load_state_dict" not in text
    assert "optimizer" not in text
    assert "backward(" not in text
