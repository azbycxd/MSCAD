import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "thesis" / "00_audit" / "evaluate_literature_compatible.py"
SPEC = importlib.util.spec_from_file_location("evaluation_e2", MODULE_PATH)
E2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E2)


def test_exact_oracle_uses_all_unique_boundaries_and_largest_tie_threshold():
    y = np.array([1, 1, 0, 0], dtype=np.int8)
    s = np.array([0.9, 0.8, 0.8, 0.1], dtype=float)
    result = E2.exact_oracle(y, s)
    assert result["n_unique_thresholds"] == len(np.unique(s)) + 1
    assert result["threshold"] == 0.1
    assert result["f1"] == 0.8


def test_pa_fills_hit_interval_only():
    y = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    intervals = E2.anomaly_intervals(y, [8])
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    adjusted = E2.point_adjustment(pred, y, intervals)
    assert adjusted.tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


def test_pa_never_removes_false_positives():
    y = np.array([0, 1, 1, 0], dtype=np.int8)
    pred = np.array([1, 0, 1, 1], dtype=np.int8)
    adjusted = E2.point_adjustment(pred, y, E2.anomaly_intervals(y, [4]))
    assert adjusted[0] == 1 and adjusted[3] == 1


def test_pa_does_not_join_equal_labels_across_segments():
    y = np.array([0, 1, 1, 1, 1, 0], dtype=np.int8)
    intervals = E2.anomaly_intervals(y, [3, 3])
    assert [(a, b, seg) for a, b, seg in intervals] == [(1, 3, 0), (3, 5, 1)]
    pred = np.array([0, 0, 1, 0, 0, 0], dtype=np.int8)
    assert E2.point_adjustment(pred, y, intervals).tolist() == [0, 1, 1, 0, 0, 0]


def test_exact_pa_oracle_matches_bruteforce():
    y = np.array([0, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    s = np.array([0.2, 0.1, 0.9, 0.8, 0.3, 0.4, 0.7], dtype=float)
    intervals = E2.anomaly_intervals(y, [4, 3])
    exact = E2.exact_pa_oracle(y, s, intervals)
    thresholds = np.r_[np.nextafter(s.min(), -np.inf), np.unique(s)]
    brute = [E2.pa_at_threshold(y, s, float(t), intervals) | {"threshold": float(t)} for t in thresholds]
    best = max(x["f1"] for x in brute)
    expected = max(x["threshold"] for x in brute if x["f1"] == best)
    assert exact["f1"] == best and exact["threshold"] == expected


def test_strict_metrics_reproduce_existing_result():
    result = E2.run()
    assert result["checks"]["strict_reproduction"] == "PASS"
    assert max(result["checks"]["strict_absolute_differences"].values()) <= 1e-6


def test_swat_ours_rec_oracle_crosschecks_old_diagnostic():
    result = E2.run()
    assert result["checks"]["old_oracle_cross_check"] == "PASS"


def test_common_endpoint_mask_is_preserved():
    result = E2.run()
    row = next(x for x in result["availability_rows"] if x["dataset"] == "SWaT" and x["method"] == "Ours-REC")
    assert row["n_scores"] == row["n_labels"] == 449793
    assert result["event_pa_audit"]["SWaT"]["cross_segment_adjustment"] is False


def test_anomaly_scores_are_not_modified_or_transformed():
    result = E2.run()
    assert result["checks"]["no_score_modification"] is True
    assert result["checks"]["no_score_transformation"] is True
    assert result["source_hashes"]["swat_ours_rec_scores_sha256"] == "b71b9b6a82985d93f61dcc2d9672afcb7d64dec9040b50cbce7cb75e4335afda"


def test_no_training_is_invoked():
    source = MODULE_PATH.read_text(encoding="utf-8")
    result = E2.run()
    assert result["checks"]["no_training"] is True
    assert "torch" not in source


def test_no_checkpoint_is_read_or_modified():
    source = MODULE_PATH.read_text(encoding="utf-8")
    result = E2.run()
    assert result["checks"]["no_checkpoint_read"] is True
    assert ".pt" not in source and "state_dict" not in source


def test_missing_scores_remain_explicitly_missing():
    result = E2.run()
    assert sum(int(x["available"]) for x in result["availability_rows"]) == 1
    assert all(x["notes"].startswith("SCORE_NOT_PERSISTED") for x in result["availability_rows"] if not x["available"])
