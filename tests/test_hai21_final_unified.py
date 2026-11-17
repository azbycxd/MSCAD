import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_hai21_final_unified"
MANIFEST = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((OUT / "hai21_final_unified_summary.json").read_text(encoding="utf-8"))
TARGETS = pd.read_csv(OUT / "dual_targets_unified_frozen.csv")
EVENTS = pd.read_csv(OUT / "event_metrics_50.csv")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_01_hai21_only():
    assert MANIFEST["dataset"] == "HAI 21.03"


def test_02_three_train_five_test():
    assert len(MANIFEST["train_files"]) == 3 and len(MANIFEST["test_files"]) == 5


def test_03_test_after_target_freeze():
    assert MANIFEST["target_freeze_before_test"] is True and MANIFEST["test_read_count"] == 5


def test_04_process_groups_frozen():
    assert set(TARGETS.process) == {"P1", "P2", "P3", "P4"}


def test_05_labels_excluded_features():
    assert not set(MANIFEST["label_columns"]) & set(MANIFEST["feature_names"])


def test_06_no_boundary_crossing():
    assert SUMMARY["tests"]["no_boundary_crossing"] is True


def test_07_baseline_median_fit_only():
    assert SUMMARY["tests"]["baseline_median_fit_only"] is True


def test_08_self_lags_fixed():
    assert MANIFEST["self_lags"] == [1, 5, 10, 20, 30]


def test_09_cross_lags_fixed():
    assert MANIFEST["cross_lags"] == [0, 1, 5, 10, 20, 30]


def test_10_target_excluded_from_cross():
    assert SUMMARY["tests"]["target_excluded_cross"] is True


def test_11_alpha_one():
    assert MANIFEST["alpha"] == 1.0


def test_12_unified_gate_exact():
    skill_cols = [c for c in TARGETS if c.endswith("_skill")]
    assert len(skill_cols) == 8
    assert (TARGETS[skill_cols] > 0).all().all()
    assert (TARGETS.median_ratio <= 2.0).all() and (TARGETS.q95_ratio <= 3.0).all()


def test_13_no_top_k():
    assert SUMMARY["no_target_top_k"] is True


def test_14_target_sha_matches():
    assert file_sha(OUT / "dual_targets_unified_frozen.csv") == SUMMARY["target_sha256"] == MANIFEST["target_sha256"]


def test_15_dual_formula_frozen():
    assert MANIFEST["dual_score"] == "max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/scale_j)"


def test_16_thresholds_normal_only():
    assert MANIFEST["threshold_source"] == "Normal Validation suffixes only"


def test_17_high_events_frozen():
    high = pd.read_csv(OUT / "high_event_metrics.csv")
    assert len(high) == 19 and set(high.event_id) == set(EVENTS.loc[EVENTS.category == "HIGH", "event_id"])


def test_18_no_point_adjustment():
    assert SUMMARY["point_adjustment"] is False


def test_19_common_mask():
    assert SUMMARY["tests"]["common_mask"] is True and SUMMARY["valid_test_points"] == 401690


def test_20_no_nan_inf():
    for name in ("dual_targets_unified_frozen.csv", "global_metrics.csv", "event_metrics_50.csv", "high_event_metrics.csv", "category_summary.csv", "event_complementarity.csv", "target_attribution.csv", "normal_domain_shift.csv"):
        frame = pd.read_csv(OUT / name)
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        assert np.isfinite(numeric).all(), name


def test_21_no_search_or_test_selection():
    assert SUMMARY["no_test_driven_selection"] is True
    assert SUMMARY["lag_search"] is False and SUMMARY["threshold_search"] is False


def test_22_expected_outputs_present():
    expected = {"dual_targets_unified_frozen.csv", "old_vs_unified_targets.csv", "global_metrics.csv", "event_metrics_50.csv", "high_event_metrics.csv", "category_summary.csv", "event_complementarity.csv", "target_attribution.csv", "normal_domain_shift.csv", "thresholds.json", "manifest.json", "hai21_final_unified_summary.json"}
    assert expected <= {path.name for path in OUT.iterdir()}
