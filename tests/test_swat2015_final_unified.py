from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_swat_final_unified"


def load_summary() -> dict:
    return json.loads((OUT / "swat_final_unified_summary.json").read_text(encoding="utf-8"))


def test_expected_artifacts_exist() -> None:
    expected = {
        "dual_targets_unified_frozen.csv", "old_vs_unified_targets.csv", "global_metrics.csv",
        "event_metrics_36.csv", "contextual_events.csv", "category_summary.csv",
        "event_complementarity.csv", "target_attribution.csv", "normal_domain_shift.csv",
        "thresholds.json", "manifest.json", "swat_final_unified_summary.json",
    }
    assert expected == {path.name for path in OUT.iterdir()}


def test_target_freeze_sha_and_gate() -> None:
    summary = load_summary()
    payload = (OUT / "dual_targets_unified_frozen.csv").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == summary["target_sha256"]
    frozen = pd.read_csv(OUT / "dual_targets_unified_frozen.csv")
    skills = [column for column in frozen if column.endswith("_skill")]
    assert len(skills) == 8
    assert (frozen[skills] > 0).all().all()
    assert (frozen.median_ratio <= 2).all() and (frozen.q95_ratio <= 3).all()


def test_protocol_and_method_are_frozen() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["normal_train"] == [0, 396000]
    assert manifest["normal_validation"] == [396000, 495000]
    assert manifest["discovery_fit"] == [0, 316800]
    assert manifest["discovery_select"] == [316800, 396000]
    assert manifest["window"] == 64 and manifest["alpha"] == 1.0
    assert manifest["self_lags"] == [1, 5, 10, 20, 30]
    assert manifest["cross_lags"] == [0, 1, 5, 10, 20, 30]
    assert manifest["target_freeze_before_attack"] is True


def test_a0_reuse_and_normal_only_thresholds() -> None:
    summary = load_summary()
    thresholds = json.loads((OUT / "thresholds.json").read_text(encoding="utf-8"))
    assert summary["rec_reused"] is True
    assert summary["rec_best_epoch"] == 13 and summary["rec_parameter_count"] == 11730
    assert thresholds["source"] == "Official Normal Validation only"
    assert thresholds["quantile"] == 0.99 and thresholds["comparator"] == ">"
    assert all(np.isfinite([thresholds[name] for name in ("rec", "dual", "fusion")]))


def test_metrics_events_context_and_no_pa() -> None:
    summary = load_summary()
    metrics = pd.read_csv(OUT / "global_metrics.csv")
    events = pd.read_csv(OUT / "event_metrics_36.csv")
    contextual = pd.read_csv(OUT / "contextual_events.csv")
    assert metrics.method.tolist() == ["REC", "DUAL", "FUSION"]
    assert np.isfinite(metrics.select_dtypes("number")).all().all()
    assert len(events) == 36 and set(contextual.event_id) == {13, 14, 29}
    assert summary["tests"]["two_hard_segments"] is True
    assert summary["point_adjustment"] is False
    assert summary["no_test_driven_selection"] is True


def test_fusion_gate_status_is_pre_registered() -> None:
    summary = load_summary()
    expected = "NORMAL_CALIBRATED_MAX_FUSION_SWAT_SUPPORTED" if all(summary["fusion_gates"].values()) else "NORMAL_CALIBRATED_MAX_FUSION_SWAT_NOT_SUPPORTED"
    assert summary["status"] == expected

