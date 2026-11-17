from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_innovation3a_tail_calibration"
SCRIPT = ROOT / "scripts" / "thesis" / "02_collaboration" / "audit_innovation3a_tail_calibration.py"


def load_module():
    spec = importlib.util.spec_from_file_location("innovation3a_tested", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_empirical_survival_probability_and_evidence_exact() -> None:
    module = load_module()
    calibration = np.array([1.0, 2.0, 2.0, 4.0])
    values = np.array([0.0, 2.0, 3.0, 5.0])
    expected_p = np.array([5 / 5, 4 / 5, 2 / 5, 1 / 5])
    assert np.allclose(module.tail_evidence(calibration, values), -np.log10(expected_p))


def test_three_frozen_target_files_and_hashes() -> None:
    summary = json.loads((OUT / "innovation3a_summary.json").read_text(encoding="utf-8"))
    paths = {
        "SWaT": ROOT / "outputs" / "thesis_swat_final_unified" / "dual_targets_unified_frozen.csv",
        "HAI21": ROOT / "outputs" / "thesis_hai21_final_unified" / "dual_targets_unified_frozen.csv",
        "WADI.A2": ROOT / "outputs" / "thesis_wadi_b0_unified_reliability" / "reliable_targets.csv",
    }
    assert summary["target_counts"] == {"SWaT": 15, "HAI21": 54, "WADI.A2": 47}
    for name, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == summary["target_sha256"][name]


def test_normal_only_chronological_segment_contract() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"].startswith("front 50% Calibration-A")
    assert manifest["threshold_source"] == "Calibration-A q99 only"
    assert manifest["self_lags"] == [1, 5, 10, 20, 30]
    assert manifest["cross_lags"] == [0, 1, 5, 10, 20, 30]
    assert manifest["alpha"] == 1.0
    assert {name: len(rows) for name, rows in manifest["validation_segments"].items()} == {"SWaT": 1, "HAI21": 3, "WADI.A2": 3}
    for rows in manifest["validation_segments"].values():
        for row in rows:
            assert row["calibration_a"] + row["audit_b"] == row["common_points"]
            assert row["audit_b"] in {row["calibration_a"], row["calibration_a"] + 1}


def test_branch_fusion_outputs_and_gates() -> None:
    summary = json.loads((OUT / "innovation3a_summary.json").read_text(encoding="utf-8"))
    branches = pd.read_csv(OUT / "branch_tail_calibration.csv")
    fusion = pd.read_csv(OUT / "fusion_normal_stability.csv")
    dominance = pd.read_csv(OUT / "dominance_analysis.csv")
    assert len(branches) == 6 and set(branches.branch) == {"REC", "DUAL"}
    assert len(fusion) == 3 and len(dominance) == 6
    assert np.isfinite(branches.select_dtypes("number")).all().all()
    assert np.isfinite(fusion.select_dtypes("number")).all().all()
    assert (branches.fraction_e_ge_2 <= .03).all()
    assert (fusion.tail_fusion_fpr <= .03).all()
    assert int((fusion.q99_balance_ratio <= 2).sum()) >= 2
    assert summary["gates"] == {"A": True, "B": True, "C": True, "C_passed_datasets": 3}
    assert summary["status"] == "NORMAL_TAIL_EVIDENCE_CALIBRATION_FEASIBLE"


def test_no_attack_or_test_path_and_no_search() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("attack_v0_features", "test1.csv.gz", "WADI_attackdataLABLE.csv"):
        assert forbidden not in text
    summary = json.loads((OUT / "innovation3a_summary.json").read_text(encoding="utf-8"))
    assert summary["attack_test_data_read"] is False
    assert summary["parameter_search"] is False
    assert summary["fusion_weight_search"] is False
    assert summary["test_tuning"] is False
    assert summary["tests"]["ecdf_fit_a_only"] is True
    assert summary["tests"]["audit_b_not_in_ecdf"] is True
    assert summary["tests"]["tail_fusion_max"] is True
    assert summary["tests"]["old_comparator_a_only"] is True

