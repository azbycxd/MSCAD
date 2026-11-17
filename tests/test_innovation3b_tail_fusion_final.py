from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_innovation3b_tail_fusion_final"
SCRIPT = ROOT / "scripts" / "thesis" / "02_collaboration" / "run_innovation3b_tail_fusion_final.py"


def module():
    spec = importlib.util.spec_from_file_location("innovation3b_tested", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def summary() -> dict:
    return json.loads((OUT / "innovation3b_summary.json").read_text(encoding="utf-8"))


def test_empirical_probability_evidence_and_max_fusion() -> None:
    m = module(); calibration = np.array([1., 2., 2., 4.]); values = np.array([0., 2., 3., 5.])
    expected = -np.log10(np.array([1., .8, .4, .2]))
    assert np.allclose(m.tail_evidence(calibration, values), expected)
    freeze = {"rec_a": calibration, "dual_a": calibration[::-1]}
    er, ed, fusion = m.apply_tail(freeze, values, values[::-1])
    assert np.array_equal(fusion, np.maximum(er, ed))


def test_frozen_targets_and_validation_contract() -> None:
    s = summary(); manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    paths = {"SWaT":ROOT/"outputs"/"thesis_swat_final_unified"/"dual_targets_unified_frozen.csv",
             "HAI21":ROOT/"outputs"/"thesis_hai21_final_unified"/"dual_targets_unified_frozen.csv",
             "WADI.A2":ROOT/"outputs"/"thesis_wadi_b0_unified_reliability"/"reliable_targets.csv"}
    for name, path in paths.items(): assert hashlib.sha256(path.read_bytes()).hexdigest() == s["target_sha256"][name]
    assert manifest["ecdf_fit"] == "Calibration-A only"
    assert manifest["threshold"] == "Threshold-B q99 only"
    assert all(len(rows) in {1, 3} for rows in (item["validation_split"] for item in manifest["datasets"].values()))


def test_metrics_are_finite_and_old_fusion_is_frozen() -> None:
    metrics = pd.read_csv(OUT / "cross_dataset_metrics.csv")
    assert set(metrics.dataset) == {"SWaT", "HAI21", "WADI.A2"}
    assert set(metrics.method) == {"REC", "DUAL", "OLD_FUSION", "TAIL_FUSION"}
    assert len(metrics) == 12 and np.isfinite(metrics.select_dtypes("number")).all().all()
    frozen_paths = {"SWaT":ROOT/"outputs"/"thesis_swat_final_unified"/"global_metrics.csv",
                    "HAI21":ROOT/"outputs"/"thesis_hai21_final_unified"/"global_metrics.csv",
                    "WADI.A2":ROOT/"outputs"/"thesis_wadi_b1_final_external"/"global_metrics.csv"}
    for dataset, path in frozen_paths.items():
        historical = pd.read_csv(path).query("method == 'FUSION'").iloc[0]
        current = metrics[(metrics.dataset == dataset) & (metrics.method == "OLD_FUSION")].iloc[0]
        assert historical.auroc == current.auroc and historical.aupr == current.aupr


def test_event_sets_masks_and_attack13() -> None:
    swat = pd.read_csv(OUT / "swat_event_metrics.csv"); hai = pd.read_csv(OUT / "hai_event_metrics.csv"); wadi = pd.read_csv(OUT / "wadi_event_metrics.csv")
    assert len(swat) == 36
    assert len(hai) == 50 and int(hai.high_collaboration.sum()) == 19
    assert len(wadi) == 14 and int(wadi.high_collaboration.sum()) == 5 and 13 not in set(wadi.attack_id)
    dominance = pd.read_csv(OUT / "branch_dominance_test.csv")
    assert set(dominance[dominance.dataset == "SWaT"].scope) == {"NORMAL", "ATTACK"}
    assert set(dominance[dominance.dataset != "SWaT"].scope) == {"NORMAL", "ATTACK", "HIGH"}


def test_gates_and_no_search_or_adaptation() -> None:
    s = summary(); gates = pd.read_csv(OUT / "fusion_gate_results.csv").set_index("gate")
    assert s["gates"] == {"A": True, "B": True, "C": False, "D": False, "E": False}
    assert gates.passed.astype(str).str.lower().map({"true":True,"false":False}).to_dict() == s["gates"]
    assert s["status"] == "TAIL_FUSION_COMPLEMENTARY_BUT_SWAT_UNSTABLE"
    assert s["wadi_status"] == "DOMAIN_SHIFT_STRESS_TEST"
    assert s["no_retraining"] and s["no_test_tuning"] and s["no_fusion_search"] and s["no_pa"] and s["no_domain_adaptation"]
    text = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("best_f1", "point_adjust", "weighted_sum", "temperature", "POT"):
        assert forbidden not in text

