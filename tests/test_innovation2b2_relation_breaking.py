"""Integrity checks for the frozen Innovation-2B2 output."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_innovation2b2_relation_breaking"
SUMMARY = json.loads((OUT / "innovation2b2_summary.json").read_text(encoding="utf-8"))
METRICS = pd.read_csv(OUT / "metrics.csv")
STRENGTH = pd.read_csv(OUT / "corruption_strength.csv")
DONORS = pd.read_csv(OUT / "donor_manifest.csv")


def test_required_outputs_exist():
    for name in ("metrics.csv", "corruption_strength.csv", "donor_manifest.csv", "old_vs_new.csv", "innovation2b2_summary.json"):
        assert (OUT / name).is_file()


def test_six_targets_frozen():
    assert tuple(METRICS.target) == ("AIT202", "FIT201", "DPIT301", "FIT301", "FIT401", "FIT601")


def test_2000_samples_per_target():
    assert DONORS.groupby("target").size().eq(2000).all()


def test_donor_normal_evaluation_range():
    assert DONORS.donor_endpoint.between(445563, 494999).all()


def test_donor_separation():
    assert (DONORS.donor_separation >= 600).all()


def test_candidate_limit():
    assert (DONORS.candidate_count <= 256).all()


def test_no_nan_inf():
    assert METRICS.select_dtypes("number").notna().all().all()
    assert STRENGTH.select_dtypes("number").notna().all().all()


def test_self_sanity():
    assert METRICS.self_auroc.between(0.49, 0.51).all()


def test_validity_gate_count():
    assert int((STRENGTH.median_shift_ratio >= 1).sum()) == SUMMARY["validity_count"] == 4


def test_clean_fpr_gate():
    assert METRICS.normal_fpr.median() <= 0.03


def test_input_only_donor_contract():
    assert SUMMARY["tests"]["donor_selection_input_distance_only"]
    assert SUMMARY["tests"]["no_model_coefficient_in_donor_selection"]
    assert SUMMARY["tests"]["no_residual_or_score_in_donor_selection"]


def test_no_attack_contract():
    assert SUMMARY["tests"]["no_attack_data"]


def test_source_hashes_unchanged():
    assert SUMMARY["hashes_before"] == SUMMARY["hashes_after"]


def test_frozen_formula_and_calibration():
    assert SUMMARY["tests"]["model_and_score_formula_unchanged"]
    assert SUMMARY["tests"]["calibration_unchanged"]


def test_preregistered_status():
    assert SUMMARY["gates"]["V_at_least_4_of_6_median_shift_ratio_at_least_1"]
    assert not SUMMARY["gates"]["B_at_least_4_of_6_collab_beats_a0_by_0_10"]
    assert SUMMARY["status"] == "STAGE_JOINT_LAGGED_SCORE_NOT_SUPPORTED"
