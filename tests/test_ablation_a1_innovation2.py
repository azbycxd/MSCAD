import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_ablation" / "innovation2"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_d1_lags_are_exactly_preregistered():
    for dataset in ("swat", "hai"):
        manifest = read_json(OUT / "d1_short_lag" / dataset / "manifest.json")
        assert manifest["self_lags"] == [1]
        assert manifest["cross_lags"] == [0, 1]


def test_alpha_is_one():
    for variant, datasets in (("d1_short_lag", ("swat", "hai")), ("d2_no_gate", ("swat", "hai"))):
        for dataset in datasets:
            assert read_json(OUT / variant / dataset / "manifest.json")["alpha"] == 1.0


def test_d2_swat_has_25_candidates():
    assert read_json(OUT / "d2_no_gate" / "swat" / "manifest.json")["candidate_count"] == 25


def test_d2_hai_has_56_candidates():
    assert read_json(OUT / "d2_no_gate" / "hai" / "manifest.json")["candidate_count"] == 56


def test_d2_has_no_test_target_selection():
    for dataset in ("swat", "hai"):
        assert read_json(OUT / "d2_no_gate" / dataset / "manifest.json")["test_target_selection"] is False


def test_gate_sets_are_monotonic():
    for dataset in ("swat", "hai"):
        rows = read_json(OUT / "d3_gate_decomposition" / dataset / "stage_results.json")
        sets = [set(row["target_ids"].split(";")) for row in rows]
        assert sets[1] <= sets[0] and sets[2] <= sets[1] and sets[3] <= sets[2]


def test_d3_d_target_lists_equal_final_gate():
    for dataset, source in (("swat", "thesis_swat_final_unified"), ("hai", "thesis_hai21_final_unified")):
        rows = read_json(OUT / "d3_gate_decomposition" / dataset / "stage_results.json")
        expected = pd.read_csv(ROOT / "outputs" / source / "dual_targets_unified_frozen.csv").target.astype(str).tolist()
        assert rows[-1]["target_ids"].split(";") == expected


def test_d3_d_metrics_reproduce_d0():
    for dataset, source in (("swat", "thesis_swat_final_unified"), ("hai", "thesis_hai21_final_unified")):
        actual = read_json(OUT / "d3_gate_decomposition" / dataset / "stage_results.json")[-1]
        expected = pd.read_csv(ROOT / "outputs" / source / "global_metrics.csv").query("method == 'DUAL'").iloc[0]
        for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr"):
            assert abs(float(actual[key]) - float(expected[key])) <= 1e-6


def test_controlled_donor_manifest_hash_is_unchanged():
    for variant in ("d1_short_lag", "d2_no_gate"):
        manifest = read_json(OUT / variant / "swat_controlled" / "manifest.json")
        assert manifest["donor_sha256_before"] == manifest["donor_sha256_after"]
        assert manifest["new_corruption"] is False


def test_comut_has_no_industrial_gate():
    manifest = read_json(OUT / "d1_short_lag" / "comut" / "manifest.json")
    assert manifest["case_count"] == 13
    assert manifest["industrial_gate"] is False


def test_q99_sources_are_normal_validation_only():
    for variant in ("d1_short_lag", "d2_no_gate"):
        for dataset in ("swat", "hai"):
            threshold = read_json(OUT / variant / dataset / "threshold.json")
            assert threshold["quantile"] == 0.99
            assert threshold["source"] == "Normal Validation only"


def test_raw_score_and_state_artifacts_are_persisted():
    for variant in ("d1_short_lag", "d2_no_gate"):
        for dataset in ("swat", "hai"):
            base = OUT / variant / dataset
            for name in ("ridge_state.npz", "ridge_state.json", "validation_scores.npz", "test_scores.npz", "calibration_stats.json", "target_list.json"):
                assert (base / name).is_file() and (base / name).stat().st_size > 0
            with np.load(base / "test_scores.npz", allow_pickle=False) as payload:
                assert {"candidate_names", "raw_disagreement", "calibrated_scores", "selected_global_score", "labels", "timeline"} <= set(payload.files)


def test_no_point_adjustment_or_test_best():
    summary = read_json(OUT / "summary" / "abl1_summary.json")
    assert summary["restrictions"]["point_adjustment"] is False
    assert summary["restrictions"]["test_tuning"] is False
    for variant in ("d1_short_lag", "d2_no_gate"):
        for dataset in ("swat", "hai"):
            manifest = read_json(OUT / variant / dataset / "manifest.json")
            assert manifest["point_adjustment"] is False
            assert manifest["oracle"] is False
