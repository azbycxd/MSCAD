from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_ablation" / "innovation3"
MANIFEST = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
PAYLOAD = json.loads((OUT / "summary_payload.json").read_text(encoding="utf-8"))


def _rows(name: str) -> list[dict]:
    return PAYLOAD[name]


def _metric(dataset: str, variant: str) -> dict:
    return next(
        row
        for row in _rows("fusion_detection_ablation")
        if row["dataset"] == dataset and row["variant"] == variant
    )


def _balance(dataset: str, variant: str) -> dict:
    return next(
        row
        for row in _rows("normal_tail_balance")
        if row["dataset"] == dataset and row["variant"] == variant
    )


def _formal_metric(dataset: str, method: str) -> dict[str, str]:
    source = ROOT / "outputs" / "thesis_innovation3b_tail_fusion_final" / "cross_dataset_metrics.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        return next(row for row in csv.DictReader(handle) if row["dataset"] == dataset and row["method"] == method)


def test_f0_f1_are_frozen_artifacts() -> None:
    assert MANIFEST["evidence_class"] == "RETROSPECTIVE_ABLATION"
    assert MANIFEST["method_freeze"] == "FROZEN"
    for dataset in ("SWaT", "HAI21"):
        for branch in ("rec", "dual"):
            artifact = MANIFEST["source_artifacts"][f"{dataset}_{branch}_test_raw"]
            source = ROOT / artifact["path"]
            assert source.exists()
            assert hashlib.sha256(source.read_bytes()).hexdigest() == artifact["sha256"]


def test_f3_formula_is_exact_historical_robust_z() -> None:
    assert MANIFEST["f3_definition"] == {
        "rec": "max(0,(REC-median_REC)/scale_REC)",
        "dual": "max(0,(DUAL-median_DUAL)/scale_DUAL)",
        "fusion": "max(z_REC,z_DUAL)",
        "calibration": "historical full Normal Validation for strict metrics; Calibration-A/Audit-B for dominance audit",
    }


def test_f4_formula_is_exact_tail_evidence() -> None:
    assert MANIFEST["f4_definition"] == {
        "p_tail": "(1 + count{s_i >= x}) / (N + 1)",
        "evidence": "-log10(p_tail)",
        "fusion": "max(E_REC,E_DUAL)",
        "ecdf": "Calibration-A only",
        "threshold": "Threshold-B q99 only",
    }


def test_only_preregistered_variants_are_present() -> None:
    assert MANIFEST["variants"] == ["F0 REC", "F1 DUAL", "F3 Robust-Z", "F4 Tail-Calibrated"]
    assert all("F2" not in row["variant"] for row in _rows("fusion_detection_ablation"))


def test_no_test_driven_calibration_or_tuning() -> None:
    restrictions = MANIFEST["restrictions"]
    assert restrictions["test_tuning"] is False
    assert restrictions["hp_search"] is False
    assert MANIFEST["f4_definition"]["ecdf"] == "Calibration-A only"
    assert MANIFEST["f4_definition"]["threshold"] == "Threshold-B q99 only"


def test_robust_z_top_one_percent_is_dual_dominated() -> None:
    for dataset in ("SWaT", "HAI21", "WADI.A2"):
        row = _balance(dataset, "F3 Robust-Z")
        assert row["top1_rec_share"] == 0.0
        assert row["top1_dual_share"] == 1.0


def test_tail_q99_balance_is_reproduced() -> None:
    expected = {"SWaT": 1.04488845531713, "HAI21": 1.0002800414976116, "WADI.A2": 1.0352968814016037}
    for dataset, value in expected.items():
        assert np.isclose(_balance(dataset, "F4 Tail-Calibrated")["balance_ratio"], value, atol=1e-12)


def test_tail_top_one_percent_branch_shares_are_reproduced() -> None:
    expected_rec = {"SWaT": 0.6519114688128773, "HAI21": 0.5915492957746479, "WADI.A2": 0.5178117048346056}
    for dataset, rec_share in expected_rec.items():
        row = _balance(dataset, "F4 Tail-Calibrated")
        assert np.isclose(row["top1_rec_share"], rec_share, atol=1e-12)
        assert np.isclose(row["top1_rec_share"] + row["top1_dual_share"], 1.0, atol=1e-12)


def test_hai_tail_metrics_are_frozen() -> None:
    row = _metric("HAI21", "F4 Tail-Calibrated")
    frozen = _formal_metric("HAI21", "TAIL_FUSION")
    assert np.isclose(row["aupr"], 0.6178966926696478, atol=1e-12)
    assert np.isclose(row["f1"], 0.6173502361421824, atol=1e-12)
    for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr"):
        assert abs(row[key] - float(frozen[key])) <= 1e-6


def test_swat_tail_metrics_are_frozen() -> None:
    row = _metric("SWaT", "F4 Tail-Calibrated")
    frozen = _formal_metric("SWaT", "TAIL_FUSION")
    assert np.isclose(row["aupr"], 0.2067470678171854, atol=1e-12)
    assert np.isclose(row["fpr"], 0.6925313534359722, atol=1e-12)
    for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr"):
        assert abs(row[key] - float(frozen[key])) <= 1e-6


def test_wadi_remains_domain_shift_stress_only() -> None:
    rows = [row for row in _rows("fusion_detection_ablation") if row["dataset"] == "WADI.A2"]
    assert len(rows) == 4
    assert {row["role"] for row in rows} == {"DOMAIN_SHIFT_STRESS_TEST"}
    assert np.isclose(_metric("WADI.A2", "F4 Tail-Calibrated")["aupr"], 0.0545101453356326, atol=1e-12)
    for variant, method in (("F0 REC", "REC"), ("F1 DUAL", "DUAL"), ("F3 Robust-Z", "OLD_FUSION"), ("F4 Tail-Calibrated", "TAIL_FUSION")):
        row = _metric("WADI.A2", variant)
        frozen = _formal_metric("WADI.A2", method)
        for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr"):
            assert abs(row[key] - float(frozen[key])) <= 1e-6


def test_persisted_scores_hashes_and_csvs_are_complete() -> None:
    for dataset, item in MANIFEST["persisted_fusion_archives"].items():
        archive = ROOT / item["path"]
        assert archive.exists(), dataset
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item["sha256"]
        with np.load(archive, allow_pickle=False) as data:
            assert {"labels", "timeline", "rec_score", "dual_score", "f3_robust_z_score", "f4_tail_score"} <= set(data.files)
            assert len(data["labels"]) == item["points"]
    expected_csvs = {
        "normal_tail_balance.csv": 8,
        "fusion_detection_ablation.csv": 12,
        "fusion_event_summary.csv": 12,
        "fusion_ablation_plot_data.csv": 24,
    }
    for filename, row_count in expected_csvs.items():
        with (OUT / filename).open(encoding="utf-8-sig", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == row_count
