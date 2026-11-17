from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/thesis/00_audit/evaluate_e2_r2_complete.py"
SPEC = importlib.util.spec_from_file_location("e2_r2_complete", SCRIPT)
R2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R2)
FINAL = ROOT / "outputs/thesis_evaluation/literature_compatible/final"


def test_twelve_raw_score_artifacts_exist() -> None:
    for dataset in R2.DATASETS:
        for method in R2.METHODS:
            assert (R2.RAW / R2.DS_SLUG[dataset] / R2.SLUG[method] / "test_raw_scores.npz").is_file()


def test_r1d_preconditions_and_strict_tolerance_pass() -> None:
    gate = R2.preconditions()
    assert gate["status"] == "PASS"
    assert gate["max_strict_difference"] <= 1e-6


def test_common_mask_sizes_are_frozen() -> None:
    assert len(R2.load_common("SWaT")["local_index"]) == 449791
    assert len(R2.load_common("HAI21")["local_index"]) == 401685


def test_exact_oracle_tie_rule() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    value = R2.E2.exact_oracle(labels, scores)
    assert value["threshold"] == 0.1
    assert value["f1"] == 0.8


def test_efficient_pa_oracle_matches_naive_pa() -> None:
    labels = np.array([0, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    scores = np.array([0.2, 0.1, 0.9, 0.8, 0.3, 0.4, 0.7])
    intervals = R2.E2.anomaly_intervals(labels, [4, 3])
    exact = R2.E2.exact_pa_oracle(labels, scores, intervals)
    candidates = np.r_[np.nextafter(scores.min(), -np.inf), np.unique(scores)]
    naive = [R2.E2.pa_at_threshold(labels, scores, float(x), intervals) | {"threshold": float(x)} for x in candidates]
    best = max(x["f1"] for x in naive)
    expected_threshold = max(x["threshold"] for x in naive if x["f1"] == best)
    assert exact["f1"] == best
    assert exact["threshold"] == expected_threshold


def test_pa_never_removes_normal_false_positive() -> None:
    labels = np.array([0, 1, 1, 0], dtype=np.int8)
    prediction = np.array([1, 0, 1, 1], dtype=np.int8)
    adjusted = R2.E2.point_adjustment(prediction, labels, R2.E2.anomaly_intervals(labels, [4]))
    assert adjusted[0] == adjusted[3] == 1


def test_pa_boundaries_split_swat_gap_and_hai_files() -> None:
    labels = np.array([0, 1, 1, 1, 1, 0], dtype=np.int8)
    expected = [(1, 3, 0), (3, 5, 1)]
    assert R2.E2.anomaly_intervals(labels, [3, 3]) == expected
    assert R2.E2.anomaly_intervals(labels, [3, 3]) == expected  # same rule for HAI official files


def test_historical_ours_rec_regression_passes() -> None:
    strict = {(x["dataset"], x["method"]): x for x in R2.E2.strict_rows()}
    result = R2.historical_ours_rec_regression(strict)
    assert result["status"] == "PASS"
    assert result["max_absolute_difference"] <= 1e-6


def test_final_tables_are_complete() -> None:
    expected = {
        "table_a_historical_strict.csv": 12,
        "table_b_common_mask_strict.csv": 12,
        "table_c_literature_compatible.csv": 12,
        "swat_literature_compatible.csv": 6,
        "hai_literature_compatible.csv": 6,
        "threshold_transfer_analysis.csv": 12,
        "literature_comparison_schema.csv": 28,
    }
    for name, count in expected.items():
        with (FINAL / name).open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == count


def test_evaluation_source_cannot_train_infer_or_open_checkpoint() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch.load" not in source
    assert ".pt\"" not in source and ".pt'" not in source
    manifest = json.loads((FINAL / "manifest.json").read_text(encoding="utf-8"))
    restrictions = manifest["restrictions"]
    assert restrictions["training"] is False
    assert restrictions["inference"] is False
    assert restrictions["checkpoint_read"] is False
    assert restrictions["scaler_recalculation"] is False
    assert restrictions["ridge_recalculation"] is False
    assert restrictions["tail_recalibration"] is False

