from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "outputs" / "thesis_evaluation" / "literature_compatible"
RECOVERY = E2 / "recovery"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_twelve_strict_reproductions_pass() -> None:
    summary = load_json(RECOVERY / "recovery_summary.json")
    assert summary["status"] == "E2_RAW_SCORES_COMPLETED"
    assert summary["passed"] == summary["total"] == 12
    assert max(row["max_diff"] for row in summary["results"]) <= 1e-6


def test_common_masks_remain_frozen() -> None:
    summary = load_json(RECOVERY / "recovery_summary.json")
    assert summary["common_masks_unchanged"] == {"SWaT": 449791, "HAI21": 401685}


def test_availability_and_archives_are_complete() -> None:
    with (E2 / "score_availability.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert all(row["available"] == "true" for row in rows)
    assert all(row["status"] == "RAW_SCORE_REPRODUCED" for row in rows)
    for row in rows:
        path = ROOT / row["score_path"]
        assert path.is_file()
        with np.load(path, allow_pickle=False) as payload:
            assert len(payload["score"]) == int(row["n_scores"])
            assert len(payload["label"]) == int(row["n_labels"])
            assert np.isfinite(payload["score"]).all()


def test_recovery_manifest_has_no_training_or_refit() -> None:
    with (RECOVERY / "artifact_recovery_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert {row["artifact_origin"] for row in rows} == {"DETERMINISTIC_RECONSTRUCTION"}
    assert all(row["model_training"] == "false" for row in rows)
    assert all(row["ridge_refit"] == "false" for row in rows)
    assert all(row["test_tuning"] == "false" for row in rows)


def test_mismatch_root_cause_is_scaler_not_score_variant_search() -> None:
    for name in ("swat_lstm_reproduction_diagnosis.json", "swat_gdn_reproduction_diagnosis.json"):
        value = load_json(RECOVERY / name)
        assert value["earliest_inconsistent_pipeline_stage"] == "B_STANDARD_SCALER_STATISTICS"
        assert value["recovered_status"] == "RAW_SCORE_REPRODUCED"
        assert value["parameter_variants_tried"] == 0
        assert value["test_based_selection"] is False


def test_reconstructed_artifact_provenance_is_explicit() -> None:
    metadata = list((RECOVERY / "reconstructed_artifacts").rglob("*.json"))
    assert metadata
    for path in metadata:
        value = load_json(path)
        assert value["artifact_origin"] == "DETERMINISTIC_RECONSTRUCTION"
        assert value["test_labels_used"] is False
        assert value["new_fit_or_retuning"] is False
        assert (ROOT / value["artifact_path"]).is_file()

