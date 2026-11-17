from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_evaluation" / "literature_compatible"


def test_common_masks_are_exact_frozen_counts_and_unique():
    expected = {"swat": 449791, "hai": 401685}
    for dataset, count in expected.items():
        with np.load(OUT / "common_masks" / f"{dataset}_common_mask.npz", allow_pickle=False) as payload:
            assert len(payload["local_index"]) == count
            assert payload["valid_endpoint"].all()
            keys = np.char.add(np.asarray(payload["segment_id"], dtype=str), np.char.add(":", np.asarray(payload["local_index"], dtype=str)))
            assert len(np.unique(keys)) == count
            assert int(payload["local_index"].min()) == 64


def test_swat_ours_rec_archive_is_bound_and_other_statuses_are_explicit():
    archive = OUT / "raw_scores" / "swat" / "ours_rec" / "test_raw_scores.npz"
    with np.load(archive, allow_pickle=False) as payload:
        required = {"score", "label", "segment_id", "file_id", "local_index", "global_index", "valid_endpoint", "score_orientation", "timestamp"}
        assert required.issubset(payload.files)
        assert str(payload["score_orientation"]) == "higher=more_anomalous"
        assert len(payload["score"]) == 449793
    strict = json.loads((archive.parent / "strict_reproduction.json").read_text(encoding="utf-8"))
    assert strict["status"] == "PASS"
    assert strict["max_absolute_difference"] <= 1e-6

    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))["e2_r1"]
    assert manifest["status"] == "E2_RAW_SCORE_REPRODUCTION_REQUIRES_REVIEW"
    assert manifest["raw_scores_persisted"] == 3
    assert manifest["raw_scores_strict_validated"] == 1
    assert set(manifest["mismatches"]) == {"SWaT/LSTM-AE", "SWaT/GDN"}
    assert manifest["no_training"] and manifest["no_scaler_fit"] and manifest["no_ridge_fit"]


def test_all_twelve_method_directories_have_provenance_and_validation_records():
    for dataset in ("swat", "hai"):
        for method in ("lstm_ae", "usad", "gdn", "ours_rec", "ours_dual", "ours_tail"):
            directory = OUT / "raw_scores" / dataset / method
            assert (directory / "provenance.json").is_file()
            assert (directory / "strict_reproduction.json").is_file()
            assert (directory / "common_mask_validation.json").is_file()
