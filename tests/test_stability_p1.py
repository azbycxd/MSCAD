from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_stability"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_exactly_three_distinct_seeds_per_dataset():
    manifest = load_json(OUT / "seed_manifest.json")
    assert manifest["created_before_training"] is True
    for dataset in ("SWaT", "HAI21"):
        seeds = manifest["datasets"][dataset]["seeds"]
        assert len(seeds) == len(set(seeds)) == 3


def test_formal_seed_artifacts_not_overwritten():
    manifest = load_json(OUT / "manifest.json")
    assert manifest["formal_hashes_before"] == manifest["formal_hashes_after"]


def test_formal_seed_metrics_reproduce():
    manifest = load_json(OUT / "manifest.json")
    assert len(manifest["formal_metric_deltas"]) == 4
    assert all(max(values.values()) <= 1e-6 for values in manifest["formal_metric_deltas"].values())


def test_only_r0_architecture_and_no_ablation_variants():
    seed_manifest = load_json(OUT / "seed_manifest.json")
    for dataset, folder, expected in (("SWaT", "swat", 11730), ("HAI21", "hai", 15482)):
        for seed in seed_manifest["datasets"][dataset]["seeds"]:
            protocol = load_json(OUT / folder / "rec" / f"seed_{seed}" / "protocol.json")
            assert protocol["architecture"] == "R0 Final"
            assert protocol["architecture_signature"]["parameter_count"] == expected
            assert protocol["R1"] is False and protocol["R2"] is False


def test_checkpoint_selected_by_normal_validation_loss():
    per_seed = pd.read_csv(OUT / "stability_per_seed.csv")
    rec = per_seed[per_seed.method == "REC"]
    assert rec.best_epoch.notna().all()
    assert rec.best_validation_loss.notna().all()
    assert (rec.best_epoch.astype(int) >= 1).all()


def test_q99_is_normal_validation_only():
    seed_manifest = load_json(OUT / "seed_manifest.json")
    for dataset, folder in (("SWaT", "swat"), ("HAI21", "hai")):
        for seed in seed_manifest["datasets"][dataset]["seeds"]:
            rec = load_json(OUT / folder / "rec" / f"seed_{seed}" / "threshold.json")
            tail = load_json(OUT / folder / "tail" / f"seed_{seed}" / "threshold.json")
            assert rec["quantile"] == tail["quantile"] == 0.99
            assert rec["test_labels_used"] is False and tail["test_labels_used"] is False
            assert "Normal Validation" in rec["source"]
            assert "Normal Validation" in tail["source"]


def test_dual_artifacts_are_unchanged_and_not_refit():
    manifest = load_json(OUT / "manifest.json")
    assert manifest["dual_hashes_before"] == manifest["dual_hashes_after"]
    assert manifest["dual_deterministic_under_frozen_protocol"] is True
    assert manifest["restrictions"]["ridge_refit"] is False
    assert manifest["restrictions"]["target_reselection"] is False


def test_tail_formula_and_per_seed_rec_binding():
    seed_manifest = load_json(OUT / "seed_manifest.json")
    for dataset, folder in (("SWaT", "swat"), ("HAI21", "hai")):
        for seed in seed_manifest["datasets"][dataset]["seeds"]:
            rec_validation = np.load(OUT / folder / "rec" / f"seed_{seed}" / "validation_raw_scores.npz", allow_pickle=False)["endpoint_mae"]
            calibration = np.load(OUT / folder / "tail" / f"seed_{seed}" / "calibration.npz", allow_pickle=False)
            segment_sizes = json.loads(calibration["segment_sizes_json"].item())
            expected = sum(int(row["calibration_a"]) for row in segment_sizes)
            assert len(calibration["rec_ecdf_sorted"]) == expected
            assert expected < len(rec_validation)
            tail_manifest = load_json(OUT / folder / "tail" / f"seed_{seed}" / "manifest.json")
            assert tail_manifest["tail_formula"].startswith("max(-log10")
            assert tail_manifest["restrictions"]["ridge_refit"] is False


def test_no_pa_oracle_test_best():
    manifest = load_json(OUT / "manifest.json")
    assert manifest["restrictions"]["test_tuning"] is False
    assert manifest["restrictions"]["PA"] is False
    assert manifest["restrictions"]["oracle"] is False
    assert manifest["restrictions"]["best_seed_selection"] is False


def test_all_raw_scores_are_persisted_and_finite():
    seed_manifest = load_json(OUT / "seed_manifest.json")
    for dataset, folder in (("SWaT", "swat"), ("HAI21", "hai")):
        for seed in seed_manifest["datasets"][dataset]["seeds"]:
            paths = (
                OUT / folder / "rec" / f"seed_{seed}" / "validation_raw_scores.npz",
                OUT / folder / "rec" / f"seed_{seed}" / "test_raw_scores.npz",
                OUT / folder / "tail" / f"seed_{seed}" / "validation_tail_scores.npz",
                OUT / folder / "tail" / f"seed_{seed}" / "test_tail_scores.npz",
            )
            for path in paths:
                assert path.exists()
                payload = np.load(path, allow_pickle=False)
                numeric = [payload[key] for key in payload.files if np.issubdtype(payload[key].dtype, np.number)]
                assert numeric and all(np.isfinite(values).all() for values in numeric)


def test_summary_has_four_rows_and_sample_std():
    summary = pd.read_csv(OUT / "stability_summary.csv")
    assert set(zip(summary.dataset, summary.method)) == {("SWaT", "REC"), ("SWaT", "TAIL"), ("HAI21", "REC"), ("HAI21", "TAIL")}
    assert (summary.seed_count == 3).all()
    for metric in ("auroc", "aupr", "f1", "fpr"):
        assert np.isfinite(summary[f"{metric}_std"]).all()


def test_manifest_integrity_gates_pass():
    manifest = load_json(OUT / "manifest.json")
    assert manifest["status"] == "THESIS_MULTI_SEED_STABILITY_COMPLETED"
    assert all(manifest["tests"].values())
