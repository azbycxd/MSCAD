from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP4B = ROOT / "outputs" / "hai_step4b_structure_controlled_validation"
STEP4BR = ROOT / "outputs" / "hai_step4b_preserving_control_audit"
STEP4BA = ROOT / "outputs" / "hai_step4b_relation_aggregation_repair"
OUTPUT = ROOT / "outputs" / "hai_step4b_fair_temporal_generalization_audit"
PROTOCOLS = (
    "original_max_in_sample", "original_max_crossfit",
    "tail_top2_in_sample", "tail_top2_crossfit",
)
REQUIRED_SCORE_FIELDS = (
    "source_window_id", "sequence_id", "fold",
    "original_distance_max_clean", "original_distance_max_R1", "original_distance_max_R2", "original_distance_max_R3",
    "relation_tail_top2_mean_clean", "relation_tail_top2_mean_R1", "relation_tail_top2_mean_R2", "relation_tail_top2_mean_R3",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def inspect_score_caches() -> dict[str, object]:
    candidates = sorted(STEP4BA.glob("*.npz"))
    fields: set[str] = set()
    for path in candidates:
        with np.load(path, allow_pickle=False) as cache:
            fields.update(cache.files)
    return {"candidate_files": [str(path) for path in candidates], "available_fields": sorted(fields),
            "missing_fields": [field for field in REQUIRED_SCORE_FIELDS if field not in fields],
            "complete": all(field in fields for field in REQUIRED_SCORE_FIELDS)}


def inspect_manifest() -> tuple[pd.DataFrame, dict[str, object]]:
    path = STEP4BA / "crossfit_manifest.csv"
    frame = pd.read_csv(path)
    required_summary = {"fold", "sequence_id", "calibration_half", "calibration_count", "evaluation_count",
                        "source_window_overlap_count", "chronological_split", "random_shuffle"}
    per_window = {"source_window_id", "fold", "role"}
    audit = {"source_path": str(path), "row_count": len(frame), "summary_fields_complete": required_summary.issubset(frame.columns),
             "source_window_overlap_count": int(frame.source_window_overlap_count.sum()) if "source_window_overlap_count" in frame else "missing",
             "per_window_assignment_fields": sorted(per_window), "per_window_assignment_available": per_window.issubset(frame.columns),
             "reused_without_modification": True}
    return frame, audit


def status_table(name: str, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(OUTPUT / name, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for root in (STEP4B, STEP4BR, STEP4BA):
        if not root.exists():
            raise FileNotFoundError(root)
    score_audit = inspect_score_caches()
    _, manifest_audit = inspect_manifest()
    blockers = []
    if not score_audit["complete"]:
        blockers.append("missing per-window clean/synthetic aggregate score cache for original_distance_max and relation_tail_top2_mean")
    if not manifest_audit["per_window_assignment_available"]:
        blockers.append("crossfit_manifest.csv lacks source_window_id/fold/role per-window assignment")
    if args.dry_run:
        print(f"REUSE: {STEP4BA / 'crossfit_manifest.csv'}", flush=True)
        print(f"SCORE_CACHE_COMPLETE: {score_audit['complete']}", flush=True)
        print(f"PER_WINDOW_FOLD_MAP_COMPLETE: {manifest_audit['per_window_assignment_available']}", flush=True)
        print(f"BLOCKERS: {blockers}", flush=True)
        print("FORBIDDEN_FALLBACK: deriving aggregate scores from relation features or rebuilding folds", flush=True)
        return 2 if blockers else 0
    if (OUTPUT / "fair_temporal_generalization_summary.json").exists():
        print(f"REUSE completed blocked audit: {OUTPUT}", flush=True)
        return 0
    before = {"step4b": tree_hashes(STEP4B), "step4b_r": tree_hashes(STEP4BR), "step4b_a": tree_hashes(STEP4BA)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    original_manifest = pd.read_csv(STEP4BA / "crossfit_manifest.csv")
    for row in original_manifest.to_dict("records"):
        manifest_rows.append(row | {"source_manifest_reused": True,
                                    "per_window_assignment_available": manifest_audit["per_window_assignment_available"],
                                    "fair_evaluation_allowed": False})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "reused_crossfit_manifest_audit.csv", index=False)
    matrix_rows = []
    for protocol in PROTOCOLS:
        matrix_rows.append({"protocol_id": protocol,
                            "score": "original_distance_max" if protocol.startswith("original") else "relation_tail_top2_mean",
                            "evaluation_protocol": "in_sample_full_validation" if protocol.endswith("in_sample") else "chronological_two_fold_crossfit",
                            "result_scope": "diagnostic_only" if protocol.endswith("in_sample") else "formal_generalization",
                            "status": "not_evaluated_missing_required_score_cache",
                            "reason": "; ".join(blockers)})
    pd.DataFrame(matrix_rows).to_csv(OUTPUT / "fair_protocol_matrix.csv", index=False)
    status_table("fair_metrics_by_candidate.csv", [{"protocol_id": row["protocol_id"], "status": row["status"], "reason": row["reason"]} for row in matrix_rows])
    status_table("fair_metrics_by_sequence.csv", [{"protocol_id": protocol, "sequence_id": f"train{sequence}_validation",
                 "status": "not_evaluated_missing_required_score_cache"} for protocol in PROTOCOLS for sequence in range(1, 7)])
    status_table("temporal_half_distribution_shift.csv", [{"score": score, "sequence_id": f"train{sequence}_validation",
                 "status": "not_evaluated_missing_per_window_score"} for score in ("original_distance_max", "relation_tail_top2_mean") for sequence in range(1, 7)])
    status_table("threshold_transfer_audit.csv", [{"protocol_id": protocol, "fold": fold,
                 "status": "not_evaluated_missing_score_and_source_fold_map"} for protocol in PROTOCOLS for fold in ("A", "B")])
    after = {"step4b": tree_hashes(STEP4B), "step4b_r": tree_hashes(STEP4BR), "step4b_a": tree_hashes(STEP4BA)}
    audit_rows = []
    roots = {"step4b": STEP4B, "step4b_r": STEP4BR, "step4b_a": STEP4BA}
    for source, hashes in before.items():
        audit_rows.extend({"source": source, "artifact_id": name, "source_path": str(roots[source] / name),
                           "sha256_before": digest, "sha256_after": after[source].get(name),
                           "hash_match": after[source].get(name) == digest, "reused": True, "recomputed": False}
                          for name, digest in hashes.items())
    pd.DataFrame(audit_rows).to_csv(OUTPUT / "artifact_reuse_audit.csv", index=False)
    summary = {"status": "stopped", "classification": "invalid_fair_comparison",
               "reason": "required_existing_cache_incomplete", "blockers": blockers,
               "score_cache_audit": score_audit, "crossfit_manifest_audit": manifest_audit,
               "protocol_rows_evaluated": 0, "historical_hashes_unchanged": before == after,
               "raw_csv_loaded": False, "hai_test_loaded": False, "ae_checkpoint_rec_loaded": False,
               "relation_features_recomputed": False, "reference_refit": False, "corruptions_regenerated": False,
               "folds_rebuilt": False}
    (OUTPUT / "fair_temporal_generalization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    missing = "\n".join(f"- `{field}`" for field in score_audit["missing_fields"])
    report = f"""# HAI Step 4B-B fair temporal generalization audit

## Status

**Stopped: invalid_fair_comparison**

The required per-window aggregate-score cache does not exist in Step 4B-A, and the reused crossfit manifest does not contain a per-window source-to-fold map. The requested 2x2 matrix cannot be computed fairly without violating the explicit no-recomputation/no-repartition rules.

## Missing score fields

{missing}

## Manifest limitation

The existing manifest has 12 sequence/fold summary rows and zero reported calibration/evaluation overlap, but lacks `source_window_id`, per-window `fold`, and `role`. Reconstructing those assignments would be a new split and was not attempted.

## Integrity

- Step 4B, Step 4B-R, and Step 4B-A hashes unchanged: **{before == after}**.
- Raw CSV/test/model/checkpoint/rec data loaded: **False**.
- Aggregate scores derived from relation features: **False**.
- Historical folds rebuilt: **False**.

No metrics, thresholds, or conclusions about temporal generalization were fabricated. The historical `not_frozen` state remains unchanged.
"""
    (OUTPUT / "final_hai_step4b_fair_temporal_generalization_audit.md").write_text(report, encoding="utf-8")
    if before != {"step4b": tree_hashes(STEP4B), "step4b_r": tree_hashes(STEP4BR), "step4b_a": tree_hashes(STEP4BA)}:
        raise RuntimeError("Historical artifact changed during blocked audit")
    print(json.dumps({"classification": "invalid_fair_comparison", "status": "stopped",
                      "blockers": blockers, "historical_hashes_unchanged": True}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
