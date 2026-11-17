"""E2-R1: persist recoverable frozen raw scores and exact common endpoint masks.

This is evaluation-only.  It never fits a scaler/model, creates an optimizer,
performs backward propagation, selects a checkpoint, or recalibrates a tail.
Unavailable frozen artifacts are recorded explicitly instead of reconstructed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "thesis_evaluation" / "literature_compatible"
RAW = OUT / "raw_scores"
MASKS = OUT / "common_masks"
METHODS = ("lstm_ae", "usad", "gdn", "ours_rec", "ours_dual", "ours_tail")
DISPLAY = {
    "lstm_ae": "LSTM-AE",
    "usad": "USAD",
    "gdn": "GDN",
    "ours_rec": "Ours-REC",
    "ours_dual": "Ours-DUAL",
    "ours_tail": "Ours-TAIL",
}
DATASET_DISPLAY = {"swat": "SWaT", "hai": "HAI21"}
WINDOW = 64
TOLERANCE = 1e-6


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("e2r1_lstm_base", ROOT / "scripts" / "thesis" / "01_baseline" / "run_lstm_ae_unified.py")
GDN = load_module("e2r1_gdn_base", ROOT / "scripts" / "thesis" / "01_baseline" / "run_gdn_unified.py")
E2 = load_module("e2r1_e2", ROOT / "scripts" / "thesis" / "00_audit" / "evaluate_literature_compatible.py")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True, text=True, capture_output=True).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "status_short": run("status", "--short")}


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pred = scores > threshold
    tp = int(((labels == 1) & pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    tn = int(((labels == 0) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": float(threshold),
        "n": int(len(labels)),
    }


def strict_reference() -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["dataset"], row["method"]): row for row in E2.strict_rows()}


def strict_check(dataset: str, method: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    frozen = strict_reference()[(dataset, method)]
    actual = metrics(labels, scores, float(frozen["threshold_q99"]))
    keys = {"auroc": "auroc", "aupr": "aupr", "precision": "precision_q99", "recall": "recall_q99", "f1": "f1_q99", "fpr": "fpr_q99"}
    delta = {name: abs(float(actual[name]) - float(frozen[source])) for name, source in keys.items()}
    ok = len(labels) == int(frozen["n_points"]) and max(delta.values()) <= TOLERANCE
    return {
        "status": "PASS" if ok else "RAW_SCORE_REPRODUCTION_MISMATCH",
        "tolerance": TOLERANCE,
        "actual": actual,
        "frozen": frozen,
        "absolute_difference": delta,
        "max_absolute_difference": max(delta.values()),
    }


def save_archive(directory: Path, *, score: np.ndarray, label: np.ndarray, segment_id: np.ndarray,
                 local_index: np.ndarray, global_index: np.ndarray, timestamp: np.ndarray | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "score": np.asarray(score, dtype=np.float64),
        "label": np.asarray(label, dtype=np.int8),
        "segment_id": np.asarray(segment_id),
        "file_id": np.asarray(segment_id),
        "local_index": np.asarray(local_index, dtype=np.int64),
        "global_index": np.asarray(global_index, dtype=np.int64),
        "valid_endpoint": np.ones(len(score), dtype=np.bool_),
        "score_orientation": np.asarray("higher=more_anomalous"),
    }
    if timestamp is not None:
        arrays["timestamp"] = np.asarray(timestamp, dtype=np.int64)
    path = directory / "test_raw_scores.npz"
    np.savez_compressed(path, **arrays)
    return path


def swat_source_and_identities() -> dict[str, Any]:
    cfg = BASE.load_config(str(BASE.SWAT_CONFIG))
    BASE.validate_config_contract(cfg)
    cache = BASE.resolve_repo_path(ROOT, cfg["paths"]["cache_dir"])
    normal, cache_manifest = BASE.load_normal_cache(cache, cfg)
    attack, labels, timestamps, _ = BASE.load_attack_cache(cache, cfg)
    with np.load(ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "scaler.npz", allow_pickle=False) as frozen:
        mean = np.asarray(frozen["mean"], dtype=np.float64).reshape(1, -1)
        scale = np.asarray(frozen["effective_scale"], dtype=np.float64).reshape(1, -1)
    if mean.shape != (1, 51) or scale.shape != (1, 51) or np.any(scale <= 0):
        raise RuntimeError("invalid frozen SWaT scaler")
    validation = ((np.asarray(normal[396000:495000], dtype=np.float64) - mean) / scale).astype(np.float32)
    scaled_attack = ((np.asarray(attack, dtype=np.float64) - mean) / scale).astype(np.float32)
    segment_manifest = json.loads((ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "segment_manifest.json").read_text(encoding="utf-8"))
    lengths = [int(row["row_count"]) for row in segment_manifest["test_segments"]]
    if lengths != [298837, 151082] or sum(lengths) != len(attack):
        raise RuntimeError("SWaT segment identity changed")

    identities: dict[int, dict[str, np.ndarray]] = {}
    for first in (63, 64):
        seg, local, glob, ts, lab = [], [], [], [], []
        offset = 0
        for sid, length in enumerate(lengths):
            loc = np.arange(first, length, dtype=np.int64)
            idx = offset + loc
            seg.append(np.full(len(loc), sid, dtype=np.int16)); local.append(loc); glob.append(idx)
            ts.append(np.asarray(timestamps[idx], dtype=np.int64)); lab.append(np.asarray(labels[idx], dtype=np.int8))
            offset += length
        identities[first] = {"segment_id": np.concatenate(seg), "local_index": np.concatenate(local), "global_index": np.concatenate(glob), "timestamp": np.concatenate(ts), "label": np.concatenate(lab)}
    pieces, offset = [], 0
    for length in lengths:
        pieces.append(scaled_attack[offset:offset + length]); offset += length
    return {
        "validation": [validation], "test": pieces, "identities": identities, "feature_names": list(cache_manifest["column_order"]),
        "scaler_path": ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "scaler.npz",
    }


def common_masks(swat: dict[str, Any]) -> dict[str, dict[str, Any]]:
    MASKS.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"policy": "E2_COMMON_MASK", "join_policy": "exact identity intersection; never array-length trimming", "datasets": {}}

    swat_id = swat["identities"][64]
    swat_path = MASKS / "swat_common_mask.npz"
    np.savez_compressed(swat_path, **swat_id, valid_endpoint=np.ones(len(swat_id["local_index"]), dtype=np.bool_))
    per_swat = {DISPLAY[m]: len(swat["identities"][64 if m in {"lstm_ae", "usad", "gdn"} else 63]["local_index"]) for m in METHODS}
    manifest["datasets"]["SWaT"] = {
        "dataset": "Official SWaT 2015 Physical", "n_total_candidate": int(len(swat["identities"][63]["local_index"])),
        "n_common": int(len(swat_id["local_index"])), "per_method_n": per_swat,
        "excluded_from_method_to_common": {key: int(value - len(swat_id["local_index"])) for key, value in per_swat.items()},
        "segment_counts": {str(i): int((swat_id["segment_id"] == i).sum()) for i in np.unique(swat_id["segment_id"])},
        "join_keys": ["dataset", "segment_id", "local_index", "global_index", "timestamp"],
        "path": str(swat_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(swat_path),
    }

    file_manifest = pd.read_csv(ROOT / "outputs" / "thesis_hai21_a0_audit" / "file_manifest.csv")
    test_rows = file_manifest[file_manifest["split"] == "test"].copy()
    files, local, glob, timestamps = [], [], [], []
    offset = 0
    file_counts: dict[str, int] = {}
    for row in test_rows.itertuples(index=False):
        count = int(row.rows); name = str(row.file); loc = np.arange(64, count, dtype=np.int64)
        files.append(np.repeat(name, len(loc))); local.append(loc); glob.append(offset + loc)
        start = int(np.datetime64(str(row.start_timestamp), "s").astype(np.int64)); timestamps.append(start + loc)
        file_counts[name] = int(len(loc)); offset += count
    hai_id = {"file_id": np.concatenate(files), "segment_id": np.concatenate(files), "local_index": np.concatenate(local), "global_index": np.concatenate(glob), "timestamp": np.concatenate(timestamps)}
    hai_path = MASKS / "hai_common_mask.npz"
    np.savez_compressed(hai_path, **hai_id, valid_endpoint=np.ones(len(hai_id["local_index"]), dtype=np.bool_))
    baseline_n = int(len(hai_id["local_index"])); ours_n = baseline_n + len(test_rows)
    per_hai = {DISPLAY[m]: (baseline_n if m in {"lstm_ae", "usad", "gdn"} else ours_n) for m in METHODS}
    manifest["datasets"]["HAI21"] = {
        "dataset": "HAI 21.03", "n_total_candidate": ours_n, "n_common": baseline_n, "per_method_n": per_hai,
        "excluded_from_method_to_common": {key: int(value - baseline_n) for key, value in per_hai.items()},
        "file_counts": file_counts, "join_keys": ["dataset", "official_test_file", "local_index", "global_index", "timestamp"],
        "path": str(hai_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(hai_path),
    }
    path = MASKS / "common_mask_manifest.json"; dump_json(path, manifest)
    return manifest


def provenance(dataset: str, method: str, *, model_path: Path | None, normalization_path: Path | None,
               score_definition: str, source_manifest: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "dataset": dataset, "method": method, "inference_only": True, "training": False,
        "model_or_checkpoint_path": str(model_path.relative_to(ROOT)).replace("\\", "/") if model_path else None,
        "model_or_checkpoint_sha256": sha256(model_path) if model_path and model_path.is_file() else None,
        "scaler_or_calibration_path": str(normalization_path.relative_to(ROOT)).replace("\\", "/") if normalization_path else None,
        "scaler_or_calibration_sha256": sha256(normalization_path) if normalization_path and normalization_path.is_file() else None,
        "source_manifest": source_manifest, "score_definition": score_definition,
        "score_orientation": "higher=more_anomalous", "script_commit_worktree_state": git_state(),
        "test_labels_used_for_score": False, "optimizer": False, "backward": False,
        "scaler_fit": False, "ridge_fit": False, "tail_recalibration": False,
    }
    if extra:
        value.update(extra)
    return value


def write_unavailable(dataset_slug: str, method_slug: str, reason: str, evidence: list[str]) -> dict[str, Any]:
    directory = RAW / dataset_slug / method_slug; directory.mkdir(parents=True, exist_ok=True)
    state = {"status": "FROZEN_ARTIFACT_UNAVAILABLE", "reason": reason, "evidence_checked": evidence, "inference_performed": False}
    dump_json(directory / "provenance.json", {**state, "dataset": DATASET_DISPLAY[dataset_slug], "method": DISPLAY[method_slug], "script_commit_worktree_state": git_state()})
    dump_json(directory / "strict_reproduction.json", state)
    dump_json(directory / "common_mask_validation.json", {**state, "evaluation_kind": "COMMON_MASK_REEVALUATION"})
    return state


def persist_swat_ours_rec(swat: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "test_scores.npz"
    with np.load(source, allow_pickle=False) as payload:
        score = np.asarray(payload["endpoint_mae"], dtype=np.float64)
        label = np.asarray(payload["label"], dtype=np.int8)
        timestamp = np.asarray(payload["endpoint_timestamp"], dtype=np.int64)
    identity = swat["identities"][63]
    if not np.array_equal(label, identity["label"]) or not np.array_equal(timestamp, identity["timestamp"]):
        raise RuntimeError("existing SWaT Ours-REC archive identity mismatch")
    directory = RAW / "swat" / "ours_rec"
    path = save_archive(directory, score=score, label=label, timestamp=timestamp, **{k: identity[k] for k in ("segment_id", "local_index", "global_index")})
    strict = strict_check("SWaT", "Ours-REC", label, score); dump_json(directory / "strict_reproduction.json", strict)
    mask = identity["local_index"] >= 64
    common = metrics(label[mask], score[mask], strict_reference()[("SWaT", "Ours-REC")]["threshold_q99"])
    dump_json(directory / "common_mask_validation.json", {"status": "PASS", "evaluation_kind": "COMMON_MASK_REEVALUATION", "metrics": common})
    prov = provenance("SWaT", "Ours-REC", model_path=ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "best.pt",
                      normalization_path=swat["scaler_path"], score_definition="endpoint mean absolute reconstruction error",
                      source_manifest="outputs/thesis_midterm_s1/a0_rec_baseline/metrics.json",
                      extra={"existing_archive_reused_without_reinference": True, "source_archive": str(source.relative_to(ROOT)).replace("\\", "/"), "source_archive_sha256": sha256(source), "persisted_archive_sha256": sha256(path)})
    dump_json(directory / "provenance.json", prov)
    return {"status": strict["status"], "n": len(score), "path": path}


def persist_swat_lstm(swat: dict[str, Any], device: torch.device) -> dict[str, Any]:
    checkpoint = ROOT / "outputs" / "thesis_baselines" / "lstm_ae" / "swat_state_dict.pt"
    model = BASE.LSTMAutoencoder(51).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); model.eval()
    store = BASE.WindowStore.from_segments(swat["test"], common_endpoint_mask=True)
    score = BASE.score_model(model, store, device)
    identity = swat["identities"][64]; label = identity["label"]
    directory = RAW / "swat" / "lstm_ae"
    path = save_archive(directory, score=score, label=label, timestamp=identity["timestamp"], **{k: identity[k] for k in ("segment_id", "local_index", "global_index")})
    strict = strict_check("SWaT", "LSTM-AE", label, score); dump_json(directory / "strict_reproduction.json", strict)
    dump_json(directory / "common_mask_validation.json", {"status": strict["status"], "evaluation_kind": "COMMON_MASK_REEVALUATION", "metrics": strict["actual"]})
    dump_json(directory / "provenance.json", provenance("SWaT", "LSTM-AE", model_path=checkpoint, normalization_path=swat["scaler_path"],
              score_definition="endpoint mean channel squared reconstruction error", source_manifest="outputs/thesis_baselines/lstm_ae/manifest.json",
              extra={"persisted_archive_sha256": sha256(path), "shared_frozen_scaler_identity": "S1 A0 scaler; exact same Official SWaT Normal Train [0,396000) StandardScaler protocol"}))
    return {"status": strict["status"], "n": len(score), "path": path}


def persist_swat_gdn(swat: dict[str, Any], device: torch.device) -> dict[str, Any]:
    checkpoint = ROOT / "outputs" / "thesis_baselines" / "gdn" / "swat_state_dict.pt"
    model = GDN.FrozenGDN(51).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); model.eval()
    val_store = GDN.ForecastStore.from_segments(swat["validation"])
    val_error = GDN.raw_errors(model, val_store, device)
    median = np.median(val_error, axis=0)
    iqr = np.quantile(val_error, 0.75, axis=0) - np.quantile(val_error, 0.25, axis=0)
    test_store = GDN.ForecastStore.from_segments(swat["test"])
    score = GDN.native_scores(GDN.raw_errors(model, test_store, device), median, iqr, swat["test"])
    identity = swat["identities"][64]; label = identity["label"]
    directory = RAW / "swat" / "gdn"
    calibration = directory / "validation_calibration_reconstructed.npz"; directory.mkdir(parents=True, exist_ok=True)
    np.savez(calibration, median=median, iqr=iqr, epsilon=np.asarray([GDN.EPSILON], dtype=np.float64))
    path = save_archive(directory, score=score, label=label, timestamp=identity["timestamp"], **{k: identity[k] for k in ("segment_id", "local_index", "global_index")})
    strict = strict_check("SWaT", "GDN", label, score); dump_json(directory / "strict_reproduction.json", strict)
    dump_json(directory / "common_mask_validation.json", {"status": strict["status"], "evaluation_kind": "COMMON_MASK_REEVALUATION", "metrics": strict["actual"]})
    dump_json(directory / "provenance.json", provenance("SWaT", "GDN", model_path=checkpoint, normalization_path=swat["scaler_path"],
              score_definition="validation median/IQR standardized per-sensor absolute forecast error; four-point segment-local trailing mean; max sensor",
              source_manifest="outputs/thesis_baselines/gdn/manifest.json",
              extra={"persisted_archive_sha256": sha256(path), "validation_calibration_path": str(calibration.relative_to(ROOT)).replace("\\", "/"), "validation_calibration_sha256": sha256(calibration), "calibration_status": "DETERMINISTIC_EVAL_ARTIFACT_RECONSTRUCTION", "test_labels_used_for_calibration": False}))
    return {"status": strict["status"], "n": len(score), "path": path}


def main() -> int:
    started = time.perf_counter(); RAW.mkdir(parents=True, exist_ok=True)
    swat = swat_source_and_identities(); mask_manifest = common_masks(swat)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, dict[str, Any]] = {}
    for method_slug, runner in (("ours_rec", persist_swat_ours_rec), ("lstm_ae", persist_swat_lstm), ("gdn", persist_swat_gdn)):
        directory = RAW / "swat" / method_slug
        prior_path = directory / "strict_reproduction.json"
        archive_path = directory / "test_raw_scores.npz"
        if prior_path.is_file() and archive_path.is_file():
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            results[f"SWaT/{DISPLAY[method_slug]}"] = {"status": prior["status"], "n": int(prior["actual"]["n"]), "path": archive_path, "reused_completed_inference": True}
        elif method_slug == "ours_rec":
            results["SWaT/Ours-REC"] = runner(swat)
        else:
            results[f"SWaT/{DISPLAY[method_slug]}"] = runner(swat, device)

    unavailable = {
        ("swat", "usad"): ("frozen Train-only MinMaxScaler parameters were not persisted", ["outputs/thesis_baselines/usad/manifest.json", "outputs/thesis_baselines/usad/swat_state_dict.pt"]),
        ("swat", "ours_dual"): ("Ridge coefficients exist, but frozen per-target disagreement median/MAD references were not persisted", ["outputs/thesis_efficiency_final/swat_dual_coefficients.npz", "outputs/thesis_swat_final_unified/dual_targets_unified_frozen.csv"]),
        ("swat", "ours_tail"): ("tail ECDF exists, but the frozen DUAL raw branch is unavailable because per-target median/MAD references are missing", ["outputs/thesis_efficiency_final/swat_tail_calibration.npz", "outputs/thesis_efficiency_final/swat_dual_coefficients.npz"]),
        ("hai", "lstm_ae"): ("frozen HAI Train-only StandardScaler parameters were not persisted", ["outputs/thesis_baselines/lstm_ae/manifest.json", "outputs/thesis_baselines/lstm_ae/hai_state_dict.pt"]),
        ("hai", "usad"): ("frozen HAI Train-only MinMaxScaler parameters were not persisted", ["outputs/thesis_baselines/usad/manifest.json", "outputs/thesis_baselines/usad/hai_state_dict.pt"]),
        ("hai", "gdn"): ("frozen HAI Train-only StandardScaler parameters were not persisted; Validation calibration cannot be reconstructed without it", ["outputs/thesis_baselines/gdn/manifest.json", "outputs/thesis_baselines/gdn/hai_state_dict.pt"]),
        ("hai", "ours_rec"): ("frozen HAI Train-only StandardScaler parameters were not persisted", ["outputs/thesis_hai21_b1_real/rec_best.pt", "outputs/thesis_hai21_final_unified/manifest.json"]),
        ("hai", "ours_dual"): ("Ridge coefficients exist, but frozen HAI scaler and per-target disagreement median/MAD references were not persisted", ["outputs/thesis_efficiency_final/hai21_dual_coefficients.npz", "outputs/thesis_hai21_final_unified/dual_targets_unified_frozen.csv"]),
        ("hai", "ours_tail"): ("tail ECDF exists, but frozen REC/DUAL Test branches cannot be generated without their missing frozen scaler/reference artifacts", ["outputs/thesis_efficiency_final/hai21_tail_calibration.npz", "outputs/thesis_efficiency_final/hai21_dual_coefficients.npz"]),
    }
    for (dataset, method), (reason, evidence) in unavailable.items():
        state = write_unavailable(dataset, method, reason, evidence)
        results[f"{DATASET_DISPLAY[dataset]}/{DISPLAY[method]}"] = state

    mismatch = [key for key, value in results.items() if value["status"] == "RAW_SCORE_REPRODUCTION_MISMATCH"]
    complete = [key for key, value in results.items() if value["status"] == "PASS"]
    status = "E2_RAW_SCORE_REPRODUCTION_REQUIRES_REVIEW" if mismatch else ("E2_RAW_SCORES_COMPLETED" if len(complete) == 12 else "E2_RAW_SCORES_PARTIAL")
    availability = []
    refs = strict_reference()
    for dataset_slug in ("swat", "hai"):
        dataset = DATASET_DISPLAY[dataset_slug]
        for method_slug in METHODS:
            method = DISPLAY[method_slug]; key = f"{dataset}/{method}"; value = results[key]
            directory = RAW / dataset_slug / method_slug; score_path = directory / "test_raw_scores.npz"
            persisted = score_path.is_file()
            available = persisted and value["status"] == "PASS"
            availability.append({
                "dataset": dataset, "method": method, "score_path": str(score_path.relative_to(ROOT)).replace("\\", "/") if persisted else "",
                "label_path": str(score_path.relative_to(ROOT)).replace("\\", "/") + "#label" if persisted else "",
                "mask_path": str((MASKS / f"{dataset_slug}_common_mask.npz").relative_to(ROOT)).replace("\\", "/"),
                "q99_threshold": refs[(dataset, method)]["threshold_q99"], "n_scores": value.get("n") if persisted else None,
                "n_labels": value.get("n") if persisted else None, "available": available, "status": value["status"],
                "provenance": str((directory / "provenance.json").relative_to(ROOT)).replace("\\", "/"),
                "notes": ("strict raw score reproduced within 1e-6" if available else value.get("reason", "raw score persisted but failed the frozen Strict reproduction tolerance; stopped without score adjustment")),
            })
    dump_json(OUT / "score_availability_rows.json", availability)
    current = json.loads((OUT / "manifest.json").read_text(encoding="utf-8")) if (OUT / "manifest.json").is_file() else {}
    current["e2_r1"] = {
        "status": status, "elapsed_seconds": time.perf_counter() - started, "raw_scores_persisted": sum((RAW / d / m / "test_raw_scores.npz").is_file() for d in ("swat", "hai") for m in METHODS),
        "raw_scores_strict_validated": len(complete),
        "raw_scores_total": 12, "completed": complete, "unavailable": [key for key in results if key not in complete],
        "mismatches": mismatch, "common_mask_manifest": "outputs/thesis_evaluation/literature_compatible/common_masks/common_mask_manifest.json",
        "common_mask_counts": {key: value["n_common"] for key, value in mask_manifest["datasets"].items()},
        "no_training": True, "no_backward": True, "no_optimizer": True, "no_scaler_fit": True, "no_ridge_fit": True,
        "no_tail_recalibration": True, "no_test_tuning": True, "no_pa": True, "no_oracle": True,
        "strict_results_overwritten": False, "commit": False, "push": False,
    }
    current["availability"] = {"strict_validated": len(complete), "persisted": current["e2_r1"]["raw_scores_persisted"], "total": 12, "mismatch": len(mismatch), "frozen_artifact_unavailable": 12 - current["e2_r1"]["raw_scores_persisted"]}
    current.setdefault("checks", {})["checkpoint_read_for_e2_r1_inference"] = True
    current.setdefault("restrictions", {})["checkpoint_read"] = True
    current["restrictions"]["checkpoint_selection"] = False
    current["status"] = status; dump_json(OUT / "manifest.json", current)
    print(json.dumps(current["e2_r1"], indent=2, ensure_ascii=False), flush=True)
    return 2 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
