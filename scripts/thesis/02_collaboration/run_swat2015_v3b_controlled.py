"""Normal-only controlled validation of frozen V3-A cross-channel contribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    EXPECTED_FEATURES,
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    OfficialSWaTError,
    load_normal_cache,
    load_scaler,
    resolve_repo_path,
    validate_config_contract,
)
from models.backbone_ae import BackboneAE  # noqa: E402
from utils.config_io import load_config  # noqa: E402


LAGS = np.asarray([1, 5, 10, 20, 30], dtype=np.int64)
ALPHA = 1.0
GAIN_THRESHOLD = 0.10
SHIFT_SECONDS = 60
DONOR_MIN_SEPARATION = 300
MAX_SAMPLES = 2000
DONOR_POOL_SIZE = 2048
SCALE_FLOOR = 1e-6
WINDOW = 64


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_even_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return np.asarray(values, dtype=np.int64)
    positions = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return np.asarray(values[positions], dtype=np.int64)


def split_validation_times(row_count: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.arange(int(LAGS.max()), row_count, dtype=np.int64)
    split = len(valid) // 2
    calibration = valid[:split]
    evaluation = valid[split:]
    if calibration[-1] >= evaluation[0]:
        raise OfficialSWaTError("calibration/evaluation time ranges overlap")
    return calibration, evaluation


def history_matrix(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.concatenate([values[times - lag] for lag in LAGS], axis=1).astype(np.float64, copy=False)


def self_columns(target_index: int) -> np.ndarray:
    return np.asarray([lag_index * EXPECTED_FEATURES + target_index for lag_index in range(len(LAGS))])


def predict_errors(
    values: np.ndarray,
    times: np.ndarray,
    target_index: int,
    full_coef: np.ndarray,
    full_intercept: np.ndarray,
    self_coef: np.ndarray,
    self_intercept: np.ndarray,
    donor_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original_history = history_matrix(values, times)
    self_input = original_history[:, self_columns(target_index)]
    self_prediction = self_input @ self_coef[target_index] + self_intercept[target_index]
    target = np.asarray(values[times, target_index], dtype=np.float64)
    self_error = np.abs(target - self_prediction)
    if donor_times is None:
        full_input = original_history
    else:
        donor_history = history_matrix(values, donor_times)
        full_input = donor_history.copy()
        full_input[:, self_columns(target_index)] = self_input
        if not np.array_equal(full_input[:, self_columns(target_index)], self_input):
            raise OfficialSWaTError("target self-history changed during controlled corruption")
    full_prediction = full_input @ full_coef[target_index] + full_intercept[target_index]
    full_error = np.abs(target - full_prediction)
    gain = self_error - full_error
    if not np.isfinite(self_error).all() or not np.isfinite(full_error).all() or not np.isfinite(gain).all():
        raise OfficialSWaTError("non-finite V3 controlled score component")
    return self_error, full_error, gain


def fit_gain_reference(gain: np.ndarray, *, source_split: str) -> tuple[float, float]:
    if source_split != "calibration_normal":
        raise OfficialSWaTError("gain reference may only use calibration_normal")
    center = float(np.median(gain))
    mad = float(np.median(np.abs(gain - center)))
    scale = max(1.4826 * mad, SCALE_FLOOR)
    return center, scale


def collaboration_score(gain: np.ndarray, center: float, scale: float) -> np.ndarray:
    score = np.maximum(0.0, (center - gain) / scale)
    if not np.isfinite(score).all():
        raise OfficialSWaTError("non-finite collaboration degradation score")
    return score


def calibration_q99(score: np.ndarray, *, source_split: str) -> float:
    if source_split != "calibration_normal":
        raise OfficialSWaTError("V3-B q99 may only use calibration_normal")
    return float(np.quantile(score, 0.99))


def select_matched_donors(
    values: np.ndarray,
    sample_times: np.ndarray,
    candidate_times: np.ndarray,
    target_index: int,
) -> np.ndarray:
    sample_self = history_matrix(values, sample_times)[:, self_columns(target_index)]
    candidate_self = history_matrix(values, candidate_times)[:, self_columns(target_index)]
    donors = np.empty(len(sample_times), dtype=np.int64)
    for index, (time_value, vector) in enumerate(zip(sample_times, sample_self)):
        legal = np.abs(candidate_times - time_value) >= DONOR_MIN_SEPARATION
        if not legal.any():
            raise OfficialSWaTError("no legal donor separated by at least 300 seconds")
        legal_times = candidate_times[legal]
        distance = np.sum((candidate_self[legal] - vector) ** 2, axis=1)
        minimum = np.min(distance)
        # candidate_times is sorted, so first exact minimum is the deterministic tie-break.
        donors[index] = legal_times[np.flatnonzero(distance == minimum)[0]]
    if np.any(np.abs(donors - sample_times) < DONOR_MIN_SEPARATION):
        raise OfficialSWaTError("matched donor violates fixed temporal separation")
    return donors


def build_hybrid_windows(
    values: np.ndarray, times: np.ndarray, donor_times: np.ndarray, target_index: int
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-WINDOW + 1, 1, dtype=np.int64)
    clean = np.asarray(values[times[:, None] + offsets[None, :]], dtype=np.float32)
    corrupt = np.asarray(values[donor_times[:, None] + offsets[None, :]], dtype=np.float32).copy()
    corrupt[:, :, target_index] = clean[:, :, target_index]
    if not np.array_equal(clean[:, :, target_index], corrupt[:, :, target_index]):
        raise OfficialSWaTError("target 64-point history is not preserved")
    return clean, corrupt


def a0_endpoint_score(model: torch.nn.Module, windows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(windows[offset : offset + batch_size])).to(device)
            chunks.append(torch.abs(model(x)[:, -1, :] - x[:, -1, :]).mean(dim=1).cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise OfficialSWaTError("non-finite A0 controlled score")
    return result


def ranking(clean: np.ndarray, corrupt: np.ndarray) -> tuple[float, float]:
    labels = np.concatenate([np.zeros(len(clean), dtype=np.int8), np.ones(len(corrupt), dtype=np.int8)])
    score = np.concatenate([clean, corrupt])
    return float(roc_auc_score(labels, score)), float(average_precision_score(labels, score))


def build_model(config: dict[str, Any]) -> BackboneAE:
    model = config["model"]
    return BackboneAE(
        in_channels=int(model["in_channels"]),
        hidden_channels=int(model["hidden_channels"]),
        latent_channels=int(model["latent_channels"]),
        dropout=float(model["dropout"]),
    )


def run(config_path: Path, v3a_output: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(str(config_path))
    validate_config_contract(config)
    if config["v3a"]["lags"] != LAGS.tolist() or float(config["v3a"]["alpha"]) != ALPHA:
        raise OfficialSWaTError("V3-B must retain V3-A lags and alpha")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing V3-B output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = list(csv.DictReader((v3a_output / "target_metrics.csv").open(encoding="utf-8")))
    frozen_targets = [row for row in target_rows if row["incremental_collaboration_target"].lower() == "true"]
    if len(frozen_targets) != 4:
        raise OfficialSWaTError(f"V3-A frozen target count must equal 4, got {len(frozen_targets)}")
    v3a_summary = json.loads((v3a_output / "summary.json").read_text(encoding="utf-8"))
    if v3a_summary["gain_ge_10_percent_count"] != 4:
        raise OfficialSWaTError("V3-A summary no longer freezes exactly four >=10% targets")
    with np.load(v3a_output / "ridge_coefficients.npz", allow_pickle=False) as payload:
        full_coef = np.asarray(payload["full_coef"], dtype=np.float64)
        full_intercept = np.asarray(payload["full_intercept"], dtype=np.float64)
        self_coef = np.asarray(payload["self_coef"], dtype=np.float64)
        self_intercept = np.asarray(payload["self_intercept"], dtype=np.float64)
        if not np.array_equal(payload["lags"], LAGS):
            raise OfficialSWaTError("frozen Ridge artifact lags differ from V3-A")

    normal, manifest = load_normal_cache(resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
    scaler, scaler_meta = load_scaler(resolve_repo_path(ROOT, config["paths"]["a0_output"]) / "scaler.npz")
    if scaler_meta["attack_rows_seen"] != 0:
        raise OfficialSWaTError("V3-B requires the Train-only A0 scaler")
    validation = scaler.transform(np.asarray(normal[TRAIN_RAW_ROWS:EXPECTED_NORMAL_ROWS], dtype=np.float32)).astype(
        np.float32, copy=False
    )
    calibration_times, evaluation_times = split_validation_times(len(validation))
    evaluation_start = int(evaluation_times[0])
    # All clean and shifted/donor 64-point windows stay wholly inside evaluation_normal.
    legal_evaluation = evaluation_times[evaluation_times >= evaluation_start + SHIFT_SECONDS + WINDOW - 1]
    sample_times = deterministic_even_sample(legal_evaluation, MAX_SAMPLES)
    donor_legal = evaluation_times[evaluation_times >= evaluation_start + WINDOW - 1]
    donor_pool = deterministic_even_sample(donor_legal, DONOR_POOL_SIZE)
    shift_donors = sample_times - SHIFT_SECONDS
    if int(shift_donors.min()) - WINDOW + 1 < evaluation_start:
        raise OfficialSWaTError("60-second context shift crosses calibration/evaluation boundary")

    a0_output = resolve_repo_path(ROOT, config["paths"]["a0_output"])
    checkpoint_path = a0_output / "best.pt"
    checkpoint_hash_before = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["best_epoch"] != 13:
        raise OfficialSWaTError("V3-B did not receive the frozen A0 epoch-13 checkpoint")
    if not torch.cuda.is_available():
        raise OfficialSWaTError("V3-B requires the available A0 CUDA runtime")
    device = torch.device("cuda")
    a0_model = build_model(config).to(device)
    a0_model.load_state_dict(checkpoint["state_dict"])
    if sum(parameter.numel() for parameter in a0_model.parameters()) != 11730:
        raise OfficialSWaTError("A0 checkpoint architecture changed")

    rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    donor_manifest: dict[str, Any] = {}
    feature_names = list(manifest["column_order"])
    for frozen in frozen_targets:
        target = frozen["target"]
        target_index = feature_names.index(target)
        calibration_self, calibration_full, calibration_gain = predict_errors(
            validation, calibration_times, target_index, full_coef, full_intercept, self_coef, self_intercept
        )
        del calibration_self, calibration_full
        center, gain_scale = fit_gain_reference(calibration_gain, source_split="calibration_normal")
        calibration_collab = collaboration_score(calibration_gain, center, gain_scale)
        threshold = calibration_q99(calibration_collab, source_split="calibration_normal")
        clean_self, clean_full, clean_gain = predict_errors(
            validation, sample_times, target_index, full_coef, full_intercept, self_coef, self_intercept
        )
        clean_collab = collaboration_score(clean_gain, center, gain_scale)
        normal_fpr = float(np.mean(clean_collab > threshold))

        matched_donors = select_matched_donors(validation, sample_times, donor_pool, target_index)
        donor_manifest[target] = {
            "matched_donor_sha256": hashlib.sha256(matched_donors.tobytes()).hexdigest(),
            "minimum_observed_separation": int(np.min(np.abs(matched_donors - sample_times))),
        }
        for injection, donors in (("TemporalShift60", shift_donors), ("MatchedContextSwap", matched_donors)):
            corrupt_self, corrupt_full, corrupt_gain = predict_errors(
                validation,
                sample_times,
                target_index,
                full_coef,
                full_intercept,
                self_coef,
                self_intercept,
                donor_times=donors,
            )
            if not np.array_equal(clean_self, corrupt_self):
                raise OfficialSWaTError("self residual changed even though target history was preserved")
            corrupt_collab = collaboration_score(corrupt_gain, center, gain_scale)
            clean_windows, corrupt_windows = build_hybrid_windows(validation, sample_times, donors, target_index)
            clean_a0 = a0_endpoint_score(a0_model, clean_windows, device, int(config["training"]["batch_size"]))
            corrupt_a0 = a0_endpoint_score(a0_model, corrupt_windows, device, int(config["training"]["batch_size"]))
            v3_auroc, v3_aupr = ranking(clean_collab, corrupt_collab)
            a0_auroc, a0_aupr = ranking(clean_a0, corrupt_a0)
            self_auroc, self_aupr = ranking(clean_self, corrupt_self)
            full_auroc, full_aupr = ranking(clean_full, corrupt_full)
            median_normal_gain = float(np.median(clean_gain))
            median_corrupted_gain = float(np.median(corrupt_gain))
            row = {
                "target": target,
                "v3a_relative_gain": float(frozen["relative_gain"]),
                "injection": injection,
                "sample_count_per_class": len(sample_times),
                "v3_auroc": v3_auroc,
                "v3_aupr": v3_aupr,
                "a0_auroc": a0_auroc,
                "a0_aupr": a0_aupr,
                "self_auroc": self_auroc,
                "self_aupr": self_aupr,
                "full_auroc": full_auroc,
                "full_aupr": full_aupr,
                "q99_threshold": threshold,
                "q99_corruption_recall": float(np.mean(corrupt_collab > threshold)),
                "normal_fpr": normal_fpr,
                "median_gain_normal": median_normal_gain,
                "median_gain_corrupted": median_corrupted_gain,
                "median_gain_drop": median_normal_gain - median_corrupted_gain,
            }
            rows.append(row)
            gain_rows.append(
                {
                    "target": target,
                    "injection": injection,
                    "reference_median_gain": center,
                    "reference_scale_gain": gain_scale,
                    "median_gain_normal": median_normal_gain,
                    "median_gain_corrupted": median_corrupted_gain,
                    "median_gain_drop": median_normal_gain - median_corrupted_gain,
                }
            )

    if sha256(checkpoint_path) != checkpoint_hash_before:
        raise OfficialSWaTError("A0 checkpoint changed during read-only inference")
    v3_aurocs = np.asarray([row["v3_auroc"] for row in rows])
    gate_a = bool(np.median(v3_aurocs) >= 0.80)
    beats = sum(row["v3_auroc"] > max(row["a0_auroc"], row["self_auroc"]) for row in rows)
    gate_b = beats >= 6
    recall_median = float(np.median([row["q99_corruption_recall"] for row in rows]))
    fpr_median = float(np.median([row["normal_fpr"] for row in rows]))
    gate_c = recall_median >= 0.50
    gate_d = fpr_median <= 0.03
    if gate_a and gate_b and gate_c and gate_d:
        status = "V3B_CONTROLLED_MECHANISM_SUPPORTED"
    elif float(np.median(v3_aurocs)) >= 0.70:
        status = "V3B_PARTIAL_MECHANISM_SUPPORT"
    else:
        status = "V3B_MECHANISM_NOT_SUPPORTED"
    def med(injection: str, key: str) -> float:
        return float(np.median([row[key] for row in rows if row["injection"] == injection]))
    summary = {
        "experiment_id": "V3-B",
        "scope": "NORMAL-ONLY CONTROLLED EXPERIMENT; no real Attack data or labels",
        "status": status,
        "frozen_targets": [
            {"target": row["target"], "v3a_relative_gain": float(row["relative_gain"])} for row in frozen_targets
        ],
        "target_count": len(frozen_targets),
        "cell_count": len(rows),
        "samples_per_class_per_cell": len(sample_times),
        "temporal_shift": {
            "median_v3_auroc": med("TemporalShift60", "v3_auroc"),
            "median_v3_aupr": med("TemporalShift60", "v3_aupr"),
            "median_a0_auroc": med("TemporalShift60", "a0_auroc"),
            "median_self_auroc": med("TemporalShift60", "self_auroc"),
        },
        "matched_swap": {
            "median_v3_auroc": med("MatchedContextSwap", "v3_auroc"),
            "median_v3_aupr": med("MatchedContextSwap", "v3_aupr"),
            "median_a0_auroc": med("MatchedContextSwap", "a0_auroc"),
            "median_self_auroc": med("MatchedContextSwap", "self_auroc"),
        },
        "all_cells": {
            "median_v3_auroc": float(np.median(v3_aurocs)),
            "median_v3_aupr": float(np.median([row["v3_aupr"] for row in rows])),
            "v3_beats_a0_and_self_cell_count": beats,
            "median_q99_corruption_recall": recall_median,
            "median_normal_fpr": fpr_median,
            "median_normal_gain": float(np.median([row["median_gain_normal"] for row in rows])),
            "median_corrupted_gain": float(np.median([row["median_gain_corrupted"] for row in rows])),
            "median_gain_drop": float(np.median([row["median_gain_drop"] for row in rows])),
        },
        "gates": {"A_median_auroc_ge_080": gate_a, "B_beats_ge_6_of_8": gate_b, "C_recall_ge_050": gate_c, "D_fpr_le_003": gate_d},
        "attack_cache_or_labels_read": False,
        "a0_checkpoint_sha256_before_after": checkpoint_hash_before,
        "a0_retraining": False,
        "target_reselection": False,
        "parameter_search": False,
        "fusion": False,
        "test_driven_tuning": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "v3b_summary.json", summary)
    with (output_dir / "v3b_target_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / "v3b_gain_shift.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gain_rows[0]))
        writer.writeheader(); writer.writerows(gain_rows)
    write_json(
        output_dir / "injection_manifest.json",
        {
            "source": "Official SWaT 2015 Normal v1 Validation only",
            "calibration_time_bounds": [int(calibration_times[0]), int(calibration_times[-1])],
            "evaluation_time_bounds": [int(evaluation_times[0]), int(evaluation_times[-1])],
            "sample_times_sha256": hashlib.sha256(sample_times.tobytes()).hexdigest(),
            "sample_count": len(sample_times),
            "sampling": "deterministic evenly spaced",
            "shift_seconds": SHIFT_SECONDS,
            "donor_minimum_separation_seconds": DONOR_MIN_SEPARATION,
            "donor_candidate_pool_size": len(donor_pool),
            "target_history_preserved": True,
            "other_context_source": "real Normal validation windows",
            "attack_data_read": False,
            "target_donor_audit": donor_manifest,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis/swat2015_midterm_seed1.yaml")
    parser.add_argument("--v3a-output", default="outputs/thesis_midterm_s1/v3a_incremental_gate_float64")
    parser.add_argument("--output", default="outputs/thesis_v3b_controlled")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(resolve_repo_path(ROOT, args.config), resolve_repo_path(ROOT, args.v3a_output), resolve_repo_path(ROOT, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
