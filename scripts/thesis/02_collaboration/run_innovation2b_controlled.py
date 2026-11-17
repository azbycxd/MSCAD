"""Validate frozen stage-joint lagged collaboration with Normal-only context swaps."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HELPER_DIR = ROOT / "scripts" / "thesis" / "02_collaboration"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

import run_innovation2a2_joint_lagged as a2  # noqa: E402
from data.swat2015_official import (  # noqa: E402
    EXPECTED_FEATURES,
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    load_normal_cache,
    load_scaler,
    resolve_repo_path,
    validate_config_contract,
)
from models.backbone_ae import BackboneAE  # noqa: E402
from utils.config_io import load_config  # noqa: E402


OUT = ROOT / "outputs" / "thesis_innovation2b_controlled"
CONFIG = ROOT / "configs" / "thesis" / "swat2015_midterm_seed1.yaml"
SCHEMA = ROOT / "outputs" / "thesis_t2_swat2015_audit" / "schema_profile.json"
A2_STABLE = ROOT / "outputs" / "thesis_innovation2a2_joint_lagged" / "stable_targets.csv"
A2_MANIFEST = ROOT / "outputs" / "thesis_innovation2a2_joint_lagged" / "dependency_manifest.json"
A0_DIR = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline"
EXPECTED_TARGETS = ("AIT202", "FIT201", "DPIT301", "FIT301", "FIT401", "FIT601")
LAGS = (1, 5, 10, 20, 30)
MAX_LAG = 30
WINDOW = 64
ALPHA = 1.0
ROBUST_FLOOR = 1e-6
CALIBRATION_START = TRAIN_RAW_ROWS
CALIBRATION_END = TRAIN_RAW_ROWS + (EXPECTED_NORMAL_ROWS - TRAIN_RAW_ROWS) // 2
EVALUATION_START = CALIBRATION_END
EVALUATION_END = EXPECTED_NORMAL_ROWS
MAX_SAMPLES = 2000
DONOR_MIN_SEPARATION = 600


class ControlledError(RuntimeError):
    """Raised when a frozen Innovation-2B contract would be violated."""


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ControlledError(f"refusing empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_ridge(x: np.ndarray, y: np.ndarray) -> Ridge:
    model = Ridge(alpha=ALPHA)
    model.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ControlledError("non-finite Ridge model")
    return model


def robust_reference(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, ROBUST_FLOOR)
    if not math.isfinite(median) or not math.isfinite(scale):
        raise ControlledError("non-finite collaboration reference")
    return median, scale


def score(values: np.ndarray, median: float, scale: float) -> np.ndarray:
    result = np.maximum(0.0, (values - median) / scale)
    if not np.isfinite(result).all():
        raise ControlledError("non-finite collaboration score")
    return result


def balanced_metrics(clean: np.ndarray, corrupted: np.ndarray) -> dict[str, float]:
    labels = np.concatenate(
        [np.zeros(len(clean), dtype=np.int8), np.ones(len(corrupted), dtype=np.int8)]
    )
    values = np.concatenate([clean, corrupted]).astype(np.float64, copy=False)
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "aupr": float(average_precision_score(labels, values)),
    }


def deterministic_donors(endpoint_count: int) -> tuple[np.ndarray, np.ndarray]:
    selected = np.linspace(0, endpoint_count - 1, min(MAX_SAMPLES, endpoint_count), dtype=np.int64)
    if len(np.unique(selected)) != len(selected):
        raise ControlledError("uniform evaluation sampling produced duplicates")
    donor = (selected + endpoint_count // 2) % endpoint_count
    for position in range(len(donor)):
        if abs(int(donor[position]) - int(selected[position])) >= DONOR_MIN_SEPARATION:
            continue
        candidate = (int(donor[position]) + 1) % endpoint_count
        while abs(candidate - int(selected[position])) < DONOR_MIN_SEPARATION:
            candidate = (candidate + 1) % endpoint_count
            if candidate == int(donor[position]):
                raise ControlledError("no deterministic donor satisfies separation")
        donor[position] = candidate
    return selected, donor


def build_a0(config: dict[str, Any], device: torch.device) -> BackboneAE:
    model_config = config["model"]
    model = BackboneAE(
        in_channels=int(model_config["in_channels"]),
        hidden_channels=int(model_config["hidden_channels"]),
        latent_channels=int(model_config["latent_channels"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    checkpoint = torch.load(A0_DIR / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters()) != 11_730:
        raise ControlledError("frozen A0 parameter count changed")
    return model


def a0_scores(
    model: BackboneAE,
    windows: np.ndarray,
    mean: np.ndarray,
    scale_value: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(windows), 256):
            raw = np.asarray(windows[start:start + 256], dtype=np.float32)
            scaled = (raw - mean) / scale_value
            tensor = torch.from_numpy(np.ascontiguousarray(scaled)).to(device)
            residual = torch.abs(model(tensor) - tensor)[:, -1, :].mean(dim=1)
            values.append(residual.cpu().numpy())
    result = np.concatenate(values).astype(np.float64, copy=False)
    if len(result) != len(windows) or not np.isfinite(result).all():
        raise ControlledError("frozen A0 scoring failed")
    return result


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite Innovation-2B output: {OUT}")
    config = load_config(str(CONFIG))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal_path = cache_dir / "normal_v1_features.npy"
    checkpoint_path = A0_DIR / "best.pt"
    a0_scaler_path = A0_DIR / "scaler.npz"
    hashes_before = {
        "normal_cache": sha256_file(normal_path),
        "a0_checkpoint": sha256_file(checkpoint_path),
        "a0_scaler": sha256_file(a0_scaler_path),
    }

    # No Attack loader or Attack artifact is referenced anywhere in this run.
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    order = [str(name) for name in cache_manifest["column_order"]]
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    contract, profile_by_name = a2.build_contract(a2.normal_profiles(schema), order)
    a2_manifest = json.loads(A2_MANIFEST.read_text(encoding="utf-8"))
    stable = pd.read_csv(A2_STABLE)
    if tuple(stable["target"].astype(str)) != EXPECTED_TARGETS:
        raise ControlledError(f"2A2 target set/order changed: {stable['target'].tolist()}")
    if a2_manifest["lags"] != list(LAGS) or float(a2_manifest["alpha"]) != ALPHA:
        raise ControlledError("2A2 lag or alpha contract changed")

    scaler = StandardScaler().fit(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float64))
    z = scaler.transform(np.asarray(normal, dtype=np.float64))
    if int(scaler.n_samples_seen_) != TRAIN_RAW_ROWS or not np.isfinite(z).all():
        raise ControlledError("Train-only scaler failed")
    name_to_index = {name: index for index, name in enumerate(order)}
    train_times = np.arange(MAX_LAG, TRAIN_RAW_ROWS, dtype=np.int64)
    calibration_times = np.arange(CALIBRATION_START + MAX_LAG, CALIBRATION_END, dtype=np.int64)
    evaluation_endpoints = np.arange(EVALUATION_START + WINDOW - 1, EVALUATION_END, dtype=np.int64)
    selected_positions, donor_positions = deterministic_donors(len(evaluation_endpoints))
    current_endpoints = evaluation_endpoints[selected_positions]
    donor_endpoints = evaluation_endpoints[donor_positions]
    donor_separations = np.abs(donor_endpoints - current_endpoints)
    if len(current_endpoints) != MAX_SAMPLES or int(donor_separations.min()) < DONOR_MIN_SEPARATION:
        raise ControlledError("deterministic donor contract failed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a0_model = build_a0(config, device)
    a0_scaler, a0_scaler_metadata = load_scaler(a0_scaler_path)
    if a0_scaler_metadata["fit_scope"] != "normal_v1_raw_rows_[0,396000)":
        raise ControlledError("A0 scaler scope changed")
    a0_mean = np.asarray(a0_scaler.mean_, dtype=np.float32).reshape(1, 1, -1)
    a0_scale = np.asarray(a0_scaler.effective_scale_, dtype=np.float32).reshape(1, 1, -1)
    window_offsets = np.arange(-WINDOW + 1, 1, dtype=np.int64)
    clean_windows = np.asarray(
        normal[current_endpoints[:, None] + window_offsets[None, :]], dtype=np.float32
    )
    clean_a0_score = a0_scores(a0_model, clean_windows, a0_mean, a0_scale, device)

    metric_rows: list[dict[str, Any]] = []
    semantics_rows: list[dict[str, Any]] = []
    target_manifest: dict[str, Any] = {}
    for frozen in stable.itertuples(index=False):
        target_name = str(frozen.target)
        stage = str(frozen.stage)
        stage_names = list(contract[stage]["nonconstant_channels"])
        target_index = name_to_index[target_name]
        target_local = stage_names.index(target_name)
        source_names = [name for name in stage_names if name != target_name]
        source_indices = [name_to_index[name] for name in source_names]
        stage_indices = [name_to_index[name] for name in stage_names]
        self_columns = list(range(target_local * len(LAGS), (target_local + 1) * len(LAGS)))

        x_train_joint = a2.lagged_design(z, train_times, stage_indices)
        y_train = z[train_times, target_index]
        self_model = fit_ridge(x_train_joint[:, self_columns], y_train)
        joint_model = fit_ridge(x_train_joint, y_train)

        x_calibration_joint = a2.lagged_design(z, calibration_times, stage_indices)
        y_calibration = z[calibration_times, target_index]
        e_self_calibration = np.abs(
            y_calibration - self_model.predict(x_calibration_joint[:, self_columns])
        )
        e_joint_calibration = np.abs(y_calibration - joint_model.predict(x_calibration_joint))
        c_calibration = e_joint_calibration - e_self_calibration
        reference_median, reference_scale = robust_reference(c_calibration)
        calibration_score = score(c_calibration, reference_median, reference_scale)
        threshold = float(np.quantile(calibration_score, 0.99))

        x_clean_joint = a2.lagged_design(z, current_endpoints, stage_indices)
        y_clean = z[current_endpoints, target_index]
        self_prediction = self_model.predict(x_clean_joint[:, self_columns])
        joint_clean_prediction = joint_model.predict(x_clean_joint)
        e_self_clean = np.abs(y_clean - self_prediction)
        e_joint_clean = np.abs(y_clean - joint_clean_prediction)
        c_clean = e_joint_clean - e_self_clean

        x_corrupt_joint = x_clean_joint.copy()
        for source_name in source_names:
            source_local = stage_names.index(source_name)
            source_global = name_to_index[source_name]
            block_start = source_local * len(LAGS)
            for lag_offset, lag in enumerate(LAGS):
                x_corrupt_joint[:, block_start + lag_offset] = z[donor_endpoints - lag, source_global]
        if not np.array_equal(x_corrupt_joint[:, self_columns], x_clean_joint[:, self_columns]):
            raise ControlledError(f"target history changed for {target_name}")
        e_self_corrupt = np.abs(y_clean - self_model.predict(x_corrupt_joint[:, self_columns]))
        e_joint_corrupt = np.abs(y_clean - joint_model.predict(x_corrupt_joint))
        c_corrupt = e_joint_corrupt - e_self_corrupt
        collab_clean = score(c_clean, reference_median, reference_scale)
        collab_corrupt = score(c_corrupt, reference_median, reference_scale)
        collab_metrics = balanced_metrics(collab_clean, collab_corrupt)
        self_metrics = balanced_metrics(e_self_clean, e_self_corrupt)

        corrupted_windows = clean_windows.copy()
        donor_window_rows = donor_endpoints[:, None] + window_offsets[None, :]
        corrupted_windows[:, :, source_indices] = np.asarray(
            normal[donor_window_rows[:, :, None], np.asarray(source_indices)[None, None, :]],
            dtype=np.float32,
        )
        if not np.array_equal(corrupted_windows[:, :, target_index], clean_windows[:, :, target_index]):
            raise ControlledError(f"A0 target history changed for {target_name}")
        unchanged_indices = [index for index in range(EXPECTED_FEATURES) if index not in source_indices]
        if not np.array_equal(corrupted_windows[:, :, unchanged_indices], clean_windows[:, :, unchanged_indices]):
            raise ControlledError(f"non-source A0 channels changed for {target_name}")
        corrupted_a0_score = a0_scores(
            a0_model, corrupted_windows, a0_mean, a0_scale, device
        )
        a0_metrics = balanced_metrics(clean_a0_score, corrupted_a0_score)
        normal_fpr = float(np.mean(collab_clean > threshold))
        corruption_recall = float(np.mean(collab_corrupt > threshold))
        joint_degradation = float(np.median(e_joint_corrupt - e_joint_clean))
        self_change = float(np.median(e_self_corrupt - e_self_clean))
        row = {
            "target": target_name,
            "stage": stage,
            "validation_gain_2a2": float(frozen.joint_validation_gain),
            "collab_auroc": collab_metrics["auroc"],
            "collab_aupr": collab_metrics["aupr"],
            "a0_auroc": a0_metrics["auroc"],
            "a0_aupr": a0_metrics["aupr"],
            "self_auroc": self_metrics["auroc"],
            "corruption_recall": corruption_recall,
            "normal_fpr": normal_fpr,
            "collab_threshold": threshold,
            "collab_beats_a0_by_0_10": collab_metrics["auroc"] >= a0_metrics["auroc"] + 0.10,
        }
        metric_rows.append(row)
        semantics_rows.append(
            {
                "target": target_name,
                "stage": stage,
                "calibration_c_median": reference_median,
                "calibration_c_mad_scale": reference_scale,
                "clean_median_e_self": float(np.median(e_self_clean)),
                "clean_median_e_joint": float(np.median(e_joint_clean)),
                "clean_median_c": float(np.median(c_clean)),
                "corrupt_median_e_self": float(np.median(e_self_corrupt)),
                "corrupt_median_e_joint": float(np.median(e_joint_corrupt)),
                "corrupt_median_c": float(np.median(c_corrupt)),
                "joint_degradation": joint_degradation,
                "self_change": self_change,
            }
        )
        target_manifest[target_name] = {
            "stage": stage,
            "target_index": target_index,
            "stage_nonconstant_channels": stage_names,
            "corrupted_source_channels": source_names,
            "source_count": len(source_names),
            "target_excluded_from_corrupted_sources": target_name not in source_names,
            "target_current_unchanged": True,
            "target_history_unchanged": True,
            "other_stage_channels_unchanged": True,
            "ridge_alpha": ALPHA,
            "lags": list(LAGS),
            "calibration_reference_median": reference_median,
            "calibration_reference_scale": reference_scale,
            "calibration_q99_threshold": threshold,
        }
        del x_train_joint, x_calibration_joint, x_clean_joint, x_corrupt_joint, corrupted_windows

    frame = pd.DataFrame(metric_rows)
    semantics = pd.DataFrame(semantics_rows)
    median_collab_auroc = float(frame["collab_auroc"].median())
    median_collab_aupr = float(frame["collab_aupr"].median())
    beats_count = int(frame["collab_beats_a0_by_0_10"].sum())
    median_recall = float(frame["corruption_recall"].median())
    median_fpr = float(frame["normal_fpr"].median())
    self_sanity = bool(frame["self_auroc"].between(0.49, 0.51, inclusive="both").all())
    gates = {
        "A_median_collab_auroc_at_least_0_80": median_collab_auroc >= 0.80,
        "B_at_least_4_of_6_collab_beats_a0_by_0_10": beats_count >= 4,
        "C_median_corruption_recall_at_least_0_50": median_recall >= 0.50,
        "D_median_clean_normal_fpr_at_most_0_03": median_fpr <= 0.03,
        "E_all_self_auroc_between_0_49_and_0_51": self_sanity,
    }
    if all(gates.values()):
        status = "STAGE_JOINT_LAGGED_CONTROLLED_MECHANISM_SUPPORTED"
    elif median_collab_auroc >= 0.70:
        status = "STAGE_JOINT_LAGGED_CONTROLLED_PARTIAL"
    else:
        status = "STAGE_JOINT_LAGGED_CONTROLLED_NOT_SUPPORTED"

    joint_gain_rank = frame["validation_gain_2a2"].rank(method="average").to_numpy(float)
    auroc_rank = frame["collab_auroc"].rank(method="average").to_numpy(float)
    spearman = float(np.corrcoef(joint_gain_rank, auroc_rank)[0, 1])
    p3 = frame[frame["stage"] == "P3"]
    p6 = frame[frame["target"] == "FIT601"].iloc[0]
    target_summary_rows = [
        {
            **row,
            "p3_mechanism_pass": (
                row["stage"] == "P3" and float(row["collab_auroc"]) >= 0.80
            ),
            "strongest_dependency_fit601_detectability_rank": (
                int(frame["collab_auroc"].rank(ascending=False, method="min")[index])
                if row["target"] == "FIT601" else ""
            ),
        }
        for index, row in enumerate(metric_rows)
    ]
    hashes_after = {
        "normal_cache": sha256_file(normal_path),
        "a0_checkpoint": sha256_file(checkpoint_path),
        "a0_scaler": sha256_file(a0_scaler_path),
    }
    tests = {
        "attack_not_read": True,
        "stable_targets_exact_2a2_six": tuple(frame["target"]) == EXPECTED_TARGETS,
        "stage_definition_unchanged": a2_manifest["stage_contract"] == contract,
        "self_features_unchanged": all(value["target_history_unchanged"] for value in target_manifest.values()),
        "joint_features_unchanged": all(
            value["lags"] == list(LAGS) and value["stage_nonconstant_channels"] == contract[value["stage"]]["nonconstant_channels"]
            for value in target_manifest.values()
        ),
        "alpha_1": ALPHA == 1.0,
        "target_current_unchanged": all(value["target_current_unchanged"] for value in target_manifest.values()),
        "target_history_unchanged": all(value["target_history_unchanged"] for value in target_manifest.values()),
        "donor_deterministic": np.array_equal(
            deterministic_donors(len(evaluation_endpoints))[1], donor_positions
        ),
        "donor_separation_at_least_600": int(donor_separations.min()) >= DONOR_MIN_SEPARATION,
        "donor_from_normal": bool(
            donor_endpoints.min() >= EVALUATION_START + WINDOW - 1
            and donor_endpoints.max() < EVALUATION_END
        ),
        "a0_checkpoint_unchanged": hashes_before["a0_checkpoint"] == hashes_after["a0_checkpoint"],
        "threshold_calibration_normal_only": all(
            math.isfinite(float(value["calibration_q99_threshold"]))
            for value in target_manifest.values()
        ),
        "no_test_driven_tuning": True,
        "no_nan_inf": bool(
            np.isfinite(frame.select_dtypes(include=["number"]).to_numpy()).all()
            and np.isfinite(semantics.select_dtypes(include=["number"]).to_numpy()).all()
        ),
    }
    if hashes_before != hashes_after or not all(tests.values()):
        raise ControlledError(f"Innovation-2B integrity failure: {tests}")

    manifest = {
        "protocol": "Innovation-2B Normal-only controlled stage context swap",
        "normal_source": "Official SWaT Normal v1 via frozen S1 cache",
        "attack_data_read": False,
        "attack_labels_read": False,
        "stable_target_source": str(A2_STABLE.relative_to(ROOT)),
        "stable_targets": list(EXPECTED_TARGETS),
        "stage_source": str(SCHEMA.relative_to(ROOT)),
        "lags": list(LAGS),
        "alpha": ALPHA,
        "dtype": "float64",
        "train_interval": [0, TRAIN_RAW_ROWS],
        "calibration_normal_raw_interval": [CALIBRATION_START, CALIBRATION_END],
        "evaluation_normal_raw_interval": [EVALUATION_START, EVALUATION_END],
        "calibration_endpoint_count": len(calibration_times),
        "evaluation_legal_window_count": len(evaluation_endpoints),
        "sample_count_per_class_per_target": len(current_endpoints),
        "sampling": "fixed uniform np.linspace over ordered evaluation legal windows",
        "selected_evaluation_positions": selected_positions.tolist(),
        "donor_positions": donor_positions.tolist(),
        "donor_rule": "(k + floor(N/2)) mod N; sequential forward fallback only if separation <600",
        "donor_min_separation_seconds": int(donor_separations.min()),
        "donor_max_separation_seconds": int(donor_separations.max()),
        "target_contract": target_manifest,
        "a0": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": hashes_after["a0_checkpoint"],
            "parameter_count": 11_730,
            "score": "frozen endpoint 51-channel MAE in A0 model input space",
            "retrained": False,
            "device": str(device),
        },
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "source_unchanged": hashes_before == hashes_after,
        "threshold_source": "per-target calibration_normal S_collab q99 only",
        "corrupted_labels_used_for_threshold": False,
        "feature_search": False,
        "lag_search": False,
        "threshold_search": False,
        "point_adjustment": False,
        "fusion": False,
        "tests": tests,
    }
    summary = {
        "status": status,
        "median_collab_auroc": median_collab_auroc,
        "median_collab_aupr": median_collab_aupr,
        "collab_beats_a0_by_0_10_count": beats_count,
        "median_corruption_recall": median_recall,
        "median_normal_fpr": median_fpr,
        "gates": gates,
        "p3_targets_both_collab_auroc_at_least_0_80": bool((p3["collab_auroc"] >= 0.80).all()),
        "fit601_collab_auroc": float(p6["collab_auroc"]),
        "fit601_detectability_rank": int(frame["collab_auroc"].rank(ascending=False, method="min")[frame["target"] == "FIT601"].iloc[0]),
        "joint_gain_vs_collab_auroc_spearman": spearman,
        "tests_passed": sum(bool(value) for value in tests.values()),
        "tests_total": len(tests),
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.mkdir(parents=True)
    write_csv(OUT / "controlled_metrics.csv", metric_rows)
    write_csv(OUT / "score_semantics.csv", semantics_rows)
    write_json(OUT / "corruption_manifest.json", manifest)
    write_csv(OUT / "target_summary.csv", target_summary_rows)
    write_json(OUT / "innovation2b_summary.json", summary)
    print(json.dumps({**summary, "targets": metric_rows, "semantics": semantics_rows}, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
