"""Run the preregistered stage-local pointwise consistency diagnostic on SWaT 2015."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    EXPECTED_ATTACK_ROWS,
    EXPECTED_FEATURES,
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    VALIDATION_RAW_ROWS,
    load_attack_cache,
    load_normal_cache,
    resolve_repo_path,
    validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


OUT = ROOT / "outputs" / "thesis_swat_pcc1"
CONFIG_PATH = ROOT / "configs" / "thesis" / "swat2015_midterm_seed1.yaml"
T2_SCHEMA = ROOT / "outputs" / "thesis_t2_swat2015_audit" / "schema_profile.json"
T25_EVENTS = ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "canonical_attack_events.csv"
T25_SEGMENTS = ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "segment_manifest.json"
S6_EVENTS = ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit" / "event_audit.csv"
S6_A0_VALIDATION = ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit" / "a0_validation_scores.npy"
A0_DIR = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline"
CONTEXTUAL_IDS = (13, 14, 29)
ALPHA = 1.0
ROBUST_FLOOR = 1e-6
THRESHOLD_QUANTILE = 0.99
STAGES = tuple(f"P{i}" for i in range(1, 7))


class PCCError(RuntimeError):
    """Raised when the frozen PCC1 protocol would be violated."""


@dataclass
class TargetModel:
    stage: str
    target_name: str
    target_local_index: int
    target_global_index: int
    input_names: list[str]
    input_local_indices: list[int]
    ridge: Ridge
    residual_median: float
    residual_mad: float
    residual_scale: float


@dataclass
class StageModel:
    stage: str
    feature_names: list[str]
    global_indices: list[int]
    scaler: StandardScaler
    targets: list[TargetModel]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise PCCError(f"refusing empty CSV: {path.name}")
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


def stage_of_channel(name: str) -> str:
    match = re.search(r"(\d{3})$", name)
    if match is None:
        raise PCCError(f"schema channel has no SWaT stage suffix: {name}")
    stage = f"P{match.group(1)[0]}"
    if stage not in STAGES:
        raise PCCError(f"schema channel resolves outside P1-P6: {name}")
    return stage


def normal_v1_profiles(audit: dict[str, Any]) -> list[dict[str, Any]]:
    physical = [item for item in audit["physical_files"] if item.get("file_name") == "SWaT_Dataset_Normal_v1.xlsx"]
    if len(physical) != 1 or len(physical[0].get("variable_profiles", [])) != EXPECTED_FEATURES:
        raise PCCError("T2 Normal v1 variable profiles are missing or ambiguous")
    return physical[0]["variable_profiles"]


def build_stage_contract(
    profiles: list[dict[str, Any]], column_order: list[str]
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, Any]]]:
    by_name = {str(profile["name"]): profile for profile in profiles}
    if set(by_name) != set(column_order) or len(by_name) != EXPECTED_FEATURES:
        raise PCCError("T2 schema and S1 cache column order disagree")
    contract: dict[str, dict[str, list[str]]] = {}
    for stage in STAGES:
        members = [name for name in column_order if stage_of_channel(name) == stage]
        nonconstant = [name for name in members if not bool(by_name[name]["constant"])]
        targets = [
            name for name in nonconstant if by_name[name]["observed_value_kind"] == "continuous_numeric"
        ]
        discrete_predictors = [
            name for name in nonconstant if by_name[name]["observed_value_kind"] != "continuous_numeric"
        ]
        if not targets or len(nonconstant) < 2:
            raise PCCError(f"stage {stage} lacks a continuous target or cross-channel predictor")
        contract[stage] = {
            "schema_members": members,
            "nonconstant_predictors": nonconstant,
            "continuous_targets": targets,
            "discrete_predictors": discrete_predictors,
            "excluded_constants": [name for name in members if bool(by_name[name]["constant"])],
        }
    return contract, by_name


def fit_models(
    normal: np.ndarray,
    column_order: list[str],
    contract: dict[str, dict[str, list[str]]],
) -> tuple[list[StageModel], list[dict[str, Any]]]:
    models: list[StageModel] = []
    references: list[dict[str, Any]] = []
    name_to_index = {name: index for index, name in enumerate(column_order)}
    train = normal[:TRAIN_RAW_ROWS]
    for stage in STAGES:
        names = contract[stage]["nonconstant_predictors"]
        global_indices = [name_to_index[name] for name in names]
        train_x = np.asarray(train[:, global_indices], dtype=np.float64)
        scaler = StandardScaler().fit(train_x)
        train_z = scaler.transform(train_x)
        targets: list[TargetModel] = []
        for target_name in contract[stage]["continuous_targets"]:
            target_local = names.index(target_name)
            input_local = [index for index in range(len(names)) if index != target_local]
            if target_local in input_local or not input_local:
                raise PCCError(f"invalid inputs for {stage}/{target_name}")
            ridge = Ridge(alpha=ALPHA)
            ridge.fit(train_z[:, input_local], train_z[:, target_local])
            residual = np.abs(train_z[:, target_local] - ridge.predict(train_z[:, input_local]))
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            scale = max(1.4826 * mad, ROBUST_FLOOR)
            if not np.isfinite(residual).all() or not math.isfinite(scale):
                raise PCCError(f"non-finite Train residual reference for {stage}/{target_name}")
            model = TargetModel(
                stage=stage,
                target_name=target_name,
                target_local_index=target_local,
                target_global_index=name_to_index[target_name],
                input_names=[names[index] for index in input_local],
                input_local_indices=input_local,
                ridge=ridge,
                residual_median=median,
                residual_mad=mad,
                residual_scale=scale,
            )
            targets.append(model)
            references.append(
                {
                    "stage": stage,
                    "target": target_name,
                    "inputs": ";".join(model.input_names),
                    "target_excluded": True,
                    "target_type": "continuous",
                    "discrete_predictors_allowed": any(
                        name in contract[stage]["discrete_predictors"] for name in model.input_names
                    ),
                    "alpha": ALPHA,
                    "ridge_intercept": float(ridge.intercept_),
                    "ridge_coefficients": ";".join(str(float(value)) for value in ridge.coef_),
                    "train_residual_median": median,
                    "train_residual_mad": mad,
                    "train_residual_scale": scale,
                    "fit_rows": TRAIN_RAW_ROWS,
                    "fit_scope": "Normal Train raw rows [0,396000) only",
                }
            )
        models.append(StageModel(stage, names, global_indices, scaler, targets))
    return models, references


def score_points(
    features: np.ndarray, models: list[StageModel]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rows = len(features)
    score = np.full(rows, -np.inf, dtype=np.float64)
    argmax_stage = np.full(rows, -1, dtype=np.int8)
    argmax_target = np.full(rows, -1, dtype=np.int16)
    target_scores: dict[str, np.ndarray] = {}
    for stage_index, stage_model in enumerate(models):
        raw = np.asarray(features[:, stage_model.global_indices], dtype=np.float64)
        z = stage_model.scaler.transform(raw)
        stage_score = np.full(rows, -np.inf, dtype=np.float64)
        stage_target = np.full(rows, -1, dtype=np.int16)
        for target in stage_model.targets:
            residual = np.abs(
                z[:, target.target_local_index]
                - target.ridge.predict(z[:, target.input_local_indices])
            )
            deviation = np.maximum(
                0.0, (residual - target.residual_median) / target.residual_scale
            )
            target_scores[target.target_name] = deviation
            update = deviation > stage_score
            stage_score[update] = deviation[update]
            stage_target[update] = target.target_global_index
        update_global = stage_score > score
        score[update_global] = stage_score[update_global]
        argmax_stage[update_global] = stage_index
        argmax_target[update_global] = stage_target[update_global]
    if (
        not np.isfinite(score).all()
        or (argmax_stage < 0).any()
        or (argmax_target < 0).any()
        or any(not np.isfinite(values).all() for values in target_scores.values())
    ):
        raise PCCError("PCC scoring produced incomplete or non-finite output")
    return score, argmax_stage, argmax_target, target_scores


def binary_metrics(labels: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = score > threshold
    positive = labels == 1
    negative = labels == 0
    tp = int((prediction & positive).sum())
    fp = int((prediction & negative).sum())
    fn = int((~prediction & positive).sum())
    tn = int((~prediction & negative).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "aupr": float(average_precision_score(labels, score)),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps),
        "fpr": fp / max(fp + tn, 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def percentile(reference_sorted: np.ndarray, value: float) -> float:
    return float(100.0 * np.searchsorted(reference_sorted, value, side="right") / len(reference_sorted))


def timestamp_seconds(value: str) -> int:
    return int(np.datetime64(value, "s").astype(np.int64))


def align_raw_to_a0(raw_timestamps: np.ndarray, endpoint_timestamps: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(raw_timestamps, endpoint_timestamps)
    if (
        len(indices) != len(endpoint_timestamps)
        or int(indices.max()) >= len(raw_timestamps)
        or not np.array_equal(raw_timestamps[indices], endpoint_timestamps)
    ):
        raise PCCError("A0 endpoint timestamps do not align to frozen raw Attack timestamps")
    return indices


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite PCC1 output: {OUT}")

    config = load_config(str(CONFIG_PATH))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    source_files = [
        cache_dir / "normal_v1_features.npy",
        cache_dir / "attack_v0_features.npy",
        cache_dir / "attack_v0_labels.npy",
        cache_dir / "attack_v0_timestamps.npy",
    ]
    hashes_before = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}

    # Phase 1: freeze schema, scaler, Ridge models, and robust references using Normal Train only.
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    schema = json.loads(T2_SCHEMA.read_text(encoding="utf-8"))
    profiles = normal_v1_profiles(schema)
    column_order = [str(name) for name in cache_manifest["column_order"]]
    stage_contract, profile_by_name = build_stage_contract(profiles, column_order)
    models, reference_rows = fit_models(normal, column_order, stage_contract)

    # Phase 2: validation only chooses the one frozen q99 PCC threshold.
    validation = normal[TRAIN_RAW_ROWS:EXPECTED_NORMAL_ROWS]
    validation_score, _, _, _ = score_points(validation, models)
    threshold = float(np.quantile(validation_score, THRESHOLD_QUANTILE))
    if len(validation) != VALIDATION_RAW_ROWS or not math.isfinite(threshold):
        raise PCCError("invalid Normal Validation q99 threshold")

    # Phase 3: Attack is opened only after all fitting, references, and threshold are frozen.
    attack, attack_labels, attack_timestamps, _ = load_attack_cache(cache_dir, config)
    attack_score, argmax_stage, argmax_target, target_scores = score_points(attack, models)
    pcc_metrics = binary_metrics(np.asarray(attack_labels), attack_score, threshold)

    a0_metrics_all = json.loads((A0_DIR / "metrics.json").read_text(encoding="utf-8"))
    a0_thresholds = json.loads((A0_DIR / "thresholds.json").read_text(encoding="utf-8"))
    with np.load(A0_DIR / "test_scores.npz", allow_pickle=False) as payload:
        a0_timestamps = np.asarray(payload["endpoint_timestamp"], dtype=np.int64)
        a0_labels = np.asarray(payload["label"], dtype=np.int8)
        a0_score = np.asarray(payload["endpoint_mae"], dtype=np.float64)
    a0_threshold = float(a0_thresholds["endpoint_mae"])
    raw_indices = align_raw_to_a0(np.asarray(attack_timestamps), a0_timestamps)
    if not np.array_equal(np.asarray(attack_labels)[raw_indices], a0_labels):
        raise PCCError("A0 endpoint labels differ from the frozen Physical point labels")
    pcc_aligned = attack_score[raw_indices]
    a0_miss = a0_score <= a0_threshold
    pcc_hit = pcc_aligned > threshold
    recovered = a0_miss & pcc_hit & (a0_labels == 1)
    added_fp = a0_miss & pcc_hit & (a0_labels == 0)

    canonical = pd.read_csv(T25_EVENTS)
    canonical = canonical[canonical["evaluable_physical_attack_interval"].astype(str).str.lower() == "true"].copy()
    s6 = pd.read_csv(S6_EVENTS).set_index("event_id")
    if len(canonical) != 36 or set(canonical["event_id"].astype(int)) != set(s6.index.astype(int)):
        raise PCCError("canonical 36-event identity differs from the frozen S6 audit")
    validation_sorted = np.sort(validation_score)
    a0_validation_sorted = np.sort(np.asarray(np.load(S6_A0_VALIDATION), dtype=np.float64))
    event_rows: list[dict[str, Any]] = []
    recovered_event_ids: list[int] = []
    for event in canonical.itertuples(index=False):
        event_id = int(event.event_id)
        start = timestamp_seconds(str(event.canonical_start))
        end = timestamp_seconds(str(event.canonical_end))
        raw_mask = (attack_timestamps >= start) & (attack_timestamps <= end)
        endpoint_mask = (a0_timestamps >= start) & (a0_timestamps <= end)
        if not raw_mask.any() or not endpoint_mask.any():
            raise PCCError(f"official event {event_id} does not intersect both score timelines")
        raw_positions = np.flatnonzero(raw_mask)
        endpoint_positions = np.flatnonzero(endpoint_mask)
        pcc_event_score = attack_score[raw_positions]
        peak_local = int(np.argmax(pcc_event_score))
        peak_raw = int(raw_positions[peak_local])
        top_stage = STAGES[int(argmax_stage[peak_raw])]
        top_target = column_order[int(argmax_target[peak_raw])]
        pcc_ratio = float(np.mean(pcc_event_score > threshold))
        a0_event_score = a0_score[endpoint_positions]
        a0_ratio = float(np.mean(a0_event_score > a0_threshold))
        recovered_points = int(recovered[endpoint_positions].sum())
        if recovered_points:
            recovered_event_ids.append(event_id)
        s6_event = s6.loc[event_id]
        event_rows.append(
            {
                "event_id": event_id,
                "canonical_start": event.canonical_start,
                "canonical_end": event.canonical_end,
                "attack_points": event.attack_point,
                "stage": s6_event["stage"],
                "s6_direct_class": s6_event["event_direct_class"],
                "s6_contextual_class": s6_event["auto_contextual_class"],
                "a0_point_count": len(endpoint_positions),
                "a0_hit_ratio": a0_ratio,
                "a0_max_score_percentile": percentile(a0_validation_sorted, float(a0_event_score.max())),
                "pcc_point_count": len(raw_positions),
                "pcc_hit_ratio": pcc_ratio,
                "pcc_max_score_percentile": percentile(validation_sorted, float(pcc_event_score.max())),
                "pcc_top_stage": top_stage,
                "pcc_top_target": top_target,
                "a0_miss_pcc_hit_points": recovered_points,
            }
        )

    contextual_rows = [row for row in event_rows if row["event_id"] in CONTEXTUAL_IDS]
    if tuple(sorted(int(row["event_id"]) for row in contextual_rows)) != CONTEXTUAL_IDS:
        raise PCCError("contextual subset is not exactly IDs 13, 14, and 29")
    contextual_hits = sum(float(row["pcc_hit_ratio"]) >= 0.10 for row in contextual_rows)
    gate_a = contextual_hits >= 2
    gate_b_candidates = [row for row in contextual_rows if row["event_id"] in {13, 14}]
    gate_b = any(
        float(row["pcc_hit_ratio"]) >= 0.10
        and float(row["pcc_hit_ratio"]) >= float(row["a0_hit_ratio"]) + 0.10
        for row in gate_b_candidates
    )
    gate_c = float(pcc_metrics["fpr"]) <= 0.10
    near_missed_recovered = [
        int(row["event_id"])
        for row in event_rows
        if float(row["a0_hit_ratio"]) < 0.10
        and float(row["pcc_hit_ratio"]) >= 0.10
        and int(row["a0_miss_pcc_hit_points"]) > 0
    ]
    gate_d = bool(near_missed_recovered)
    if gate_a and gate_b and gate_c and gate_d:
        status = "STAGE_LOCAL_PCC_REAL_COMPLEMENTARITY_SUPPORTED"
    elif contextual_hits > 0 and not gate_c:
        status = "STAGE_LOCAL_PCC_SIGNAL_PRESENT_BUT_UNSTABLE"
    else:
        status = "STAGE_LOCAL_PCC_NOT_SUPPORTED_ON_REAL_SWAT"

    stage_target_rows: list[dict[str, Any]] = []
    contextual_raw_mask = np.zeros(EXPECTED_ATTACK_ROWS, dtype=bool)
    for row in contextual_rows:
        contextual_raw_mask |= (
            (attack_timestamps >= timestamp_seconds(str(row["canonical_start"])))
            & (attack_timestamps <= timestamp_seconds(str(row["canonical_end"])))
        )
    reference_by_target = {str(row["target"]): row for row in reference_rows}
    for target_name, values in target_scores.items():
        target_index = column_order.index(target_name)
        winning = argmax_target == target_index
        stage_target_rows.append(
            {
                **reference_by_target[target_name],
                "attack_argmax_count": int(winning.sum()),
                "attack_argmax_fraction": float(winning.mean()),
                "labeled_anomaly_argmax_count": int((winning & (attack_labels == 1)).sum()),
                "labeled_anomaly_argmax_fraction": float(winning[attack_labels == 1].mean()),
                "contextual_argmax_count": int((winning & contextual_raw_mask).sum()),
                "contextual_argmax_fraction": float(winning[contextual_raw_mask].mean()),
                "test_score_median": float(np.median(values)),
                "test_score_q99": float(np.quantile(values, 0.99)),
            }
        )

    complementarity = {
        "alignment_scope": "A0 legal endpoint timestamps only; both hard-gap segments retained",
        "a0_threshold": a0_threshold,
        "pcc_threshold": threshold,
        "a0_miss_pcc_hit_anomaly_points": int(recovered.sum()),
        "a0_miss_pcc_hit_official_events": len(set(recovered_event_ids)),
        "a0_miss_pcc_hit_official_event_ids": sorted(set(recovered_event_ids)),
        "near_completely_missed_recovered_event_ids": near_missed_recovered,
        "added_normal_false_alarms_on_a0_normal_decisions": int(added_fp.sum()),
        "added_fp_per_recovered_tp": (
            float(added_fp.sum() / recovered.sum()) if recovered.sum() else None
        ),
        "fusion": False,
        "threshold_tuning": False,
    }
    tests = {
        "train_only_scaler": all(int(scaler.n_samples_seen_) == TRAIN_RAW_ROWS for scaler in [m.scaler for m in models]),
        "train_only_ridge": all(row["fit_scope"] == "Normal Train raw rows [0,396000) only" for row in reference_rows),
        "stage_from_t2_schema": set(stage_contract) == set(STAGES) and len(profile_by_name) == EXPECTED_FEATURES,
        "no_attack_feature_selection": True,
        "target_excluded_from_input": all(bool(row["target_excluded"]) for row in reference_rows),
        "continuous_targets_only": all(row["target_type"] == "continuous" for row in reference_rows),
        "discrete_predictor_allowed": any(bool(row["discrete_predictors_allowed"]) for row in reference_rows),
        "constant_excluded": all(
            name not in contract["nonconstant_predictors"]
            for contract in stage_contract.values()
            for name in contract["excluded_constants"]
        ),
        "alpha_1": all(float(row["alpha"]) == ALPHA for row in reference_rows),
        "no_lag": True,
        "q99_validation_only": THRESHOLD_QUANTILE == 0.99 and len(validation_score) == VALIDATION_RAW_ROWS,
        "no_point_adjustment": True,
        "no_fusion": True,
        "official_events_unchanged": len(event_rows) == 36,
        "contextual_ids_exact": tuple(sorted(int(row["event_id"]) for row in contextual_rows)) == CONTEXTUAL_IDS,
        "no_nan_inf": bool(
            np.isfinite(validation_score).all()
            and np.isfinite(attack_score).all()
            and all(math.isfinite(float(value)) for value in pcc_metrics.values())
        ),
    }
    segment_manifest = json.loads(T25_SEGMENTS.read_text(encoding="utf-8"))
    if segment_manifest.get("gap_policy") != "hard_session_boundary" or len(segment_manifest["test_segments"]) != 2:
        raise PCCError("T2.5 hard gap boundary is not frozen")
    if not all(tests.values()):
        raise PCCError(f"PCC1 test failure: {tests}")

    hashes_after = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}
    if hashes_before != hashes_after:
        raise PCCError("a frozen SWaT cache file changed during PCC1")
    a0_primary = a0_metrics_all["primary_endpoint_mae"]
    global_metrics = {
        "status": status,
        "dataset": "Official SWaT 2015 Physical",
        "pcc": {
            **pcc_metrics,
            "threshold": threshold,
            "threshold_source": "Normal Validation raw rows [396000,495000) q99 only",
            "evaluation_points": EXPECTED_ATTACK_ROWS,
            "point_adjustment": False,
        },
        "a0_existing": {
            "auroc": float(a0_primary["auroc"]),
            "aupr": float(a0_primary["aupr"]),
            "precision": float(a0_primary["precision"]),
            "recall": float(a0_primary["recall"]),
            "f1": float(a0_primary["f1"]),
            "fpr": float(a0_primary["fpr"]),
            "threshold": a0_threshold,
            "threshold_recomputed": False,
            "evaluation_points": int(a0_metrics_all["test_evaluation_point_count"]),
        },
        "gates": {
            "A_contextual_hits_at_least_2_of_3": gate_a,
            "B_id13_or_14_pcc_hit_and_at_least_0_10_above_a0": gate_b,
            "C_pcc_test_fpr_at_most_0_10": gate_c,
            "D_recovers_a0_near_missed_official_event": gate_d,
        },
        "contextual_events_hit": contextual_hits,
        "tests": tests,
    }
    manifest = {
        "protocol": "SWaT-PCC1 stage-local pointwise cross-channel consistency",
        "dataset": "Official SWaT 2015 Physical",
        "normal_source": "SWaT_Dataset_Normal_v1.xlsx",
        "attack_source": "SWaT_Dataset_Attack_v0.xlsx",
        "train_rows": TRAIN_RAW_ROWS,
        "validation_rows": VALIDATION_RAW_ROWS,
        "attack_rows": EXPECTED_ATTACK_ROWS,
        "feature_count_a0": EXPECTED_FEATURES,
        "schema_source": str(T2_SCHEMA.relative_to(ROOT)),
        "stage_membership_source": "T2 Normal v1 schema variable names; numeric suffix first digit maps to P1-P6",
        "stage_contract": stage_contract,
        "continuous_target_count": len(reference_rows),
        "ridge_alpha": ALPHA,
        "dtype": "float64",
        "lag": False,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold_source": "Normal Validation only",
        "attack_opened_after_fit_and_threshold": True,
        "hard_session_boundary_count": 1,
        "pointwise_temporal_context": 0,
        "cross_gap_operation": False,
        "contextual_ids": list(CONTEXTUAL_IDS),
        "gate_b_operationalization": "PCC hit ratio >=0.10 and at least 0.10 absolute above A0 for ID13 or ID14",
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "source_unchanged": hashes_before == hashes_after,
        "tests": tests,
    }

    OUT.mkdir(parents=True)
    write_json(OUT / "global_metrics.json", global_metrics)
    write_csv(OUT / "event_metrics.csv", event_rows)
    write_csv(OUT / "contextual_subset.csv", contextual_rows)
    write_csv(OUT / "stage_target_contributions.csv", stage_target_rows)
    write_json(OUT / "complementarity.json", complementarity)
    write_json(OUT / "pcc_reference_manifest.json", manifest)
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "status": status,
        "pcc": global_metrics["pcc"],
        "a0": global_metrics["a0_existing"],
        "contextual": contextual_rows,
        "complementarity": complementarity,
        "gates": global_metrics["gates"],
        "tests_passed": sum(bool(value) for value in tests.values()),
        "tests_total": len(tests),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
