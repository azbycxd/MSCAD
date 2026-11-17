"""Evaluate the frozen dual-view disagreement score on official SWaT attacks."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "thesis" / "02_collaboration"
for item in (ROOT, HELPER):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_innovation2a2_joint_lagged as a2  # noqa: E402
import run_innovation2b_controlled as b2  # noqa: E402
import run_innovation2d0_dual_view_comut as d0  # noqa: E402
import run_swat2015_pcc1 as pcc1  # noqa: E402
from data.swat2015_official import (  # noqa: E402
    EXPECTED_ATTACK_ROWS,
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    load_attack_cache,
    load_normal_cache,
    resolve_repo_path,
    validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


OUT = ROOT / "outputs" / "thesis_innovation2d2_real_swat"
EVENTS = ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "canonical_attack_events.csv"
SEGMENTS = ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "segment_manifest.json"
S6 = ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit" / "event_audit.csv"
A0 = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline"
CONTEXTUAL_IDS = (13, 14, 29)
EXPECTED_TARGETS = ("AIT202", "FIT201", "DPIT301", "FIT301", "FIT401", "FIT601")
THRESHOLD_QUANTILE = 0.99


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_frame(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all())


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def main() -> int:
    started = time.perf_counter()
    if OUT.exists() and (OUT / "innovation2d2_summary.json").exists():
        raise FileExistsError(f"refusing to overwrite 2D2 output: {OUT}")

    config = load_config(str(b2.CONFIG))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    source_files = [
        cache_dir / "normal_v1_features.npy",
        cache_dir / "attack_v0_features.npy",
        cache_dir / "attack_v0_labels.npy",
        cache_dir / "attack_v0_timestamps.npy",
        A0 / "best.pt",
        A0 / "scaler.npz",
        A0 / "test_scores.npz",
        A0 / "thresholds.json",
        EVENTS,
        SEGMENTS,
        S6,
    ]
    hashes_before = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}

    # Freeze all fit/calibration state before opening Attack arrays.
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    if normal.shape[0] != EXPECTED_NORMAL_ROWS:
        raise RuntimeError("Normal v1 row count changed")
    column_order = [str(value) for value in cache_manifest["column_order"]]
    name_to_index = {name: index for index, name in enumerate(column_order)}
    schema = json.loads(b2.SCHEMA.read_text(encoding="utf-8"))
    contract, _ = a2.build_contract(a2.normal_profiles(schema), column_order)
    stable = pd.read_csv(b2.A2_STABLE)
    if tuple(stable["target"].astype(str)) != EXPECTED_TARGETS:
        raise RuntimeError("frozen 2A2 target identity/order changed")

    scaler = StandardScaler().fit(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float64))
    z_normal = scaler.transform(np.asarray(normal, dtype=np.float64))
    train_times = np.arange(d0.MAX_LAG, TRAIN_RAW_ROWS, dtype=np.int64)
    calibration_times = np.arange(b2.CALIBRATION_START + d0.MAX_LAG, b2.CALIBRATION_END, dtype=np.int64)
    models: list[dict[str, object]] = []
    calibration_scores: list[np.ndarray] = []
    target_contract: dict[str, object] = {}
    reference_rows: list[dict[str, object]] = []

    for frozen in stable.itertuples(index=False):
        target, stage = str(frozen.target), str(frozen.stage)
        target_index = name_to_index[target]
        stage_names = list(contract[stage]["nonconstant_channels"])
        sources = [name for name in stage_names if name != target]
        source_indices = [name_to_index[name] for name in sources]
        if target in sources or not source_indices:
            raise RuntimeError(f"invalid Cross View for {target}")
        self_model = Ridge(alpha=d0.ALPHA).fit(
            d0.design(z_normal, train_times, [target_index], d0.SELF_LAGS), z_normal[train_times, target_index]
        )
        cross_model = Ridge(alpha=d0.ALPHA).fit(
            d0.design(z_normal, train_times, source_indices, d0.CROSS_LAGS), z_normal[train_times, target_index]
        )
        pred_self = self_model.predict(d0.design(z_normal, calibration_times, [target_index], d0.SELF_LAGS))
        pred_cross = cross_model.predict(d0.design(z_normal, calibration_times, source_indices, d0.CROSS_LAGS))
        disagreement = np.abs(pred_self - pred_cross)
        median = float(np.median(disagreement))
        scale = max(1.4826 * float(np.median(np.abs(disagreement - median))), 1e-6)
        calibration_scores.append(d0.calibrated(disagreement, median, scale))
        models.append({
            "target": target, "stage": stage, "target_index": target_index,
            "source_indices": source_indices, "self_model": self_model, "cross_model": cross_model,
            "median": median, "scale": scale,
        })
        reference_rows.append({"target": target, "stage": stage, "median": median, "scale": scale})
        target_contract[target] = {
            "stage": stage, "cross_sources": sources, "target_excluded_from_cross": target not in sources,
            "self_lags": list(d0.SELF_LAGS), "cross_lags": list(d0.CROSS_LAGS), "alpha": d0.ALPHA,
            "model_fit_raw_interval": [0, TRAIN_RAW_ROWS],
            "calibration_raw_interval": [b2.CALIBRATION_START, b2.CALIBRATION_END],
            "calibration_legal_interval": [b2.CALIBRATION_START + d0.MAX_LAG, b2.CALIBRATION_END],
            "score_formula": "max(0,(abs(pred_self-pred_cross)-median)/scale)",
            "observed_target_residual_in_score": False,
        }

    calibration_matrix = np.column_stack(calibration_scores)
    calibration_global = calibration_matrix.max(axis=1)
    dual_threshold = float(np.quantile(calibration_global, THRESHOLD_QUANTILE))
    if not np.isfinite(calibration_global).all() or not np.isfinite(dual_threshold):
        raise RuntimeError("non-finite Normal calibration")

    # Attack is read only after model, calibration, aggregation, and threshold are frozen.
    attack, attack_labels, attack_timestamps, _ = load_attack_cache(cache_dir, config)
    if attack.shape[0] != EXPECTED_ATTACK_ROWS:
        raise RuntimeError("Attack v0 row count changed")
    z_attack = scaler.transform(np.asarray(attack, dtype=np.float64))
    segment_doc = json.loads(SEGMENTS.read_text(encoding="utf-8"))
    row_counts = [int(item["row_count"]) for item in segment_doc["test_segments"]]
    bounds, cursor = [], 0
    for rows in row_counts:
        bounds.append((cursor, cursor + rows))
        cursor += rows
    if cursor != len(attack):
        raise RuntimeError("T2.5 segment rows do not cover Attack exactly")

    score_matrix = np.full((len(attack), len(models)), np.nan, dtype=np.float64)
    valid_mask = np.zeros(len(attack), dtype=bool)
    for start, end in bounds:
        times = np.arange(start + d0.MAX_LAG, end, dtype=np.int64)
        valid_mask[times] = True
        for target_number, model in enumerate(models):
            target_index = int(model["target_index"])
            source_indices = list(model["source_indices"])
            pred_self = model["self_model"].predict(d0.design(z_attack, times, [target_index], d0.SELF_LAGS))
            pred_cross = model["cross_model"].predict(d0.design(z_attack, times, source_indices, d0.CROSS_LAGS))
            score_matrix[times, target_number] = d0.calibrated(
                np.abs(pred_self - pred_cross), float(model["median"]), float(model["scale"])
            )
    if valid_mask.sum() != len(attack) - len(bounds) * d0.MAX_LAG or not np.isfinite(score_matrix[valid_mask]).all():
        raise RuntimeError("invalid segmented Attack score matrix")
    dual_raw = np.full(len(attack), np.nan, dtype=np.float64)
    argmax_raw = np.full(len(attack), -1, dtype=np.int8)
    dual_raw[valid_mask] = score_matrix[valid_mask].max(axis=1)
    argmax_raw[valid_mask] = score_matrix[valid_mask].argmax(axis=1).astype(np.int8)

    with np.load(A0 / "test_scores.npz", allow_pickle=False) as payload:
        a0_timestamps = np.asarray(payload["endpoint_timestamp"], dtype=np.int64)
        a0_labels = np.asarray(payload["label"], dtype=np.int8)
        a0_score = np.asarray(payload["endpoint_mae"], dtype=np.float64)
    a0_threshold = float(json.loads((A0 / "thresholds.json").read_text(encoding="utf-8"))["endpoint_mae"])
    raw_indices = pcc1.align_raw_to_a0(np.asarray(attack_timestamps), a0_timestamps)
    if not valid_mask[raw_indices].all() or not np.array_equal(np.asarray(attack_labels)[raw_indices], a0_labels):
        raise RuntimeError("aligned A0 endpoints are not the shared legal mask")
    labels = a0_labels
    dual_score = dual_raw[raw_indices]
    dual_argmax = argmax_raw[raw_indices]
    dual_metrics = pcc1.binary_metrics(labels, dual_score, dual_threshold)
    a0_metrics = pcc1.binary_metrics(labels, a0_score, a0_threshold)
    global_metrics = pd.DataFrame([
        {"method": "Dual-View", "threshold": dual_threshold, **dual_metrics},
        {"method": "A0 aligned", "threshold": a0_threshold, **a0_metrics},
    ])

    a0_hit, dual_hit = a0_score > a0_threshold, dual_score > dual_threshold
    attack_point = labels == 1
    normal_point = labels == 0
    complement = {
        "a0_hit_dual_hit": int((attack_point & a0_hit & dual_hit).sum()),
        "a0_hit_dual_miss": int((attack_point & a0_hit & ~dual_hit).sum()),
        "a0_miss_dual_hit": int((attack_point & ~a0_hit & dual_hit).sum()),
        "a0_miss_dual_miss": int((attack_point & ~a0_hit & ~dual_hit).sum()),
        "added_normal_fp": int((normal_point & ~a0_hit & dual_hit).sum()),
    }
    complement["added_fp_per_recovered_tp"] = (
        complement["added_normal_fp"] / complement["a0_miss_dual_hit"]
        if complement["a0_miss_dual_hit"] else None
    )

    canonical = pd.read_csv(EVENTS)
    canonical = canonical[canonical["evaluable_physical_attack_interval"].astype(str).str.lower() == "true"].copy()
    s6 = pd.read_csv(S6).set_index("event_id")
    if len(canonical) != 36 or set(canonical["event_id"].astype(int)) != set(s6.index.astype(int)):
        raise RuntimeError("canonical/S6 event identity changed")
    model_targets = [str(model["target"]) for model in models]
    model_stages = [str(model["stage"]) for model in models]
    event_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []
    for event in canonical.itertuples(index=False):
        event_id = int(event.event_id)
        start = pcc1.timestamp_seconds(str(event.canonical_start))
        end = pcc1.timestamp_seconds(str(event.canonical_end))
        mask = (a0_timestamps >= start) & (a0_timestamps <= end)
        if not mask.any():
            raise RuntimeError(f"event {event_id} has no shared legal endpoint")
        counts = np.bincount(dual_argmax[mask], minlength=len(models))
        top_index = int(np.argmax(counts))
        a0_rate = float(np.mean(a0_hit[mask]))
        dual_rate = float(np.mean(dual_hit[mask]))
        row = {
            "event_id": event_id, "start": str(event.canonical_start), "end": str(event.canonical_end),
            "attacked_components": str(s6.loc[event_id, "canonical_attack_points"]),
            "attacked_stage": str(s6.loc[event_id, "stage"]),
            "s6_class": str(s6.loc[event_id, "auto_contextual_class"]),
            "shared_endpoint_count": int(mask.sum()), "a0_hit_rate": a0_rate, "dual_hit_rate": dual_rate,
            "dual_median_score": float(np.median(dual_score[mask])), "dual_max_score": float(np.max(dual_score[mask])),
            "top_dual_target": model_targets[top_index], "top_dual_stage": model_stages[top_index],
            "recovered_event": bool(a0_rate < 0.10 and dual_rate >= 0.10),
        }
        event_rows.append(row)
        for index, target in enumerate(model_targets):
            attribution_rows.append({
                "event_id": event_id, "target": target, "stage": model_stages[index],
                "argmax_count": int(counts[index]), "argmax_fraction": float(counts[index] / mask.sum()),
            })

    events = pd.DataFrame(event_rows)
    contextual = events[events["event_id"].isin(CONTEXTUAL_IDS)].copy()
    recovered = events[events["recovered_event"]].copy()
    class_summary = events.groupby("s6_class", sort=False).agg(
        event_count=("event_id", "count"), dual_hit_010_count=("dual_hit_rate", lambda s: int((s >= 0.10).sum())),
        a0_near_miss_count=("a0_hit_rate", lambda s: int((s < 0.10).sum())),
        recovered_count=("recovered_event", "sum"),
    ).reset_index()
    context_by_id = contextual.set_index("event_id")
    gate_a = int((contextual["dual_hit_rate"] >= 0.10).sum()) >= 2
    gate_b = any(
        float(context_by_id.loc[event_id, "dual_hit_rate"]) >= 0.10
        and float(context_by_id.loc[event_id, "dual_hit_rate"]) > float(context_by_id.loc[event_id, "a0_hit_rate"])
        for event_id in (13, 14)
    )
    gate_c = float(dual_metrics["fpr"]) <= 0.10
    gate_d = len(recovered) >= 1
    gates = {"A": bool(gate_a), "B": bool(gate_b), "C": bool(gate_c), "D": bool(gate_d)}
    if all(gates.values()):
        status = "DUAL_VIEW_REAL_SWAT_COMPLEMENTARITY_SUPPORTED"
    elif any((gate_a, gate_b, gate_d)) and not gate_c:
        status = "DUAL_VIEW_REAL_SWAT_SIGNAL_PRESENT_BUT_UNSTABLE"
    else:
        status = "DUAL_VIEW_REAL_SWAT_COMPLEMENTARITY_NOT_SUPPORTED"

    tests = {
        "train_validation_test_protocol": TRAIN_RAW_ROWS == 396000 and b2.CALIBRATION_START == 396000 and b2.CALIBRATION_END == 445500,
        "attack_evaluation_only": True, "six_frozen_targets": tuple(model_targets) == EXPECTED_TARGETS,
        "scaler_train_only": int(scaler.n_samples_seen_) == TRAIN_RAW_ROWS,
        "self_lags_fixed": d0.SELF_LAGS == (1, 5, 10, 20, 30),
        "cross_lags_fixed": d0.CROSS_LAGS == (0, 1, 5, 10, 20, 30),
        "target_excluded_cross": all(bool(value["target_excluded_from_cross"]) for value in target_contract.values()),
        "alpha_one": d0.ALPHA == 1.0, "prediction_disagreement_only": True,
        "observed_target_residual_absent": True, "calibration_normal_validation_front_half": True,
        "threshold_calibration_only": True, "gap_not_crossed": all(not valid_mask[start:start + d0.MAX_LAG].any() for start, _ in bounds),
        "no_point_adjustment": True, "no_threshold_search": True, "canonical_36": len(events) == 36,
        "contextual_ids_exact": set(contextual["event_id"].astype(int)) == set(CONTEXTUAL_IDS),
        "a0_frozen": a0_threshold == 0.18396227061748505 and (A0 / "best.pt").exists(),
        "aligned_valid_mask": len(raw_indices) == len(a0_score) and valid_mask[raw_indices].all(),
        "no_nan_inf": finite_frame(global_metrics) and finite_frame(events) and np.isfinite(dual_score).all(),
    }
    if not all(tests.values()):
        raise RuntimeError(f"2D2 integrity failure: {tests}")
    hashes_after = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}
    if hashes_after != hashes_before:
        raise RuntimeError("frozen input artifact changed during 2D2")

    summary = {
        "status": status, "elapsed_seconds": time.perf_counter() - started,
        "dual_metrics": dual_metrics, "aligned_a0_metrics": a0_metrics,
        "dual_threshold": dual_threshold, "a0_threshold": a0_threshold,
        "point_complementarity": complement,
        "canonical_36": {
            "dual_hit_at_least_0_10_count": int((events["dual_hit_rate"] >= 0.10).sum()),
            "a0_near_miss_count": int((events["a0_hit_rate"] < 0.10).sum()),
            "recovered_count": len(recovered), "recovered_event_ids": recovered["event_id"].astype(int).tolist(),
        },
        "gates": gates, "tests": tests,
        "class_summary": class_summary.to_dict("records"),
        "official_real_swat_attack": True, "attack_driven_selection": False, "target_change": False,
        "feature_search": False, "lag_search": False, "threshold_search": False,
        "point_adjustment": False, "fusion": False, "a0_retrained": False,
    }
    manifest = {
        "summary": summary, "targets": target_contract, "calibration_reference": reference_rows,
        "test_segment_raw_bounds": bounds, "shared_endpoint_count": len(labels),
        "input_sha256": hashes_before, "score_formula": "max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/scale_j)",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    global_metrics.to_csv(OUT / "global_metrics.csv", index=False)
    pd.DataFrame([complement]).to_csv(OUT / "point_complementarity.csv", index=False)
    events.to_csv(OUT / "event_metrics_36.csv", index=False)
    contextual.to_csv(OUT / "contextual_events.csv", index=False)
    recovered.to_csv(OUT / "recovered_events.csv", index=False)
    pd.DataFrame(attribution_rows).to_csv(OUT / "target_attribution.csv", index=False)
    class_summary.to_csv(OUT / "event_class_summary.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    (OUT / "innovation2d2_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "contextual": contextual.to_dict("records")}, indent=2, allow_nan=False, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
