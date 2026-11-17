"""S6 phenomenon audit of official SWaT 2015 Physical attack events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    EXPECTED_NORMAL_ROWS, TRAIN_RAW_ROWS, WINDOW, OfficialSWaTError,
    load_attack_cache, load_normal_cache, load_scaler, resolve_repo_path, validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


EXCLUDED_IDS = {5, 9, 12, 15, 18}
LAGS = np.asarray([1, 5, 10, 20, 30], dtype=np.int64)
ALPHA = 1.0
FIT_STRIDE = 4
TOP_PREDICTOR_COUNT = 5
ROBUST_FLOOR = 1e-6
EPS = 1e-12
LEVEL_FRACTION_STRONG = 0.20
LEVEL_FRACTION_MEDIUM = 0.05
STATE_STRONG_FREQUENCY = 0.005
STATE_MEDIUM_FREQUENCY = 0.02
TRANSITION_MEDIUM_PROBABILITY = 0.01
UNMAPPED_INTERVAL = ("2015-12-29 06:30:00", "2015-12-29 06:42:00")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def normal_v1_profiles(audit: dict[str, Any]) -> list[dict[str, Any]]:
    def visit(value: Any) -> list[dict[str, Any]] | None:
        if isinstance(value, dict):
            if value.get("file_name") == "SWaT_Dataset_Normal_v1.xlsx":
                result = value.get("variable_profiles") or value.get("schema", {}).get("variable_profiles")
                if result:
                    return result
            for child in value.values():
                result = visit(child)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = visit(child)
                if result is not None:
                    return result
        return None
    result = visit(audit)
    if result is None or len(result) != 51:
        raise OfficialSWaTError("Normal v1 schema profile missing")
    return result


def canonical_attack_point(raw: str) -> str:
    return re.sub(r"[-\s]", "", raw).upper()


def split_attack_points(raw: str) -> list[str]:
    return [token.strip() for token in re.split(r"[,;]", raw) if token.strip()]


def stage_of(point: str) -> str:
    match = re.search(r"(\d{3})$", point)
    return f"P{match.group(1)[0]}" if match else "UNKNOWN"


def robust_reference(values: np.ndarray, *, source_split: str) -> tuple[float, float]:
    if source_split != "Normal Train":
        raise OfficialSWaTError("robust statistics must use Normal Train only")
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    scale = max(1.4826 * float(np.median(np.abs(values - center))), ROBUST_FLOOR)
    return center, scale


def calibration_q995(values: np.ndarray, *, source_split: str) -> float:
    if source_split != "Normal Validation":
        raise OfficialSWaTError("q995 calibration must use Normal Validation only")
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.995))


def direct_class_continuous(level: float, delta: float, self_fraction: float) -> str:
    peak = max(level, delta, self_fraction)
    if peak >= LEVEL_FRACTION_STRONG:
        return "STRONG_DIRECT"
    if peak >= LEVEL_FRACTION_MEDIUM:
        return "MEDIUM_DIRECT"
    return "WEAK_DIRECT"


def direct_class_discrete(unseen: bool, frequency: float, dwell: float, dwell_q995: float, transition_probability: float) -> str:
    if unseen or frequency < STATE_STRONG_FREQUENCY or dwell > dwell_q995:
        return "STRONG_DIRECT"
    if transition_probability < TRANSITION_MEDIUM_PROBABILITY or frequency < STATE_MEDIUM_FREQUENCY:
        return "MEDIUM_DIRECT"
    return "WEAK_DIRECT"


def event_direct_class(classes: Sequence[str]) -> str:
    if "STRONG_DIRECT" in classes:
        return "STRONG_DIRECT"
    if "MEDIUM_DIRECT" in classes:
        return "MEDIUM_DIRECT"
    if "WEAK_DIRECT" in classes:
        return "WEAK_DIRECT"
    return "UNRESOLVED_MAPPING"


def contextual_class(direct: str, mechanism: str, common_actuator_state: bool) -> str:
    relevant = mechanism in {"SENSOR_SPOOFING", "MULTI_POINT_COORDINATED"} or (
        mechanism == "ACTUATOR_MANIPULATION" and common_actuator_state
    )
    if direct == "STRONG_DIRECT":
        return "DIRECT_DOMINATED"
    if direct == "WEAK_DIRECT" and relevant:
        return "HIGH_CONTEXTUAL_CANDIDATE"
    if direct == "MEDIUM_DIRECT" and relevant:
        return "MIXED_CONTEXTUAL"
    return "UNCLEAR"


def self_history(values: np.ndarray, times: np.ndarray, channel: int) -> np.ndarray:
    return np.column_stack([values[times - lag, channel] for lag in LAGS]).astype(np.float64, copy=False)


def self_residual(values: np.ndarray, channel: int, coefficient: np.ndarray, intercept: float, segments: Sequence[tuple[int, int]]) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for start, end in segments:
        times = np.arange(start + int(LAGS.max()), end, dtype=np.int64)
        prediction = self_history(values, times, channel) @ coefficient + intercept
        result[times] = np.abs(values[times, channel].astype(np.float64) - prediction)
    return result


def dwell_lengths(values: np.ndarray) -> dict[float, np.ndarray]:
    by_state: dict[float, list[int]] = {}
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            by_state.setdefault(float(values[start]), []).append(index - start)
            start = index
    return {state: np.asarray(lengths, dtype=np.int64) for state, lengths in by_state.items()}


def max_run(values: np.ndarray, state: float) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value == state else 0
        best = max(best, current)
    return best


def discrete_reference(train: np.ndarray) -> dict[str, Any]:
    counts = Counter(float(v) for v in train)
    frequencies = {state: count / len(train) for state, count in counts.items()}
    pair_counts = Counter((float(a), float(b)) for a, b in zip(train[:-1], train[1:]))
    previous_counts = Counter(float(v) for v in train[:-1])
    probabilities = {pair: count / previous_counts[pair[0]] for pair, count in pair_counts.items()}
    dwell = dwell_lengths(train)
    dwell_q995 = {state: float(np.quantile(lengths, .995)) for state, lengths in dwell.items()}
    return {"frequencies": frequencies, "transition_probabilities": probabilities, "dwell_q995": dwell_q995, "dwell_lengths": dwell}


def discrete_evidence(values: np.ndarray, indices: np.ndarray, reference: dict[str, Any]) -> dict[str, Any]:
    event_values = np.asarray(values[indices], dtype=np.float64)
    counts = Counter(float(v) for v in event_values)
    attacked_state = sorted(counts, key=lambda state: (-counts[state], state))[0]
    frequency = float(reference["frequencies"].get(attacked_state, 0.0))
    pairs = [(float(values[i - 1]), float(values[i])) for i in indices if i > 0]
    probabilities = [float(reference["transition_probabilities"].get(pair, 0.0)) for pair in pairs]
    unseen = any(probability == 0.0 for probability in probabilities)
    minimum_probability = min(probabilities) if probabilities else 1.0
    surprise = max((-math.log(probability + EPS) for probability in probabilities), default=0.0)
    dwell = float(max_run(event_values, attacked_state))
    dwell_q995 = float(reference["dwell_q995"].get(attacked_state, 0.0))
    normal_dwell = reference["dwell_lengths"].get(attacked_state, np.asarray([], dtype=np.int64))
    percentile = 100.0 if not len(normal_dwell) else float(100.0 * np.mean(normal_dwell <= dwell))
    return {
        "attacked_state": attacked_state, "attacked_state_frequency": frequency,
        "minimum_transition_probability": minimum_probability, "transition_surprise": surprise,
        "unseen_transition": unseen, "event_max_dwell": dwell, "normal_dwell_q995": dwell_q995,
        "dwell_percentile_proxy": percentile,
        "direct_class": direct_class_discrete(unseen, frequency, dwell, dwell_q995, minimum_probability),
    }


def mechanism_for(raw_points: Sequence[str], mapped_profiles: Sequence[dict[str, Any]]) -> str:
    if len(raw_points) >= 2:
        return "MULTI_POINT_COORDINATED"
    if mapped_profiles:
        semantic = mapped_profiles[0]["semantic_type"]
        return "SENSOR_SPOOFING" if semantic == "sensor" else "ACTUATOR_MANIPULATION" if semantic == "actuator" else "OTHER"
    prefix = re.match(r"[A-Z]+", canonical_attack_point(raw_points[0]))
    if prefix and prefix.group() in {"MV", "P", "UV"}:
        return "ACTUATOR_MANIPULATION"
    if prefix and prefix.group() in {"AIT", "FIT", "LIT", "DPIT"}:
        return "SENSOR_SPOOFING"
    return "OTHER"


def physical_note(mechanism: str, points: str, expected: str) -> str:
    if not expected:
        return "INSUFFICIENT_OFFICIAL_SEMANTICS"
    if mechanism == "SENSOR_SPOOFING":
        return f"Official metadata links manipulated measurement {points} to expected impact: {expected}"
    if mechanism == "ACTUATOR_MANIPULATION":
        return f"Official metadata links forced actuator state {points} to expected impact: {expected}"
    if mechanism == "MULTI_POINT_COORDINATED":
        return f"Official metadata links coordinated points {points} to expected impact: {expected}"
    return "INSUFFICIENT_OFFICIAL_SEMANTICS"


def percentile(reference_sorted: np.ndarray, value: float) -> float:
    return float(100.0 * np.searchsorted(reference_sorted, value, side="right") / len(reference_sorted))


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(str(config_path)); validate_config_contract(config)
    output_dir = output_dir.resolve()
    allowed_prefiles = {"a0_validation_scores.npy", "a0_validation_score_manifest.json"}
    if output_dir.exists() and any(path.is_file() and path.name not in allowed_prefiles for path in output_dir.rglob("*")):
        raise FileExistsError(f"refusing to overwrite S6 output: {output_dir}")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, manifest = load_normal_cache(cache_dir, config)
    attack, attack_labels, attack_timestamps, _ = load_attack_cache(cache_dir, config)
    feature_names = list(manifest["column_order"]); feature_index = {name: i for i, name in enumerate(feature_names)}
    audit = json.loads(resolve_repo_path(ROOT, config["paths"]["t2_audit_json"]).read_text(encoding="utf-8"))
    profiles = normal_v1_profiles(audit); profile_by_name = {row["name"]: row for row in profiles}
    event_path = ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "canonical_attack_events.csv"
    with event_path.open(encoding="utf-8") as handle:
        raw_event_rows = list(csv.DictReader(handle))
    excluded = {int(row["attack_id"]) for row in raw_event_rows if row["no_physical_impact_marker"] == "True"}
    events = [row for row in raw_event_rows if row["evaluable_physical_attack_interval"] == "True"]
    if excluded != EXCLUDED_IDS or len(events) != 36:
        raise OfficialSWaTError("T2.5 event protocol changed")
    if any(not row["canonical_start"].startswith("2016-") for row in events if 37 <= int(row["attack_id"]) <= 41):
        raise OfficialSWaTError("canonical date correction for IDs 37-41 missing")

    scaler, scaler_meta = load_scaler(resolve_repo_path(ROOT, config["paths"]["a0_output"]) / "scaler.npz")
    if scaler_meta["fit_row_count"] != TRAIN_RAW_ROWS or scaler_meta["attack_rows_seen"] != 0:
        raise OfficialSWaTError("S6 scaler is not Train-only")
    train = np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float64)
    validation = np.asarray(normal[TRAIN_RAW_ROWS:EXPECTED_NORMAL_ROWS], dtype=np.float64)
    scaled_train = scaler.transform(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float32)).astype(np.float32, copy=False)
    scaled_validation = scaler.transform(np.asarray(normal[TRAIN_RAW_ROWS:EXPECTED_NORMAL_ROWS], dtype=np.float32)).astype(np.float32, copy=False)
    scaled_attack = scaler.transform(np.asarray(attack, dtype=np.float32)).astype(np.float32, copy=False)

    segment_manifest = json.loads(resolve_repo_path(ROOT, config["paths"]["t25_segment_manifest"]).read_text(encoding="utf-8"))
    segment_bounds = []
    for segment in segment_manifest["test_segments"]:
        start = int(np.flatnonzero(attack_timestamps == int(np.datetime64(segment["start"], "s").astype(np.int64)))[0])
        end = int(np.flatnonzero(attack_timestamps == int(np.datetime64(segment["end_inclusive"], "s").astype(np.int64)))[0]) + 1
        segment_bounds.append((start, end))

    v3i_manifest = json.loads((ROOT / "outputs" / "thesis_v3i_innovation" / "v3i1_predictor_manifest.json").read_text(encoding="utf-8"))
    continuous_names = set(v3i_manifest["continuous_targets"])
    continuous_refs: dict[str, dict[str, Any]] = {}; attack_self_z: dict[str, np.ndarray] = {}
    for name in continuous_names:
        channel = feature_index[name]
        center_x, scale_x = robust_reference(train[:, channel], source_split="Normal Train")
        val_level_z = np.abs(validation[:, channel] - center_x) / scale_x
        train_dx = np.diff(train[:, channel]); val_dx = np.diff(validation[:, channel])
        center_dx, scale_dx = robust_reference(train_dx, source_split="Normal Train")
        val_delta_z = np.abs(val_dx - center_dx) / scale_dx
        model = v3i_manifest["self_models"][name]
        coefficient = np.asarray(model["coefficient"], dtype=np.float64); intercept = float(model["intercept"])
        train_self = self_residual(scaled_train, channel, coefficient, intercept, [(0, len(scaled_train))])
        val_self = self_residual(scaled_validation, channel, coefficient, intercept, [(0, len(scaled_validation))])
        attack_self = self_residual(scaled_attack, channel, coefficient, intercept, segment_bounds)
        center_self, scale_self = robust_reference(train_self[np.isfinite(train_self)], source_split="Normal Train")
        val_self_z = np.abs(val_self[np.isfinite(val_self)] - center_self) / scale_self
        full_attack_self_z = np.full(len(attack), np.nan, dtype=np.float64)
        valid = np.isfinite(attack_self); full_attack_self_z[valid] = np.abs(attack_self[valid] - center_self) / scale_self
        attack_self_z[name] = full_attack_self_z
        continuous_refs[name] = {
            "median_x": center_x, "scale_x": scale_x, "median_dx": center_dx, "scale_dx": scale_dx,
            "self_center": center_self, "self_scale": scale_self,
            "q995_level": calibration_q995(val_level_z, source_split="Normal Validation"),
            "q995_delta": calibration_q995(val_delta_z, source_split="Normal Validation"),
            "q995_self": calibration_q995(val_self_z, source_split="Normal Validation"),
            "normal_reference_low": float(np.quantile(train[:, channel], .005)),
            "normal_reference_high": float(np.quantile(train[:, channel], .995)),
        }
    discrete_refs = {name: discrete_reference(train[:, feature_index[name]]) for name in feature_names if name not in continuous_names}

    checkpoint_path = resolve_repo_path(ROOT, config["paths"]["a0_output"]) / "best.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    a0_score_manifest = json.loads((output_dir / "a0_validation_score_manifest.json").read_text(encoding="utf-8"))
    validation_a0 = np.load(output_dir / "a0_validation_scores.npy")
    if a0_score_manifest["checkpoint_sha256"] != checkpoint_hash or a0_score_manifest["checkpoint_epoch"] != 13 or a0_score_manifest["parameter_count"] != 11730:
        raise OfficialSWaTError("precomputed A0 Validation scores do not match frozen checkpoint")
    if len(validation_a0) != 99000 - WINDOW + 1 or not np.isfinite(validation_a0).all():
        raise OfficialSWaTError("precomputed A0 Validation scores are invalid")
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise OfficialSWaTError("A0 checkpoint changed")
    validation_a0_sorted = np.sort(validation_a0); a0_threshold = float(np.quantile(validation_a0, .99))
    test_scores = np.load(resolve_repo_path(ROOT, config["paths"]["a0_output"]) / "test_scores.npz")
    test_times = test_scores["endpoint_timestamp"]; test_a0 = test_scores["endpoint_mae"]

    variable_rows: list[dict[str, Any]] = []; event_rows: list[dict[str, Any]] = []
    for event in events:
        event_id = int(event["event_id"]); start = int(np.datetime64(event["canonical_start"], "s").astype(np.int64)); end = int(np.datetime64(event["canonical_end"], "s").astype(np.int64))
        indices = np.flatnonzero((attack_timestamps >= start) & (attack_timestamps <= end))
        if not len(indices):
            raise OfficialSWaTError(f"event {event_id} absent from Attack timeline")
        raw_points = split_attack_points(event["attack_point"])
        canonical_points = [canonical_attack_point(value) for value in raw_points]
        mapped_profiles = [profile_by_name[value] for value in canonical_points if value in profile_by_name]
        mechanism = mechanism_for(raw_points, mapped_profiles)
        per_event: list[dict[str, Any]] = []
        for raw_point, point in zip(raw_points, canonical_points):
            base = {
                "event_id": event_id, "raw_attack_point": raw_point, "canonical_attack_point": point,
                "mapping_status": "MAPPED" if point in feature_index else "MAPPING_ERROR",
                "channel_type": "", "semantic_type": "", "stage": stage_of(point), "analysis_kind": "",
                "fraction_level_above_q995": "", "fraction_delta_above_q995": "", "fraction_self_above_q995": "",
                "max_z_level": "", "median_z_level": "", "max_z_delta": "", "median_z_delta": "",
                "max_self_z": "", "median_self_z": "", "attacked_state": "", "attacked_state_frequency": "",
                "transition_surprise": "", "minimum_transition_probability": "", "unseen_transition": "",
                "event_max_dwell": "", "normal_dwell_q995": "", "dwell_percentile_proxy": "", "direct_class": "UNRESOLVED_MAPPING",
            }
            if point in feature_index:
                channel = feature_index[point]; profile = profile_by_name[point]
                base.update({"channel_type": profile["observed_value_kind"], "semantic_type": profile["semantic_type"]})
                if point in continuous_names:
                    reference = continuous_refs[point]
                    level_z = np.abs(np.asarray(attack[indices, channel], dtype=np.float64) - reference["median_x"]) / reference["scale_x"]
                    delta_z = np.abs(np.asarray(attack[indices, channel], dtype=np.float64) - np.asarray(attack[indices - 1, channel], dtype=np.float64) - reference["median_dx"]) / reference["scale_dx"]
                    self_z = attack_self_z[point][indices]; self_z = self_z[np.isfinite(self_z)]
                    lf = float(np.mean(level_z > reference["q995_level"])); df = float(np.mean(delta_z > reference["q995_delta"])); sf = float(np.mean(self_z > reference["q995_self"]))
                    base.update({
                        "analysis_kind": "continuous_sensor", "fraction_level_above_q995": lf,
                        "fraction_delta_above_q995": df, "fraction_self_above_q995": sf,
                        "max_z_level": float(np.max(level_z)), "median_z_level": float(np.median(level_z)),
                        "max_z_delta": float(np.max(delta_z)), "median_z_delta": float(np.median(delta_z)),
                        "max_self_z": float(np.max(self_z)), "median_self_z": float(np.median(self_z)),
                        "direct_class": direct_class_continuous(lf, df, sf),
                    })
                else:
                    evidence = discrete_evidence(np.asarray(attack[:, channel]), indices, discrete_refs[point])
                    base.update({"analysis_kind": "discrete_state", **evidence})
            per_event.append(base); variable_rows.append(base)
        direct = event_direct_class([row["direct_class"] for row in per_event])
        common_actuator = any(row.get("analysis_kind") == "discrete_state" and float(row.get("attacked_state_frequency") or 0.0) >= STATE_MEDIUM_FREQUENCY for row in per_event)
        contextual = contextual_class(direct, mechanism, common_actuator)
        event_score_mask = (test_times >= start) & (test_times <= end); scores = np.asarray(test_a0[event_score_mask], dtype=np.float64)
        if not len(scores):
            raise OfficialSWaTError(f"event {event_id} has no legal A0 endpoint")
        mapped_points = [row for row in per_event if row["mapping_status"] == "MAPPED"]
        evidence_text = "; ".join(f"{row['canonical_attack_point']}:{row['direct_class']}" for row in per_event)
        event_rows.append({
            "event_id": event_id, "start": event["canonical_start"], "end": event["canonical_end"],
            "raw_attack_points": event["attack_point"], "canonical_attack_points": ";".join(canonical_points),
            "mapping_errors": ";".join(row["canonical_attack_point"] for row in per_event if row["mapping_status"] == "MAPPING_ERROR"),
            "stage": ";".join(sorted({stage_of(point) for point in canonical_points})), "attack_mechanism": mechanism,
            "start_state": event["start_state"], "official_change": event["description"], "actual_change": event["actual_change"],
            "expected_impact": event["expected_impact_or_intent"], "unexpected_outcome": event["unexpected_outcome"],
            "physical_consistency_note": physical_note(mechanism, event["attack_point"], event["expected_impact_or_intent"]),
            "per_variable_direct_evidence": evidence_text, "event_direct_class": direct,
            "event_score_median": float(np.median(scores)), "event_score_max": float(np.max(scores)),
            "median_percentile": percentile(validation_a0_sorted, float(np.median(scores))),
            "max_percentile": percentile(validation_a0_sorted, float(np.max(scores))),
            "formal_q99_hit_ratio": float(np.mean(scores > a0_threshold)), "auto_contextual_class": contextual,
            "physical_impact": True, "mapped_variable_count": len(mapped_points),
        })

    if not all(np.isfinite(float(row[key])) for row in event_rows for key in ("event_score_median", "event_score_max", "median_percentile", "max_percentile", "formal_q99_hit_ratio")):
        raise OfficialSWaTError("non-finite event metrics")
    direct_counts = Counter(row["event_direct_class"] for row in event_rows); contextual_counts = Counter(row["auto_contextual_class"] for row in event_rows)
    high_count = contextual_counts["HIGH_CONTEXTUAL_CANDIDATE"]
    status = "SWAT_HAS_MEANINGFUL_CONTEXTUAL_ATTACK_SUBSET" if high_count >= 8 else "SWAT_CONTEXTUAL_ATTACKS_ARE_NARROW" if high_count >= 3 else "SWAT_POORLY_ALIGNED_WITH_CONTEXTUAL_ANOMALY_OBJECTIVE"
    stage_rows = []
    for stage in [f"P{i}" for i in range(1, 7)]:
        selected = [row for row in event_rows if stage in row["stage"].split(";")]
        stage_rows.append({
            "stage": stage, "event_count": len(selected),
            "direct_dominated_count": sum(row["auto_contextual_class"] == "DIRECT_DOMINATED" for row in selected),
            "high_contextual_candidate_count": sum(row["auto_contextual_class"] == "HIGH_CONTEXTUAL_CANDIDATE" for row in selected),
            "mixed_count": sum(row["auto_contextual_class"] == "MIXED_CONTEXTUAL" for row in selected),
            "unclear_count": sum(row["auto_contextual_class"] == "UNCLEAR" for row in selected),
        })

    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_csv("event_audit.csv", event_rows); write_csv("attacked_variable_audit.csv", variable_rows); write_csv("stage_summary.csv", stage_rows)
    review_rows = [{
        "event_id": row["event_id"], "time": f"{row['start']}--{row['end']}", "attack_points": row["canonical_attack_points"],
        "stage": row["stage"], "mechanism": row["attack_mechanism"], "official_change": row["official_change"],
        "expected_impact": row["expected_impact"], "direct_class": row["event_direct_class"],
        "level_evidence": ";".join(f"{v['canonical_attack_point']}:{v['fraction_level_above_q995']}" for v in variable_rows if v["event_id"] == row["event_id"] and v["fraction_level_above_q995"] != ""),
        "delta_evidence": ";".join(f"{v['canonical_attack_point']}:{v['fraction_delta_above_q995']}" for v in variable_rows if v["event_id"] == row["event_id"] and v["fraction_delta_above_q995"] != ""),
        "self_evidence": ";".join(f"{v['canonical_attack_point']}:{v['fraction_self_above_q995']}" for v in variable_rows if v["event_id"] == row["event_id"] and v["fraction_self_above_q995"] != ""),
        "A0_evidence": f"median_pct={row['median_percentile']:.3f};max_pct={row['max_percentile']:.3f};q99_hit={row['formal_q99_hit_ratio']:.6f}",
        "auto_contextual_class": row["auto_contextual_class"], "review_note": "AUTO CLASSIFICATION; MANUAL REVIEW REQUIRED",
    } for row in event_rows if row["auto_contextual_class"] in {"HIGH_CONTEXTUAL_CANDIDATE", "MIXED_CONTEXTUAL", "UNCLEAR"}]
    write_csv("manual_review_candidates.csv", review_rows)
    interval_start = int(np.datetime64(UNMAPPED_INTERVAL[0], "s").astype(np.int64)); interval_end = int(np.datetime64(UNMAPPED_INTERVAL[1], "s").astype(np.int64))
    interval_mask = (attack_timestamps >= interval_start) & (attack_timestamps <= interval_end)
    unmapped_rows = [{"event_id": "", "record_type": "UNMAPPED_PHYSICAL_LABEL_INTERVAL", "start": UNMAPPED_INTERVAL[0], "end": UNMAPPED_INTERVAL[1], "official_event_created": False, "physical_label_attack_fraction": float(np.mean(attack_labels[interval_mask] == 1))}]
    write_csv("unmapped_physical_label_intervals.csv", unmapped_rows)

    # Fixed representative selection: first two high, then first direct, then first mixed/unclear.
    representatives: list[dict[str, Any]] = []
    for group, limit in (("HIGH_CONTEXTUAL_CANDIDATE", 2), ("DIRECT_DOMINATED", 1)):
        representatives.extend([row for row in event_rows if row["auto_contextual_class"] == group][:limit])
    representatives.extend([row for row in event_rows if row["auto_contextual_class"] in {"MIXED_CONTEXTUAL", "UNCLEAR"}][:1])
    unique_representatives = []
    for row in representatives:
        if row["event_id"] not in {item["event_id"] for item in unique_representatives}:
            unique_representatives.append(row)
    summary = {
        "status": status, "official_physical_event_count": len(event_rows), "excluded_no_physical_impact_ids": sorted(EXCLUDED_IDS),
        "direct_class_counts": {name: direct_counts[name] for name in ("STRONG_DIRECT", "MEDIUM_DIRECT", "WEAK_DIRECT", "UNRESOLVED_MAPPING")},
        "contextual_class_counts": {name: contextual_counts[name] for name in ("HIGH_CONTEXTUAL_CANDIDATE", "MIXED_CONTEXTUAL", "DIRECT_DOMINATED", "UNCLEAR")},
        "high_contextual_candidate_ids": [row["event_id"] for row in event_rows if row["auto_contextual_class"] == "HIGH_CONTEXTUAL_CANDIDATE"],
        "stage_summary": stage_rows, "a0_checkpoint_sha256": checkpoint_hash, "a0_parameter_count": 11730,
        "a0_validation_q99": a0_threshold, "threshold_source": "Normal Validation only",
        "attack_labels_used_for_threshold": False, "point_adjustment": False, "new_detector": False,
        "attack_driven_training": False, "parameter_search": False, "fusion": False,
        "unmapped_physical_label_interval": {"start": UNMAPPED_INTERVAL[0], "end": UNMAPPED_INTERVAL[1], "official_event_id": None, "physical_label_attack_fraction": unmapped_rows[0]["physical_label_attack_fraction"]},
        "representative_figure_event_ids": [row["event_id"] for row in unique_representatives[:4]],
    }
    write_json(output_dir / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False)); return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis/swat2015_midterm_seed1.yaml")
    parser.add_argument("--output", default="outputs/thesis_s6_swat_phenomenon_audit")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(); run(resolve_repo_path(ROOT, args.config), resolve_repo_path(ROOT, args.output))
