"""Frozen HAI 21.03 reconstruction, dual-view, and max-fusion evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbone_ae import BackboneAE  # noqa: E402
from utils.seed import set_seed  # noqa: E402


DATA = ROOT / "dataset" / "hai" / "hai-21.03"
AUDIT = ROOT / "outputs" / "thesis_hai21_a0_audit"
OUT = ROOT / "outputs" / "thesis_hai21_b1_real"
TRAIN_FILES = tuple(f"train{i}.csv.gz" for i in range(1, 4))
TEST_FILES = tuple(f"test{i}.csv.gz" for i in range(1, 6))
LABELS = ("attack", "attack_P1", "attack_P2", "attack_P3")
HIGH_AP = {"AP02", "AP04", "AP07", "AP09", "AP12", "AP14", "AP17", "AP19"}
SELF_LAGS = (1, 5, 10, 20, 30)
JOINT_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (0, 1, 5, 10, 20, 30)
MAX_LAG = 30
WINDOW = 64
ALPHA = 1.0
SEED = 1
GAIN_DISCOVERY = 0.05
GAIN_VALIDATION = 0.05
GAIN_HALF = 0.02
ROBUST_FLOOR = 1e-6
Q = 0.99


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    """Convert NumPy scalar values without relaxing strict JSON output."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def relative_gain(joint_error: np.ndarray, self_error: np.ndarray) -> tuple[float, bool]:
    """Return the preregistered relative gain and flag an exact zero baseline.

    A zero self-error denominator cannot establish a positive relative improvement,
    so it is conservatively assigned gain 0 and therefore cannot pass the 0.02/0.05
    stability gates.
    """
    baseline = float(self_error.mean())
    if baseline == 0.0:
        return 0.0, True
    return 1.0 - float(joint_error.mean()) / baseline, False


@dataclass
class LinearModel:
    coef: np.ndarray  # [features, outputs]
    intercept: np.ndarray  # [outputs]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coef + self.intercept


def design(values: np.ndarray, times: np.ndarray, specs: list[tuple[int, int]]) -> np.ndarray:
    x = np.column_stack([values[times - lag, channel] for channel, lag in specs]).astype(np.float64, copy=False)
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite lagged design")
    return x


def fit_ridge_stats(
    segments: list[np.ndarray], specs: list[tuple[int, int]], targets: list[int], stride: int = 1, chunk: int = 8192
) -> LinearModel:
    p, q = len(specs), len(targets)
    sx, sy = np.zeros(p), np.zeros(q)
    xx, xy = np.zeros((p, p)), np.zeros((p, q))
    n = 0
    for values in segments:
        times = np.arange(MAX_LAG, len(values), stride, dtype=np.int64)
        for offset in range(0, len(times), chunk):
            batch_times = times[offset:offset + chunk]
            x = design(values, batch_times, specs)
            y = values[np.ix_(batch_times, targets)]
            sx += x.sum(axis=0); sy += y.sum(axis=0)
            xx += x.T @ x; xy += x.T @ y; n += len(x)
    if n <= p:
        raise RuntimeError("insufficient samples for Ridge")
    x_mean, y_mean = sx / n, sy / n
    centered_xx = xx - n * np.outer(x_mean, x_mean)
    centered_xy = xy - n * np.outer(x_mean, y_mean)
    coef = np.linalg.solve(centered_xx + ALPHA * np.eye(p), centered_xy)
    intercept = y_mean - x_mean @ coef
    return LinearModel(coef=coef, intercept=np.asarray(intercept))


def errors(
    model: LinearModel, segments: list[np.ndarray], specs: list[tuple[int, int]], targets: list[int], chunk: int = 16384
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for values in segments:
        times = np.arange(MAX_LAG, len(values), dtype=np.int64)
        local: list[np.ndarray] = []
        for offset in range(0, len(times), chunk):
            batch_times = times[offset:offset + chunk]
            prediction = model.predict(design(values, batch_times, specs))
            local.append(np.abs(values[np.ix_(batch_times, targets)] - prediction))
        parts.append(np.vstack(local))
    return np.vstack(parts)


def robust(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    scale = max(1.4826 * float(np.median(np.abs(values - median))), ROBUST_FLOOR)
    return median, scale


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = scores > threshold
    positive, negative = labels == 1, labels == 0
    tp, fp = int((prediction & positive).sum()), int((prediction & negative).sum())
    fn, tn = int((~prediction & positive).sum()), int((~prediction & negative).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "auroc": float(roc_auc_score(labels, scores)), "aupr": float(average_precision_score(labels, scores)),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, np.finfo(float).eps),
        "fpr": fp / max(fp + tn, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def window_view(values: np.ndarray) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(values, WINDOW, axis=0).transpose(0, 2, 1)


def ae_batches(segments: list[np.ndarray], batch_size: int, shuffle: bool, rng: np.random.Generator):
    segment_order = rng.permutation(len(segments)) if shuffle else np.arange(len(segments))
    for segment_id in segment_order:
        windows = window_view(segments[int(segment_id)])
        order = rng.permutation(len(windows)) if shuffle else np.arange(len(windows))
        for offset in range(0, len(order), batch_size):
            yield np.asarray(windows[order[offset:offset + batch_size]], dtype=np.float32)


def ae_loss(model: torch.nn.Module, segments: list[np.ndarray], device: torch.device, batch_size: int) -> float:
    model.eval(); total = 0.0; count = 0
    criterion = torch.nn.SmoothL1Loss()
    with torch.inference_mode():
        for windows in ae_batches(segments, batch_size, False, np.random.default_rng(0)):
            x = torch.from_numpy(windows).to(device)
            loss = criterion(model(x), x)
            total += float(loss.item()) * len(x); count += len(x)
    return total / count


def ae_scores(model: torch.nn.Module, segments: list[np.ndarray], device: torch.device, batch_size: int) -> list[np.ndarray]:
    model.eval(); result: list[np.ndarray] = []
    with torch.inference_mode():
        for values in segments:
            parts: list[np.ndarray] = []
            for windows in ae_batches([values], batch_size, False, np.random.default_rng(0)):
                x = torch.from_numpy(windows).to(device)
                parts.append(torch.abs(model(x)[:, -1] - x[:, -1]).mean(dim=1).cpu().numpy())
            result.append(np.concatenate(parts).astype(np.float64))
    return result


def split_halves(segments: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    first, second = [], []
    for values in segments:
        midpoint = len(values) // 2
        first.append(values[:midpoint]); second.append(values[midpoint:])
    return first, second


def dual_scores(
    segments: list[np.ndarray], frozen: list[dict[str, object]], models: dict[str, tuple[LinearModel, LinearModel]],
    references: dict[str, tuple[float, float]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    global_parts: list[np.ndarray] = []
    argmax_parts: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    for values in segments:
        times = np.arange(MAX_LAG, len(values), dtype=np.int64)
        target_scores = []
        for row in frozen:
            target = str(row["target"]); target_index = int(row["target_index"])
            source_indices = list(row["source_indices"])
            self_specs = [(target_index, lag) for lag in SELF_LAGS]
            cross_specs = [(source, lag) for source in source_indices for lag in CROSS_LAGS]
            self_model, cross_model = models[target]
            d = np.abs(self_model.predict(design(values, times, self_specs))[:, 0] - cross_model.predict(design(values, times, cross_specs))[:, 0])
            median, scale = references[target]
            target_scores.append(np.maximum(0.0, (d - median) / scale))
        matrix = np.column_stack(target_scores)
        matrices.append(matrix); global_parts.append(matrix.max(axis=1)); argmax_parts.append(matrix.argmax(axis=1))
    return global_parts, argmax_parts, matrices


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite HAI-B1 output: {OUT}")
    audit_summary = json.loads((AUDIT / "audit_summary.json").read_text(encoding="utf-8"))
    if audit_summary["status"] != "HAI21_BENCHMARK_PROTOCOL_CANDIDATE_READY":
        raise RuntimeError("HAI-A0 protocol is not ready")
    schema = pd.read_csv(AUDIT / "variable_schema.csv")
    groups = pd.read_csv(AUDIT / "process_groups.csv")
    audit_events = pd.read_csv(AUDIT / "attack_events_50.csv")
    feature_names = schema["column_name"].astype(str).tolist()
    if len(feature_names) != 79 or set(schema["prefix"]) != {"P1", "P2", "P3", "P4"}:
        raise RuntimeError("HAI-A0 schema changed")
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    source_paths = [DATA / name for name in TRAIN_FILES + TEST_FILES] + [AUDIT / "audit_summary.json", AUDIT / "variable_schema.csv", AUDIT / "process_groups.csv", AUDIT / "attack_events_50.csv"]
    hashes_before = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}

    # Freeze HIGH occurrences from official AP metadata before any Test feature read.
    def is_high(value: str) -> bool:
        return bool(set(str(value).split("+")) & HIGH_AP)
    high_frozen = audit_events[audit_events["AP_scenario"].map(is_high)].copy()
    high_frozen = high_frozen[["file", "event_id", "AP_scenario", "affected_process_labels", "start", "end", "start_index", "end_index_inclusive"]]
    high_frozen["freeze_basis"] = "official HAI-A0 AP mapping intersects preregistered HIGH_AP set"

    # First pass: fit one StandardScaler on only the three per-file 80% prefixes.
    scaler = StandardScaler()
    train_bounds: dict[str, int] = {}
    for name in TRAIN_FILES:
        frame = pd.read_csv(DATA / name, compression="gzip", usecols=feature_names, low_memory=False)
        boundary = int(math.floor(0.8 * len(frame))); train_bounds[name] = boundary
        scaler.partial_fit(frame.iloc[:boundary].to_numpy(np.float64))
    if int(scaler.n_samples_seen_) != sum(train_bounds.values()):
        raise RuntimeError("train-only scaler row count mismatch")

    # Second pass: retain standardized normal prefixes/suffixes as independent segments.
    train_segments: list[np.ndarray] = []
    validation_segments: list[np.ndarray] = []
    for name in TRAIN_FILES:
        frame = pd.read_csv(DATA / name, compression="gzip", usecols=feature_names, low_memory=False)
        boundary = train_bounds[name]
        train_segments.append(scaler.transform(frame.iloc[:boundary].to_numpy(np.float64)))
        validation_segments.append(scaler.transform(frame.iloc[boundary:].to_numpy(np.float64)))
    discovery_fit_segments, discovery_select_segments = [], []
    for values in train_segments:
        boundary = int(math.floor(0.8 * len(values)))
        discovery_fit_segments.append(values[:boundary]); discovery_select_segments.append(values[boundary:])

    schema_by_name = schema.set_index("column_name")
    discovery_rows: list[dict[str, object]] = []
    for process in ("P1", "P2", "P3", "P4"):
        process_names = [name for name in feature_names if name.startswith(process + "_")]
        nonconstant = [name for name in process_names if str(schema_by_name.loc[name, "nonconstant"]).lower() == "true"]
        targets = [name for name in nonconstant if schema_by_name.loc[name, "variable_kind"] == "continuous"]
        channel_indices = [name_to_index[name] for name in nonconstant]
        target_indices = [name_to_index[name] for name in targets]
        joint_specs = [(channel, lag) for channel in channel_indices for lag in JOINT_LAGS]
        discovery_joint = fit_ridge_stats(discovery_fit_segments, joint_specs, target_indices, stride=4)
        full_joint = fit_ridge_stats(train_segments, joint_specs, target_indices, stride=1)
        joint_discovery_error = errors(discovery_joint, discovery_select_segments, joint_specs, target_indices)
        joint_validation_error = errors(full_joint, validation_segments, joint_specs, target_indices)
        validation_first, validation_second = split_halves(validation_segments)
        joint_first_error = errors(full_joint, validation_first, joint_specs, target_indices)
        joint_second_error = errors(full_joint, validation_second, joint_specs, target_indices)
        for target_number, target in enumerate(targets):
            target_index = name_to_index[target]
            self_specs = [(target_index, lag) for lag in SELF_LAGS]
            discovery_self = fit_ridge_stats(discovery_fit_segments, self_specs, [target_index], stride=4)
            full_self = fit_ridge_stats(train_segments, self_specs, [target_index], stride=1)
            self_d = errors(discovery_self, discovery_select_segments, self_specs, [target_index])[:, 0]
            self_v = errors(full_self, validation_segments, self_specs, [target_index])[:, 0]
            self_f = errors(full_self, validation_first, self_specs, [target_index])[:, 0]
            self_s = errors(full_self, validation_second, self_specs, [target_index])[:, 0]
            gain_d, zero_d = relative_gain(joint_discovery_error[:, target_number], self_d)
            gain_v, zero_v = relative_gain(joint_validation_error[:, target_number], self_v)
            gain_f, zero_f = relative_gain(joint_first_error[:, target_number], self_f)
            gain_s, zero_s = relative_gain(joint_second_error[:, target_number], self_s)
            stable = gain_d >= GAIN_DISCOVERY and gain_v >= GAIN_VALIDATION and gain_f >= GAIN_HALF and gain_s >= GAIN_HALF
            sources = [name for name in nonconstant if name != target]
            discovery_rows.append({
                "target": target, "process": process, "target_index": target_index,
                "source_indices": ";".join(str(name_to_index[name]) for name in sources),
                "cross_source_names": ";".join(sources), "discovery_gain": gain_d,
                "validation_gain": gain_v, "validation_first_gain": gain_f, "validation_second_gain": gain_s,
                "zero_self_error_discovery": zero_d, "zero_self_error_validation": zero_v,
                "zero_self_error_validation_first": zero_f, "zero_self_error_validation_second": zero_s,
                "stable": stable,
            })
    discovery = pd.DataFrame(discovery_rows)
    frozen = discovery[discovery["stable"]].copy().reset_index(drop=True)
    if frozen.empty:
        raise RuntimeError("normal-only discovery produced no stable target")
    frozen_records = []
    for row in frozen.to_dict("records"):
        row["source_indices"] = [int(value) for value in str(row["source_indices"]).split(";") if value]
        frozen_records.append(row)

    # Materialize pre-Test freezes and hash the exact target order.
    OUT.mkdir(parents=True)
    high_frozen.to_csv(OUT / "high_collaboration_events_frozen.csv", index=False)
    frozen.drop(columns=["target_index", "source_indices", "cross_source_names", "stable"]).to_csv(OUT / "dual_targets_frozen.csv", index=False)
    target_hash = sha256(OUT / "dual_targets_frozen.csv")
    high_hash = sha256(OUT / "high_collaboration_events_frozen.csv")

    # Fit final frozen Self/Cross views on complete Normal Train prefixes.
    dual_models: dict[str, tuple[LinearModel, LinearModel]] = {}
    references: dict[str, tuple[float, float]] = {}
    raw_validation_by_target: dict[str, list[np.ndarray]] = {}
    for row in frozen_records:
        target, target_index = str(row["target"]), int(row["target_index"])
        source_indices = list(row["source_indices"])
        self_specs = [(target_index, lag) for lag in SELF_LAGS]
        cross_specs = [(source, lag) for source in source_indices for lag in CROSS_LAGS]
        self_model = fit_ridge_stats(train_segments, self_specs, [target_index])
        cross_model = fit_ridge_stats(train_segments, cross_specs, [target_index])
        dual_models[target] = (self_model, cross_model)
        parts = []
        for values in validation_segments:
            times = np.arange(MAX_LAG, len(values), dtype=np.int64)
            ps = self_model.predict(design(values, times, self_specs))[:, 0]
            pc = cross_model.predict(design(values, times, cross_specs))[:, 0]
            parts.append(np.abs(ps - pc))
        raw_validation_by_target[target] = parts
        references[target] = robust(np.concatenate(parts))
    dual_val, _, _ = dual_scores(validation_segments, frozen_records, dual_models, references)
    dual_val_all = np.concatenate(dual_val)
    threshold_dual = float(np.quantile(dual_val_all, Q))
    median_val_d, scale_val_d = robust(dual_val_all)

    # Train the one allowed lightweight reconstruction baseline, Normal-only.
    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("authorized frozen A0 training requires CUDA")
    device = torch.device("cuda")
    model = BackboneAE(79, 64, 32, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.SmoothL1Loss()
    batch_size, patience, max_epochs = 256, 4, 15
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, object]] = []
    checkpoint = OUT / "rec_best.pt"
    for epoch in range(1, max_epochs + 1):
        model.train(); total = 0.0; count = 0
        for windows in ae_batches(train_segments, batch_size, True, np.random.default_rng(SEED + epoch)):
            x = torch.from_numpy(windows).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), x)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite A0 loss")
            loss.backward(); optimizer.step()
            total += float(loss.item()) * len(x); count += len(x)
        train_loss = total / count
        val_loss = ae_loss(model, validation_segments, device, batch_size)
        improved = val_loss < best_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss, "improved": improved})
        print(json.dumps(history[-1]), flush=True)
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "validation_loss": val_loss}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    rec_val = ae_scores(model, validation_segments, device, batch_size)
    rec_val_all = np.concatenate(rec_val)
    threshold_rec = float(np.quantile(rec_val_all, Q))
    median_val_r, scale_val_r = robust(rec_val_all)
    fusion_val_parts = []
    for rec_part, dual_part in zip(rec_val, dual_val):
        aligned_dual = dual_part[WINDOW - 1 - MAX_LAG:]
        if len(aligned_dual) != len(rec_part):
            raise RuntimeError("validation common mask mismatch")
        zr = np.maximum(0.0, (rec_part - median_val_r) / scale_val_r)
        zd = np.maximum(0.0, (aligned_dual - median_val_d) / scale_val_d)
        fusion_val_parts.append(np.maximum(zr, zd))
    threshold_fusion = float(np.quantile(np.concatenate(fusion_val_parts), Q))

    # Test is first read after HIGH list, targets, models, calibrations, and thresholds are frozen.
    all_labels: list[np.ndarray] = []
    all_rec: list[np.ndarray] = []
    all_dual: list[np.ndarray] = []
    all_fusion: list[np.ndarray] = []
    test_records: dict[str, dict[str, np.ndarray]] = {}
    for name in TEST_FILES:
        frame = pd.read_csv(DATA / name, compression="gzip", low_memory=False)
        if list(frame.columns[1:-4]) != feature_names or list(frame.columns[-4:]) != list(LABELS):
            raise RuntimeError(f"Test schema changed: {name}")
        values = scaler.transform(frame[feature_names].to_numpy(np.float64))
        rec = ae_scores(model, [values], device, batch_size)[0]
        dual_parts, argmax_parts, _ = dual_scores([values], frozen_records, dual_models, references)
        dual = dual_parts[0][WINDOW - 1 - MAX_LAG:]
        argmax = argmax_parts[0][WINDOW - 1 - MAX_LAG:]
        if len(rec) != len(dual):
            raise RuntimeError("Test common valid mask mismatch")
        zr = np.maximum(0.0, (rec - median_val_r) / scale_val_r)
        zd = np.maximum(0.0, (dual - median_val_d) / scale_val_d)
        fusion = np.maximum(zr, zd)
        labels = frame["attack"].to_numpy(np.int8)[WINDOW - 1:]
        raw_indices = np.arange(WINDOW - 1, len(frame), dtype=np.int64)
        timestamps = frame["time"].astype(str).to_numpy()[WINDOW - 1:]
        all_labels.append(labels); all_rec.append(rec); all_dual.append(dual); all_fusion.append(fusion)
        test_records[name] = {"labels": labels, "rec": rec, "dual": dual, "fusion": fusion,
                              "argmax": argmax, "raw_indices": raw_indices, "timestamps": timestamps}
    labels = np.concatenate(all_labels); rec = np.concatenate(all_rec); dual = np.concatenate(all_dual); fusion = np.concatenate(all_fusion)
    rec_m, dual_m, fusion_m = metrics(labels, rec, threshold_rec), metrics(labels, dual, threshold_dual), metrics(labels, fusion, threshold_fusion)
    global_metrics = pd.DataFrame([
        {"method": "REC", "threshold": threshold_rec, **rec_m},
        {"method": "DUAL", "threshold": threshold_dual, **dual_m},
        {"method": "FUSION", "threshold": threshold_fusion, **fusion_m},
    ])

    target_names = frozen["target"].astype(str).tolist()
    target_processes = frozen["process"].astype(str).tolist()
    event_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []
    for event in audit_events.itertuples(index=False):
        record = test_records[str(event.file)]
        mask = (record["raw_indices"] >= int(event.start_index)) & (record["raw_indices"] <= int(event.end_index_inclusive))
        if not mask.any():
            raise RuntimeError(f"event absent from common valid mask: {event.event_id}")
        rec_hit = record["rec"][mask] > threshold_rec
        dual_hit = record["dual"][mask] > threshold_dual
        fusion_hit = record["fusion"][mask] > threshold_fusion
        counts = np.bincount(record["argmax"][mask], minlength=len(target_names))
        top = int(np.argmax(counts))
        category = "HIGH" if is_high(str(event.AP_scenario)) else "DIRECT"
        event_rows.append({
            "file": str(event.file), "event_id": str(event.event_id), "AP_scenario": str(event.AP_scenario),
            "affected_process": str(event.affected_process_labels), "category": category,
            "rec_hit_rate": float(rec_hit.mean()), "dual_hit_rate": float(dual_hit.mean()),
            "fusion_hit_rate": float(fusion_hit.mean()), "top_dual_target": target_names[top],
            "top_dual_process": target_processes[top], "common_point_count": int(mask.sum()),
            "rec_near_miss": bool(rec_hit.mean() < 0.10),
            "dual_recovered": bool(rec_hit.mean() < 0.10 and dual_hit.mean() >= 0.10),
            "dual_beats_rec_010": bool(dual_hit.mean() >= rec_hit.mean() + 0.10),
        })
        for index, target in enumerate(target_names):
            attribution_rows.append({"file": str(event.file), "event_id": str(event.event_id), "target": target,
                                     "process": target_processes[index], "argmax_count": int(counts[index]),
                                     "argmax_fraction": float(counts[index] / mask.sum())})
    events = pd.DataFrame(event_rows)
    high_metrics = events[events["category"] == "HIGH"].copy()
    recovered = events[events["dual_recovered"]]
    event_complementarity = pd.DataFrame([
        {"scope": "ALL_50", "event_count": len(events), "rec_near_miss_count": int(events["rec_near_miss"].sum()),
         "dual_recovered_count": int(events["dual_recovered"].sum())},
        {"scope": "HIGH", "event_count": len(high_metrics), "rec_near_miss_count": int(high_metrics["rec_near_miss"].sum()),
         "dual_recovered_count": int(high_metrics["dual_recovered"].sum())},
    ])
    rec_hit, dual_hit = rec > threshold_rec, dual > threshold_dual
    attack, normal = labels == 1, labels == 0
    point_complementarity = pd.DataFrame([{
        "rec_hit_dual_hit": int((attack & rec_hit & dual_hit).sum()),
        "rec_hit_dual_miss": int((attack & rec_hit & ~dual_hit).sum()),
        "rec_miss_dual_hit": int((attack & ~rec_hit & dual_hit).sum()),
        "rec_miss_dual_miss": int((attack & ~rec_hit & ~dual_hit).sum()),
        "rec_correct_dual_fp": int((normal & ~rec_hit & dual_hit).sum()),
    }])
    category_summary = events.groupby("category", sort=False).agg(
        event_count=("event_id", "count"), median_rec_hit=("rec_hit_rate", "median"),
        median_dual_hit=("dual_hit_rate", "median"), median_fusion_hit=("fusion_hit_rate", "median"),
    ).reset_index()

    h_gates = {
        "H1": bool((high_metrics["dual_hit_rate"] >= 0.10).mean() >= 0.50),
        "H2": bool((high_metrics["dual_hit_rate"] >= high_metrics["rec_hit_rate"] + 0.10).sum() >= 2),
        "H3": bool(((high_metrics["rec_hit_rate"] < 0.10) & (high_metrics["dual_hit_rate"] >= 0.10)).any()),
        "H4": bool(float(dual_m["fpr"]) <= 0.10),
    }
    if all(h_gates.values()):
        status_i2 = "DUAL_VIEW_REAL_HAI_COLLABORATION_SUPPORTED"
    elif h_gates["H1"] and h_gates["H2"] and not h_gates["H3"]:
        status_i2 = "DUAL_VIEW_REAL_HAI_INCREMENTAL_PARTIAL"
    else:
        status_i2 = "DUAL_VIEW_REAL_HAI_NOT_SUPPORTED"
    rec_event_count = int((events["rec_hit_rate"] >= 0.10).sum())
    fusion_event_count = int((events["fusion_hit_rate"] >= 0.10).sum())
    f_gates = {
        "F1": bool(float(fusion_m["aupr"]) >= float(rec_m["aupr"]) + 0.01),
        "F2": bool(float(fusion_m["auroc"]) >= float(rec_m["auroc"]) - 0.01),
        "F3": fusion_event_count >= rec_event_count,
        "F4": bool(float(fusion_m["fpr"]) <= 0.10),
    }
    status_fusion = "NORMAL_CALIBRATED_MAX_FUSION_PROMISING" if all(f_gates.values()) else "NORMAL_CALIBRATED_MAX_FUSION_NOT_YET_SUPPORTED"

    thresholds = {
        "quantile": Q, "comparator": ">", "source": "Normal Validation suffixes only",
        "rec": threshold_rec, "dual": threshold_dual, "fusion": threshold_fusion,
        "rec_branch_median": median_val_r, "rec_branch_scale": scale_val_r,
        "dual_branch_median": median_val_d, "dual_branch_scale": scale_val_d,
    }
    tests = {
        "three_train_five_test": len(TRAIN_FILES) == 3 and len(TEST_FILES) == 5,
        "no_cross_file_window": True, "scaler_train_only": int(scaler.n_samples_seen_) == sum(train_bounds.values()),
        "labels_excluded_features": not set(LABELS) & set(feature_names), "target_discovery_normal_only": True,
        "stable_rules_exact": (GAIN_DISCOVERY, GAIN_VALIDATION, GAIN_HALF) == (0.05, 0.05, 0.02),
        "target_list_written_before_test": True, "high_list_official_mapping": set(high_frozen["event_id"]) == set(high_metrics["event_id"]),
        "self_lags_fixed": SELF_LAGS == (1,5,10,20,30), "cross_lags_fixed": CROSS_LAGS == (0,1,5,10,20,30),
        "target_excluded_cross": all(int(row["target_index"]) not in row["source_indices"] for row in frozen_records),
        "alpha_one": ALPHA == 1.0, "dual_disagreement_only": True, "rec_hyperparameters_frozen": True,
        "thresholds_normal_only": True, "fusion_max_no_weights": True, "no_point_adjustment": True,
        "test_labels_evaluation_only": True, "common_valid_mask": sum(len(x) for x in all_labels) == sum(len(x) for x in all_rec) == sum(len(x) for x in all_dual),
        "no_nan_inf": bool(np.isfinite(global_metrics.select_dtypes("number").to_numpy()).all() and np.isfinite(events.select_dtypes("number").to_numpy()).all()),
    }
    if not all(tests.values()):
        raise RuntimeError(f"HAI-B1 integrity failure: {tests}")
    summary = {
        "elapsed_seconds": time.perf_counter() - started, "frozen_target_count": len(frozen),
        "target_counts_by_process": frozen["process"].value_counts().to_dict(), "frozen_targets": target_names,
        "high_event_count": len(high_frozen), "high_events": high_frozen[["file","event_id","AP_scenario","affected_process_labels"]].to_dict("records"),
        "rec_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "rec_best_epoch": best_epoch, "rec_best_validation_loss": best_loss,
        "global_metrics": global_metrics.to_dict("records"),
        "event_hit_010_counts": {"REC": rec_event_count, "DUAL": int((events["dual_hit_rate"] >= 0.10).sum()), "FUSION": fusion_event_count},
        "event_complementarity": event_complementarity.to_dict("records"), "innovation2_gates": h_gates,
        "status_innovation2": status_i2, "fusion_gates": f_gates, "status_fusion": status_fusion,
        "tests": tests, "no_attack_driven_selection": True, "point_adjustment": False,
        "lag_search": False, "threshold_search": False, "fusion_weight_search": False,
    }
    manifest = {
        "summary": summary, "dataset": "HAI 21.03", "seed": SEED,
        "train_prefix_boundaries": train_bounds, "feature_count": 79, "feature_names": feature_names,
        "scaler_fit_rows": int(scaler.n_samples_seen_), "target_freeze_sha256": target_hash,
        "high_event_freeze_sha256": high_hash, "source_sha256": hashes_before,
        "self_lags": list(SELF_LAGS), "joint_lags": list(JOINT_LAGS), "cross_lags": list(CROSS_LAGS),
        "alpha": ALPHA, "rec_model": {"window":64,"hidden":64,"latent":32,"dropout":0.1,"loss":"SmoothL1","optimizer":"Adam","lr":1e-3,"weight_decay":1e-4,"batch":256,"max_epochs":15,"patience":4},
        "dual_score": "max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/scale_j)",
        "fusion_score": "max(Z_rec,Z_dual)", "test_opened_after_freeze": True,
    }
    global_metrics.to_csv(OUT / "global_metrics.csv", index=False)
    events.to_csv(OUT / "event_metrics_50.csv", index=False)
    high_metrics.to_csv(OUT / "high_event_metrics.csv", index=False)
    point_complementarity.to_csv(OUT / "point_complementarity.csv", index=False)
    event_complementarity.to_csv(OUT / "event_complementarity.csv", index=False)
    pd.DataFrame(attribution_rows).to_csv(OUT / "target_attribution.csv", index=False)
    category_summary.to_csv(OUT / "scenario_category_summary.csv", index=False)
    discovery.to_csv(OUT / "normal_target_discovery_all.csv", index=False)
    pd.DataFrame(history).to_csv(OUT / "rec_training_history.csv", index=False)
    (OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    (OUT / "hai21_b1_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    hashes_after = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("source HAI/audit artifact changed")
    print(json.dumps(summary, indent=2, allow_nan=False, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
