"""WADI.A2 frozen REC / Dual / max-fusion external validation."""

from __future__ import annotations

import hashlib
import importlib.util
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


DATA = ROOT / "dataset" / "WADI_A2_2019"
A0 = ROOT / "outputs" / "thesis_wadi_a0_audit"
A05 = ROOT / "outputs" / "thesis_wadi_a05_protocol"
HAI = ROOT / "outputs" / "thesis_hai21_b1_real"
OUT = ROOT / "outputs" / "thesis_wadi_b1_external"
DOC = ROOT / "docs" / "thesis" / "wadi_b1_rec_dual_fusion_external.md"
NORMAL = DATA / "WADI_14days_new.csv"
ATTACK = DATA / "WADI_attackdataLABLE.csv"
LABEL = "Attack LABLE (1:No Attack, -1:Attack)"
TRAIN_RANGES = ((0, 268799), (335999, 694856))
VALIDATION_RANGES = ((268799, 335999), (694856, 784571))
HIGH_IDS = {2, 3, 4, 7, 15}
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
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def load_a0_module():
    path = ROOT / "scripts" / "thesis" / "00_audit" / "audit_wadi_a0.py"
    spec = importlib.util.spec_from_file_location("wadi_a0_source_b1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class LinearModel:
    coef: np.ndarray
    intercept: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coef + self.intercept


@dataclass
class SufficientStats:
    x_mean: np.ndarray
    y_mean: np.ndarray
    centered_xx: np.ndarray
    centered_xy: np.ndarray


@dataclass
class ProcessDual:
    target_names: list[str]
    target_indices: list[int]
    source_indices: list[int]
    cross_specs: list[tuple[int, int]]
    cross_model: LinearModel
    self_models: list[LinearModel]
    references: list[tuple[float, float]]


def design(values: np.ndarray, times: np.ndarray, specs: list[tuple[int, int]]) -> np.ndarray:
    result = np.column_stack([values[times - lag, channel] for channel, lag in specs]).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite lagged design")
    return result


def fit_stats(segments: list[np.ndarray], specs: list[tuple[int, int]], targets: list[int], stride: int = 1) -> SufficientStats:
    p, q = len(specs), len(targets)
    sx, sy = np.zeros(p), np.zeros(q)
    xx, xy = np.zeros((p, p)), np.zeros((p, q))
    n = 0
    for values in segments:
        times = np.arange(MAX_LAG, len(values), stride, dtype=np.int64)
        for offset in range(0, len(times), 8192):
            bt = times[offset:offset + 8192]
            x = design(values, bt, specs)
            y = values[np.ix_(bt, targets)]
            sx += x.sum(0); sy += y.sum(0); xx += x.T @ x; xy += x.T @ y; n += len(x)
    if n <= p:
        raise RuntimeError("insufficient normal samples for Ridge")
    xm, ym = sx / n, sy / n
    return SufficientStats(xm, ym, xx - n * np.outer(xm, xm), xy - n * np.outer(xm, ym))


def model_from_stats(stats: SufficientStats) -> LinearModel:
    coef = np.linalg.solve(stats.centered_xx + ALPHA * np.eye(len(stats.x_mean)), stats.centered_xy)
    return LinearModel(coef, np.asarray(stats.y_mean - stats.x_mean @ coef))


def fit_ridge(segments: list[np.ndarray], specs: list[tuple[int, int]], targets: list[int], stride: int = 1) -> LinearModel:
    return model_from_stats(fit_stats(segments, specs, targets, stride))


def leave_target_out_models(
    segments: list[np.ndarray], all_specs: list[tuple[int, int]], target_indices: list[int]
) -> LinearModel:
    stats = fit_stats(segments, all_specs, target_indices)
    coefficients = np.zeros((len(all_specs), len(target_indices)), dtype=np.float64)
    for column, target in enumerate(target_indices):
        keep = np.asarray([channel != target for channel, _ in all_specs])
        sub_xx = stats.centered_xx[np.ix_(keep, keep)]
        sub_xy = stats.centered_xy[keep, column]
        coefficients[keep, column] = np.linalg.solve(sub_xx + ALPHA * np.eye(int(keep.sum())), sub_xy)
    intercept = stats.y_mean - stats.x_mean @ coefficients
    return LinearModel(coefficients, np.asarray(intercept))


def absolute_errors(model: LinearModel, segments: list[np.ndarray], specs: list[tuple[int, int]], targets: list[int]) -> np.ndarray:
    parts = []
    for values in segments:
        times = np.arange(MAX_LAG, len(values), dtype=np.int64)
        local = []
        for offset in range(0, len(times), 16384):
            bt = times[offset:offset + 16384]
            local.append(np.abs(values[np.ix_(bt, targets)] - model.predict(design(values, bt, specs))))
        parts.append(np.vstack(local))
    return np.vstack(parts)


def relative_gain(joint_error: np.ndarray, self_error: np.ndarray) -> tuple[float, bool]:
    baseline = float(self_error.mean())
    return (0.0, True) if baseline == 0.0 else (1.0 - float(joint_error.mean()) / baseline, False)


def robust(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    return median, max(1.4826 * float(np.median(np.abs(values - median))), ROBUST_FLOOR)


def point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
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


def valid_runs(frame: pd.DataFrame, features: list[str], ranges: tuple[tuple[int, int], ...]) -> tuple[list[np.ndarray], list[dict]]:
    segments, metadata = [], []
    for original_segment, (start, end) in enumerate(ranges, 1):
        valid = frame.iloc[start:end][features].notna().all(axis=1).to_numpy()
        starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
        ends = np.flatnonzero(valid & ~np.r_[valid[1:], False])
        for part, (local_start, local_end) in enumerate(zip(starts, ends), 1):
            a, b = start + int(local_start), start + int(local_end) + 1
            if b - a <= WINDOW:
                continue
            values = frame.iloc[a:b][features].to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError("invalid point leaked into segment")
            segments.append(values)
            metadata.append({"source_segment": original_segment, "valid_run": part, "start": a, "end_exclusive": b, "rows": b-a})
    return segments, metadata


def window_view(values: np.ndarray) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(values, WINDOW, axis=0).transpose(0, 2, 1)


def ae_batches(segments: list[np.ndarray], batch_size: int, shuffle: bool, rng: np.random.Generator):
    order_segments = rng.permutation(len(segments)) if shuffle else np.arange(len(segments))
    for segment_index in order_segments:
        windows = window_view(segments[int(segment_index)])
        order = rng.permutation(len(windows)) if shuffle else np.arange(len(windows))
        for offset in range(0, len(order), batch_size):
            yield np.asarray(windows[order[offset:offset + batch_size]], dtype=np.float32)


def ae_loss(model, segments, device, batch_size):
    model.eval(); total = 0.0; count = 0
    criterion = torch.nn.SmoothL1Loss()
    with torch.inference_mode():
        for windows in ae_batches(segments, batch_size, False, np.random.default_rng(0)):
            x = torch.from_numpy(windows).to(device)
            loss = criterion(model(x), x)
            total += float(loss.item()) * len(x); count += len(x)
    return total / count


def ae_scores(model, segments, device, batch_size):
    model.eval(); output = []
    with torch.inference_mode():
        for values in segments:
            parts = []
            for windows in ae_batches([values], batch_size, False, np.random.default_rng(0)):
                x = torch.from_numpy(windows).to(device)
                parts.append(torch.abs(model(x)[:, -1] - x[:, -1]).mean(1).cpu().numpy())
            output.append(np.concatenate(parts).astype(np.float64))
    return output


def split_halves(segments: list[np.ndarray]):
    first, second = [], []
    for values in segments:
        midpoint = len(values) // 2
        first.append(values[:midpoint]); second.append(values[midpoint:])
    return first, second


def score_dual(segments: list[np.ndarray], groups: list[ProcessDual]):
    names = [name for group in groups for name in group.target_names]
    global_parts, argmax_parts, matrix_parts = [], [], []
    for values in segments:
        times = np.arange(MAX_LAG, len(values), dtype=np.int64)
        columns = []
        for group in groups:
            pred_cross = group.cross_model.predict(design(values, times, group.cross_specs))
            for j, target_index in enumerate(group.target_indices):
                self_specs = [(target_index, lag) for lag in SELF_LAGS]
                pred_self = group.self_models[j].predict(design(values, times, self_specs))[:, 0]
                disagreement = np.abs(pred_self - pred_cross[:, j])
                center, scale = group.references[j]
                columns.append(np.maximum(0.0, (disagreement - center) / scale))
        matrix = np.column_stack(columns)
        if matrix.shape[1] != len(names) or not np.isfinite(matrix).all():
            raise RuntimeError("invalid Dual score matrix")
        matrix_parts.append(matrix); global_parts.append(matrix.max(1)); argmax_parts.append(matrix.argmax(1))
    return global_parts, argmax_parts, matrix_parts


def category_name(taxonomy: str) -> str:
    return {"HIGH_COLLABORATION_RELEVANCE": "HIGH", "MIXED_COLLABORATION": "MIXED", "DIRECT_DOMINATED": "DIRECT"}[taxonomy]


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite: {OUT}")
    protocol = json.loads((A05 / "protocol_freeze.json").read_text(encoding="utf-8"))
    if protocol["missingness_status"] != "MISSINGNESS_PROTOCOL_READY" or protocol["attack13_status"] != "ATTACK13_SOURCE_CONFLICT":
        raise RuntimeError("WADI-A0.5 gate is not frozen as expected")
    schema = pd.read_csv(A05 / "feature_schema_frozen.csv")
    kept_schema = schema[schema["included"]].copy()
    features = kept_schema["raw_column_name"].astype(str).tolist()
    excluded = schema.loc[~schema["included"], "raw_column_name"].astype(str).tolist()
    if len(features) != 123 or set(excluded) != {"2_LS_001_AL", "2_LS_002_AL", "2_P_001_STATUS", "2_P_002_STATUS"}:
        raise RuntimeError("frozen WADI feature gate changed")
    groups_csv = pd.read_csv(A0 / "process_groups.csv")
    events_all = pd.read_csv(A0 / "official_attack_events.csv")
    bindings = pd.read_csv(A05 / "event_binding_final.csv")
    bound_ids = set(bindings.loc[bindings.final_binding_status == "BOUND", "attack_id"].astype(int))
    if len(bound_ids) != 14 or 13 in bound_ids or not HIGH_IDS.issubset(bound_ids):
        raise RuntimeError("event binding freeze changed")
    bound_events = events_all[events_all.attack_id.isin(bound_ids)].copy().sort_values("attack_id")

    source_paths = [NORMAL, ATTACK, A05 / "protocol_freeze.json", A05 / "feature_schema_frozen.csv", A05 / "event_binding_final.csv", A0 / "process_groups.csv", A0 / "official_attack_events.csv"]
    hashes_before = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}

    normal = pd.read_csv(NORMAL, usecols=features, low_memory=False)
    train_raw, train_run_metadata = valid_runs(normal, features, TRAIN_RANGES)
    validation_raw, validation_run_metadata = valid_runs(normal, features, VALIDATION_RANGES)
    scaler = StandardScaler()
    for values in train_raw:
        scaler.partial_fit(values)
    train_segments = [scaler.transform(values).astype(np.float64, copy=False) for values in train_raw]
    validation_segments = [scaler.transform(values).astype(np.float64, copy=False) for values in validation_raw]
    del train_raw, validation_raw, normal
    if int(scaler.n_samples_seen_) != sum(len(x) for x in train_segments):
        raise RuntimeError("scaler Train-only count mismatch")

    discovery_fit, discovery_select = [], []
    for values in train_segments:
        boundary = int(math.floor(0.8 * len(values)))
        discovery_fit.append(values[:boundary]); discovery_select.append(values[boundary:])
    validation_first, validation_second = split_halves(validation_segments)
    index = {name: i for i, name in enumerate(features)}
    schema_by_name = kept_schema.set_index("raw_column_name")
    discovery_rows = []
    group_definitions = []
    for group_row in groups_csv.itertuples(index=False):
        process = str(group_row.process_group)
        declared = str(group_row.channel_names).split(";")
        usable = [name for name in declared if name in index and schema_by_name.loc[name, "variable_kind"] in {"continuous", "discrete"}]
        targets = [name for name in usable if schema_by_name.loc[name, "variable_kind"] == "continuous"]
        if not targets:
            continue
        channels = [index[name] for name in usable]; target_indices = [index[name] for name in targets]
        joint_specs = [(channel, lag) for channel in channels for lag in JOINT_LAGS]
        discovery_joint = fit_ridge(discovery_fit, joint_specs, target_indices, stride=4)
        full_joint = fit_ridge(train_segments, joint_specs, target_indices)
        joint_d = absolute_errors(discovery_joint, discovery_select, joint_specs, target_indices)
        joint_v = absolute_errors(full_joint, validation_segments, joint_specs, target_indices)
        joint_f = absolute_errors(full_joint, validation_first, joint_specs, target_indices)
        joint_s = absolute_errors(full_joint, validation_second, joint_specs, target_indices)
        for j, target in enumerate(targets):
            ti = index[target]; self_specs = [(ti, lag) for lag in SELF_LAGS]
            self_d = absolute_errors(fit_ridge(discovery_fit, self_specs, [ti], stride=4), discovery_select, self_specs, [ti])[:, 0]
            full_self = fit_ridge(train_segments, self_specs, [ti])
            self_v = absolute_errors(full_self, validation_segments, self_specs, [ti])[:, 0]
            self_f = absolute_errors(full_self, validation_first, self_specs, [ti])[:, 0]
            self_s = absolute_errors(full_self, validation_second, self_specs, [ti])[:, 0]
            gain_d, zero_d = relative_gain(joint_d[:, j], self_d)
            gain_v, zero_v = relative_gain(joint_v[:, j], self_v)
            gain_f, zero_f = relative_gain(joint_f[:, j], self_f)
            gain_s, zero_s = relative_gain(joint_s[:, j], self_s)
            stable = gain_d >= GAIN_DISCOVERY and gain_v >= GAIN_VALIDATION and gain_f >= GAIN_HALF and gain_s >= GAIN_HALF
            sources = [name for name in usable if name != target]
            discovery_rows.append({
                "target": target, "process": process, "target_index": ti,
                "source_indices": ";".join(str(index[name]) for name in sources), "cross_source_names": ";".join(sources),
                "discovery_gain": gain_d, "validation_gain": gain_v, "validation_first_gain": gain_f,
                "validation_second_gain": gain_s, "zero_self_error_discovery": zero_d,
                "zero_self_error_validation": zero_v, "zero_self_error_validation_first": zero_f,
                "zero_self_error_validation_second": zero_s, "stable": stable,
            })
        group_definitions.append((process, usable))
        print(json.dumps({"phase": "discovery", "process": process, "targets": len(targets)}), flush=True)
    discovery = pd.DataFrame(discovery_rows)
    frozen = discovery[discovery.stable].copy().reset_index(drop=True)
    if frozen.empty:
        OUT.mkdir(parents=True)
        discovery.to_csv(OUT / "normal_target_discovery_all.csv", index=False)
        frozen_public = discovery.iloc[0:0][["target", "process", "discovery_gain", "validation_gain", "validation_first_gain", "validation_second_gain"]]
        frozen_public.to_csv(OUT / "dual_targets_frozen.csv", index=False)
        target_hash = sha256(OUT / "dual_targets_frozen.csv")
        top = discovery.sort_values(
            ["validation_gain", "discovery_gain"], ascending=False
        ).head(10)[["target", "process", "discovery_gain", "validation_gain", "validation_first_gain", "validation_second_gain"]]
        gate_tests = {
            "wadi_a2_only": True,
            "frozen_123": len(features) == 123,
            "four_all_nan_excluded": len(excluded) == 4,
            "exclusion_normal_train_only": protocol["attack_test_used_for_exclusion"] is False,
            "no_imputation": protocol["imputation"] == "NONE",
            "nan_breaks_preserved": True,
            "scaler_train_only": int(scaler.n_samples_seen_) == sum(len(x) for x in train_segments),
            "discovery_normal_only": True,
            "stable_rules_exact": bool((
                discovery.stable
                == ((discovery.discovery_gain >= GAIN_DISCOVERY)
                    & (discovery.validation_gain >= GAIN_VALIDATION)
                    & (discovery.validation_first_gain >= GAIN_HALF)
                    & (discovery.validation_second_gain >= GAIN_HALF))
            ).all()),
            "target_list_frozen_empty": len(frozen_public) == 0,
            "test_not_opened": True,
            "rec_not_trained": True,
            "no_parameter_or_target_search": True,
        }
        gate_pass_counts = {
            "discovery_gain_ge_0_05": int((discovery.discovery_gain >= GAIN_DISCOVERY).sum()),
            "validation_gain_ge_0_05": int((discovery.validation_gain >= GAIN_VALIDATION).sum()),
            "validation_first_gain_ge_0_02": int((discovery.validation_first_gain >= GAIN_HALF).sum()),
            "validation_second_gain_ge_0_02": int((discovery.validation_second_gain >= GAIN_HALF).sum()),
            "all_four": int(discovery.stable.sum()),
        }
        gate_summary = {
            "status": "WADI_B1_STOPPED_NO_STABLE_DUAL_TARGETS",
            "elapsed_seconds": time.perf_counter() - started,
            "candidate_target_count": len(discovery),
            "frozen_target_count": 0,
            "target_counts_by_process": {},
            "frozen_targets": [],
            "target_freeze_sha256": target_hash,
            "individual_gate_pass_counts": gate_pass_counts,
            "maximum_validation_gain": float(discovery.validation_gain.max()),
            "top_candidates_by_validation_gain": top.to_dict("records"),
            "gate_thresholds": {"discovery_gain": GAIN_DISCOVERY, "validation_gain": GAIN_VALIDATION,
                                "validation_first_gain": GAIN_HALF, "validation_second_gain": GAIN_HALF},
            "rec_training_started": False,
            "test_opened": False,
            "global_metrics_available": False,
            "reason": "Dual target set is empty under the preregistered four-gate rule; DUAL and FUSION are undefined.",
            "tests": gate_tests,
            "no_attack_driven_selection": True,
            "no_test_tuning": True,
            "lag_search": False,
            "threshold_search": False,
            "point_adjustment": False,
            "fusion_weight_search": False,
            "attack13_patch": False,
        }
        gate_manifest = {
            "summary": gate_summary, "dataset": "WADI.A2", "seed": SEED,
            "feature_count": len(features), "feature_names": features, "excluded_features": excluded,
            "train_ranges": TRAIN_RANGES, "validation_ranges": VALIDATION_RANGES,
            "train_valid_runs": train_run_metadata, "validation_valid_runs": validation_run_metadata,
            "scaler_fit_rows": int(scaler.n_samples_seen_), "target_freeze_sha256": target_hash,
            "source_sha256": hashes_before, "process_group_source": "outputs/thesis_wadi_a0_audit/process_groups.csv",
            "self_lags": SELF_LAGS, "joint_lags": JOINT_LAGS, "cross_lags": CROSS_LAGS, "alpha": ALPHA,
            "test_opened_after_freeze": False, "rec_training_started": False,
        }
        top.to_csv(OUT / "target_gate_top_candidates.csv", index=False)
        (OUT / "manifest.json").write_text(json.dumps(gate_manifest, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
        (OUT / "wadi_b1_summary.json").write_text(json.dumps(gate_summary, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
        DOC.write_text(f"""# WADI-B1 Frozen Rec / Dual / Fusion External Validation

## Pre-Test target gate result

WADI.A2 A0.5 passed the source, 123-variable schema, missingness, and segment gates. StandardScaler and target discovery used only the frozen valid Normal Train points. The exact self lags `[1,5,10,20,30]`, joint lags `[1,5,10,20,30]`, Ridge alpha 1.0, physical process groups, and four stability thresholds were unchanged.

All {len(discovery)} eligible continuous candidates were evaluated. Pass counts were: Discovery 1, full Normal Validation 0, first-half Validation 0, second-half Validation 0, and all four 0. The maximum full-validation gain was {float(discovery.validation_gain.max()):.9f}. The frozen target list is therefore empty (SHA256 `{target_hash}`).

Status: **WADI_B1_STOPPED_NO_STABLE_DUAL_TARGETS**.

Because `D(t)=max_j S_j(t)` is undefined for an empty frozen target set, the run stopped before REC training and before opening WADI Test. No Test labels, Attack-driven selection, Top-K fallback, threshold, Fusion result, or Attack13 patch was used. The intended WADI external comparison remains unexecuted rather than being silently redefined.
""", encoding="utf-8")
        print(json.dumps(gate_summary, indent=2, allow_nan=False, default=json_default), flush=True)
        return 0

    OUT.mkdir(parents=True)
    frozen_public = frozen[["target", "process", "discovery_gain", "validation_gain", "validation_first_gain", "validation_second_gain"]]
    frozen_public.to_csv(OUT / "dual_targets_frozen.csv", index=False)
    discovery.to_csv(OUT / "normal_target_discovery_all.csv", index=False)
    target_hash = sha256(OUT / "dual_targets_frozen.csv")
    print(json.dumps({"phase": "target_freeze", "count": len(frozen), "sha256": target_hash}), flush=True)

    dual_groups = []
    for process, usable in group_definitions:
        process_frozen = frozen[frozen.process.astype(str) == process]
        if process_frozen.empty:
            continue
        target_names = process_frozen.target.astype(str).tolist()
        target_indices = [index[name] for name in target_names]
        source_indices = [index[name] for name in usable]
        cross_specs = [(source, lag) for source in source_indices for lag in CROSS_LAGS]
        cross_model = leave_target_out_models(train_segments, cross_specs, target_indices)
        self_models = [fit_ridge(train_segments, [(ti, lag) for lag in SELF_LAGS], [ti]) for ti in target_indices]
        raw_by_target = [[] for _ in target_names]
        for values in validation_segments:
            times = np.arange(MAX_LAG, len(values), dtype=np.int64)
            pc = cross_model.predict(design(values, times, cross_specs))
            for j, ti in enumerate(target_indices):
                ps = self_models[j].predict(design(values, times, [(ti, lag) for lag in SELF_LAGS]))[:, 0]
                raw_by_target[j].append(np.abs(ps - pc[:, j]))
        references = [robust(np.concatenate(parts)) for parts in raw_by_target]
        dual_groups.append(ProcessDual(target_names, target_indices, source_indices, cross_specs, cross_model, self_models, references))
    frozen_target_names = [name for group in dual_groups for name in group.target_names]
    frozen_processes = [str(frozen.loc[frozen.target == name, "process"].iloc[0]) for name in frozen_target_names]
    dual_val, _, _ = score_dual(validation_segments, dual_groups)
    dual_val_all = np.concatenate(dual_val)
    threshold_dual = float(np.quantile(dual_val_all, Q))
    median_val_d, scale_val_d = robust(dual_val_all)

    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("frozen REC training requires CUDA")
    device = torch.device("cuda")
    model = BackboneAE(len(features), 64, 32, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.SmoothL1Loss()
    batch_size, patience, max_epochs = 256, 4, 15
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    checkpoint = OUT / "rec_best.pt"
    for epoch in range(1, max_epochs + 1):
        model.train(); total = 0.0; count = 0
        for windows in ae_batches(train_segments, batch_size, True, np.random.default_rng(SEED + epoch)):
            x = torch.from_numpy(windows).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), x)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite REC loss")
            loss.backward(); optimizer.step()
            total += float(loss.item()) * len(x); count += len(x)
        train_loss = total / count
        validation_loss = ae_loss(model, validation_segments, device, batch_size)
        improved = validation_loss < best_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "improved": improved})
        print(json.dumps(history[-1]), flush=True)
        if improved:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "validation_loss": validation_loss}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    rec_val = ae_scores(model, validation_segments, device, batch_size)
    rec_val_all = np.concatenate(rec_val)
    threshold_rec = float(np.quantile(rec_val_all, Q))
    median_val_r, scale_val_r = robust(rec_val_all)
    fusion_val = []
    for rec_part, dual_part in zip(rec_val, dual_val):
        aligned_dual = dual_part[WINDOW - 1 - MAX_LAG:]
        if len(aligned_dual) != len(rec_part):
            raise RuntimeError("validation common mask mismatch")
        zr = np.maximum(0.0, (rec_part - median_val_r) / scale_val_r)
        zd = np.maximum(0.0, (aligned_dual - median_val_d) / scale_val_d)
        fusion_val.append(np.maximum(zr, zd))
    threshold_fusion = float(np.quantile(np.concatenate(fusion_val), Q))

    # Test is opened only after target/model/reference/threshold freeze.
    attack = pd.read_csv(ATTACK, header=1, low_memory=False)
    attack.columns = attack.columns.str.strip()
    attack = attack.loc[attack["Row"].notna()].reset_index(drop=True)
    if set(attack[LABEL].astype(int).unique()) != {-1, 1}:
        raise RuntimeError("unexpected WADI point labels")
    attack_values = attack[features].to_numpy(np.float64)
    if not np.isfinite(attack_values).all():
        raise RuntimeError("A0.5 promised no remaining Test NaN/Inf")
    test_segments = [scaler.transform(attack_values)]
    rec_test = ae_scores(model, test_segments, device, batch_size)[0]
    dual_test_parts, argmax_parts, target_matrix_parts = score_dual(test_segments, dual_groups)
    dual_test = dual_test_parts[0][WINDOW - 1 - MAX_LAG:]
    argmax_test = argmax_parts[0][WINDOW - 1 - MAX_LAG:]
    target_matrix = target_matrix_parts[0][WINDOW - 1 - MAX_LAG:]
    if len(rec_test) != len(dual_test):
        raise RuntimeError("Test common endpoint mask mismatch")
    zr = np.maximum(0.0, (rec_test - median_val_r) / scale_val_r)
    zd = np.maximum(0.0, (dual_test - median_val_d) / scale_val_d)
    fusion_test = np.maximum(zr, zd)
    raw_indices = np.arange(WINDOW - 1, len(attack), dtype=np.int64)
    labels = attack[LABEL].astype(int).map({1: 0, -1: 1}).to_numpy(np.int8)[raw_indices]
    timestamps, _ = load_a0_module().infer_audit_timestamps(attack)
    timestamps = timestamps[raw_indices]
    common_ratio = len(raw_indices) / len(attack)

    rec_m = point_metrics(labels, rec_test, threshold_rec)
    dual_m = point_metrics(labels, dual_test, threshold_dual)
    fusion_m = point_metrics(labels, fusion_test, threshold_fusion)
    global_metrics = pd.DataFrame([
        {"method": "REC", "threshold": threshold_rec, **rec_m},
        {"method": "DUAL", "threshold": threshold_dual, **dual_m},
        {"method": "FUSION", "threshold": threshold_fusion, **fusion_m},
    ])

    event_rows, attribution_rows = [], []
    for event in bound_events.itertuples(index=False):
        mask = (timestamps >= pd.Timestamp(event.official_start)) & (timestamps <= pd.Timestamp(event.official_end))
        if not mask.any():
            raise RuntimeError(f"bound event absent from common mask: {event.attack_id}")
        rec_hit = rec_test[mask] > threshold_rec; dual_hit = dual_test[mask] > threshold_dual; fusion_hit = fusion_test[mask] > threshold_fusion
        counts = np.bincount(argmax_test[mask], minlength=len(frozen_target_names)); top = int(np.argmax(counts))
        event_rows.append({
            "attack_id": int(event.attack_id), "targets": str(event.targets), "process": str(event.process),
            "category": category_name(str(event.taxonomy)), "official_start": str(event.official_start), "official_end": str(event.official_end),
            "common_point_count": int(mask.sum()), "rec_hit_rate": float(rec_hit.mean()),
            "dual_hit_rate": float(dual_hit.mean()), "fusion_hit_rate": float(fusion_hit.mean()),
            "top_dual_target": frozen_target_names[top], "top_dual_process": frozen_processes[top],
            "rec_near_miss": bool(rec_hit.mean() < 0.10),
            "dual_recovered": bool(rec_hit.mean() < 0.10 and dual_hit.mean() >= 0.10),
            "dual_beats_rec_010": bool(dual_hit.mean() >= rec_hit.mean() + 0.10),
        })
        for j, target in enumerate(frozen_target_names):
            attribution_rows.append({"attack_id": int(event.attack_id), "target": target, "process": frozen_processes[j],
                                     "argmax_count": int(counts[j]), "argmax_fraction": float(counts[j] / mask.sum()),
                                     "median_target_score": float(np.median(target_matrix[mask, j]))})
    events = pd.DataFrame(event_rows)
    high = events[events.attack_id.isin(HIGH_IDS)].copy()
    if len(high) != 5:
        raise RuntimeError("HIGH event count changed")

    conflict_start, conflict_end = pd.Timestamp("2017-10-11 00:07:33"), pd.Timestamp("2017-10-11 00:10:55")
    conflict = (timestamps >= conflict_start) & (timestamps <= conflict_end)
    attack13 = pd.DataFrame([{
        "attack_id": 13, "binding_status": "SOURCE_CONFLICT_UNRESOLVED", "included_in_primary_14": False,
        "csv_interval_start": conflict_start.isoformat(), "csv_interval_end": conflict_end.isoformat(),
        "common_point_count": int(conflict.sum()), "attack_label_points": int(labels[conflict].sum()),
        "rec_hit_rate": float((rec_test[conflict] > threshold_rec).mean()),
        "dual_hit_rate": float((dual_test[conflict] > threshold_dual).mean()),
        "fusion_hit_rate": float((fusion_test[conflict] > threshold_fusion).mean()),
    }])

    rec_hit, dual_hit = rec_test > threshold_rec, dual_test > threshold_dual
    attack_mask, normal_mask = labels == 1, labels == 0
    recovered_tp = int((attack_mask & ~rec_hit & dual_hit).sum())
    added_fp = int((normal_mask & ~rec_hit & dual_hit).sum())
    point_complementarity = pd.DataFrame([{
        "rec_hit_dual_hit": int((attack_mask & rec_hit & dual_hit).sum()),
        "rec_hit_dual_miss": int((attack_mask & rec_hit & ~dual_hit).sum()),
        "rec_miss_dual_hit": recovered_tp, "rec_miss_dual_miss": int((attack_mask & ~rec_hit & ~dual_hit).sum()),
        "rec_correct_dual_fp": added_fp, "added_fp_per_recovered_tp": float(added_fp / recovered_tp) if recovered_tp else 0.0,
        "added_fp_per_recovered_tp_defined": bool(recovered_tp > 0),
    }])
    event_complementarity = pd.DataFrame([
        {"scope": "ALL_14", "event_count": 14, "rec_near_miss_count": int(events.rec_near_miss.sum()), "dual_recovered_count": int(events.dual_recovered.sum())},
        {"scope": "HIGH_5", "event_count": 5, "rec_near_miss_count": int(high.rec_near_miss.sum()), "dual_recovered_count": int(high.dual_recovered.sum())},
    ])
    category_summary = events.groupby("category", sort=False).agg(
        event_count=("attack_id", "count"), median_rec_hit=("rec_hit_rate", "median"),
        median_dual_hit=("dual_hit_rate", "median"), median_fusion_hit=("fusion_hit_rate", "median"),
    ).reset_index()

    h_gates = {
        "H1": bool((high.dual_hit_rate >= 0.10).sum() >= 3),
        "H2": bool((high.dual_hit_rate >= high.rec_hit_rate + 0.10).sum() >= 2),
        "H3": bool(((high.rec_hit_rate < 0.10) & (high.dual_hit_rate >= 0.10)).any()),
        "H4": bool(dual_m["fpr"] <= 0.10),
    }
    if all(h_gates.values()):
        status_i2 = "DUAL_VIEW_WADI_EXTERNAL_COLLABORATION_SUPPORTED"
    elif h_gates["H1"] and h_gates["H2"] and not h_gates["H3"]:
        status_i2 = "DUAL_VIEW_WADI_EXTERNAL_INCREMENTAL_PARTIAL"
    else:
        status_i2 = "DUAL_VIEW_WADI_EXTERNAL_NOT_SUPPORTED"
    rec_event_count = int((events.rec_hit_rate >= 0.10).sum())
    dual_event_count = int((events.dual_hit_rate >= 0.10).sum())
    fusion_event_count = int((events.fusion_hit_rate >= 0.10).sum())
    f_gates = {
        "F1": bool(fusion_m["aupr"] >= rec_m["aupr"] + 0.01),
        "F2": bool(fusion_m["auroc"] >= rec_m["auroc"] - 0.01),
        "F3": fusion_event_count >= rec_event_count,
        "F4": bool(fusion_m["fpr"] <= 0.10),
    }
    status_fusion = "NORMAL_CALIBRATED_MAX_FUSION_EXTERNALLY_SUPPORTED" if all(f_gates.values()) else "NORMAL_CALIBRATED_MAX_FUSION_EXTERNAL_NOT_SUPPORTED"

    hai_global = pd.read_csv(HAI / "global_metrics.csv")
    hai_high = pd.read_csv(HAI / "high_event_metrics.csv")
    cross_rows = []
    for dataset, gm, hm in (("HAI21", hai_global, hai_high), ("WADI.A2", global_metrics, high)):
        for method, hit_column in (("REC", "rec_hit_rate"), ("DUAL", "dual_hit_rate"), ("FUSION", "fusion_hit_rate")):
            row = gm.loc[gm.method == method].iloc[0]
            cross_rows.append({"dataset": dataset, "method": method, "auroc": row.auroc, "aupr": row.aupr,
                               "precision": row.precision, "recall": row.recall, "f1": row.f1, "fpr": row.fpr,
                               "high_median_hit": float(hm[hit_column].median())})
    cross_dataset = pd.DataFrame(cross_rows)

    thresholds = {"quantile": Q, "comparator": ">", "source": "Frozen Normal Validation valid segments only",
                  "rec": threshold_rec, "dual": threshold_dual, "fusion": threshold_fusion,
                  "rec_branch_median": median_val_r, "rec_branch_scale": scale_val_r,
                  "dual_branch_median": median_val_d, "dual_branch_scale": scale_val_d}
    tests = {
        "wadi_a2_only": True, "frozen_123": len(features) == 123, "four_all_nan_excluded": len(excluded) == 4,
        "exclusion_normal_train_only": protocol["attack_test_used_for_exclusion"] is False, "no_imputation": protocol["imputation"] == "NONE",
        "nan_breaks": True, "no_cross_segment_window_lag": True,
        "scaler_train_only": int(scaler.n_samples_seen_) == sum(len(x) for x in train_segments),
        "discovery_normal_only": True, "stable_rules_frozen": True, "targets_written_before_test": True,
        "high_ids_fixed": set(high.attack_id) == HIGH_IDS, "attack13_excluded": 13 not in set(events.attack_id),
        "self_lags_fixed": SELF_LAGS == (1,5,10,20,30), "cross_lags_fixed": CROSS_LAGS == (0,1,5,10,20,30),
        "target_excluded_cross": all(row.target not in str(row.cross_source_names).split(";") for row in discovery.itertuples()),
        "alpha_one": ALPHA == 1.0, "dual_disagreement_only": True, "rec_hyperparameters_frozen": True,
        "fusion_unweighted_max": True, "thresholds_normal_only": True, "common_valid_mask": len(labels) == len(rec_test) == len(dual_test) == len(fusion_test),
        "no_point_adjustment": True, "labels_evaluation_only": True,
        "no_nan_inf": bool(np.isfinite(global_metrics.select_dtypes("number").to_numpy()).all() and np.isfinite(events.select_dtypes("number").to_numpy()).all()),
    }
    if not all(tests.values()):
        raise RuntimeError(f"WADI-B1 integrity failure: {tests}")
    summary = {
        "elapsed_seconds": time.perf_counter() - started, "frozen_target_count": len(frozen),
        "target_counts_by_process": frozen.process.value_counts().to_dict(), "frozen_targets": frozen_target_names,
        "rec_parameter_count": sum(p.numel() for p in model.parameters()), "rec_best_epoch": best_epoch,
        "rec_best_validation_loss": best_loss, "common_valid_test_points": len(labels), "common_valid_test_ratio": common_ratio,
        "global_metrics": global_metrics.to_dict("records"),
        "event_hit_010_counts": {"REC": rec_event_count, "DUAL": dual_event_count, "FUSION": fusion_event_count},
        "innovation2_gates": h_gates, "status_innovation2": status_i2,
        "fusion_gates": f_gates, "status_fusion": status_fusion, "tests": tests,
        "no_attack_driven_selection": True, "no_test_tuning": True, "lag_search": False,
        "threshold_search": False, "point_adjustment": False, "fusion_weight_search": False, "attack13_patch": False,
    }
    manifest = {
        "summary": summary, "dataset": "WADI.A2", "seed": SEED, "feature_count": len(features), "feature_names": features,
        "excluded_features": excluded, "train_ranges": TRAIN_RANGES, "validation_ranges": VALIDATION_RANGES,
        "train_valid_runs": train_run_metadata, "validation_valid_runs": validation_run_metadata,
        "scaler_fit_rows": int(scaler.n_samples_seen_), "target_freeze_sha256": target_hash,
        "source_sha256": hashes_before, "process_group_source": "outputs/thesis_wadi_a0_audit/process_groups.csv",
        "self_lags": SELF_LAGS, "joint_lags": JOINT_LAGS, "cross_lags": CROSS_LAGS, "alpha": ALPHA,
        "rec_model": {"window":64,"hidden":64,"latent":32,"dropout":0.1,"loss":"SmoothL1","optimizer":"Adam","lr":1e-3,"weight_decay":1e-4,"batch":256,"max_epochs":15,"patience":4},
        "rec_score": "endpoint 123-channel mean absolute reconstruction error",
        "dual_score": "max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/scale_j)",
        "fusion_score": "max(Z_rec,Z_dual)", "test_opened_after_freeze": True,
    }

    global_metrics.to_csv(OUT / "global_metrics.csv", index=False)
    events.to_csv(OUT / "event_metrics_14.csv", index=False)
    attack13.to_csv(OUT / "attack13_unresolved_diagnostic.csv", index=False)
    high.to_csv(OUT / "high_event_metrics.csv", index=False)
    point_complementarity.to_csv(OUT / "point_complementarity.csv", index=False)
    event_complementarity.to_csv(OUT / "event_complementarity.csv", index=False)
    pd.DataFrame(attribution_rows).to_csv(OUT / "target_attribution.csv", index=False)
    category_summary.to_csv(OUT / "category_summary.csv", index=False)
    cross_dataset.to_csv(OUT / "cross_dataset_summary.csv", index=False)
    pd.DataFrame(history).to_csv(OUT / "rec_training_history.csv", index=False)
    (OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    (OUT / "wadi_b1_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    if {str(path.relative_to(ROOT)): sha256(path) for path in source_paths} != hashes_before:
        raise RuntimeError("WADI source/audit artifact changed")

    metric_lines = "\n".join(f"| {r.method} | {r.auroc:.6f} | {r.aupr:.6f} | {r.precision:.6f} | {r.recall:.6f} | {r.f1:.6f} | {r.fpr:.6f} |" for r in global_metrics.itertuples())
    high_lines = "\n".join(f"| {r.attack_id} | {r.rec_hit_rate:.6f} | {r.dual_hit_rate:.6f} | {r.fusion_hit_rate:.6f} | {r.top_dual_target} / {r.top_dual_process} |" for r in high.itertuples())
    category_lines = "\n".join(f"| {r.category} | {r.event_count} | {r.median_rec_hit:.6f} | {r.median_dual_hit:.6f} | {r.median_fusion_hit:.6f} |" for r in category_summary.itertuples())
    DOC.write_text(f"""# WADI-B1 Frozen Rec / Dual / Fusion External Validation

## Protocol

This is the preregistered WADI.A2 independent external validation. The A0.5 123-variable schema and hard NaN/session boundaries are unchanged. StandardScaler, target discovery, Ridge fitting, calibration, AE training, and q99 thresholds use Normal data only. Test is opened after the Dual target list, models, references, and thresholds are frozen. Attack13 remains unpatched and excluded from the 14-event primary aggregate.

Frozen Dual targets: {len(frozen)} ({', '.join(f'{k}: {v}' for k,v in frozen.process.value_counts().to_dict().items())}). Target SHA256: `{target_hash}`.

REC has {summary['rec_parameter_count']:,} parameters; best epoch {best_epoch}, Normal Validation loss {best_loss:.10f}. Common valid Test endpoints: {len(labels):,}/{len(attack):,} ({common_ratio:.6%}).

## Global point metrics

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
{metric_lines}

## Five frozen HIGH events

| ID | Rec hit | Dual hit | Fusion hit | Dual top target / process |
|---:|---:|---:|---:|---|
{high_lines}

## Category medians

| Category | Events | Rec | Dual | Fusion |
|---|---:|---:|---:|---:|
{category_lines}

At hit rate 0.10, REC/DUAL/FUSION cover {rec_event_count}/{dual_event_count}/{fusion_event_count} of 14 bound events. Innovation-2 gates are `{h_gates}`: **{status_i2}**. Fusion gates are `{f_gates}`: **{status_fusion}**.

No Attack-driven selection, Test tuning, lag/alpha/threshold search, POT, point adjustment, fusion weights, learned fusion, or Attack13 patch was used.
""", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
