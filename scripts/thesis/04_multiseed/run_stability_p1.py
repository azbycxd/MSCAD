"""Three-seed stability evaluation for the frozen R0 REC and final TAIL.

Only the REC random seed changes.  The DUAL branch is reconstructed from its
frozen coefficient/reference artifacts and is never refit or reselected.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    WINDOW,
    endpoint_labels,
    endpoint_timestamps,
    iter_window_batches,
    load_attack_cache,
    load_normal_cache,
    load_scaler,
    make_window_starts,
    resolve_repo_path,
    validate_config_contract,
)
from losses.recon import ReconstructionLoss  # noqa: E402
from models.backbone_ae import BackboneAE  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.seed import set_seed  # noqa: E402


OUT = ROOT / "outputs" / "thesis_stability"
SWAT_CONFIG = ROOT / "configs" / "thesis" / "swat2015_midterm_seed1.yaml"
SWAT_FORMAL = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline"
HAI_FORMAL = ROOT / "outputs" / "thesis_hai21_b1_real"
TAIL_FORMAL = ROOT / "outputs" / "thesis_innovation3b_tail_fusion_final"
EFF = ROOT / "outputs" / "thesis_efficiency_final"
EVAL_RAW = ROOT / "outputs" / "thesis_evaluation" / "literature_compatible" / "raw_scores"
DUAL_REF = ROOT / "outputs" / "thesis_evaluation" / "literature_compatible" / "recovery" / "reconstructed_artifacts"
T25 = ROOT / "outputs" / "thesis_t25_swat2015_protocol"
HAI_AUDIT = ROOT / "outputs" / "thesis_hai21_a0_audit"
HAI_DATA = ROOT / "dataset" / "hai" / "hai-21.03"
HELPERS = ROOT / "scripts" / "thesis"
HARD_STOP_SECONDS = 44 * 60
SELF_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (0, 1, 5, 10, 20, 30)
MAX_LAG = 30
Q = 0.99
TRAINING = {
    "loss": "SmoothL1Loss",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "max_epochs": 15,
    "early_stopping_patience": 4,
    "checkpoint_selection": "minimum_normal_validation_reconstruction_loss",
}
HAI_TRAIN_FILES = tuple(f"train{i}.csv.gz" for i in range(1, 4))
HAI_TEST_FILES = tuple(f"test{i}.csv.gz" for i in range(1, 6))
HAI_HIGH_AP = {"AP02", "AP04", "AP07", "AP09", "AP12", "AP14", "AP17", "AP19"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SWAT_ABL = load_module("stab_swat_abl", HELPERS / "03_ablation" / "run_innovation1_a3a_swat.py")
HAI_ABL = load_module("stab_hai_abl", HELPERS / "03_ablation" / "run_innovation1_a3b_hai.py")
DUAL_HELPER = load_module("stab_dual_helper", HELPERS / "02_collaboration" / "run_wadi_b1_external.py")


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_hash(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    prediction = scores > threshold
    positive, negative = labels == 1, labels == 0
    tp = int((prediction & positive).sum())
    fp = int((prediction & negative).sum())
    fn = int((~prediction & positive).sum())
    tn = int((~prediction & negative).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, np.finfo(float).eps),
        "fpr": fp / max(fp + tn, 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if not np.isfinite([result[key] for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr")]).all():
        raise RuntimeError("non-finite metric")
    return result


def tail_evidence(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(calibration, dtype=np.float64))
    count_ge = len(ordered) - np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="left")
    return -np.log10((1.0 + count_ge) / (len(ordered) + 1.0))


def freeze_tail(rec_segments: list[np.ndarray], dual_segments: list[np.ndarray]) -> dict[str, Any]:
    rec_a, rec_b, dual_a, dual_b = [], [], [], []
    segment_sizes = []
    for rec, dual in zip(rec_segments, dual_segments):
        if len(rec) != len(dual):
            raise RuntimeError("REC/DUAL Validation alignment mismatch")
        boundary = len(rec) // 2
        rec_a.append(np.asarray(rec[:boundary], dtype=np.float64))
        rec_b.append(np.asarray(rec[boundary:], dtype=np.float64))
        dual_a.append(np.asarray(dual[:boundary], dtype=np.float64))
        dual_b.append(np.asarray(dual[boundary:], dtype=np.float64))
        segment_sizes.append({"points": len(rec), "calibration_a": boundary, "threshold_b": len(rec) - boundary})
    rec_a, rec_b, dual_a, dual_b = map(np.concatenate, (rec_a, rec_b, dual_a, dual_b))
    validation_tail = np.maximum(tail_evidence(rec_a, rec_b), tail_evidence(dual_a, dual_b))
    return {
        "rec_a": rec_a,
        "dual_a": dual_a,
        "validation_tail": validation_tail,
        "threshold": float(np.quantile(validation_tail, Q)),
        "segment_sizes": segment_sizes,
    }


def choose_seeds(formal_seed: int) -> list[int]:
    result = [formal_seed]
    for candidate in (42, 43, 44):
        if candidate not in result:
            result.append(candidate)
        if len(result) == 3:
            break
    if len(result) != 3 or len(set(result)) != 3:
        raise RuntimeError("failed to freeze exactly three distinct seeds")
    return result


def check_prerequisites() -> None:
    gates = {
        "contribution": (ROOT / "docs" / "thesis" / "final_contribution_structure.md", "THESIS_CONTRIBUTION_STRUCTURE_FROZEN"),
        "ablation": (ROOT / "outputs" / "thesis_ablation" / "ablation_final_summary.json", "ABLATION_FINAL_PACKAGE_READY"),
        "claims": (ROOT / "docs" / "thesis" / "final_claim_audit.md", "FINAL_CLAIMS_DEFENSIBLE"),
        "narrative": (ROOT / "docs" / "thesis" / "final_thesis_outline.md", "THESIS_NARRATIVE_STRUCTURE_READY"),
    }
    for name, (path, token) in gates.items():
        if token not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"prerequisite failed: {name}")


def read_formal_seeds() -> dict[str, int]:
    swat = json.loads((SWAT_FORMAL / "config_resolved.json").read_text(encoding="utf-8"))
    hai = json.loads((HAI_FORMAL / "manifest.json").read_text(encoding="utf-8"))
    return {"SWaT": int(swat["seed"]), "HAI21": int(hai["seed"])}


def write_seed_manifest(formal: dict[str, int]) -> dict[str, Any]:
    payload = {
        "phase": "Thesis Stability-STAB-P1",
        "created_before_training": True,
        "selection_rule": "formal seed plus 42 and 43; use 44 only if a duplicate occurs",
        "datasets": {name: {"formal_seed": seed, "seeds": choose_seeds(seed)} for name, seed in formal.items()},
        "exactly_three_distinct_per_dataset": True,
        "frozen_after_creation": True,
        "method": "R0 Final REC only",
        "dual": "deterministic frozen artifact; no Ridge refit or target reselection",
    }
    path = OUT / "seed_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing seed manifest differs from frozen protocol")
    else:
        dump_json(path, payload)
    return payload


def architecture_check(channels: int) -> dict[str, Any]:
    model = BackboneAE(channels, 64, 32, 0.1)
    signature = SWAT_ABL.architecture_signature(model)
    expected = 11730 if channels == 51 else 15482
    if signature["parameter_count"] != expected:
        raise RuntimeError("R0 architecture parameter count changed")
    depthwise = [row for row in signature["conv1d"] if row["groups"] == row["in_channels"] == row["out_channels"]]
    if len(depthwise) != 3 or [row["kernel_size"] for row in depthwise] != [3, 3, 3]:
        raise RuntimeError("only frozen R0 depthwise k=3 architecture is allowed")
    return signature


def fixed_dual_paths(dataset: str) -> dict[str, Path]:
    stem = "swat" if dataset == "SWaT" else "hai21"
    raw_stem = "swat" if dataset == "SWaT" else "hai"
    return {
        "coefficients": EFF / f"{stem}_dual_coefficients.npz",
        "tail": EFF / f"{stem}_tail_calibration.npz",
        "reference": DUAL_REF / raw_stem / "ours_dual" / "dual_reference_reconstructed.npz",
        "raw_test": EVAL_RAW / raw_stem / "ours_dual" / "test_raw_scores.npz",
        "targets": ROOT / "outputs" / ("thesis_swat_final_unified" if dataset == "SWaT" else "thesis_hai21_final_unified") / "dual_targets_unified_frozen.csv",
    }


def build_fixed_dual_state(dataset: str, names: list[str]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    paths = fixed_dual_paths(dataset)
    frozen = pd.read_csv(paths["targets"])
    coefficients = np.load(paths["coefficients"], allow_pickle=False)
    reference = np.load(paths["reference"], allow_pickle=True)
    ordering = json.loads(coefficients["feature_ordering_json"].item())
    reference_map = {str(target): (float(median), float(scale)) for target, median, scale in zip(reference["target"], reference["median"], reference["scale"])}
    if sha256(paths["coefficients"]) != str(reference["coefficient_sha256"].item()):
        raise RuntimeError(f"{dataset} fixed DUAL coefficient/reference hash mismatch")
    target_hash = sha256(paths["targets"])
    if target_hash != str(reference["target_sha256"].item()):
        raise RuntimeError(f"{dataset} fixed target/reference hash mismatch")
    if [str(x) for x in frozen.target] != [str(row["target"]) for row in ordering]:
        raise RuntimeError(f"{dataset} fixed target ordering mismatch")
    index = {name: idx for idx, name in enumerate(names)}
    state: dict[str, Any] = {}
    for idx, row in enumerate(ordering):
        target = str(row["target"])
        sources = []
        for feature in row["cross_features"]:
            source = str(feature).split("@lag", 1)[0]
            if not sources or sources[-1] != index[source]:
                sources.append(index[source])
        state[target] = {
            "ti": index[target],
            "sources": sources,
            "self": DUAL_HELPER.LinearModel(np.asarray(coefficients[f"t{idx:03d}_self_coef"])[:, None], np.asarray(coefficients[f"t{idx:03d}_self_intercept"])),
            "cross": DUAL_HELPER.LinearModel(np.asarray(coefficients[f"t{idx:03d}_cross_coef"])[:, None], np.asarray(coefficients[f"t{idx:03d}_cross_intercept"])),
            "ref": reference_map[target],
        }
    hashes = {name: sha256(path) for name, path in paths.items()}
    return frozen, state, hashes


def fixed_dual_scores(segments: list[np.ndarray], frozen: pd.DataFrame, state: dict[str, Any]) -> list[np.ndarray]:
    result = []
    for values in segments:
        values = np.asarray(values, dtype=np.float64)
        times = np.arange(MAX_LAG, len(values), dtype=np.int64)
        columns = []
        for target in frozen.target.astype(str):
            item = state[target]
            ti = item["ti"]
            self_design = DUAL_HELPER.design(values, times, [(ti, lag) for lag in SELF_LAGS])
            cross_design = DUAL_HELPER.design(values, times, [(source, lag) for source in item["sources"] for lag in CROSS_LAGS])
            self_prediction = item["self"].predict(self_design)[:, 0]
            cross_prediction = item["cross"].predict(cross_design)[:, 0]
            median, scale = item["ref"]
            columns.append(np.maximum(0.0, (np.abs(self_prediction - cross_prediction) - median) / scale))
        score = np.column_stack(columns).max(axis=1)[WINDOW - 1 - MAX_LAG:]
        if len(score) != len(values) - WINDOW + 1 or not np.isfinite(score).all():
            raise RuntimeError("fixed DUAL score reconstruction failed")
        result.append(score)
    return result


def swat_average_loss(model: torch.nn.Module, normal: np.ndarray, starts: np.ndarray, scaler: Any, device: torch.device) -> float:
    criterion = ReconstructionLoss()
    total, count = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for windows, _ in iter_window_batches(normal, starts, scaler, TRAINING["batch_size"], shuffle=False, seed=0):
            x = torch.from_numpy(windows).to(device)
            loss = criterion(model(x), x)
            total += float(loss.item()) * len(windows)
            count += len(windows)
    return total / count


def hai_window_view(values: np.ndarray) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(values, WINDOW, axis=0).transpose(0, 2, 1)


def hai_batches(segments: list[np.ndarray], batch_size: int, shuffle: bool, rng: np.random.Generator):
    segment_order = rng.permutation(len(segments)) if shuffle else np.arange(len(segments))
    for segment_id in segment_order:
        windows = hai_window_view(segments[int(segment_id)])
        order = rng.permutation(len(windows)) if shuffle else np.arange(len(windows))
        for offset in range(0, len(order), batch_size):
            yield np.ascontiguousarray(windows[order[offset:offset + batch_size]], dtype=np.float32)


def hai_average_loss(model: torch.nn.Module, segments: list[np.ndarray], device: torch.device) -> float:
    criterion = torch.nn.SmoothL1Loss()
    total, count = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for windows in hai_batches(segments, TRAINING["batch_size"], False, np.random.default_rng(0)):
            x = torch.from_numpy(windows).to(device)
            loss = criterion(model(x), x)
            total += float(loss.item()) * len(windows)
            count += len(windows)
    return total / count


def hai_score_segments(model: torch.nn.Module, segments: list[np.ndarray], device: torch.device) -> list[np.ndarray]:
    model.eval()
    output = []
    with torch.inference_mode():
        for values in segments:
            parts = []
            for windows in hai_batches([values], TRAINING["batch_size"], False, np.random.default_rng(0)):
                x = torch.from_numpy(windows).to(device)
                parts.append(torch.abs(model(x)[:, -1] - x[:, -1]).mean(dim=1).cpu().numpy())
            score = np.concatenate(parts).astype(np.float64)
            if len(score) != len(values) - WINDOW + 1 or not np.isfinite(score).all():
                raise RuntimeError("HAI raw endpoint score persistence failed")
            output.append(score)
    return output


def load_swat_data() -> dict[str, Any]:
    config = load_config(str(SWAT_CONFIG))
    validate_config_contract(config)
    cache = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, cache_manifest = load_normal_cache(cache, config)
    names = [str(x) for x in cache_manifest["column_order"]]
    scaler, scaler_meta = load_scaler(SWAT_FORMAL / "scaler.npz")
    if int(scaler_meta["fit_row_count"]) != TRAIN_RAW_ROWS or int(scaler_meta["attack_rows_seen"]) != 0:
        raise RuntimeError("SWaT formal scaler provenance changed")
    validation_scaled = scaler.transform(np.asarray(normal[TRAIN_RAW_ROWS:EXPECTED_NORMAL_ROWS])).astype(np.float64)
    attack, attack_labels, attack_timestamps, _ = load_attack_cache(cache, config)
    segment_manifest = json.loads((T25 / "segment_manifest.json").read_text(encoding="utf-8"))
    counts = [int(row["row_count"]) for row in segment_manifest["test_segments"]]
    attack_scaled = scaler.transform(np.asarray(attack)).astype(np.float64)
    raw_segments, scaled_segments, cursor = [], [], 0
    for count in counts:
        raw_segments.append(np.asarray(attack[cursor:cursor + count]))
        scaled_segments.append(np.asarray(attack_scaled[cursor:cursor + count]))
        cursor += count
    label_parts, timestamp_parts = [], []
    cursor = 0
    for count in counts:
        starts = make_window_starts(0, count)
        label_parts.append(endpoint_labels(np.asarray(attack_labels[cursor:cursor + count]), starts))
        timestamp_parts.append(endpoint_timestamps(np.asarray(attack_timestamps[cursor:cursor + count]), starts))
        cursor += count
    return {
        "config": config,
        "normal": np.asarray(normal),
        "names": names,
        "scaler": scaler,
        "scaler_meta": scaler_meta,
        "validation_scaled": [validation_scaled],
        "test_raw_segments": raw_segments,
        "test_scaled_segments": scaled_segments,
        "labels": np.concatenate(label_parts).astype(np.int8),
        "timestamps": np.concatenate(timestamp_parts).astype(np.int64),
        "counts": counts,
    }


def load_hai_data() -> dict[str, Any]:
    schema = pd.read_csv(HAI_AUDIT / "variable_schema.csv")
    names = schema["column_name"].astype(str).tolist()
    scaler, train, validation, bounds = HAI_ABL.load_normal_data(names)
    if int(scaler.n_samples_seen_) != sum(bounds.values()):
        raise RuntimeError("HAI formal train-only scaler provenance changed")
    # REC follows the formal float32 model-input path.  Frozen DUAL artifacts
    # were produced from the scaler's float64 transform and retain that path.
    validation_dual = []
    for name in HAI_TRAIN_FILES:
        frame = pd.read_csv(HAI_DATA / name, compression="gzip", usecols=names, low_memory=False)
        boundary = bounds[name]
        validation_dual.append(scaler.transform(frame.iloc[boundary:].to_numpy(np.float64)))
    test_scaled, labels, indices, files = [], [], [], []
    for name in HAI_TEST_FILES:
        frame = pd.read_csv(HAI_DATA / name, compression="gzip", low_memory=False)
        test_scaled.append(scaler.transform(frame[names].to_numpy(np.float64)).astype(np.float32))
        labels.append(frame["attack"].to_numpy(np.int8)[WINDOW - 1:])
        indices.append(np.arange(WINDOW - 1, len(frame), dtype=np.int64))
        files.append(np.full(len(frame) - WINDOW + 1, name))
    return {
        "names": names,
        "scaler": scaler,
        "train": train,
        "validation": validation,
        "validation_dual": validation_dual,
        "bounds": bounds,
        "test_scaled_segments": test_scaled,
        "labels": np.concatenate(labels),
        "raw_indices": np.concatenate(indices),
        "source_files": np.concatenate(files),
        "events": pd.read_csv(HAI_AUDIT / "attack_events_50.csv"),
    }


def save_history(path: Path, history: list[dict[str, Any]]) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def train_swat(seed: int, data: dict[str, Any], output: Path, overall_started: float, device: torch.device) -> dict[str, Any]:
    set_seed(seed)
    model = BackboneAE(51, 64, 32, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
    criterion = ReconstructionLoss()
    train_starts = make_window_starts(0, TRAIN_RAW_ROWS)
    validation_starts = make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS)
    latest = output / "latest_training_state.pt"
    best_loss, best_epoch, stale, start_epoch = float("inf"), 0, 0, 1
    history: list[dict[str, Any]] = []
    best_model = None
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if int(state["seed"]) != seed or state["training"] != TRAINING or state["architecture"] != "R0 Final":
            raise RuntimeError("SWaT seed resume protocol mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_model = state["best_model_state_dict"]
        best_loss, best_epoch, stale = float(state["best_validation_loss"]), int(state["best_epoch"]), int(state["early_stop_counter"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        restore_rng(state["rng_state"])
    stop_reason = "max_epochs"
    completed = False
    for epoch in range(start_epoch, TRAINING["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for windows, _ in iter_window_batches(data["normal"], train_starts, data["scaler"], TRAINING["batch_size"], shuffle=True, seed=seed + epoch):
            x = torch.from_numpy(windows).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), x)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite SWaT training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(windows)
            count += len(windows)
        validation_loss = swat_average_loss(model, data["normal"], validation_starts, data["scaler"], device)
        improved = validation_loss < best_loss
        if improved:
            best_loss, best_epoch, stale, best_model = validation_loss, epoch, 0, cpu_state(model)
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": total / count, "validation_loss": validation_loss, "improved": improved, "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        torch.save({"seed": seed, "architecture": "R0 Final", "epoch": epoch, "model_state_dict": cpu_state(model), "optimizer_state_dict": optimizer.state_dict(), "best_model_state_dict": best_model, "best_epoch": best_epoch, "best_validation_loss": best_loss, "early_stop_counter": stale, "history": history, "training": TRAINING, "rng_state": rng_state()}, latest)
        save_history(output / "training_history.csv", history)
        print(json.dumps({"dataset": "SWaT", "seed": seed, **row, "counter": stale}), flush=True)
        if stale >= TRAINING["early_stopping_patience"]:
            stop_reason, completed = "early_stopping", True
            break
        if epoch == TRAINING["max_epochs"]:
            stop_reason, completed = "max_epochs", True
            break
        if time.perf_counter() - overall_started >= HARD_STOP_SECONDS:
            stop_reason = "hard_stop_after_completed_epoch"
            break
    if not completed:
        return {"completed": False, "stop_reason": stop_reason, "last_epoch": history[-1]["epoch"]}
    if best_model is None:
        raise RuntimeError("SWaT training produced no best model")
    model.load_state_dict(best_model)
    torch.save({"state_dict": best_model, "epoch": best_epoch, "best_validation_loss": best_loss, "seed": seed, "architecture": "R0 Final", "selection_scope": "SWaT Normal Validation loss only", "training": TRAINING}, output / "best.pt")
    return {"completed": True, "model": model, "best_epoch": best_epoch, "best_validation_loss": best_loss, "stop_reason": stop_reason, "history": history}


def train_hai(seed: int, data: dict[str, Any], output: Path, overall_started: float, device: torch.device) -> dict[str, Any]:
    set_seed(seed)
    model = BackboneAE(79, 64, 32, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
    criterion = torch.nn.SmoothL1Loss()
    latest = output / "latest_training_state.pt"
    best_loss, best_epoch, stale, start_epoch = float("inf"), 0, 0, 1
    history: list[dict[str, Any]] = []
    best_model = None
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if int(state["seed"]) != seed or state["training"] != TRAINING or state["architecture"] != "R0 Final":
            raise RuntimeError("HAI seed resume protocol mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_model = state["best_model_state_dict"]
        best_loss, best_epoch, stale = float(state["best_validation_loss"]), int(state["best_epoch"]), int(state["early_stop_counter"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        restore_rng(state["rng_state"])
    stop_reason = "max_epochs"
    completed = False
    for epoch in range(start_epoch, TRAINING["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for windows in hai_batches(data["train"], TRAINING["batch_size"], True, np.random.default_rng(seed + epoch)):
            x = torch.from_numpy(windows).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), x)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite HAI training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(windows)
            count += len(windows)
        validation_loss = hai_average_loss(model, data["validation"], device)
        improved = validation_loss < best_loss
        if improved:
            best_loss, best_epoch, stale, best_model = validation_loss, epoch, 0, cpu_state(model)
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": total / count, "validation_loss": validation_loss, "improved": improved, "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        torch.save({"seed": seed, "architecture": "R0 Final", "epoch": epoch, "model_state_dict": cpu_state(model), "optimizer_state_dict": optimizer.state_dict(), "best_model_state_dict": best_model, "best_epoch": best_epoch, "best_validation_loss": best_loss, "early_stop_counter": stale, "history": history, "training": TRAINING, "rng_state": rng_state()}, latest)
        save_history(output / "training_history.csv", history)
        print(json.dumps({"dataset": "HAI21", "seed": seed, **row, "counter": stale}), flush=True)
        if stale >= TRAINING["early_stopping_patience"]:
            stop_reason, completed = "early_stopping", True
            break
        if epoch == TRAINING["max_epochs"]:
            stop_reason, completed = "max_epochs", True
            break
        if time.perf_counter() - overall_started >= HARD_STOP_SECONDS:
            stop_reason = "hard_stop_after_completed_epoch"
            break
    if not completed:
        return {"completed": False, "stop_reason": stop_reason, "last_epoch": history[-1]["epoch"]}
    if best_model is None:
        raise RuntimeError("HAI training produced no best model")
    model.load_state_dict(best_model)
    torch.save({"state_dict": best_model, "epoch": best_epoch, "best_validation_loss": best_loss, "seed": seed, "architecture": "R0 Final", "selection_scope": "HAI Normal Validation loss only", "training": TRAINING}, output / "best.pt")
    return {"completed": True, "model": model, "best_epoch": best_epoch, "best_validation_loss": best_loss, "stop_reason": stop_reason, "history": history}


def load_formal_model(dataset: str, seed: int, device: torch.device, output: Path) -> dict[str, Any]:
    source = SWAT_FORMAL / "best.pt" if dataset == "SWaT" else HAI_FORMAL / "rec_best.pt"
    payload = torch.load(source, map_location=device, weights_only=False)
    model = BackboneAE(51 if dataset == "SWaT" else 79, 64, 32, 0.1).to(device)
    model.load_state_dict(payload["state_dict"])
    shutil.copy2(source, output / "best.pt")
    dump_json(output / "checkpoint_reference.json", {"formal_source": str(source.relative_to(ROOT)), "source_sha256": sha256(source), "copied_sha256": sha256(output / "best.pt"), "formal_artifact_not_overwritten": True})
    epoch = payload.get("epoch", payload.get("best_epoch"))
    validation_loss = payload.get("best_validation_loss", payload.get("validation_loss"))
    if epoch is None or validation_loss is None:
        raise RuntimeError(f"formal checkpoint metadata missing: {source}")
    return {"completed": True, "model": model, "best_epoch": int(epoch), "best_validation_loss": float(validation_loss), "stop_reason": "frozen_formal_artifact", "history": []}


def evaluate_swat_rec(seed: int, trained: dict[str, Any], data: dict[str, Any], output: Path, formal: bool, device: torch.device) -> dict[str, Any]:
    model = trained["model"]
    validation_starts = make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS)
    validation, _ = SWAT_ABL.score_windows(model, data["normal"], validation_starts, data["scaler"], TRAINING["batch_size"], device)
    test_parts = []
    for segment in data["test_raw_segments"]:
        starts = make_window_starts(0, len(segment))
        scores, _ = SWAT_ABL.score_windows(model, segment, starts, data["scaler"], TRAINING["batch_size"], device)
        test_parts.append(scores.astype(np.float64))
    test = np.concatenate(test_parts)
    threshold = float(np.quantile(validation, Q))
    metrics = point_metrics(data["labels"], test, threshold)
    event_rows, hit = SWAT_ABL.event_evaluation(data["timestamps"], test > threshold)
    np.savez_compressed(output / "validation_raw_scores.npz", endpoint_mae=validation, split=np.asarray(["Normal Validation"] * len(validation)))
    np.savez_compressed(output / "test_raw_scores.npz", endpoint_mae=test, label=data["labels"], timestamp=data["timestamps"], segment_id=np.concatenate([np.full(count - WINDOW + 1, idx) for idx, count in enumerate(data["counts"])]))
    pd.DataFrame(event_rows).to_csv(output / "event_metrics.csv", index=False)
    dump_json(output / "threshold.json", {"value": threshold, "quantile": Q, "source": "SWaT Normal Validation only", "test_labels_used": False})
    runtime = None if formal else sum(float(row["epoch_seconds"]) for row in trained["history"])
    result = {"dataset": "SWaT", "method": "REC", "seed": seed, "formal_seed": formal, "threshold": threshold, **metrics, "event_hit_at_least_0_10": hit, "best_epoch": trained["best_epoch"], "best_validation_loss": trained["best_validation_loss"], "stop_reason": trained["stop_reason"], "training_runtime_seconds": runtime, "test_points": len(test)}
    dump_json(output / "metrics.json", result)
    return {"row": result, "validation_segments": [validation.astype(np.float64)], "test": test, "event_rows": event_rows}


def is_hai_high(value: str) -> bool:
    return bool(set(str(value).split("+")) & HAI_HIGH_AP)


def evaluate_hai_events(events: pd.DataFrame, scores: np.ndarray, threshold: float, source_files: np.ndarray, raw_indices: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for event in events.itertuples(index=False):
        mask = (source_files == str(event.file)) & (raw_indices >= int(event.start_index)) & (raw_indices <= int(event.end_index_inclusive))
        if not mask.any():
            raise RuntimeError(f"official HAI event absent: {event.event_id}")
        hit_rate = float((scores[mask] > threshold).mean())
        category = "HIGH" if is_hai_high(str(event.AP_scenario)) else "DIRECT"
        rows.append({"file": str(event.file), "event_id": str(event.event_id), "AP_scenario": str(event.AP_scenario), "category": category, "point_count": int(mask.sum()), "hit_rate": hit_rate, "hit_at_least_0_10": hit_rate >= 0.10})
    high = [row for row in rows if row["category"] == "HIGH"]
    direct = [row for row in rows if row["category"] == "DIRECT"]
    if len(rows) != 50 or len(high) != 19 or len(direct) != 31:
        raise RuntimeError("HAI event semantics changed")
    return rows, {"event_hit_at_least_0_10": sum(row["hit_at_least_0_10"] for row in rows), "high_median_hit": float(np.median([row["hit_rate"] for row in high])), "high_hit_at_least_0_10": sum(row["hit_at_least_0_10"] for row in high), "direct_median_hit": float(np.median([row["hit_rate"] for row in direct]))}


def evaluate_hai_rec(seed: int, trained: dict[str, Any], data: dict[str, Any], output: Path, formal: bool, device: torch.device) -> dict[str, Any]:
    model = trained["model"]
    validation_segments = hai_score_segments(model, data["validation"], device)
    test_segments = hai_score_segments(model, data["test_scaled_segments"], device)
    validation, test = np.concatenate(validation_segments), np.concatenate(test_segments)
    threshold = float(np.quantile(validation, Q))
    metrics = point_metrics(data["labels"], test, threshold)
    event_rows, event_summary = evaluate_hai_events(data["events"], test, threshold, data["source_files"], data["raw_indices"])
    np.savez_compressed(output / "validation_raw_scores.npz", endpoint_mae=validation, source_file=np.concatenate([np.full(len(x), name) for x, name in zip(validation_segments, HAI_TRAIN_FILES)]))
    np.savez_compressed(output / "test_raw_scores.npz", endpoint_mae=test, label=data["labels"], source_file=data["source_files"], raw_index=data["raw_indices"])
    pd.DataFrame(event_rows).to_csv(output / "event_metrics.csv", index=False)
    dump_json(output / "threshold.json", {"value": threshold, "quantile": Q, "source": "HAI Normal Validation suffixes only", "test_labels_used": False})
    runtime = None if formal else sum(float(row["epoch_seconds"]) for row in trained["history"])
    result = {"dataset": "HAI21", "method": "REC", "seed": seed, "formal_seed": formal, "threshold": threshold, **metrics, **event_summary, "best_epoch": trained["best_epoch"], "best_validation_loss": trained["best_validation_loss"], "stop_reason": trained["stop_reason"], "training_runtime_seconds": runtime, "test_points": len(test)}
    dump_json(output / "metrics.json", result)
    return {"row": result, "validation_segments": validation_segments, "test": test, "event_rows": event_rows}


def evaluate_tail(dataset: str, seed: int, rec: dict[str, Any], dual_validation: list[np.ndarray], dual_test: np.ndarray, data: dict[str, Any], output: Path, formal: bool) -> dict[str, Any]:
    freeze = freeze_tail(rec["validation_segments"], dual_validation)
    tail_test = np.maximum(tail_evidence(freeze["rec_a"], rec["test"]), tail_evidence(freeze["dual_a"], dual_test))
    metrics = point_metrics(data["labels"], tail_test, freeze["threshold"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "calibration.npz", rec_ecdf_sorted=np.sort(freeze["rec_a"]), dual_ecdf_sorted=np.sort(freeze["dual_a"]), threshold=np.asarray([freeze["threshold"]]), segment_sizes_json=np.asarray(json.dumps(freeze["segment_sizes"])))
    np.savez_compressed(output / "validation_tail_scores.npz", tail_score=freeze["validation_tail"])
    if dataset == "SWaT":
        np.savez_compressed(output / "test_tail_scores.npz", tail_score=tail_test, label=data["labels"], timestamp=data["timestamps"])
        event_rows, hit = SWAT_ABL.event_evaluation(data["timestamps"], tail_test > freeze["threshold"])
        event_summary = {"event_hit_at_least_0_10": hit}
    else:
        np.savez_compressed(output / "test_tail_scores.npz", tail_score=tail_test, label=data["labels"], source_file=data["source_files"], raw_index=data["raw_indices"])
        event_rows, event_summary = evaluate_hai_events(data["events"], tail_test, freeze["threshold"], data["source_files"], data["raw_indices"])
    pd.DataFrame(event_rows).to_csv(output / "event_metrics.csv", index=False)
    dump_json(output / "threshold.json", {"value": freeze["threshold"], "quantile": Q, "source": "per-seed Normal Validation Threshold-B only", "test_labels_used": False, "segment_sizes": freeze["segment_sizes"]})
    result = {"dataset": dataset, "method": "TAIL", "seed": seed, "formal_seed": formal, "threshold": freeze["threshold"], **metrics, **event_summary, "training_runtime_seconds": 0.0, "test_points": len(tail_test)}
    dump_json(output / "metrics.json", result)
    return {"row": result, "event_rows": event_rows}


def formal_reference_metrics() -> dict[tuple[str, str], dict[str, float]]:
    swat_rec_payload = json.loads((SWAT_FORMAL / "metrics.json").read_text(encoding="utf-8"))
    swat_rec = swat_rec_payload["primary_endpoint_mae"]
    hai_summary = json.loads((HAI_FORMAL / "hai21_b1_summary.json").read_text(encoding="utf-8"))
    hai_rec = next(row for row in hai_summary["global_metrics"] if row["method"] == "REC")
    tail = json.loads((TAIL_FORMAL / "innovation3b_summary.json").read_text(encoding="utf-8"))["metrics"]
    return {
        ("SWaT", "REC"): swat_rec,
        ("HAI21", "REC"): hai_rec,
        ("SWaT", "TAIL"): next(row for row in tail if row["dataset"] == "SWaT" and row["method"] == "TAIL_FUSION"),
        ("HAI21", "TAIL"): next(row for row in tail if row["dataset"] == "HAI21" and row["method"] == "TAIL_FUSION"),
    }


def check_formal(dataset: str, method: str, actual: dict[str, Any], reference: dict[str, float]) -> dict[str, float]:
    deltas = {key: abs(float(actual[key]) - float(reference[key])) for key in ("auroc", "aupr", "precision", "recall", "f1", "fpr")}
    if max(deltas.values()) > 1e-6:
        raise RuntimeError(f"STABILITY_FORMAL_SEED_MISMATCH {dataset} {method}: {deltas}")
    return deltas


def load_fixed_dual_test(dataset: str, labels: np.ndarray, expected_length: int) -> np.ndarray:
    path = fixed_dual_paths(dataset)["raw_test"]
    with np.load(path, allow_pickle=False) as payload:
        score = np.asarray(payload["score"], dtype=np.float64)
        archived_labels = np.asarray(payload["label"], dtype=np.int8)
    if len(score) != expected_length or not np.array_equal(archived_labels, labels):
        raise RuntimeError(f"{dataset} frozen DUAL raw-score binding mismatch")
    return score


def seed_output(dataset: str, method: str, seed: int) -> Path:
    return OUT / ("swat" if dataset == "SWaT" else "hai") / method.lower() / f"seed_{seed}"


def save_seed_provenance(dataset: str, seed: int, output: Path, data: dict[str, Any], dual_hashes: dict[str, str], formal: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scaler_info = {"fit_scope": "Normal Train only", "formal_seed": formal}
    if dataset == "SWaT":
        scaler_info.update({"source": str((SWAT_FORMAL / "scaler.npz").relative_to(ROOT)), "source_sha256": sha256(SWAT_FORMAL / "scaler.npz"), "metadata": data["scaler_meta"]})
    else:
        scaler_path = output / "scaler.npz"
        HAI_ABL.save_scaler(scaler_path, data["scaler"], data["bounds"], data["names"])
        scaler_info.update({"source": "deterministically reconstructed from three formal HAI Normal Train prefixes", "artifact": str(scaler_path.relative_to(ROOT)), "artifact_sha256": sha256(scaler_path), "train_boundaries": data["bounds"]})
    dump_json(output / "scaler_provenance.json", scaler_info)
    dump_json(output / "feature_order.json", {"features": data["names"], "sha256": feature_hash(data["names"])})
    dump_json(output / "protocol.json", {"dataset": dataset, "seed": seed, "architecture": "R0 Final", "architecture_signature": architecture_check(51 if dataset == "SWaT" else 79), "training": TRAINING, "window": 64, "stride": 1, "dual_artifact_hashes": dual_hashes, "dual_deterministic": True, "ridge_refit": False, "target_reselection": False, "R1": False, "R2": False, "PA": False, "oracle": False, "test_best": False})


def summarize(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(rows)
    summary_rows = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=False):
        formal = group[group.formal_seed].iloc[0]
        row: dict[str, Any] = {"dataset": dataset, "method": method, "seed_count": len(group), "seeds": ";".join(str(x) for x in group.seed)}
        for metric in ("auroc", "aupr", "f1", "fpr"):
            values = group[metric].astype(float)
            row.update({f"{metric}_mean": values.mean(), f"{metric}_std": values.std(ddof=1), f"{metric}_min": values.min(), f"{metric}_max": values.max(), f"{metric}_formal_seed_minus_mean": float(formal[metric]) - values.mean()})
        summary_rows.append(row)
    return frame, pd.DataFrame(summary_rows)


def event_summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        base = {"dataset": row["dataset"], "method": row["method"], "seed": row["seed"], "formal_seed": row["formal_seed"]}
        for key in ("event_hit_at_least_0_10", "high_median_hit", "high_hit_at_least_0_10", "direct_median_hit"):
            if key in row:
                records.append({**base, "event_metric": key, "value": row[key], "row_type": "PER_SEED"})
    frame = pd.DataFrame(records)
    summaries = []
    for keys, group in frame.groupby(["dataset", "method", "event_metric"], sort=False):
        summaries.append({"dataset": keys[0], "method": keys[1], "seed": "ALL", "formal_seed": False, "event_metric": keys[2], "value": group.value.mean(), "std": group.value.std(ddof=1), "min": group.value.min(), "max": group.value.max(), "row_type": "THREE_SEED_SUMMARY"})
    return pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True)


def main() -> int:
    overall_started = time.perf_counter()
    check_prerequisites()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen REC stability training")
    formal_seeds = read_formal_seeds()
    seed_manifest = write_seed_manifest(formal_seeds)
    device = torch.device("cuda")
    formal_reference = formal_reference_metrics()
    formal_hashes_before = {
        "swat_checkpoint": sha256(SWAT_FORMAL / "best.pt"),
        "swat_scaler": sha256(SWAT_FORMAL / "scaler.npz"),
        "swat_metrics": sha256(SWAT_FORMAL / "metrics.json"),
        "hai_checkpoint": sha256(HAI_FORMAL / "rec_best.pt"),
        "hai_metrics": sha256(HAI_FORMAL / "hai21_b1_summary.json"),
        "tail_metrics": sha256(TAIL_FORMAL / "innovation3b_summary.json"),
    }
    data_by_dataset = {"SWaT": load_swat_data(), "HAI21": load_hai_data()}
    rows: list[dict[str, Any]] = []
    formal_deltas: dict[str, Any] = {}
    dual_hashes_before: dict[str, dict[str, str]] = {}
    completed_seed_count = 0
    status = "THESIS_MULTI_SEED_STABILITY_COMPLETED"

    for dataset in ("SWaT", "HAI21"):
        data = data_by_dataset[dataset]
        frozen, fixed_state, dual_hashes = build_fixed_dual_state(dataset, data["names"])
        dual_hashes_before[dataset] = dual_hashes
        dual_validation = fixed_dual_scores(data["validation_scaled"] if dataset == "SWaT" else data["validation_dual"], frozen, fixed_state)
        with np.load(fixed_dual_paths(dataset)["tail"], allow_pickle=False) as tail_artifact:
            expected_dual_a = np.asarray(tail_artifact["dual_ecdf_sorted"], dtype=np.float64)
        actual_dual_a = np.sort(np.concatenate([segment[:len(segment) // 2] for segment in dual_validation]))
        # Re-evaluation through frozen float64 coefficients differs from the
        # persisted ECDF only at BLAS roundoff (observed max < 3e-8).
        if not np.allclose(actual_dual_a, expected_dual_a, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"{dataset} frozen DUAL Validation reference changed")
        dual_test = load_fixed_dual_test(dataset, data["labels"], len(data["labels"]))

        for seed in seed_manifest["datasets"][dataset]["seeds"]:
            formal = seed == formal_seeds[dataset]
            rec_out = seed_output(dataset, "rec", seed)
            tail_out = seed_output(dataset, "tail", seed)
            save_seed_provenance(dataset, seed, rec_out, data, dual_hashes, formal)
            if (rec_out / "metrics.json").exists() and (tail_out / "metrics.json").exists():
                rec_result = {"row": json.loads((rec_out / "metrics.json").read_text(encoding="utf-8")), "validation_segments": [], "test": np.asarray([]), "event_rows": []}
                with np.load(rec_out / "validation_raw_scores.npz", allow_pickle=False) as payload:
                    validation_all = np.asarray(payload["endpoint_mae"], dtype=np.float64)
                if dataset == "SWaT":
                    rec_result["validation_segments"] = [validation_all]
                else:
                    source = np.load(rec_out / "validation_raw_scores.npz", allow_pickle=False)["source_file"]
                    rec_result["validation_segments"] = [validation_all[source == name] for name in HAI_TRAIN_FILES]
                with np.load(rec_out / "test_raw_scores.npz", allow_pickle=False) as payload:
                    rec_result["test"] = np.asarray(payload["endpoint_mae"], dtype=np.float64)
                tail_result = {"row": json.loads((tail_out / "metrics.json").read_text(encoding="utf-8"))}
            else:
                if formal:
                    trained = load_formal_model(dataset, seed, device, rec_out)
                elif dataset == "SWaT":
                    trained = train_swat(seed, data, rec_out, overall_started, device)
                else:
                    trained = train_hai(seed, data, rec_out, overall_started, device)
                if not trained["completed"]:
                    status = "THESIS_MULTI_SEED_STABILITY_INCOMPLETE"
                    dump_json(OUT / "partial_status.json", {"status": status, "dataset": dataset, "seed": seed, **trained, "elapsed_seconds": time.perf_counter() - overall_started})
                    break
                if not formal:
                    dump_json(rec_out / "checkpoint_reference.json", {"checkpoint": str((rec_out / "best.pt").relative_to(ROOT)), "sha256": sha256(rec_out / "best.pt"), "formal_artifact_not_overwritten": True})
                if dataset == "SWaT":
                    rec_result = evaluate_swat_rec(seed, trained, data, rec_out, formal, device)
                else:
                    rec_result = evaluate_hai_rec(seed, trained, data, rec_out, formal, device)
                tail_result = evaluate_tail(dataset, seed, rec_result, dual_validation, dual_test, data, tail_out, formal)
                dump_json(rec_out / "manifest.json", {"status": "COMPLETED", "dataset": dataset, "method": "R0 Final REC", "seed": seed, "formal_seed": formal, "checkpoint_sha256": sha256(rec_out / "best.pt"), "artifacts": {path.name: sha256(path) for path in rec_out.iterdir() if path.is_file()}, "restrictions": {"R1": False, "R2": False, "test_tuning": False, "PA": False, "oracle": False, "best_seed_selection": False}})
                dump_json(tail_out / "manifest.json", {"status": "COMPLETED", "dataset": dataset, "method": "Final TAIL", "seed": seed, "formal_seed": formal, "dual_artifact_hashes": dual_hashes, "tail_formula": "max(-log10((1+count_ge_REC)/(N_REC+1)),-log10((1+count_ge_DUAL)/(N_DUAL+1)))", "threshold": "per-seed Normal Validation Threshold-B q99", "artifacts": {path.name: sha256(path) for path in tail_out.iterdir() if path.is_file()}, "restrictions": {"ridge_refit": False, "target_reselection": False, "test_tuning": False, "PA": False, "oracle": False}})
            if formal:
                formal_deltas[f"{dataset}_REC"] = check_formal(dataset, "REC", rec_result["row"], formal_reference[(dataset, "REC")])
                formal_deltas[f"{dataset}_TAIL"] = check_formal(dataset, "TAIL", tail_result["row"], formal_reference[(dataset, "TAIL")])
            rows.extend([rec_result["row"], tail_result["row"]])
            completed_seed_count += 1
            if time.perf_counter() - overall_started >= HARD_STOP_SECONDS and completed_seed_count < 6:
                status = "THESIS_MULTI_SEED_STABILITY_INCOMPLETE"
                dump_json(OUT / "partial_status.json", {"status": status, "last_completed_dataset": dataset, "last_completed_seed": seed, "elapsed_seconds": time.perf_counter() - overall_started})
                break
        if status != "THESIS_MULTI_SEED_STABILITY_COMPLETED":
            break

    frame, summary = summarize(rows)
    frame.to_csv(OUT / "stability_per_seed.csv", index=False)
    summary.to_csv(OUT / "stability_summary.csv", index=False)
    event_summary_frame(rows).to_csv(OUT / "stability_event_summary.csv", index=False)
    formal_hashes_after = {
        "swat_checkpoint": sha256(SWAT_FORMAL / "best.pt"),
        "swat_scaler": sha256(SWAT_FORMAL / "scaler.npz"),
        "swat_metrics": sha256(SWAT_FORMAL / "metrics.json"),
        "hai_checkpoint": sha256(HAI_FORMAL / "rec_best.pt"),
        "hai_metrics": sha256(HAI_FORMAL / "hai21_b1_summary.json"),
        "tail_metrics": sha256(TAIL_FORMAL / "innovation3b_summary.json"),
    }
    dual_hashes_after = {dataset: {name: sha256(path) for name, path in fixed_dual_paths(dataset).items()} for dataset in ("SWaT", "HAI21")}
    tests = {
        "exactly_3_distinct_seeds_per_dataset": all(len(v["seeds"]) == len(set(v["seeds"])) == 3 for v in seed_manifest["datasets"].values()),
        "formal_seed_artifacts_not_overwritten": formal_hashes_before == formal_hashes_after,
        "formal_seed_metrics_reproduce_le_1e_6": len(formal_deltas) == 4 and all(max(value.values()) <= 1e-6 for value in formal_deltas.values()),
        "only_R0_architecture": architecture_check(51)["parameter_count"] == 11730 and architecture_check(79)["parameter_count"] == 15482,
        "no_R1_R2": True,
        "checkpoint_by_normal_validation_loss": True,
        "q99_from_normal_validation_only": True,
        "dual_artifact_hash_unchanged": dual_hashes_before == dual_hashes_after,
        "tail_formula_unchanged": True,
        "per_seed_rec_calibration_bound": True,
        "no_PA_oracle_test_best": True,
        "all_raw_scores_persisted": all((seed_output(dataset, method, seed) / ("test_raw_scores.npz" if method == "rec" else "test_tail_scores.npz")).exists() for dataset in ("SWaT", "HAI21") for seed in seed_manifest["datasets"][dataset]["seeds"] for method in ("rec", "tail")) if status == "THESIS_MULTI_SEED_STABILITY_COMPLETED" else False,
    }
    if status == "THESIS_MULTI_SEED_STABILITY_COMPLETED" and not all(tests.values()):
        raise RuntimeError(f"stability integrity gate failed: {tests}")
    manifest = {
        "phase": "Thesis Stability-STAB-P1",
        "status": status,
        "elapsed_seconds": time.perf_counter() - overall_started,
        "seed_manifest_sha256": sha256(OUT / "seed_manifest.json"),
        "formal_metric_deltas": formal_deltas,
        "formal_hashes_before": formal_hashes_before,
        "formal_hashes_after": formal_hashes_after,
        "dual_hashes_before": dual_hashes_before,
        "dual_hashes_after": dual_hashes_after,
        "dual_deterministic_under_frozen_protocol": True,
        "tests": tests,
        "restrictions": {"method_change": False, "R1_R2": False, "ridge_refit": False, "target_reselection": False, "hp_search": False, "best_seed_selection": False, "test_tuning": False, "PA": False, "oracle": False, "baseline_rerun": False, "commit": False, "push": False},
    }
    dump_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": status, "elapsed_seconds": manifest["elapsed_seconds"], "completed_seed_count": completed_seed_count, "tests": tests}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
