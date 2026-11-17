"""Baseline-B1-Resume: restart HAI deterministically and fill missing B1 artifacts."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("run_lstm_ae_unified.py")
SPEC = importlib.util.spec_from_file_location("baseline_b1_runner", SCRIPT)
B1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = B1
SPEC.loader.exec_module(B1)

OUT = B1.OUT
LATEST = OUT / "hai_latest_training_state.pt"
SWAT_FILES = (OUT / "swat_metrics.json", OUT / "swat_event_metrics.csv", OUT / "swat_state_dict.pt")


def verify_protocol() -> dict[str, Any]:
    manifest = json.loads(B1.B0_MANIFEST.read_text(encoding="utf-8"))
    method = manifest["baselines"]["LSTM-AE"]
    checks = {
        "status": manifest["status"] == "BASELINE_PROTOCOL_FROZEN",
        "max_epochs": method["max_epochs"] == B1.MAX_EPOCHS == 100,
        "patience": manifest["common"]["early_stopping"]["patience"] == B1.PATIENCE == 10,
        "optimizer": method["optimizer"] == "Adam",
        "learning_rate": method["learning_rate"] == B1.LR == 1e-3,
        "weight_decay": method["weight_decay"] == B1.WEIGHT_DECAY == 0.0,
        "batch_size": method["batch_size"] == B1.TRAIN_BATCH == 128,
        "loss": method["loss"] == "full-window MSE",
        "hidden": "h=64" in method["architecture"] and B1.HIDDEN == 64,
        "window": manifest["common"]["window"] == B1.WINDOW == 64,
    }
    if not all(checks.values()):
        raise RuntimeError(f"LSTM_AE_PROTOCOL_MISMATCH: {checks}")
    return checks


def inspect_resume_state() -> dict[str, Any]:
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "early_stop_counter", "best_validation_loss", "best_epoch", "best_model_state_dict",
        "history", "python_rng_state", "numpy_rng_state", "torch_rng_state",
        "torch_cuda_rng_state_all", "protocol",
    }
    if not LATEST.is_file():
        return {"complete": False, "reason": "no persisted HAI training state from first run"}
    state = torch.load(LATEST, map_location="cpu", weights_only=False)
    missing = sorted(required - set(state))
    return {"complete": not missing, "missing": missing, "epoch": state.get("epoch")}


def load_hai_normal_only() -> tuple[dict[str, Any], B1.StandardScaler]:
    schema = pd.read_csv(B1.HAI_AUDIT / "variable_schema.csv")
    features = schema["column_name"].astype(str).tolist()
    if len(features) != 79 or "attack" in features:
        raise RuntimeError("HAI feature schema changed")
    scaler = B1.StandardScaler()
    raw: list[np.ndarray] = []
    boundaries: dict[str, int] = {}
    for name in B1.HAI_TRAIN_FILES:
        values = pd.read_csv(B1.HAI_ROOT / name, compression="gzip", usecols=features, low_memory=False).to_numpy(np.float64)
        boundary = int(math.floor(0.8 * len(values)))
        raw.append(values)
        boundaries[name] = boundary
    scaler.fit_many([values[: boundaries[name]] for name, values in zip(B1.HAI_TRAIN_FILES, raw)])
    train = [scaler.transform(values[: boundaries[name]]).astype(np.float32) for name, values in zip(B1.HAI_TRAIN_FILES, raw)]
    validation = [scaler.transform(values[boundaries[name] :]).astype(np.float32) for name, values in zip(B1.HAI_TRAIN_FILES, raw)]
    return {
        "train": train,
        "validation": validation,
        "feature_names": features,
        "boundaries": boundaries,
        "expected_validation_endpoints": sum(len(values) - B1.WINDOW for values in validation),
    }, scaler


def load_hai_test(features: list[str], scaler: B1.StandardScaler) -> dict[str, Any]:
    tests: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    file_parts: list[np.ndarray] = []
    raw_indices: list[np.ndarray] = []
    for name in B1.HAI_TEST_FILES:
        frame = pd.read_csv(B1.HAI_ROOT / name, compression="gzip", usecols=features + ["attack"], low_memory=False)
        raw_label = frame["attack"].to_numpy(np.int8)
        if set(np.unique(raw_label).tolist()) - {0, 1}:
            raise RuntimeError(f"HAI label changed in {name}")
        values = scaler.transform(frame[features].to_numpy(np.float64)).astype(np.float32)
        tests.append(values)
        labels.append(raw_label[B1.WINDOW:])
        file_parts.append(np.repeat(name, len(values) - B1.WINDOW))
        raw_indices.append(np.arange(B1.WINDOW, len(values), dtype=np.int64))
    return {
        "test": tests,
        "test_labels": np.concatenate(labels),
        "test_files": np.concatenate(file_parts),
        "test_raw_indices": np.concatenate(raw_indices),
        "expected_test_endpoints": sum(len(values) - B1.WINDOW for values in tests),
    }


def measure_efficiency(model: torch.nn.Module, windows: np.ndarray, state_path: Path, device: torch.device) -> dict[str, Any]:
    if len(windows) < 1000 or not state_path.is_file():
        raise RuntimeError("efficiency inputs incomplete")
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    samples = [torch.from_numpy(windows[i : i + 1]).to(device) for i in range(1000)]
    model.eval()
    with torch.inference_mode():
        for i in range(100):
            _ = model(samples[i])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        latencies: list[float] = []
        for sample in samples:
            torch.cuda.synchronize()
            started = time.perf_counter()
            _ = model(sample)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
        peak = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return {
        "trainable_parameters": int(params),
        "state_dict_bytes": int(state_path.stat().st_size),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_median": float(np.median(latencies)),
        "latency_ms_p95": float(np.quantile(latencies, 0.95)),
        "peak_inference_memory_mib": peak,
        "latency_warmup": 100,
        "latency_timed": 1000,
        "latency_batch_size": 1,
    }


def main() -> int:
    started = time.perf_counter()
    checks = verify_protocol()
    frozen_hashes = {path.name: B1.sha256(path) for path in SWAT_FILES}
    swat = json.loads((OUT / "swat_metrics.json").read_text(encoding="utf-8"))
    expected_swat = {"AUROC": 0.8182424435222648, "AUPR": 0.7060434385305675, "F1": 0.25449521557370985, "FPR": 0.7266088012754005, "event_hit_count": 29}
    if any(abs(float(swat[key]) - value) > 1e-12 for key, value in expected_swat.items()):
        raise RuntimeError("existing SWaT metrics differ from first B1 run")
    resume = inspect_resume_state()
    # The first interrupted run persisted no HAI state. A stale state from a later failed
    # retry is not silently consumed: this authorized attempt is a deterministic restart.
    mode = "restart_from_scratch_no_complete_first_run_state"
    if resume.get("complete"):
        raise RuntimeError("complete HAI resume state now exists; exact resume path requires explicit adjudication")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required by frozen efficiency protocol")
    device = torch.device("cuda")
    print(json.dumps({"protocol_verification": checks, "mode": mode, "resume_audit": resume}), flush=True)

    normal, scaler = load_hai_normal_only()
    train_store = B1.WindowStore.from_segments(normal["train"], common_endpoint_mask=False)
    validation_store = B1.WindowStore.from_segments(normal["validation"], common_endpoint_mask=True)
    if len(validation_store.starts) != normal["expected_validation_endpoints"]:
        raise RuntimeError("HAI validation endpoint count mismatch")
    model, history, train_meta = B1.train_model(
        "HAI21", 79, train_store, validation_store, device, latest_checkpoint_path=LATEST
    )
    validation_scores = B1.score_model(model, validation_store, device)
    threshold = float(np.quantile(validation_scores, B1.Q))
    # Test features and labels are opened only after the model and q99 threshold are frozen.
    test = load_hai_test(normal["feature_names"], scaler)
    test_store = B1.WindowStore.from_segments(test["test"], common_endpoint_mask=True)
    if len(test_store.starts) != test["expected_test_endpoints"] or len(test["test_labels"]) != len(test_store.starts):
        raise RuntimeError("HAI test endpoint count mismatch")
    scores = B1.score_model(model, test_store, device)
    metrics = B1.point_metrics(test["test_labels"], scores, threshold)
    metrics.update({
        "dataset": "HAI21",
        "validation_score_q99": threshold,
        "validation_exceedance_rate": float(np.mean(validation_scores > threshold)),
        "validation_endpoint_count": int(len(validation_scores)),
        "threshold_source": "Normal Validation endpoint scores only",
        "score": "endpoint mean channel squared reconstruction error",
        "window": B1.WINDOW,
        "seed": B1.SEED,
    })
    predictions = scores > threshold
    event_rows, event_summary = B1.hai_event_metrics(predictions, test["test_files"], test["test_raw_indices"])
    metrics["events"] = event_summary
    hai_state = OUT / "hai_state_dict.pt"
    torch.save(model.state_dict(), hai_state)
    B1.dump_json(OUT / "hai_metrics.json", metrics)
    B1.write_csv(OUT / "hai_event_metrics.csv", event_rows)
    B1.write_csv(OUT / "training_history.csv", history)
    hai_eff = measure_efficiency(model, test_store.first_windows(1000), hai_state, device)
    hai_eff.update(train_meta)

    swat_bundle = B1.swat_data()
    swat_store = B1.WindowStore.from_segments(swat_bundle["test"], common_endpoint_mask=True)
    swat_model = B1.LSTMAutoencoder(51).to(device)
    swat_model.load_state_dict(torch.load(OUT / "swat_state_dict.pt", map_location=device, weights_only=True))
    swat_eff = measure_efficiency(swat_model, swat_store.first_windows(1000), OUT / "swat_state_dict.pt", device)
    swat_eff.update({
        "training_seconds": None,
        "training_time_status": "N/A_NOT_PERSISTED_BY_FIRST_RUN",
        "best_epoch": 46,
        "epochs_ran": 56,
        "best_validation_loss": 0.00552277,
        "early_stopped": True,
    })
    efficiency = {
        "protocol_sha256": B1.sha256(B1.EFFICIENCY_PROTOCOL),
        "device": torch.cuda.get_device_name(device),
        "datasets": {"SWaT": swat_eff, "HAI21": hai_eff},
    }
    B1.dump_json(OUT / "efficiency.json", efficiency)
    if frozen_hashes != {path.name: B1.sha256(path) for path in SWAT_FILES}:
        raise RuntimeError("SWaT frozen artifacts changed during B1-Resume")

    ours_path = ROOT / "outputs" / "thesis_innovation3b_tail_fusion_final" / "cross_dataset_metrics.csv"
    ours = pd.read_csv(ours_path).to_dict(orient="records")
    manifest = {
        "phase": "Thesis Baseline-B1-Resume",
        "status": "LSTM_AE_BASELINE_COMPLETED",
        "protocol_verification": checks,
        "hai_execution_mode": mode,
        "first_run_resume_state": resume,
        "swat_frozen_artifact_sha256": frozen_hashes,
        "swat_training_time": "N/A_NOT_PERSISTED_BY_FIRST_RUN",
        "hai_training": train_meta,
        "b0_manifest_sha256": B1.sha256(B1.B0_MANIFEST),
        "efficiency_protocol_sha256": B1.sha256(B1.EFFICIENCY_PROTOCOL),
        "ours_source": str(ours_path.relative_to(ROOT)),
        "ours_rows": ours,
        "sanity": {
            "same_frozen_protocol": True,
            "patience_unchanged": B1.PATIENCE == 10,
            "swat_not_retrained": True,
            "swat_artifacts_unchanged": True,
            "hai_test_opened_after_model_and_threshold_freeze": True,
            "no_test_tuning": True,
            "no_point_adjustment": True,
            "no_pot": True,
            "no_usad_gdn": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "commit": False,
        "push": False,
    }
    B1.dump_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "HAI21": metrics, "efficiency": efficiency}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
