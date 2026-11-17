"""Baseline-B1-Resume-3: exact HAI continuation from the persisted epoch-42 state."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B1 = load_module("baseline_b1_core_exact", Path(__file__).with_name("run_lstm_ae_unified.py"))
R1 = load_module("baseline_b1_resume_helpers", Path(__file__).with_name("resume_lstm_ae_hai.py"))
OUT = B1.OUT
LATEST = OUT / "hai_latest_training_state.pt"
SWAT_FILES = (OUT / "swat_metrics.json", OUT / "swat_event_metrics.csv", OUT / "swat_state_dict.pt")
EXPECTED_SWAT = {
    "AUROC": 0.8182424435222648,
    "AUPR": 0.7060434385305675,
    "F1": 0.25449521557370985,
    "FPR": 0.7266088012754005,
    "event_hit_count": 29,
}


def rng_states_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "early_stop_counter", "best_validation_loss", "best_epoch", "best_model_state_dict",
        "history", "python_rng_state", "numpy_rng_state", "torch_rng_state",
        "torch_cuda_rng_state_all", "protocol",
    }
    missing = sorted(required - set(state))
    expected_protocol = {
        "seed": 1,
        "batch_size": 128,
        "max_epochs": 100,
        "patience": 10,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    }
    checks = {
        "required_fields": not missing,
        "epoch_42": state.get("epoch") == 42,
        "best_epoch_39": state.get("best_epoch") == 39,
        "best_validation_loss": abs(float(state.get("best_validation_loss", math.inf)) - 0.06134573767856918) <= 1e-12,
        "counter_three": state.get("early_stop_counter") == 3,
        "history_42": len(state.get("history", [])) == 42,
        "order_present": isinstance(state.get("order"), np.ndarray),
        "protocol_exact": state.get("protocol") == expected_protocol,
        "scheduler_absent_by_design": state.get("scheduler_state_dict") is None,
    }
    if not all(checks.values()):
        raise RuntimeError(f"HAI_EXACT_RESUME_STATE_INVALID missing={missing} checks={checks}")
    return checks


def save_state(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stale: int,
    best_loss: float,
    best_epoch: int,
    best_state: dict[str, torch.Tensor],
    history: list[dict[str, Any]],
    order: np.ndarray,
    continuation_seconds: float,
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "early_stop_counter": stale,
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "best_model_state_dict": best_state,
        "history": history,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "order": order,
        "continuation_training_seconds": continuation_seconds,
        "training_seconds_total": None,
        "protocol": {
            "seed": B1.SEED,
            "batch_size": B1.TRAIN_BATCH,
            "max_epochs": B1.MAX_EPOCHS,
            "patience": B1.PATIENCE,
            "learning_rate": B1.LR,
            "weight_decay": B1.WEIGHT_DECAY,
        },
    }, path)


def exact_resume_train(
    state: dict[str, Any],
    train_store: B1.WindowStore,
    validation_store: B1.WindowStore,
    device: torch.device,
    hard_stop_seconds: float,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    # Construct containers, then restore every persisted state. No fresh seed is applied.
    model = B1.LSTMAutoencoder(79).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=B1.LR, weight_decay=B1.WEIGHT_DECAY)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    best_state = {name: value.detach().cpu().clone() for name, value in state["best_model_state_dict"].items()}
    best_loss = float(state["best_validation_loss"])
    best_epoch = int(state["best_epoch"])
    stale = int(state["early_stop_counter"])
    history = list(state["history"])

    # Resume-3 restores the exact in-place sampler order persisted after epoch 42.
    order = np.asarray(state["order"], dtype=np.int64).copy()
    if order.shape != (len(train_store.starts),) or not np.array_equal(np.sort(order), np.arange(len(order), dtype=np.int64)):
        raise RuntimeError("HAI_EXACT_RESUME_STATE_INVALID: persisted sample order is not a complete permutation")

    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    loss_fn = nn.MSELoss(reduction="mean")
    prior_known_resumed_seconds = float(state.get("continuation_training_seconds", 0.0))
    continued_started = time.perf_counter()
    stopped_by_deadline = False
    for epoch in range(int(state["epoch"]) + 1, B1.MAX_EPOCHS + 1):
        model.train()
        np.random.shuffle(order)
        total_loss = 0.0
        total_windows = 0
        for begin in range(0, len(order), B1.TRAIN_BATCH):
            ids = order[begin : begin + B1.TRAIN_BATCH]
            batch = torch.from_numpy(train_store.batch(ids)).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
            total_windows += len(ids)
        model.eval()
        val_total = 0.0
        val_windows = 0
        with torch.inference_mode():
            for begin in range(0, len(validation_store.starts), B1.SCORE_BATCH):
                ids = np.arange(begin, min(begin + B1.SCORE_BATCH, len(validation_store.starts)), dtype=np.int64)
                batch = torch.from_numpy(validation_store.batch(ids)).to(device, non_blocking=True)
                loss = loss_fn(model(batch), batch)
                val_total += float(loss) * len(ids)
                val_windows += len(ids)
        train_loss = total_loss / total_windows
        val_loss = val_total / val_windows
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append({
            "dataset": "HAI21",
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "improved": improved,
        })
        continuation_seconds = time.perf_counter() - continued_started
        save_state(
            LATEST,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            stale=stale,
            best_loss=best_loss,
            best_epoch=best_epoch,
            best_state=best_state,
            history=history,
            order=order,
            continuation_seconds=prior_known_resumed_seconds + continuation_seconds,
        )
        print(
            f"[HAI21 exact-resume] epoch={epoch:03d} train={train_loss:.8f} val={val_loss:.8f} "
            f"best={best_loss:.8f}@{best_epoch} stale={stale}/{B1.PATIENCE}",
            flush=True,
        )
        if stale >= B1.PATIENCE:
            break
        if continuation_seconds >= hard_stop_seconds:
            stopped_by_deadline = True
            break
    if stopped_by_deadline:
        raise TimeoutError(f"HARD_STOP_INCOMPLETE: saved complete state at epoch {history[-1]['epoch']}")
    model.load_state_dict(best_state)
    return model, history, {
        "training_seconds": None,
        "training_time_status": "N/A_PRIOR_CUMULATIVE_TIME_NOT_PERSISTED",
        "continuation_training_seconds": prior_known_resumed_seconds + time.perf_counter() - continued_started,
        "resume3_training_seconds": time.perf_counter() - continued_started,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "early_stopped": len(history) < B1.MAX_EPOCHS,
        "resume_from_epoch": 42,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-stop-seconds", type=float, default=1650.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overall_started = time.perf_counter()
    protocol_checks = R1.verify_protocol()
    if not LATEST.is_file():
        raise RuntimeError("HAI_EXACT_RESUME_STATE_INVALID: checkpoint missing")
    state = torch.load(LATEST, map_location="cpu", weights_only=False)
    resume_checks = validate_state(state)
    swat_hashes = {path.name: B1.sha256(path) for path in SWAT_FILES}
    swat = json.loads((OUT / "swat_metrics.json").read_text(encoding="utf-8"))
    if any(abs(float(swat[key]) - value) > 1e-12 for key, value in EXPECTED_SWAT.items()):
        raise RuntimeError("frozen SWaT metrics changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required by frozen protocol")
    device = torch.device("cuda")
    print(json.dumps({"resume_verification": resume_checks, "protocol_verification": protocol_checks}), flush=True)

    normal, scaler = R1.load_hai_normal_only()
    train_store = B1.WindowStore.from_segments(normal["train"], common_endpoint_mask=False)
    validation_store = B1.WindowStore.from_segments(normal["validation"], common_endpoint_mask=True)
    if len(validation_store.starts) != normal["expected_validation_endpoints"]:
        raise RuntimeError("HAI validation endpoint count mismatch")
    model, history, training = exact_resume_train(state, train_store, validation_store, device, args.hard_stop_seconds)
    validation_scores = B1.score_model(model, validation_store, device)
    threshold = float(np.quantile(validation_scores, B1.Q))

    # Only now, after best-model restoration and q99 freeze, open the five Test files.
    test = R1.load_hai_test(normal["feature_names"], scaler)
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

    hai_eff = R1.measure_efficiency(model, test_store.first_windows(1000), hai_state, device)
    hai_eff.update(training)
    swat_bundle = B1.swat_data()
    swat_store = B1.WindowStore.from_segments(swat_bundle["test"], common_endpoint_mask=True)
    swat_model = B1.LSTMAutoencoder(51).to(device)
    swat_model.load_state_dict(torch.load(OUT / "swat_state_dict.pt", map_location=device, weights_only=True))
    swat_eff = R1.measure_efficiency(swat_model, swat_store.first_windows(1000), OUT / "swat_state_dict.pt", device)
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
    if swat_hashes != {path.name: B1.sha256(path) for path in SWAT_FILES}:
        raise RuntimeError("SWaT frozen artifacts changed")

    ours_path = ROOT / "outputs" / "thesis_innovation3b_tail_fusion_final" / "cross_dataset_metrics.csv"
    ours = pd.read_csv(ours_path).to_dict(orient="records")
    manifest = {
        "phase": "Thesis Baseline-B1-Resume-3",
        "status": "LSTM_AE_BASELINE_COMPLETED",
        "resume_verification": resume_checks,
        "protocol_verification": protocol_checks,
        "resume_from_epoch": 42,
        "persisted_sampler_order_restored": True,
        "training": training,
        "training_time": "N/A_PRIOR_CUMULATIVE_TIME_NOT_PERSISTED",
        "swat_frozen_artifact_sha256": swat_hashes,
        "b0_manifest_sha256": B1.sha256(B1.B0_MANIFEST),
        "efficiency_protocol_sha256": B1.sha256(B1.EFFICIENCY_PROTOCOL),
        "ours_source": str(ours_path.relative_to(ROOT)),
        "ours_rows": ours,
        "sanity": {
            "same_frozen_protocol": True,
            "patience_unchanged": B1.PATIENCE == 10,
            "swat_not_retrained": True,
            "swat_artifacts_unchanged": True,
            "hai_exact_resume": True,
            "best_model_restored": True,
            "test_opened_after_training_and_threshold_freeze": True,
            "no_test_tuning": True,
            "no_point_adjustment": True,
            "no_pot": True,
            "no_usad_gdn": True,
        },
        "elapsed_seconds": time.perf_counter() - overall_started,
        "commit": False,
        "push": False,
    }
    B1.dump_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "HAI21": metrics, "efficiency": efficiency}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
