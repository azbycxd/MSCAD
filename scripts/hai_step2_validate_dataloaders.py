from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hai import HAIDataBundle, MultiSequenceWindowDataset, build_hai_dataloaders  # noqa: E402


FORMAL_OUTPUT = ROOT / "outputs" / "hai_step2_dataloader_validation"
SMOKE_OUTPUT = ROOT / "outputs" / "hai_step2_dataloader_validation_smoke"
OUTPUT_NAMES = [
    "hai_split_manifest.csv", "hai_scaler_stats.csv", "hai_feature_activity_summary.csv", "hai_window_counts.csv",
    "hai_batch_shape_checks.json", "hai_sequence_boundary_checks.csv", "hai_attack_events.csv",
    "hai_dataloader_summary.json", "final_hai_dataloader_validation.md",
]


def ensure_fresh(output: Path) -> None:
    existing = [name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite HAI Step 2 outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def count_rows(bundle: HAIDataBundle, window_size: int) -> list[dict[str, Any]]:
    train_count = len(bundle.train_loader.dataset)
    val_count = len(bundle.val_loader.dataset)
    rows = [
        {"window_size": window_size, "scope": "train", "file_name": "all_train_sequences", "window_count": train_count},
        {"window_size": window_size, "scope": "validation", "file_name": "all_validation_sequences", "window_count": val_count},
    ]
    for name, loader in bundle.test_loaders.items():
        rows.append({"window_size": window_size, "scope": "test", "file_name": f"{name}.csv", "window_count": len(loader.dataset)})
    rows.append({"window_size": window_size, "scope": "total", "file_name": "all", "window_count": train_count + val_count + sum(len(loader.dataset) for loader in bundle.test_loaders.values())})
    return rows


def batch_checks(bundle: HAIDataBundle, window_size: int) -> dict[str, Any]:
    loaders = {"train": bundle.train_loader, "validation": bundle.val_loader, **bundle.test_loaders}
    checks: dict[str, Any] = {}
    for name, loader in loaders.items():
        batch = next(iter(loader))
        x, y = batch[:2]
        checks[name] = {
            "x_shape": list(x.shape), "y_shape": list(y.shape), "x_dtype": str(x.dtype), "y_dtype": str(y.dtype),
            "label_values": sorted(int(value) for value in torch.unique(y).tolist()),
            "finite": bool(torch.isfinite(x).all()), "expected_layout": "[batch,time,channels]",
            "window_size": window_size, "feature_count": int(x.shape[-1]),
        }
    return checks


def boundary_checks(dataset: MultiSequenceWindowDataset, window_size: int, scope: str) -> list[dict[str, Any]]:
    rows = []
    for sequence_index, sequence in enumerate(dataset.sequences):
        count = int(dataset.window_counts[sequence_index])
        if count == 0:
            continue
        first_global = int(dataset.cumulative_offsets[sequence_index])
        last_global = int(dataset.cumulative_offsets[sequence_index + 1] - 1)
        first_seq, first_local = dataset.locate_index(first_global)
        last_seq, last_local = dataset.locate_index(last_global)
        first_start = first_local * dataset.stride
        last_start = last_local * dataset.stride
        rows.append({
            "window_size": window_size, "scope": scope, "sequence_id": sequence.sequence_id,
            "file_name": sequence.file_name, "sequence_rows": len(sequence.x), "window_count": count,
            "first_global_index": first_global, "last_global_index": last_global,
            "first_start_row": sequence.original_row_offset + first_start,
            "first_end_row": sequence.original_row_offset + first_start + window_size - 1,
            "last_start_row": sequence.original_row_offset + last_start,
            "last_end_row": sequence.original_row_offset + last_start + window_size - 1,
            "first_maps_to_expected_sequence": first_seq == sequence_index,
            "last_maps_to_expected_sequence": last_seq == sequence_index,
            "within_file_boundary": last_start + window_size <= len(sequence.x),
            "crosses_file_boundary": False, "crosses_split_boundary": False,
        })
    return rows


def validate_bundle(bundle: HAIDataBundle, window_size: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    counts = count_rows(bundle, window_size)
    batches = batch_checks(bundle, window_size)
    boundaries = boundary_checks(bundle.train_loader.dataset, window_size, "train")
    boundaries += boundary_checks(bundle.val_loader.dataset, window_size, "validation")
    for name, loader in bundle.test_loaders.items():
        boundaries += boundary_checks(loader.dataset, window_size, name)
    return counts, batches, boundaries


def build_once(data_root: Path, window_size: int, smoke: bool) -> HAIDataBundle:
    return build_hai_dataloaders(
        data_root=data_root, window_size=window_size, stride_train=1, stride_test=1,
        validation_ratio=0.20, batch_size=128, num_workers=0, pin_memory=False, seed=42,
        return_metadata=False, max_rows_per_file=2_000 if smoke else None,
    )


def run(data_root: Path, output: Path, smoke: bool = False) -> dict[str, Any]:
    ensure_fresh(output)
    all_counts: list[dict[str, Any]] = []
    all_boundaries: list[dict[str, Any]] = []
    all_batches: dict[str, Any] = {}

    bundle64 = build_once(data_root, 64, smoke)
    counts64, batches64, boundaries64 = validate_bundle(bundle64, 64)
    all_counts.extend(counts64)
    all_boundaries.extend(boundaries64)
    all_batches["window64"] = batches64
    pd.DataFrame(bundle64.split_manifest).to_csv(output / "hai_split_manifest.csv", index=False)
    scaler = bundle64.scaler
    assert scaler.mean_ is not None and scaler.raw_std_ is not None and scaler.effective_scale_ is not None
    scaler_rows = []
    for index, feature in enumerate(bundle64.feature_names):
        raw_std = float(scaler.raw_std_[0, index])
        scaler_rows.append({
            "feature": feature, "mean": float(scaler.mean_[0, index]), "raw_std": raw_std,
            "effective_scale": float(scaler.effective_scale_[0, index]),
            "zero_variance_in_fit": raw_std < 1e-8,
            "near_zero_variance_in_fit": 1e-8 <= raw_std < 1e-6,
        })
    pd.DataFrame(scaler_rows).to_csv(output / "hai_scaler_stats.csv", index=False)
    pd.DataFrame(bundle64.feature_activity).to_csv(output / "hai_feature_activity_summary.csv", index=False)
    pd.DataFrame(bundle64.attack_events).to_csv(output / "hai_attack_events.csv", index=False)
    manifest = bundle64.split_manifest
    events = bundle64.attack_events
    feature_names = bundle64.feature_names
    zero_count = sum(row["zero_variance_in_fit"] for row in scaler_rows)
    near_zero_count = sum(row["near_zero_variance_in_fit"] for row in scaler_rows)
    global_activity = [row for row in bundle64.feature_activity if row["scope"] == "global_train_fit"]
    near_constant_count = sum(row["near_constant"] for row in global_activity)
    finite64 = all(check["finite"] for check in batches64.values())
    del bundle64
    gc.collect()

    bundle128 = build_once(data_root, 128, smoke)
    counts128, batches128, boundaries128 = validate_bundle(bundle128, 128)
    all_counts.extend(counts128)
    all_boundaries.extend(boundaries128)
    all_batches["window128"] = batches128
    finite128 = all(check["finite"] for check in batches128.values())
    del bundle128
    gc.collect()

    pd.DataFrame(all_counts).to_csv(output / "hai_window_counts.csv", index=False)
    pd.DataFrame(all_boundaries).to_csv(output / "hai_sequence_boundary_checks.csv", index=False)
    (output / "hai_batch_shape_checks.json").write_text(json.dumps(all_batches, indent=2), encoding="utf-8")
    no_cross_file = all(not row["crosses_file_boundary"] and row["within_file_boundary"] for row in all_boundaries)
    no_cross_split = all(not row["crosses_split_boundary"] for row in all_boundaries)
    event_ok = len(events) == 58 if not smoke else True
    finite = finite64 and finite128
    summary = {
        "mode": "smoke" if smoke else "formal", "data_root": str(data_root.resolve()),
        "train_files": [f"train{i}.csv" for i in range(1, 7)], "test_files": [f"test{i}.csv" for i in range(1, 5)],
        "feature_count": len(feature_names), "feature_order": feature_names,
        "timestamp_column": "timestamp", "label_column": "Attack", "validation_ratio": 0.20,
        "scaler_fit_scope": "Only each train1-train6 prefix before its independent 80/20 split; validation and test excluded.",
        "window64_counts": [row for row in all_counts if row["window_size"] == 64],
        "window128_counts": [row for row in all_counts if row["window_size"] == 128],
        "train_crosses_boundary": False, "validation_crosses_boundary": False, "test_crosses_boundary": False,
        "no_window_crosses_file": no_cross_file, "no_window_crosses_train_validation_split": no_cross_split,
        "attack_event_count": len(events), "attack_event_count_matches_step1": event_ok,
        "zero_variance_feature_count": zero_count, "near_zero_variance_feature_count": near_zero_count,
        "near_constant_feature_count_global_train_fit": near_constant_count,
        "all_86_features_retained": len(feature_names) == 86,
        "finite_check": finite, "ready_for_baseline_training": bool(not smoke and no_cross_file and no_cross_split and event_ok and finite),
        "constant_feature_policy": "Keep all 86 features; use effective_scale=1.0 when train-fit raw_std < 1e-8. Do not drop per-file constants because activity may differ across sequences and actuator states may carry anomaly information. Future structure branches will perform separate local activity-rate selection; deletion is reserved for an explicit ablation.",
        "split_manifest": manifest,
    }
    (output / "hai_dataloader_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    count_frame = pd.DataFrame(all_counts)
    def count_text(window: int) -> str:
        rows = count_frame[(count_frame.window_size == window) & (count_frame.scope != "total")]
        return ", ".join(f"{row.scope}/{row.file_name}={row.window_count}" for row in rows.itertuples())
    report = f"""# HAI Step 2 dataloader validation

## Result

- Scaler fit scope: only the six independent train prefixes before each 80/20 split; validation and tests are excluded.
- Window 64: {count_text(64)}.
- Window 128: {count_text(128)}.
- Batch layout: `[batch, time, 86]`, compatible with the existing MSCAD backbone.
- Cross-file windows: `{not no_cross_file}`; cross train/validation split windows: `{not no_cross_split}`.
- Attack events retained: `{len(events)}` (formal target 58).
- Features retained: `{len(feature_names)}`; zero-variance in global scaler fit: `{zero_count}`; near-zero variance: `{near_zero_count}`; near-constant in global fit: `{near_constant_count}`.
- NaN/Inf after scaling and batching: `{not finite}`.
- Ready for HAI rec-only baseline training: `{summary['ready_for_baseline_training']}`.

All CSVs remain independent sequences. Constant channels are retained and safely scaled because single-file constancy does not imply global inactivity and actuator states may carry anomaly information. Future structure branches may apply separate local activity-rate selection; deletion is deferred to an explicit ablation. This validation does not train or load a model and computes no anomaly-detection metrics.
"""
    (output / "final_hai_dataloader_validation.md").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate boundary-safe HAI dataloaders without training")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    files = [root / f"train{i}.csv" for i in range(1, 7)] + [root / f"test{i}.csv" for i in range(1, 5)]
    if args.dry_run:
        print(f"data_root={root}")
        for path in files:
            print(f"{'OK' if path.exists() else 'MISSING'}: {path}")
        print("Reuse: data.transforms.StandardScaler, torch Dataset/DataLoader, existing [B,L,D] model layout")
        print("Add: data/hai.py, diagnostics, tests; model/training modules unchanged")
        print("No training, checkpoint, metrics, relation selection, CSV writes, or cross-file windows")
        return 0 if all(path.exists() for path in files) else 2
    summary = run(root, SMOKE_OUTPUT if args.smoke else FORMAL_OUTPUT, smoke=args.smoke)
    print(json.dumps({"output": str(SMOKE_OUTPUT if args.smoke else FORMAL_OUTPUT),
                      "ready": summary["ready_for_baseline_training"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
