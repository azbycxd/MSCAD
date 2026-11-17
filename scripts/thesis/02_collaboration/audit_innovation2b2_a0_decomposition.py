"""Decompose frozen A0 endpoint errors for the frozen Innovation-2B2 corruptions."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "thesis" / "02_collaboration"
for item in (ROOT, HELPER):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_innovation2a2_joint_lagged as a2  # noqa: E402
import run_innovation2b_controlled as b2  # noqa: E402
from data.swat2015_official import (  # noqa: E402
    EXPECTED_FEATURES, load_normal_cache, load_scaler, resolve_repo_path, validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


OUT = ROOT / "outputs" / "thesis_innovation2b2_a0_decomposition"
B2 = ROOT / "outputs" / "thesis_innovation2b2_relation_breaking"


def endpoint_errors(model, windows, mean, scale, device):
    vectors, globals_ = [], []
    with torch.inference_mode():
        for start in range(0, len(windows), 256):
            raw = np.asarray(windows[start:start + 256], dtype=np.float32)
            tensor = torch.from_numpy(np.ascontiguousarray((raw - mean) / scale)).to(device)
            errors = torch.abs(model(tensor) - tensor)[:, -1, :]
            vectors.append(errors.cpu().numpy())
            globals_.append(errors.mean(dim=1).cpu().numpy())
    vector = np.concatenate(vectors).astype(np.float64, copy=False)
    global_score = np.concatenate(globals_).astype(np.float64, copy=False)
    if vector.shape != (len(windows), EXPECTED_FEATURES) or not np.isfinite(vector).all():
        raise RuntimeError("invalid A0 endpoint error vectors")
    return vector, global_score


def magnitude(target, component, clean, corrupt):
    clean_median = float(np.median(clean))
    corrupt_median = float(np.median(corrupt))
    return {
        "target": target, "component": component,
        "clean_median": clean_median, "corrupt_median": corrupt_median,
        "absolute_delta": corrupt_median - clean_median,
        "corrupt_clean_ratio": corrupt_median / clean_median if clean_median > 1e-12 else None,
        "ratio_suppressed_near_zero": clean_median <= 1e-12,
    }


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite decomposition output: {OUT}")
    config = load_config(str(b2.CONFIG))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    order = [str(name) for name in cache_manifest["column_order"]]
    name_to_index = {name: index for index, name in enumerate(order)}
    schema = json.loads(b2.SCHEMA.read_text(encoding="utf-8"))
    contract, _ = a2.build_contract(a2.normal_profiles(schema), order)
    b2_metrics = pd.read_csv(B2 / "metrics.csv").set_index("target")
    donors = pd.read_csv(B2 / "donor_manifest.csv")
    if tuple(b2_metrics.index) != b2.EXPECTED_TARGETS:
        raise RuntimeError("frozen 2B2 targets changed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = b2.build_a0(config, device)
    scaler, metadata = load_scaler(b2.A0_DIR / "scaler.npz")
    if metadata["fit_scope"] != "normal_v1_raw_rows_[0,396000)":
        raise RuntimeError("frozen A0 scaler scope changed")
    mean = np.asarray(scaler.mean_, dtype=np.float32).reshape(1, 1, -1)
    scale = np.asarray(scaler.effective_scale_, dtype=np.float32).reshape(1, 1, -1)
    offsets = np.arange(-b2.WINDOW + 1, 1, dtype=np.int64)
    first = donors[donors["target"] == b2.EXPECTED_TARGETS[0]].sort_values("sample_index")
    clean_endpoints = first["clean_endpoint"].to_numpy(np.int64)
    clean_windows = np.asarray(normal[clean_endpoints[:, None] + offsets[None, :]], dtype=np.float32)
    clean_errors, clean_global = endpoint_errors(model, clean_windows, mean, scale, device)

    metric_rows, magnitude_rows, reproduction_rows = [], [], []
    for target in b2.EXPECTED_TARGETS:
        target_donors = donors[donors["target"] == target].sort_values("sample_index")
        if not np.array_equal(target_donors["clean_endpoint"].to_numpy(np.int64), clean_endpoints):
            raise RuntimeError(f"clean endpoint mismatch for {target}")
        donor_endpoints = target_donors["donor_endpoint"].to_numpy(np.int64)
        stage = str(b2_metrics.loc[target, "stage"])
        stage_names = list(contract[stage]["nonconstant_channels"])
        sources = [name for name in stage_names if name != target]
        source_indices = [name_to_index[name] for name in sources]
        target_index = name_to_index[target]
        unaffected_indices = [index for index in range(EXPECTED_FEATURES) if index not in source_indices and index != target_index]
        stage_indices = [target_index, *source_indices]

        corrupted_windows = clean_windows.copy()
        donor_rows = donor_endpoints[:, None] + offsets[None, :]
        corrupted_windows[:, :, source_indices] = np.asarray(
            normal[donor_rows[:, :, None], np.asarray(source_indices)[None, None, :]], dtype=np.float32
        )
        corrupt_errors, corrupt_global = endpoint_errors(model, corrupted_windows, mean, scale, device)
        components = {
            "GLOBAL": (clean_global, corrupt_global),
            "TARGET_ONLY": (clean_errors[:, target_index], corrupt_errors[:, target_index]),
            "CORRUPTED_SOURCES": (clean_errors[:, source_indices].mean(axis=1), corrupt_errors[:, source_indices].mean(axis=1)),
            "UNAFFECTED": (clean_errors[:, unaffected_indices].mean(axis=1), corrupt_errors[:, unaffected_indices].mean(axis=1)),
            "STAGE_ALL": (clean_errors[:, stage_indices].mean(axis=1), corrupt_errors[:, stage_indices].mean(axis=1)),
        }
        for component, (clean_score, corrupt_score) in components.items():
            values = b2.balanced_metrics(clean_score, corrupt_score)
            metric_rows.append({"target": target, "stage": stage, "component": component, **values})
            magnitude_rows.append(magnitude(target, component, clean_score, corrupt_score))
        refit_global = b2.balanced_metrics(clean_global, corrupt_global)
        difference = abs(refit_global["auroc"] - float(b2_metrics.loc[target, "a0_auroc"]))
        reproduction_rows.append({"target": target, "refit_global_auroc": refit_global["auroc"], "original_global_auroc": float(b2_metrics.loc[target, "a0_auroc"]), "abs_difference": difference})

    metrics = pd.DataFrame(metric_rows)
    magnitudes = pd.DataFrame(magnitude_rows)
    reproduction = pd.DataFrame(reproduction_rows)
    max_difference = float(reproduction["abs_difference"].max())
    if max_difference > 1e-8:
        status = "A0_DECOMPOSITION_REPRODUCTION_MISMATCH"
    else:
        # Descriptive classification: inspect whether source-only consistently explains global ranking
        # while target-only does not. This is not a method-success gate.
        pivot = metrics.pivot(index="target", columns="component", values="auroc")
        source_dominated = int(((pivot["CORRUPTED_SOURCES"] >= 0.90) & ((pivot["GLOBAL"] - pivot["TARGET_ONLY"]) > 0.05)).sum())
        target_near_global = int((np.abs(pivot["TARGET_ONLY"] - pivot["GLOBAL"]) <= 0.05).sum())
        if source_dominated >= 4:
            status = "GLOBAL_A0_DOMINATED_BY_CORRUPTED_SOURCE_ERRORS"
        elif target_near_global >= 4:
            status = "A0_TARGET_RECONSTRUCTION_CAPTURES_CONTEXT_VIOLATION"
        else:
            status = "MIXED_A0_SCORE_ORIGIN"

    summary = {
        "a0_global_reproduction": "PASS" if max_difference <= 1e-8 else "FAIL",
        "max_global_auroc_absolute_difference": max_difference,
        "final_diagnosis": status,
        "attack_data_read": False, "retraining": False, "new_corruption": False,
        "donor_change": False, "score_tuning": False, "threshold_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.mkdir(parents=True)
    metrics.to_csv(OUT / "a0_component_metrics.csv", index=False)
    magnitudes.to_csv(OUT / "a0_component_magnitudes.csv", index=False)
    reproduction.to_csv(OUT / "a0_global_reproduction.csv", index=False)
    (OUT / "a0_decomposition_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if max_difference > 1e-8:
        raise RuntimeError(json.dumps(summary))
    print(json.dumps({"summary": summary, "metrics": metric_rows, "magnitudes": magnitude_rows}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
