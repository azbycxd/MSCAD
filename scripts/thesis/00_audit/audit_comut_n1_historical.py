"""Audit historical official CoMuT generation semantics; no model execution."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_REPO = Path(r"D:\workspace\python\external\GutenTAG-1.0.0")
CONFIG = HISTORICAL_REPO / "generation_configs" / "comut-datasets.yaml"
HISTORICAL = ROOT / "dataset" / "CoMuT_historical_v1"
COMPAT = ROOT / "dataset" / "CoMuT_compat"
OUT = ROOT / "outputs" / "thesis_comut_n1_historical"
EXPECTED_COMMIT = "5638dadf9b1e83699ca317ce9eb4569a6c350064"
EXPECTED_CONFIG_SHA = "4D90D3871BFBB6C088F0AE4A3EAFE279D365FDBE4CACB4F85397908826570717"
REPRESENTATIVES = [
    "rmj-short-mode-correlation-on-2",
    "rmj-medium-mode-correlation-on-2",
    "rmj-large-mode-correlation-on-2",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns if str(column).startswith("value-")]


def pair_stats(frame: pd.DataFrame) -> tuple[list[tuple[int, int]], np.ndarray]:
    values = frame[feature_columns(frame)].to_numpy(float)
    pairs = list(itertools.combinations(range(values.shape[1]), 2))
    correlations = np.asarray([np.corrcoef(values[:, left], values[:, right])[0, 1] for left, right in pairs])
    if not np.isfinite(correlations).all():
        raise ValueError("Non-finite Pearson correlation")
    return pairs, correlations


def jump_stats(frame: pd.DataFrame) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, list[float]]:
    values = frame[feature_columns(frame)].to_numpy(float)
    delta = np.diff(values, axis=0)
    indicators = []
    for channel in range(values.shape[1]):
        magnitude = np.abs(delta[:, channel])
        center = np.median(magnitude)
        mad = np.median(np.abs(magnitude - center))
        threshold = center + 6.0 * max(1.4826 * mad, 1e-12)
        indicators.append(magnitude > threshold)
    pairs = list(itertools.combinations(range(values.shape[1]), 2))
    timing, signed = [], []
    for left, right in pairs:
        union = indicators[left] | indicators[right]
        intersection = indicators[left] & indicators[right]
        timing.append(float(intersection.sum() / union.sum()) if union.any() else np.nan)
        if intersection.any():
            signed.append(float(np.mean(np.sign(delta[intersection, left]) == np.sign(delta[intersection, right]))))
        else:
            signed.append(np.nan)
    return pairs, np.asarray(timing), np.asarray(signed), [float(x.sum()) for x in indicators]


def anomaly_rows(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame["is_anomaly"].astype(int).to_numpy()
    selected = frame.loc[labels == 1]
    if selected.empty:
        raise ValueError("Representative test has no anomaly interval")
    return selected


def correlation_summary(frame: pd.DataFrame) -> dict:
    _, correlations = pair_stats(frame)
    return {
        "median_signed_pearson": float(np.median(correlations)),
        "median_abs_pearson": float(np.median(np.abs(correlations))),
        "min_abs_pearson": float(np.min(np.abs(correlations))),
        "max_abs_pearson": float(np.max(np.abs(correlations))),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def finite_median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else None


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite N1 output: {OUT}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HISTORICAL_REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=HISTORICAL_REPO, text=True).strip()
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=HISTORICAL_REPO, text=True).strip()
    commit_date = subprocess.check_output(["git", "show", "-s", "--format=%cI", "HEAD"], cwd=HISTORICAL_REPO, text=True).strip()
    if head != EXPECTED_COMMIT or status:
        raise RuntimeError("Historical GutenTAG clone gate failed")
    if sha256(CONFIG) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("Historical config SHA gate failed")
    parsed = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cases = parsed["timeseries"]
    base = parsed["random-mode-jump"][0]
    if len(cases) != 13 or base != {
        "kind": "random-mode-jump", "frequency": 250.0, "variance": 0.05, "random-seed": 42
    }:
        raise RuntimeError("Historical config semantic gate failed")
    generated_cases = sorted(path for path in HISTORICAL.iterdir() if path.is_dir())
    if len(generated_cases) != 13:
        raise RuntimeError("Generation did not produce 13 cases")

    dataset_rows, normal_rows, jump_rows, representative_rows, comparison_rows = [], [], [], [], []
    all_historical_abs = []
    all_jump_timing = []
    hashes_before = {str(path.relative_to(HISTORICAL)): sha256(path) for path in HISTORICAL.rglob("*") if path.is_file()}
    for case_path in generated_cases:
        test_path = case_path / "test.csv"
        normal_path = case_path / "train_no_anomaly.csv"
        anomaly_train_path = case_path / "train_anomaly.csv"
        test = pd.read_csv(test_path)
        normal = pd.read_csv(normal_path)
        features = feature_columns(normal)
        arrays = [normal[features].to_numpy(float), test[features].to_numpy(float)]
        values_finite = all(np.isfinite(array).all() for array in arrays)
        labels_valid = set(test["is_anomaly"].astype(int).unique()).issubset({0, 1})
        gate = len(normal) == 10000 and len(test) == 10000 and values_finite and labels_valid
        dataset_rows.append({
            "case_name": case_path.name,
            "test_exists": test_path.is_file(),
            "train_no_anomaly_exists": normal_path.is_file(),
            "train_anomaly_exists": anomaly_train_path.is_file(),
            "test_rows": len(test),
            "train_no_anomaly_rows": len(normal),
            "channels": len(features),
            "labels": ";".join(map(str, sorted(test["is_anomaly"].astype(int).unique()))),
            "nan_count": int(sum(np.isnan(array).sum() for array in arrays)),
            "inf_count": int(sum(np.isinf(array).sum() for array in arrays)),
            "gate_pass": gate,
        })
        if not gate:
            raise RuntimeError(f"Dataset gate failed: {case_path.name}")
        pairs, correlations = pair_stats(normal)
        abs_correlations = np.abs(correlations)
        all_historical_abs.extend(abs_correlations.tolist())
        normal_rows.append({
            "case_name": case_path.name,
            "pair_count": len(pairs),
            "median_abs_pearson": float(np.median(abs_correlations)),
            "min_abs_pearson": float(abs_correlations.min()),
            "max_abs_pearson": float(abs_correlations.max()),
        })
        _, timing, signed, jump_counts = jump_stats(normal)
        if not np.isfinite(timing).all():
            raise ValueError("Normal jump timing agreement cannot be computed")
        all_jump_timing.extend(timing.tolist())
        jump_rows.append({
            "case_name": case_path.name,
            "pair_count": len(pairs),
            "jump_indicator": "abs(delta) > median(abs(delta)) + 6 * max(1.4826*MAD,1e-12)",
            "median_jump_timing_jaccard": float(np.median(timing)),
            "min_jump_timing_jaccard": float(np.min(timing)),
            "max_jump_timing_jaccard": float(np.max(timing)),
            "median_signed_jump_agreement": float(np.nanmedian(signed)),
            "jump_counts_by_channel": ";".join(str(int(value)) for value in jump_counts),
        })

        old_normal = pd.read_csv(COMPAT / case_path.name / "train_no_anomaly.csv")
        old_test = pd.read_csv(COMPAT / case_path.name / "test.csv")
        old_normal_summary = correlation_summary(old_normal)
        new_normal_summary = correlation_summary(normal)
        old_anomaly_summary = correlation_summary(anomaly_rows(old_test))
        new_anomaly_summary = correlation_summary(anomaly_rows(test))
        comparison_rows.append({
            "case_name": case_path.name,
            "compat_normal_median_abs_pearson": old_normal_summary["median_abs_pearson"],
            "historical_normal_median_abs_pearson": new_normal_summary["median_abs_pearson"],
            "normal_abs_pearson_delta": new_normal_summary["median_abs_pearson"] - old_normal_summary["median_abs_pearson"],
            "compat_anomaly_median_signed_pearson": old_anomaly_summary["median_signed_pearson"],
            "historical_anomaly_median_signed_pearson": new_anomaly_summary["median_signed_pearson"],
        })

        if case_path.name in REPRESENTATIVES:
            anomalous = anomaly_rows(test)
            _, anomaly_corr = pair_stats(anomalous)
            normal_signed = float(np.median(correlations))
            anomaly_signed = float(np.median(anomaly_corr))
            _, anomaly_timing, anomaly_signed_jump, _ = jump_stats(anomalous)
            relation_shift = float(np.median(np.abs(anomaly_corr - correlations)))
            representative_rows.append({
                "case_name": case_path.name,
                "anomaly_start_timestamp": int(anomalous["timestamp"].iloc[0]),
                "anomaly_end_timestamp": int(anomalous["timestamp"].iloc[-1]),
                "anomaly_points": len(anomalous),
                "normal_median_signed_pearson": normal_signed,
                "anomaly_median_signed_pearson": anomaly_signed,
                "median_signed_relation_shift": relation_shift,
                "normal_jump_timing_jaccard": float(np.median(timing)),
                "anomaly_jump_timing_jaccard": finite_median(anomaly_timing),
                "normal_signed_jump_agreement": float(np.nanmedian(signed)),
                "anomaly_signed_jump_agreement": finite_median(anomaly_signed_jump),
                "relation_structure_clearly_deviates": bool(relation_shift >= 0.5),
            })

    hashes_after = {str(path.relative_to(HISTORICAL)): sha256(path) for path in HISTORICAL.rglob("*") if path.is_file()}
    all_normal_gate = all(row["median_abs_pearson"] >= 0.80 for row in normal_rows)
    representative_gate = len(representative_rows) == 3 and all(row["relation_structure_clearly_deviates"] for row in representative_rows)
    final_status = "COMUT_HISTORICAL_SEMANTICS_RECOVERED" if all_normal_gate and representative_gate else "COMUT_HISTORICAL_REPRODUCTION_FAILED"
    site_packages = Path(r"D:\workspace\python\external\comut-venv\Lib\site-packages")
    freeze = []
    for metadata in sorted(site_packages.glob("*.dist-info/METADATA")):
        name = version = None
        for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Name: ") and name is None:
                name = line[6:]
            elif line.startswith("Version: ") and version is None:
                version = line[9:]
            if name and version:
                break
        if name and version:
            freeze.append(f"{name}=={version}")
    source_manifest = {
        "repository": remote,
        "head": head,
        "commit_date": commit_date,
        "clone_clean": True,
        "gutentag_version": "1.0.0",
        "python_version": sys.version,
        "environment": str(site_packages.parent.parent),
        "environment_note": "existing comut-venv site-packages loaded read-only with p310 interpreter because its launcher is broken",
        "pip_freeze_from_dist_info": freeze,
        "config_path": str(CONFIG),
        "config_sha256": EXPECTED_CONFIG_SHA,
        "config_modified": False,
        "seed": 42,
        "generation_command_semantics": "python -m gutenTAG --config-yaml <official> --output-dir <historical_v1> --seed 42 --addons gutenTAG.addons.timeeval.TimeEvalAddOn --n_jobs 1",
    }
    summary = {
        "status": final_status,
        "generation_success": "13/13",
        "train_no_anomaly_success": "13/13",
        "normal_pairwise_abs_pearson_median": float(np.median(all_historical_abs)),
        "normal_pairwise_abs_pearson_range": [float(np.min(all_historical_abs)), float(np.max(all_historical_abs))],
        "normal_jump_timing_agreement_median": float(np.median(all_jump_timing)),
        "representative_relation_gate": representative_gate,
        "old_compat_vs_historical_normal_correlation_recovered": bool(
            np.median([row["historical_normal_median_abs_pearson"] for row in comparison_rows])
            - np.median([row["compat_normal_median_abs_pearson"] for row in comparison_rows]) >= 0.80
        ),
        "dataset_sha256_unchanged_during_audit": hashes_before == hashes_after,
        "no_model_training": True,
    }
    OUT.mkdir(parents=True)
    shutil.copyfile(CONFIG, OUT / "historical_config_snapshot.yaml")
    if sha256(OUT / "historical_config_snapshot.yaml") != EXPECTED_CONFIG_SHA:
        raise RuntimeError("Config snapshot is not exact")
    write_csv(OUT / "dataset_manifest.csv", dataset_rows)
    write_csv(OUT / "normal_correlation_audit.csv", normal_rows)
    write_csv(OUT / "jump_sync_audit.csv", jump_rows)
    write_csv(OUT / "representative_anomaly_audit.csv", representative_rows)
    write_csv(OUT / "compat_vs_historical.csv", comparison_rows)
    (OUT / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    (OUT / "n1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "representatives": representative_rows}, indent=2))


if __name__ == "__main__":
    main()
