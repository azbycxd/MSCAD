"""Freeze the CoMuT compatibility normal-only data protocol; never trains a model."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset" / "CoMuT_compat"
OUT = ROOT / "outputs" / "thesis_comut_p0_protocol"
CONFIG_SHA256 = "99852D184C06F0CFD0EBBF085EDCD431DBE8D25DCB82FE004645766727C82699"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def anomaly_intervals(frame: pd.DataFrame, case: str) -> list[dict]:
    labels = frame["is_anomaly"].astype(int).to_numpy()
    timestamps = frame["timestamp"].to_numpy()
    rows: list[dict] = []
    start = None
    for index, label in enumerate(labels):
        if label == 1 and start is None:
            start = index
        if start is not None and (label == 0 or index == len(labels) - 1):
            end = index if label == 1 else index - 1
            rows.append({
                "case_name": case,
                "interval_id": len(rows) + 1,
                "start_row": start,
                "end_row_inclusive": end,
                "start_timestamp": timestamps[start],
                "end_timestamp": timestamps[end],
                "point_count": end - start + 1,
            })
            start = None
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite an existing P0 output: {OUT}")
    cases = sorted(path for path in DATA.iterdir() if path.is_dir())
    assert len(cases) == 13, f"Expected 13 cases, found {len(cases)}"
    source_files = sorted(path for path in DATA.rglob("*") if path.is_file())
    hashes_before = {str(path.relative_to(DATA)): sha256(path) for path in source_files}

    inventory: list[dict] = []
    intervals: list[dict] = []
    train_rows: dict[str, int] = {}
    test_rows: dict[str, int] = {}
    chronology_pass = True
    clean_train_pass = True
    for case in cases:
        train_normal = case / "train_no_anomaly.csv"
        train_anomaly = case / "train_anomaly.csv"
        test = case / "test.csv"
        other = sorted(
            path.name for path in case.iterdir()
            if path.is_file() and path.name not in {"test.csv", "train_no_anomaly.csv", "train_anomaly.csv"}
        )
        inventory.append({
            "case_name": case.name,
            "test_exists": test.is_file(),
            "normal_train_exists": train_normal.is_file(),
            "anomaly_train_exists": train_anomaly.is_file(),
            "other_files": ";".join(other),
        })
        assert train_normal.is_file() and test.is_file()
        normal_frame = pd.read_csv(train_normal)
        test_frame = pd.read_csv(test)
        train_rows[case.name] = len(normal_frame)
        test_rows[case.name] = len(test_frame)
        clean_train_pass &= bool((normal_frame["is_anomaly"].astype(int) == 0).all())
        chronology_pass &= bool(normal_frame["timestamp"].is_monotonic_increasing)
        chronology_pass &= bool(test_frame["timestamp"].is_monotonic_increasing)
        intervals.extend(anomaly_intervals(test_frame, case.name))

    hashes_after = {str(path.relative_to(DATA)): sha256(path) for path in source_files}
    data_unchanged = hashes_before == hashes_after
    all_normal = all(row["normal_train_exists"] for row in inventory)
    assert all_normal and clean_train_pass and chronology_pass and data_unchanged
    assert all(count == 10000 for count in train_rows.values())
    assert all(count == 10000 for count in test_rows.values())

    OUT.mkdir(parents=True)
    with (OUT / "file_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory[0].keys())
        writer.writeheader()
        writer.writerows(inventory)
    with (OUT / "anomaly_interval_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=intervals[0].keys())
        writer.writeheader()
        writer.writerows(intervals)

    manifest = {
        "status": "COMUT_NORMAL_ONLY_PROTOCOL_READY",
        "compatibility_reproduction": True,
        "compat_config_sha256": CONFIG_SHA256,
        "case_count": 13,
        "normal_train_available_count": 13,
        "train": "per-case train_no_anomaly.csv (all 10,000 rows)",
        "test": "per-case test.csv (all 10,000 rows)",
        "train_rows_by_case": train_rows,
        "test_rows_by_case": test_rows,
        "scaler_and_normal_references": "fit only on the corresponding train_no_anomaly.csv",
        "test_label_access": "final evaluation only; P0 reads labels solely to inventory frozen anomaly intervals",
        "prefix_fallback_used": False,
        "candidate_train_end": None,
        "window": 64,
        "stride": 1,
        "primary_metrics": ["point-wise AUROC", "point-wise AUPR"],
        "point_adjustment": False,
        "test_best_threshold": False,
        "f1_primary": False,
        "future_f1_policy": "threshold must be preregistered from normal TRAIN/validation",
        "published_results": "REFERENCE_ONLY / NOT_DIRECTLY_COMPARABLE",
        "published_reference_reasons": [
            "compatibility-generated CoMuT",
            "historical GutenTAG commit unavailable",
            "historical TimeEval algorithm image unavailable",
            "current k-Means semantics drift confirmed",
        ],
        "tests": {
            "all_13_cases_scanned": len(inventory) == 13,
            "data_unmodified_sha256_before_after": data_unchanged,
            "train_test_chronology_monotonic": chronology_pass,
            "normal_train_contains_no_anomaly": clean_train_pass,
            "guard_correct": "NOT_APPLICABLE_INDEPENDENT_NORMAL_TRAIN_PRESENT",
            "no_point_adjustment": True,
            "published_references_reference_only": True,
            "no_model_training": True,
        },
        "source_sha256": hashes_before,
    }
    (OUT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "cases": len(inventory),
        "normal_train_count": 13,
        "train_rows": sorted(set(train_rows.values())),
        "test_rows": sorted(set(test_rows.values())),
        "anomaly_intervals": len(intervals),
        "tests": manifest["tests"],
    }, indent=2))


if __name__ == "__main__":
    main()
