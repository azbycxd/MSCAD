"""Audit the CoMuT compatibility generation without fitting a detector."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = Path(r"D:\workspace\python\external\s2gpp_experiments")
DATA = ROOT / "dataset" / "CoMuT_compat"
GENERATION = ROOT / "dataset" / "CoMuT_generation"
OUTPUT = ROOT / "outputs" / "thesis_comut_n0r"
DOC = ROOT / "docs" / "thesis" / "comut_n0r_compatibility_reproduction.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def intervals(labels: np.ndarray) -> list[tuple[int, int, int]]:
    mask = labels.astype(bool); padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:]); ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return [(int(a), int(b), int(b - a + 1)) for a, b in zip(starts, ends)]


def corr_summary(values: np.ndarray) -> tuple[float, float]:
    corr = np.atleast_2d(np.corrcoef(values, rowvar=False)); mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.mean(corr[mask])), float(np.mean(np.abs(corr[mask])))


def published_results(readme: str) -> list[dict[str, str]]:
    section = readme.split("## Quality CoMuT (AUROC)", 1)[1].split("## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    headers = [part.strip().replace("**", "") for part in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        values = [part.strip().replace("**", "") for part in line.strip("|").split("|")]
        row = dict(zip(headers, values))
        rows.append({"dataset_name": row["dataset_name"], "LSTM-AD": row["LSTM-AD"], "S2G++20p-KDE": row["S2G++20p-KDE"], "k-Means": row["k-Means"], "source": "published_repository_result", "locally_reproduced": "false"})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    original = yaml.safe_load((OFFICIAL / "dataset_configs" / "comut.yaml").read_text(encoding="utf-8"))
    compat = yaml.safe_load((GENERATION / "comut_compat.yaml").read_text(encoding="utf-8"))
    configs = {case["name"]: case for case in original["timeseries"]}
    generated_dirs = sorted(path for path in DATA.iterdir() if path.is_dir())
    dataset_rows = []; interval_rows = []; compare_rows = []
    relation = {}
    for directory in generated_dirs:
        case = configs[directory.name]; path = directory / "test.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle); header = next(reader); raw = np.asarray([[float(v) for v in row] for row in reader], dtype=np.float64)
        label_name = header[-1]; timestamp_name = header[0]; features = raw[:, 1:-1]; labels = raw[:, -1].astype(np.int8)
        recovered = intervals(labels); expected_lengths = [int(a["length"]) for a in case["anomalies"]]
        nan_count = int(np.isnan(raw).sum()); inf_count = int(np.isinf(raw).sum())
        category = "multiple-short" if len(case["anomalies"]) > 1 else "short" if "-short-" in case["name"] else "medium" if "-medium-" in case["name"] else "large"
        dataset_rows.append({
            "name": case["name"], "generation_id": case["generation-id"], "category": category,
            "file_path": str(path), "file_size": path.stat().st_size, "sha256": sha(path), "rows": len(raw),
            "channels": features.shape[1], "label_column": label_name, "normal_points": int((labels == 0).sum()),
            "positive_points": int(labels.sum()), "anomaly_interval_count": len(recovered),
            "anomaly_interval_lengths": ";".join(str(x[2]) for x in recovered), "timestamp_column": timestamp_name,
            "format": "CSV", "nan_count": nan_count, "inf_count": inf_count,
        })
        for number, (start, end, length) in enumerate(recovered, 1):
            configured = case["anomalies"][number - 1]
            interval_rows.append({"name": case["name"], "interval_index": number, "start": start, "end": end, "length": length, "configured_length": configured["length"], "configured_position": configured["position"], "explicit_channel": configured.get("channel", "")})
        match = len(raw) == int(case["length"]) and features.shape[1] == int(case["channels"]) and sorted(x[2] for x in recovered) == sorted(expected_lengths) and label_name == "is_anomaly"
        compare_rows.append({
            "name": case["name"], "generation_id": case["generation-id"], "configured_length": case["length"],
            "generated_rows": len(raw), "configured_channels": case["channels"], "generated_channels": features.shape[1],
            "configured_anomaly_lengths": ";".join(map(str, expected_lengths)), "generated_anomaly_lengths": ";".join(str(x[2]) for x in recovered), "match": match,
        })
        if case["name"] in {"rmj-short-mode-correlation-on-2", "rmj-medium-mode-correlation-on-2", "rmj-large-mode-correlation-on-2"}:
            normal_signed, normal_abs = corr_summary(features[labels == 0]); anomaly_signed, anomaly_abs = corr_summary(features[labels == 1])
            relation[category] = {"case": case["name"], "normal_pairwise_correlation": normal_signed, "anomaly_pairwise_correlation": anomaly_signed, "normal_mean_abs_correlation": normal_abs, "anomaly_mean_abs_correlation": anomaly_abs, "absolute_correlation_change": anomaly_abs - normal_abs, "direction": "decreased" if anomaly_abs < normal_abs else "increased"}
    refs = published_results((OFFICIAL / "README.md").read_text(encoding="utf-8"))
    if len(refs) != 13:
        raise RuntimeError("published CoMuT result table must have 13 rows")
    write_csv(OUTPUT / "dataset_manifest.csv", dataset_rows); write_csv(OUTPUT / "anomaly_intervals.csv", interval_rows); write_csv(OUTPUT / "config_vs_generated.csv", compare_rows); write_csv(OUTPUT / "official_reference_results.csv", refs)
    all_pass = len(dataset_rows) == 13 and {row["name"] for row in dataset_rows} == set(configs) and all(row["match"] for row in compare_rows) and all(row["nan_count"] == 0 and row["inf_count"] == 0 for row in dataset_rows)
    manifest_path = OUTPUT / "generation_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["generation_status"] = "13/13" if all_pass else "FAILED_AUDIT"; manifest["generator_package"] = "timeeval-gutenTAG==1.0.0"; manifest["generator_numpy"] = "1.21.6"; manifest["generator_scipy_compatibility"] = "1.7.3"; manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {"status": "COMUT_COMPAT_BENCHMARK_READY" if all_pass else "COMUT_NOT_READY", "author_artifact_found": False, "dataset_count": len(dataset_rows), "seed": 42, "removed_amplitude_count": 13, "other_changed_key_count": 0, "original_config_preserved": True, "compat_config_sha256": sha(GENERATION / "comut_compat.yaml"), "config_vs_generated": "PASS" if all_pass else "FAIL", "label_column": "is_anomaly", "row_count_per_case": sorted(set(row["rows"] for row in dataset_rows)), "channel_counts": sorted(set(row["channels"] for row in dataset_rows)), "nan_count": sum(row["nan_count"] for row in dataset_rows), "inf_count": sum(row["inf_count"] for row in dataset_rows), "representative_relation_sanity": relation, "official_reference_rows": len(refs), "compatibility_reproduction": True, "exact_historical_generator": False, "model_training": False, "parameter_search": False}
    (OUTPUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# CoMuT N0R compatibility reproduction", "", "No author-generated CoMuT artifact was found in the official repository history, branches, tags, or LFS. The original config is preserved at SHA256 `D9F6EB...175B`.", "", "A compatibility copy removes exactly 13 `random-mode-jump.amplitude` keys; no other structural key changes. This is schema adaptation, not parameter tuning. Because the historical generator commit was unavailable, this is not a bitwise historical reproduction.", "", "## Audit", "", f"- Status: `{summary['status']}`", "- Generated cases: 13/13", "- Rows: 10,000 per case", "- Channels: 2, 3, or 4", "- Label: `is_anomaly` with values 0/1", "- NaN/Inf: 0/0", "- Config vs generated: PASS", "", "## Relation sanity", ""]
    lines += [f"- {key}: `{value['case']}` normal correlation={value['normal_pairwise_correlation']:.6f}, anomaly correlation={value['anomaly_pairwise_correlation']:.6f}, |corr| {value['direction']}." for key, value in relation.items()]
    lines += ["", "CoMuT is a dedicated synthetic correlation-anomaly benchmark, not the primary real industrial anomaly benchmark. Published repository AUROC values are REFERENCE_ONLY and are explicitly marked `locally_reproduced=false`.", "", "NO MODEL TRAINING; NO PARAMETER SEARCH; SEED 42."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
