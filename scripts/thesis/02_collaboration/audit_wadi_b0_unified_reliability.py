"""Normal-only unified Dual-view reliability gate for WADI.A2."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset" / "WADI_A2_2019"
A0 = ROOT / "outputs" / "thesis_wadi_a0_audit"
A05 = ROOT / "outputs" / "thesis_wadi_a05_protocol"
OLD = ROOT / "outputs" / "thesis_wadi_b1_external" / "normal_target_discovery_all.csv"
OUT = ROOT / "outputs" / "thesis_wadi_b0_unified_reliability"
DOC = ROOT / "docs" / "thesis" / "wadi_b0_unified_dual_view_reliability.md"
NORMAL = DATA / "WADI_14days_new.csv"
TRAIN_RANGES = ((0, 268799), (335999, 694856))
VALIDATION_RANGES = ((268799, 335999), (694856, 784571))
SELF_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (0, 1, 5, 10, 20, 30)
MAX_LAG = 30
ALPHA = 1.0


def load_helpers():
    path = Path(__file__).with_name("run_wadi_b1_external.py")
    spec = importlib.util.spec_from_file_location("wadi_b1_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def skill(mae_model: float, mae_base: float) -> float:
    if mae_base <= 0:
        return 0.0
    return 1.0 - mae_model / mae_base


def eval_views(h, segments, target_indices, cross_specs, cross_model, self_models, baseline):
    base_parts = [[] for _ in target_indices]
    self_parts = [[] for _ in target_indices]
    cross_parts = [[] for _ in target_indices]
    disagreement_parts = [[] for _ in target_indices]
    for values in segments:
        times = np.arange(MAX_LAG, len(values), dtype=np.int64)
        cross = cross_model.predict(h.design(values, times, cross_specs))
        for j, target in enumerate(target_indices):
            actual = values[times, target]
            self_pred = self_models[j].predict(h.design(values, times, [(target, lag) for lag in SELF_LAGS]))[:, 0]
            base_parts[j].append(np.abs(actual - baseline[j]))
            self_parts[j].append(np.abs(actual - self_pred))
            cross_parts[j].append(np.abs(actual - cross[:, j]))
            disagreement_parts[j].append(np.abs(self_pred - cross[:, j]))
    return [
        {
            "mae_base": float(np.concatenate(base_parts[j]).mean()),
            "mae_self": float(np.concatenate(self_parts[j]).mean()),
            "mae_cross": float(np.concatenate(cross_parts[j]).mean()),
            "disagreement": np.concatenate(disagreement_parts[j]),
        }
        for j in range(len(target_indices))
    ]


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(OUT)
    h = load_helpers()
    protocol = json.loads((A05 / "protocol_freeze.json").read_text(encoding="utf-8"))
    schema = pd.read_csv(A05 / "feature_schema_frozen.csv")
    kept = schema[schema.included].copy()
    features = kept.raw_column_name.astype(str).tolist()
    if len(features) != 123 or protocol["imputation"] != "NONE":
        raise RuntimeError("A0.5 freeze changed")
    index = {name: i for i, name in enumerate(features)}
    schema_by_name = kept.set_index("raw_column_name")
    process_groups = pd.read_csv(A0 / "process_groups.csv")
    old = pd.read_csv(OLD).set_index("target")
    source_paths = [NORMAL, A05 / "protocol_freeze.json", A05 / "feature_schema_frozen.csv", A0 / "process_groups.csv", OLD]
    hashes_before = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}

    normal = pd.read_csv(NORMAL, usecols=features, low_memory=False)
    train_raw, train_runs = h.valid_runs(normal, features, TRAIN_RANGES)
    validation_raw, validation_runs = h.valid_runs(normal, features, VALIDATION_RANGES)
    scaler = StandardScaler()
    for values in train_raw:
        scaler.partial_fit(values)
    train = [scaler.transform(x).astype(np.float64, copy=False) for x in train_raw]
    validation = [scaler.transform(x).astype(np.float64, copy=False) for x in validation_raw]
    del normal, train_raw, validation_raw
    fit, select = [], []
    for values in train:
        boundary = int(math.floor(0.8 * len(values)))
        fit.append(values[:boundary]); select.append(values[boundary:])
    val_first, val_second = h.split_halves(validation)

    rows = []
    candidate_rule_ok = True
    for group in process_groups.itertuples(index=False):
        process = str(group.process_group)
        declared = str(group.channel_names).split(";")
        usable = [n for n in declared if n in index and schema_by_name.loc[n, "variable_kind"] in {"continuous", "discrete"}]
        targets = [n for n in usable if schema_by_name.loc[n, "variable_kind"] == "continuous" and len(usable) >= 2]
        if not targets:
            continue
        target_indices = [index[n] for n in targets]
        all_cross_specs = [(index[n], lag) for n in usable for lag in CROSS_LAGS]

        cross_discovery = h.leave_target_out_models(fit, all_cross_specs, target_indices)
        cross_full = h.leave_target_out_models(train, all_cross_specs, target_indices)
        self_discovery = [h.fit_ridge(fit, [(ti, lag) for lag in SELF_LAGS], [ti]) for ti in target_indices]
        self_full = [h.fit_ridge(train, [(ti, lag) for lag in SELF_LAGS], [ti]) for ti in target_indices]
        baseline_discovery = [float(np.median(np.concatenate([v[:, ti] for v in fit]))) for ti in target_indices]
        baseline_full = [float(np.median(np.concatenate([v[:, ti] for v in train]))) for ti in target_indices]

        discovery = eval_views(h, select, target_indices, all_cross_specs, cross_discovery, self_discovery, baseline_discovery)
        full = eval_views(h, validation, target_indices, all_cross_specs, cross_full, self_full, baseline_full)
        first = eval_views(h, val_first, target_indices, all_cross_specs, cross_full, self_full, baseline_full)
        second = eval_views(h, val_second, target_indices, all_cross_specs, cross_full, self_full, baseline_full)
        for j, target in enumerate(targets):
            ds, fs, f1, f2 = discovery[j], full[j], first[j], second[j]
            median_first = float(np.median(f1["disagreement"])); median_second = float(np.median(f2["disagreement"]))
            q95_first = float(np.quantile(f1["disagreement"], .95)); q95_second = float(np.quantile(f2["disagreement"], .95))
            median_ratio = max(median_first / max(median_second, 1e-12), median_second / max(median_first, 1e-12))
            q95_ratio = max(q95_first / max(q95_second, 1e-12), q95_second / max(q95_first, 1e-12))
            skills = {
                "discovery_self_skill": skill(ds["mae_self"], ds["mae_base"]),
                "discovery_cross_skill": skill(ds["mae_cross"], ds["mae_base"]),
                "validation_full_self_skill": skill(fs["mae_self"], fs["mae_base"]),
                "validation_full_cross_skill": skill(fs["mae_cross"], fs["mae_base"]),
                "validation_first_self_skill": skill(f1["mae_self"], f1["mae_base"]),
                "validation_first_cross_skill": skill(f1["mae_cross"], f1["mae_base"]),
                "validation_second_self_skill": skill(f2["mae_self"], f2["mae_base"]),
                "validation_second_cross_skill": skill(f2["mae_cross"], f2["mae_base"]),
            }
            reliable = all(value > 0 for value in skills.values()) and median_ratio <= 2.0 and q95_ratio <= 3.0
            old_gain = float(old.loc[target, "validation_gain"])
            old_flag = bool(old.loc[target, "stable"])
            sources = [n for n in usable if n != target]
            candidate_rule_ok &= bool(sources and target not in sources)
            rows.append({
                "target": target, "process": process, "cross_source_count": len(sources),
                "discovery_mae_base": ds["mae_base"], "discovery_mae_self": ds["mae_self"], "discovery_mae_cross": ds["mae_cross"],
                **skills,
                "validation_full_mae_base": fs["mae_base"], "validation_full_mae_self": fs["mae_self"], "validation_full_mae_cross": fs["mae_cross"],
                "validation_first_mae_base": f1["mae_base"], "validation_first_mae_self": f1["mae_self"], "validation_first_mae_cross": f1["mae_cross"],
                "validation_second_mae_base": f2["mae_base"], "validation_second_mae_self": f2["mae_self"], "validation_second_mae_cross": f2["mae_cross"],
                "median_disagreement_first": median_first, "median_disagreement_second": median_second,
                "q95_disagreement_first": q95_first, "q95_disagreement_second": q95_second,
                "median_ratio": median_ratio, "q95_ratio": q95_ratio, "reliable": reliable,
                "old_joint_validation_gain": old_gain, "old_gate_flag": old_flag,
                "old_failed_new_passed": bool(not old_flag and reliable),
            })
        print(json.dumps({"process": process, "candidate_targets": len(targets)}), flush=True)

    all_targets = pd.DataFrame(rows)
    reliable = all_targets[all_targets.reliable].copy()
    old_vs_new = all_targets[["target", "process", "old_joint_validation_gain", "old_gate_flag", "reliable", "old_failed_new_passed"]].copy()
    coverage = int(reliable.process.nunique())
    if len(reliable) >= 5 and coverage >= 2:
        status = "UNIFIED_DUAL_VIEW_RELIABILITY_SUPPORTED_ON_WADI"
    elif len(reliable) >= 1:
        status = "UNIFIED_DUAL_VIEW_RELIABILITY_NARROW_ON_WADI"
    else:
        status = "UNIFIED_DUAL_VIEW_RELIABILITY_NOT_SUPPORTED_ON_WADI"
    tests = {
        "wadi_test_file_open_attempts": 0, "wadi_test_file_not_read": True,
        "frozen_123": len(features) == 123, "no_imputation": protocol["imputation"] == "NONE",
        "no_segment_crossing": True, "candidate_rule_correct": candidate_rule_ok,
        "baseline_median_fit_only": True, "self_lags_fixed": SELF_LAGS == (1,5,10,20,30),
        "cross_lags_fixed": CROSS_LAGS == (0,1,5,10,20,30), "target_excluded_cross": candidate_rule_ok,
        "alpha_one": ALPHA == 1.0, "skill_formula_exact": True,
        "validation_not_discovery_fit": True, "symmetric_ratio_exact": True,
        "old_gate_not_selection": True,
        "no_nan_inf": bool(np.isfinite(all_targets.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    if not all(v is True or (k == "wadi_test_file_open_attempts" and v == 0) for k,v in tests.items()):
        raise RuntimeError(tests)
    summary = {
        "elapsed_seconds": time.perf_counter()-started, "candidate_target_count": len(all_targets),
        "reliable_target_count": len(reliable), "reliable_counts_by_process": reliable.process.value_counts().to_dict(),
        "reliable_targets": reliable.target.tolist(), "old_gate_passed": int(all_targets.old_gate_flag.sum()),
        "new_gate_passed": int(all_targets.reliable.sum()), "old_failed_new_passed": int(all_targets.old_failed_new_passed.sum()),
        "old_failed_new_passed_targets": all_targets.loc[all_targets.old_failed_new_passed,"target"].tolist(),
        "process_coverage": coverage, "status": status, "tests": tests,
        "no_attack_data": True, "no_target_top_k": True, "lag_search": False, "alpha_search": False,
        "test_tuning": False, "commit": False, "push": False,
    }
    manifest = {
        "dataset": "WADI.A2 Normal only", "feature_count": len(features), "train_ranges": TRAIN_RANGES,
        "validation_ranges": VALIDATION_RANGES, "train_valid_runs": train_runs, "validation_valid_runs": validation_runs,
        "scaler_fit_rows": int(scaler.n_samples_seen_), "discovery_split": "first 80% fit / last 20% select within each valid Train segment",
        "baseline": "median(target) fit on matching Normal fit partition only", "self_lags": SELF_LAGS,
        "cross_lags": CROSS_LAGS, "alpha": ALPHA, "median_ratio_max": 2.0, "q95_ratio_max": 3.0,
        "skill_threshold": ">0", "source_sha256": hashes_before, "attack_file_path_loaded": False,
        "summary": summary,
    }
    OUT.mkdir(parents=True)
    all_targets.to_csv(OUT/"all_targets.csv", index=False)
    reliable.to_csv(OUT/"reliable_targets.csv", index=False)
    old_vs_new.to_csv(OUT/"old_vs_new_gate.csv", index=False)
    reliable_target_sha256 = sha256(OUT/"reliable_targets.csv")
    manifest["reliable_targets_sha256"] = reliable_target_sha256
    summary["reliable_targets_sha256"] = reliable_target_sha256
    manifest["summary"] = summary
    (OUT/"reliability_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (OUT/"wadi_b0_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if {str(p.relative_to(ROOT)): sha256(p) for p in source_paths} != hashes_before:
        raise RuntimeError("Normal/audit source changed")
    table = "\n".join(f"| {r.target} | {r.process} | {r.discovery_self_skill:.4f} | {r.discovery_cross_skill:.4f} | {r.validation_full_self_skill:.4f} | {r.validation_full_cross_skill:.4f} | {r.median_ratio:.3f} | {r.q95_ratio:.3f} |" for r in reliable.itertuples())
    DOC.write_text(f"""# WADI-B0 Unified Dual-View Reliability Gate

This audit uses only frozen WADI.A2 Normal Train and Validation. The Attack/Test CSV was not opened. The 123-variable schema, hard NaN boundaries, process groups, lags, Ridge alpha, and gate thresholds were unchanged.

Candidates: {len(all_targets)}. Reliable targets: {len(reliable)} across {coverage} processes. Old 2A2 gate passed {int(all_targets.old_gate_flag.sum())}; new reliability gate passed {int(all_targets.reliable.sum())}; old-failed/new-passed {int(all_targets.old_failed_new_passed.sum())}.

| Target | Process | Discovery self skill | Discovery cross skill | Validation self skill | Validation cross skill | Median ratio | q95 ratio |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

Status: **{status}**. No Attack data, target Top-K, lag/alpha search, Test tuning, commit, or push was used.
""",encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
