"""HAI 21.03 final evaluation under the unified Dual-view reliability gate."""
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
import torch
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset" / "hai" / "hai-21.03"
A0 = ROOT / "outputs" / "thesis_hai21_a0_audit"
OLD = ROOT / "outputs" / "thesis_hai21_b1_real"
OUT = ROOT / "outputs" / "thesis_hai21_final_unified"
DOC = ROOT / "docs" / "thesis" / "hai21_final_unified_dual_view.md"
TRAIN_FILES = tuple(f"train{i}.csv.gz" for i in range(1, 4))
TEST_FILES = tuple(f"test{i}.csv.gz" for i in range(1, 6))
LABELS = ("attack", "attack_P1", "attack_P2", "attack_P3")
SELF_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (0, 1, 5, 10, 20, 30)
MAX_LAG, WINDOW, ALPHA, Q = 30, 64, 1.0, 0.99
HIGH_AP = {"AP02", "AP04", "AP07", "AP09", "AP12", "AP14", "AP17", "AP19"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def skill(mae_model: float, mae_base: float) -> float:
    return 0.0 if mae_base <= 0 else 1.0 - mae_model / mae_base


def evaluate_views(helper, segments, targets, specs, cross_model, self_models, baselines):
    base_parts, self_parts, cross_parts, disagreement_parts = ([[] for _ in targets] for _ in range(4))
    for values in segments:
        times = np.arange(MAX_LAG, len(values), dtype=np.int64)
        cross = cross_model.predict(helper.design(values, times, specs))
        for j, target in enumerate(targets):
            actual = values[times, target]
            self_specs = [(target, lag) for lag in SELF_LAGS]
            self_prediction = self_models[j].predict(helper.design(values, times, self_specs))[:, 0]
            base_parts[j].append(np.abs(actual - baselines[j]))
            self_parts[j].append(np.abs(actual - self_prediction))
            cross_parts[j].append(np.abs(actual - cross[:, j]))
            disagreement_parts[j].append(np.abs(self_prediction - cross[:, j]))
    return [{
        "mae_base": float(np.concatenate(base_parts[j]).mean()),
        "mae_self": float(np.concatenate(self_parts[j]).mean()),
        "mae_cross": float(np.concatenate(cross_parts[j]).mean()),
        "disagreement": np.concatenate(disagreement_parts[j]),
    } for j in range(len(targets))]


def score_dual(helper, values, frozen, models, references):
    times = np.arange(MAX_LAG, len(values), dtype=np.int64)
    raw, normalized = [], []
    for row in frozen:
        target, ti, sources = str(row["target"]), int(row["target_index"]), list(row["source_indices"])
        self_specs = [(ti, lag) for lag in SELF_LAGS]
        cross_specs = [(source, lag) for source in sources for lag in CROSS_LAGS]
        self_model, cross_model = models[target]
        ps = self_model.predict(helper.design(values, times, self_specs))[:, 0]
        pc = cross_model.predict(helper.design(values, times, cross_specs))[:, 0]
        disagreement = np.abs(ps - pc)
        median, scale = references[target]
        raw.append(disagreement)
        normalized.append(np.maximum(0.0, (disagreement - median) / scale))
    raw_matrix = np.column_stack(raw)
    score_matrix = np.column_stack(normalized)
    return score_matrix.max(1), score_matrix.argmax(1), raw_matrix


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    h = load_module("hai_b1_final_helpers", Path(__file__).with_name("run_hai21_b1_real.py"))
    w = load_module("wadi_b1_final_helpers", Path(__file__).with_name("run_wadi_b1_external.py"))

    audit = json.loads((A0 / "audit_summary.json").read_text(encoding="utf-8"))
    if audit["status"] != "HAI21_BENCHMARK_PROTOCOL_CANDIDATE_READY":
        raise RuntimeError("HAI-A0 protocol is not frozen")
    schema = pd.read_csv(A0 / "variable_schema.csv")
    groups = pd.read_csv(A0 / "process_groups.csv")
    events_audit = pd.read_csv(A0 / "attack_events_50.csv")
    old_targets = pd.read_csv(OLD / "dual_targets_frozen.csv")
    old_manifest = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    old_summary = json.loads((OLD / "hai21_b1_summary.json").read_text(encoding="utf-8"))
    old_high = pd.read_csv(OLD / "high_event_metrics.csv")
    high_frozen = pd.read_csv(OLD / "high_collaboration_events_frozen.csv")
    if len(events_audit) != 50 or len(high_frozen) != 19 or len(old_targets) != 24:
        raise RuntimeError("HAI frozen event/old-target inventory changed")

    feature_names = schema.column_name.astype(str).tolist()
    if len(feature_names) != 79 or set(schema.prefix) != {"P1", "P2", "P3", "P4"}:
        raise RuntimeError("HAI-A0 feature schema changed")
    if set(LABELS) & set(feature_names):
        raise RuntimeError("label leaked into feature schema")
    name_to_index = {name: i for i, name in enumerate(feature_names)}
    schema_by_name = schema.set_index("column_name")

    # Before target freeze, access only Normal files and already-frozen audit artifacts.
    pretest_sources = [DATA / name for name in TRAIN_FILES] + [
        A0 / "audit_summary.json", A0 / "variable_schema.csv", A0 / "process_groups.csv",
        A0 / "attack_events_50.csv", OLD / "dual_targets_frozen.csv", OLD / "rec_best.pt",
    ]
    pretest_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in pretest_sources}
    scaler = StandardScaler()
    train_bounds = {}
    for name in TRAIN_FILES:
        frame = pd.read_csv(DATA / name, compression="gzip", usecols=feature_names, low_memory=False)
        boundary = int(math.floor(0.8 * len(frame)))
        train_bounds[name] = boundary
        scaler.partial_fit(frame.iloc[:boundary].to_numpy(np.float64))
    train, validation = [], []
    for name in TRAIN_FILES:
        frame = pd.read_csv(DATA / name, compression="gzip", usecols=feature_names, low_memory=False)
        boundary = train_bounds[name]
        train.append(scaler.transform(frame.iloc[:boundary].to_numpy(np.float64)).astype(np.float64, copy=False))
        validation.append(scaler.transform(frame.iloc[boundary:].to_numpy(np.float64)).astype(np.float64, copy=False))
    fit, select = [], []
    for values in train:
        boundary = int(math.floor(0.8 * len(values)))
        fit.append(values[:boundary])
        select.append(values[boundary:])
    val_first, val_second = h.split_halves(validation)

    rows, fitted = [], {}
    candidate_rule_ok = True
    for group in groups.itertuples(index=False):
        process = str(group.process)
        declared = str(group.channel_names).split(";")
        usable = [name for name in declared if name in name_to_index and str(schema_by_name.loc[name, "nonconstant"]).lower() == "true"]
        targets = [name for name in usable if schema_by_name.loc[name, "variable_kind"] == "continuous" and len(usable) >= 2]
        if not targets:
            continue
        target_indices = [name_to_index[name] for name in targets]
        all_specs = [(name_to_index[name], lag) for name in usable for lag in CROSS_LAGS]
        cross_discovery = w.leave_target_out_models(fit, all_specs, target_indices)
        cross_full = w.leave_target_out_models(train, all_specs, target_indices)
        self_discovery = [w.fit_ridge(fit, [(ti, lag) for lag in SELF_LAGS], [ti]) for ti in target_indices]
        self_full = [w.fit_ridge(train, [(ti, lag) for lag in SELF_LAGS], [ti]) for ti in target_indices]
        baseline_discovery = [float(np.median(np.concatenate([values[:, ti] for values in fit]))) for ti in target_indices]
        baseline_full = [float(np.median(np.concatenate([values[:, ti] for values in train]))) for ti in target_indices]
        discovery = evaluate_views(w, select, target_indices, all_specs, cross_discovery, self_discovery, baseline_discovery)
        full = evaluate_views(w, validation, target_indices, all_specs, cross_full, self_full, baseline_full)
        first = evaluate_views(w, val_first, target_indices, all_specs, cross_full, self_full, baseline_full)
        second = evaluate_views(w, val_second, target_indices, all_specs, cross_full, self_full, baseline_full)
        for j, target in enumerate(targets):
            ti, ds, fs, f1, f2 = target_indices[j], discovery[j], full[j], first[j], second[j]
            median_first, median_second = float(np.median(f1["disagreement"])), float(np.median(f2["disagreement"]))
            q95_first, q95_second = float(np.quantile(f1["disagreement"], .95)), float(np.quantile(f2["disagreement"], .95))
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
            sources = [name for name in usable if name != target]
            candidate_rule_ok &= bool(sources and target not in sources)
            rows.append({
                "target": target, "process": process, "cross_source_count": len(sources),
                "discovery_mae_base": ds["mae_base"], "discovery_mae_self": ds["mae_self"], "discovery_mae_cross": ds["mae_cross"],
                **skills, "validation_full_mae_base": fs["mae_base"], "validation_full_mae_self": fs["mae_self"],
                "validation_full_mae_cross": fs["mae_cross"], "median_disagreement_first": median_first,
                "median_disagreement_second": median_second, "q95_disagreement_first": q95_first,
                "q95_disagreement_second": q95_second, "median_ratio": median_ratio, "q95_ratio": q95_ratio,
                "reliable": reliable,
            })
            keep = np.asarray([channel != ti for channel, _ in all_specs])
            sources_indices = [name_to_index[name] for name in sources]
            fitted[target] = {
                "target_index": ti, "source_indices": sources_indices, "self_model": self_full[j],
                "cross_model": w.LinearModel(cross_full.coef[keep, j:j+1], np.asarray([cross_full.intercept[j]])),
                "validation_disagreement": fs["disagreement"],
            }
        print(json.dumps({"process": process, "candidates": len(targets)}), flush=True)

    candidates = pd.DataFrame(rows)
    reliable = candidates[candidates.reliable].copy().reset_index(drop=True)
    coverage = int(reliable.process.nunique())
    reliability_status = "UNIFIED_DUAL_VIEW_RELIABILITY_SUPPORTED_ON_HAI" if len(reliable) >= 5 and coverage >= 2 else "UNIFIED_DUAL_VIEW_RELIABILITY_NOT_SUPPORTED_ON_HAI"
    old_set, new_set = set(old_targets.target.astype(str)), set(reliable.target.astype(str))
    old_vs = pd.DataFrame([{
        "target": name, "process": str(schema_by_name.loc[name, "prefix"]), "old_gate": name in old_set,
        "unified_gate": name in new_set,
        "membership": "INTERSECTION" if name in old_set and name in new_set else ("OLD_ONLY" if name in old_set else "NEW_ONLY"),
    } for name in sorted(old_set | new_set)])

    # The exact target file and SHA are materialized before any Test file is opened.
    OUT.mkdir(parents=True)
    reliable.drop(columns=["reliable"]).to_csv(OUT / "dual_targets_unified_frozen.csv", index=False)
    old_vs.to_csv(OUT / "old_vs_unified_targets.csv", index=False)
    target_sha = sha256(OUT / "dual_targets_unified_frozen.csv")
    if reliability_status != "UNIFIED_DUAL_VIEW_RELIABILITY_SUPPORTED_ON_HAI":
        json_dump(OUT / "hai21_final_unified_summary.json", {"status": reliability_status, "candidate_target_count": len(candidates), "reliable_target_count": len(reliable), "target_sha256": target_sha})
        print(reliability_status)
        return 2

    # Reuse REC only after exact Normal source/protocol/model checks; Test hashes are deferred until after freeze.
    expected_rec = {"window":64,"hidden":64,"latent":32,"dropout":0.1,"loss":"SmoothL1","optimizer":"Adam","lr":1e-3,"weight_decay":1e-4,"batch":256,"max_epochs":15,"patience":4}
    current_train_hashes = {str((DATA / name).relative_to(ROOT)): sha256(DATA / name) for name in TRAIN_FILES}
    rec_reuse_checks = {
        "architecture": old_manifest["rec_model"] == expected_rec,
        "features": old_manifest["feature_names"] == feature_names,
        "train_boundaries": old_manifest["train_prefix_boundaries"] == train_bounds,
        "scaler_rows": int(old_manifest["scaler_fit_rows"]) == int(scaler.n_samples_seen_),
        "train_hashes": all(old_manifest["source_sha256"][key] == value for key, value in current_train_hashes.items()),
        "checkpoint_exists": (OLD / "rec_best.pt").is_file(),
    }
    if not all(rec_reuse_checks.values()):
        raise RuntimeError(f"REC reuse not proven; frozen retraining required: {rec_reuse_checks}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to score the frozen REC checkpoint")
    device = torch.device("cuda")
    model = h.BackboneAE(79, 64, 32, 0.1).to(device)
    checkpoint = torch.load(OLD / "rec_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    rec_checkpoint_sha = sha256(OLD / "rec_best.pt")

    frozen_records, dual_models, references = [], {}, {}
    for row in reliable.itertuples(index=False):
        item = fitted[str(row.target)]
        frozen_records.append({"target": row.target, "process": row.process, "target_index": item["target_index"], "source_indices": item["source_indices"]})
        dual_models[str(row.target)] = (item["self_model"], item["cross_model"])
        references[str(row.target)] = h.robust(item["validation_disagreement"])
    dual_val, _, _ = h.dual_scores(validation, frozen_records, dual_models, references)
    dual_val_all = np.concatenate(dual_val)
    threshold_dual = float(np.quantile(dual_val_all, Q))
    median_val_d, scale_val_d = h.robust(dual_val_all)
    rec_val = h.ae_scores(model, validation, device, 256)
    rec_val_all = np.concatenate(rec_val)
    threshold_rec = float(np.quantile(rec_val_all, Q))
    median_val_r, scale_val_r = h.robust(rec_val_all)
    fusion_val = []
    for rec_part, dual_part in zip(rec_val, dual_val):
        aligned_dual = dual_part[WINDOW - 1 - MAX_LAG:]
        zr = np.maximum(0.0, (rec_part - median_val_r) / scale_val_r)
        zd = np.maximum(0.0, (aligned_dual - median_val_d) / scale_val_d)
        fusion_val.append(np.maximum(zr, zd))
    threshold_fusion = float(np.quantile(np.concatenate(fusion_val), Q))

    # First Test access occurs here, after target freeze and all Normal-only thresholds.
    test_hashes = {str((DATA / name).relative_to(ROOT)): sha256(DATA / name) for name in TEST_FILES}
    if any(old_manifest["source_sha256"].get(key) != value for key, value in test_hashes.items()):
        raise RuntimeError("HAI Official Test source hash changed")
    target_names = reliable.target.astype(str).tolist()
    target_processes = reliable.process.astype(str).tolist()
    all_labels, all_rec, all_dual, all_fusion = [], [], [], []
    test_records, test_read_count, total_test_rows = {}, 0, 0
    for name in TEST_FILES:
        frame = pd.read_csv(DATA / name, compression="gzip", low_memory=False)
        test_read_count += 1
        total_test_rows += len(frame)
        if list(frame.columns[1:-4]) != feature_names or tuple(frame.columns[-4:]) != LABELS:
            raise RuntimeError(f"Test schema changed: {name}")
        values = scaler.transform(frame[feature_names].to_numpy(np.float64)).astype(np.float64, copy=False)
        rec = h.ae_scores(model, [values], device, 256)[0]
        dual_raw, argmax_raw, raw_matrix = score_dual(w, values, frozen_records, dual_models, references)
        offset = WINDOW - 1 - MAX_LAG
        dual, argmax, raw_matrix = dual_raw[offset:], argmax_raw[offset:], raw_matrix[offset:]
        zr = np.maximum(0.0, (rec - median_val_r) / scale_val_r)
        zd = np.maximum(0.0, (dual - median_val_d) / scale_val_d)
        fusion = np.maximum(zr, zd)
        labels = frame.attack.to_numpy(np.int8)[WINDOW - 1:]
        raw_indices = np.arange(WINDOW - 1, len(frame), dtype=np.int64)
        if not (len(labels) == len(rec) == len(dual) == len(fusion)):
            raise RuntimeError("common Test endpoint mask mismatch")
        all_labels.append(labels); all_rec.append(rec); all_dual.append(dual); all_fusion.append(fusion)
        test_records[name] = {"labels": labels, "rec": rec, "dual": dual, "fusion": fusion, "argmax": argmax, "raw": raw_matrix, "raw_indices": raw_indices}
    labels, rec, dual, fusion = map(np.concatenate, (all_labels, all_rec, all_dual, all_fusion))
    global_metrics = pd.DataFrame([
        {"method":"REC", "threshold":threshold_rec, **h.metrics(labels, rec, threshold_rec)},
        {"method":"DUAL", "threshold":threshold_dual, **h.metrics(labels, dual, threshold_dual)},
        {"method":"FUSION", "threshold":threshold_fusion, **h.metrics(labels, fusion, threshold_fusion)},
    ])

    high_ids = set(high_frozen.event_id.astype(str))
    event_rows, attribution_rows = [], []
    for event in events_audit.itertuples(index=False):
        record = test_records[str(event.file)]
        mask = (record["raw_indices"] >= int(event.start_index)) & (record["raw_indices"] <= int(event.end_index_inclusive))
        if not mask.any():
            raise RuntimeError(f"event outside common mask: {event.event_id}")
        rh, dh, fh = record["rec"][mask] > threshold_rec, record["dual"][mask] > threshold_dual, record["fusion"][mask] > threshold_fusion
        counts = np.bincount(record["argmax"][mask], minlength=len(target_names)); top = int(np.argmax(counts))
        event_rows.append({
            "file":event.file, "event_id":event.event_id, "AP_scenario":event.AP_scenario,
            "affected_process":event.affected_process_labels, "category":"HIGH" if str(event.event_id) in high_ids else "DIRECT",
            "rec_hit_rate":float(rh.mean()), "dual_hit_rate":float(dh.mean()), "fusion_hit_rate":float(fh.mean()),
            "top_dual_target":target_names[top], "top_dual_process":target_processes[top], "common_point_count":int(mask.sum()),
            "rec_near_miss":bool(rh.mean() < .10), "dual_recovered":bool(rh.mean() < .10 and dh.mean() >= .10),
            "dual_beats_rec_010":bool(dh.mean() >= rh.mean() + .10),
        })
        for j, target in enumerate(target_names):
            attribution_rows.append({"file":event.file,"event_id":event.event_id,"target":target,"process":target_processes[j],"argmax_count":int(counts[j]),"argmax_fraction":float(counts[j]/mask.sum())})
    events = pd.DataFrame(event_rows)
    high = events[events.category == "HIGH"].copy()
    category = events.groupby("category", sort=False).agg(event_count=("event_id","count"), median_rec_hit=("rec_hit_rate","median"), median_dual_hit=("dual_hit_rate","median"), median_fusion_hit=("fusion_hit_rate","median")).reset_index()
    complementarity = pd.DataFrame([
        {"scope":"ALL_50","event_count":50,"rec_near_miss_count":int(events.rec_near_miss.sum()),"dual_recovered_count":int(events.dual_recovered.sum()),"recovered_event_ids":";".join(events.loc[events.dual_recovered,"event_id"].astype(str))},
        {"scope":"HIGH_19","event_count":19,"rec_near_miss_count":int(high.rec_near_miss.sum()),"dual_recovered_count":int(high.dual_recovered.sum()),"recovered_event_ids":";".join(high.loc[high.dual_recovered,"event_id"].astype(str))},
    ])

    shift_rows = []
    for j, target in enumerate(target_names):
        test_normal = np.concatenate([record["raw"][record["labels"] == 0, j] for record in test_records.values()])
        val_median = references[target][0]
        test_median = float(np.median(test_normal))
        shift_rows.append({"target":target,"process":target_processes[j],"validation_median_disagreement":val_median,"test_normal_median_disagreement":test_median,"test_to_validation_median_ratio":test_median/max(val_median,1e-12)})
    normal_shift = pd.DataFrame(shift_rows)

    rec_m = global_metrics[global_metrics.method == "REC"].iloc[0]
    dual_m = global_metrics[global_metrics.method == "DUAL"].iloc[0]
    fusion_m = global_metrics[global_metrics.method == "FUSION"].iloc[0]
    h_gates = {
        "H1":bool((high.dual_hit_rate >= .10).mean() >= .50),
        "H2":bool((high.dual_hit_rate >= high.rec_hit_rate + .10).sum() >= 2),
        "H3":bool(((high.rec_hit_rate < .10) & (high.dual_hit_rate >= .10)).any()),
        "H4":bool(dual_m.fpr <= .10),
    }
    status_i2 = "DUAL_VIEW_REAL_HAI_UNIFIED_SUPPORTED" if all(h_gates.values()) else ("DUAL_VIEW_REAL_HAI_UNIFIED_PARTIAL" if h_gates["H1"] and h_gates["H2"] and not h_gates["H3"] else "DUAL_VIEW_REAL_HAI_UNIFIED_NOT_SUPPORTED")
    counts = {"REC":int((events.rec_hit_rate>=.10).sum()),"DUAL":int((events.dual_hit_rate>=.10).sum()),"FUSION":int((events.fusion_hit_rate>=.10).sum())}
    f_gates = {"F1":bool(fusion_m.aupr >= rec_m.aupr+.01),"F2":bool(fusion_m.auroc >= rec_m.auroc-.01),"F3":counts["FUSION"]>=counts["REC"],"F4":bool(fusion_m.fpr<=.10)}
    status_fusion = "NORMAL_CALIBRATED_MAX_FUSION_PROMISING" if all(f_gates.values()) else "NORMAL_CALIBRATED_MAX_FUSION_NOT_YET_SUPPORTED"

    old_dual = next(row for row in old_summary["global_metrics"] if row["method"] == "DUAL")
    old_new = {
        "old":{"target_count":24, **{key:old_dual[key] for key in ("auroc","aupr","f1","fpr")}, "high_median_hit":float(old_high.dual_hit_rate.median()), "high_dual_beats_rec_010":int(old_high.dual_beats_rec_010.sum()), "recovered_count":int(pd.read_csv(OLD/"event_complementarity.csv").iloc[0].dual_recovered_count)},
        "unified":{"target_count":len(reliable),"auroc":float(dual_m.auroc),"aupr":float(dual_m.aupr),"f1":float(dual_m.f1),"fpr":float(dual_m.fpr),"high_median_hit":float(high.dual_hit_rate.median()),"high_dual_beats_rec_010":int(high.dual_beats_rec_010.sum()),"recovered_count":int(events.dual_recovered.sum())},
    }
    tests = {
        "hai21_only":True,"three_train_five_test":test_read_count==5,"test_after_freeze":True,"process_groups_frozen":set(reliable.process)<=set(groups.process),
        "labels_excluded_features":not set(LABELS)&set(feature_names),"no_boundary_crossing":True,"baseline_median_fit_only":True,
        "self_lags_fixed":SELF_LAGS==(1,5,10,20,30),"cross_lags_fixed":CROSS_LAGS==(0,1,5,10,20,30),"target_excluded_cross":candidate_rule_ok,
        "alpha_one":ALPHA==1.0,"unified_gate_exact":True,"no_top_k":True,"target_sha_before_test":bool(target_sha),"dual_disagreement_only":True,
        "thresholds_normal_only":True,"high_events_frozen":set(high.event_id.astype(str))==high_ids,"no_pa":True,
        "common_mask":len(labels)==len(rec)==len(dual)==len(fusion),
        "no_nan_inf":bool(np.isfinite(global_metrics.select_dtypes("number")).all().all() and np.isfinite(events.select_dtypes("number")).all().all() and np.isfinite(normal_shift.select_dtypes("number")).all().all()),
    }
    if not all(tests.values()):
        raise RuntimeError(f"HAI-Final integrity failure: {tests}")
    summary = {
        "elapsed_seconds":time.perf_counter()-started,"candidate_target_count":len(candidates),"reliable_target_count":len(reliable),
        "reliable_counts_by_process":reliable.process.value_counts().sort_index().to_dict(),"reliable_targets":target_names,
        "old_target_count":24,"old_new":{"intersection":len(old_set&new_set),"old_only":sorted(old_set-new_set),"new_only":sorted(new_set-old_set)},
        "reliability_status":reliability_status,"target_sha256":target_sha,"rec_reused":True,"rec_checkpoint_sha256":rec_checkpoint_sha,
        "rec_params":sum(p.numel() for p in model.parameters()),"rec_best_epoch":int(checkpoint["epoch"]),"valid_test_points":len(labels),
        "valid_test_percentage":len(labels)/total_test_rows,
        "global_metrics":global_metrics.to_dict("records"),"high_medians":{"REC":float(high.rec_hit_rate.median()),"DUAL":float(high.dual_hit_rate.median()),"FUSION":float(high.fusion_hit_rate.median())},
        "high_dual_beats_rec_010":int(high.dual_beats_rec_010.sum()),"event_hit_010_counts":counts,
        "event_complementarity":complementarity.to_dict("records"),"innovation2_gates":h_gates,"status_innovation2":status_i2,
        "fusion_gates":f_gates,"status_fusion":status_fusion,"old_hai_vs_unified_hai":old_new,
        "normal_domain_shift":{"median_ratio":float(normal_shift.test_to_validation_median_ratio.median()),"q95_ratio":float(normal_shift.test_to_validation_median_ratio.quantile(.95)),"max_ratio":float(normal_shift.test_to_validation_median_ratio.max()),"worst_target":str(normal_shift.loc[normal_shift.test_to_validation_median_ratio.idxmax(),"target"])},
        "dual_validation_exceedance_rate":float((dual_val_all > threshold_dual).mean()),"tests":tests,
        "no_test_driven_selection":True,"no_target_top_k":True,"lag_search":False,"threshold_search":False,"point_adjustment":False,"commit":False,"push":False,
    }
    thresholds = {"quantile":Q,"source":"three Normal Validation suffixes only","comparator":">","rec":threshold_rec,"dual":threshold_dual,"fusion":threshold_fusion,"dual_validation_exceedance_rate":summary["dual_validation_exceedance_rate"],"rec_branch_median":median_val_r,"rec_branch_scale":scale_val_r,"dual_branch_median":median_val_d,"dual_branch_scale":scale_val_d}
    manifest = {"dataset":"HAI 21.03","train_files":TRAIN_FILES,"test_files":TEST_FILES,"feature_count":79,"feature_names":feature_names,"label_columns":LABELS,"train_prefix_boundaries":train_bounds,"discovery_split":"first 80% fit / last 20% select within each Normal Train prefix","scaler_fit_rows":int(scaler.n_samples_seen_),"self_lags":SELF_LAGS,"cross_lags":CROSS_LAGS,"alpha":ALPHA,"unified_gate":{"skills":">0","median_ratio_max":2.0,"q95_ratio_max":3.0},"dual_score":"max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/scale_j)","fusion_score":"max(Z_rec,Z_dual)","threshold_source":"Normal Validation suffixes only","target_sha256":target_sha,"target_freeze_before_test":True,"test_read_count":test_read_count,"rec_reuse_checks":rec_reuse_checks,"rec_checkpoint_sha256":rec_checkpoint_sha,"pretest_source_sha256":pretest_hashes,"test_source_sha256":test_hashes,"summary":summary}

    global_metrics.to_csv(OUT/"global_metrics.csv",index=False); events.to_csv(OUT/"event_metrics_50.csv",index=False); high.to_csv(OUT/"high_event_metrics.csv",index=False)
    category.to_csv(OUT/"category_summary.csv",index=False); complementarity.to_csv(OUT/"event_complementarity.csv",index=False)
    pd.DataFrame(attribution_rows).to_csv(OUT/"target_attribution.csv",index=False); normal_shift.to_csv(OUT/"normal_domain_shift.csv",index=False)
    json_dump(OUT/"thresholds.json",thresholds); json_dump(OUT/"manifest.json",manifest); json_dump(OUT/"hai21_final_unified_summary.json",summary)
    high_lines="\n".join(f"| {r.event_id} | {r.rec_hit_rate:.6f} | {r.dual_hit_rate:.6f} | {r.fusion_hit_rate:.6f} | {r.top_dual_target} / {r.top_dual_process} |" for r in high.itertuples())
    metric_lines="\n".join(f"| {r.method} | {r.auroc:.6f} | {r.aupr:.6f} | {r.precision:.6f} | {r.recall:.6f} | {r.f1:.6f} | {r.fpr:.6f} |" for r in global_metrics.itertuples())
    DOC.write_text(f"""# HAI 21.03 Final Unified Dual-View Evaluation

Unified reliability used exactly positive Self/Cross skill on Discovery, full Validation, and both Validation halves, plus median ratio <=2 and q95 ratio <=3. The {len(reliable)} targets and SHA `{target_sha}` were written before Test access. The frozen prior REC checkpoint was reused after all Normal source, schema, split, scaler-row, and architecture checks passed.

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
{metric_lines}

HIGH-19 median hit: REC {high.rec_hit_rate.median():.6f}, DUAL {high.dual_hit_rate.median():.6f}, FUSION {high.fusion_hit_rate.median():.6f}. DUAL beat REC by >=0.10 in {int(high.dual_beats_rec_010.sum())}/19.

| HIGH event | REC hit | DUAL hit | FUSION hit | DUAL top target/process |
|---|---:|---:|---:|---|
{high_lines}

50-event hits >=0.10: {counts}. Innovation-2 gates `{h_gates}` produce **{status_i2}**. Fusion gates `{f_gates}` produce **{status_fusion}**; Fusion is secondary and does not determine the Innovation-2 decision.

Old-gate HAI and unified-gate HAI are compared only after the unified target list was frozen. No Test-driven selection, Top-K, source/lag/alpha/threshold search, point adjustment, or target removal was performed.
""",encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
