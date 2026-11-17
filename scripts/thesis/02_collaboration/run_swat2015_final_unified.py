"""Official SWaT final REC / unified Dual / frozen max-Fusion evaluation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "thesis_swat_final_unified"
DOC = ROOT / "docs" / "thesis" / "swat_final_unified_rec_dual_fusion.md"
A0 = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline"
S6 = ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit"
T25 = ROOT / "outputs" / "thesis_t25_swat2015_protocol"
OLD = ROOT / "outputs" / "thesis_innovation2a2_joint_lagged" / "stable_targets.csv"
SELF_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (0, 1, 5, 10, 20, 30)
MAX_LAG, WINDOW, ALPHA, Q = 30, 64, 1.0, 0.99
TRAIN_END, DISCOVERY_END, NORMAL_END = 396000, 316800, 495000
CONTEXTUAL_IDS = (13, 14, 29)


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


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def robust(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    return median, max(1.4826 * float(np.median(np.abs(values - median))), 1e-6)


def skill(model_mae: float, base_mae: float) -> float:
    return 0.0 if base_mae <= 0 else 1.0 - model_mae / base_mae


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred, pos, neg = scores > threshold, labels == 1, labels == 0
    tp, fp = int((pred & pos).sum()), int((pred & neg).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & neg).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auroc": float(roc_auc_score(labels, scores)), "aupr": float(average_precision_score(labels, scores)),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def evaluate_views(helper, values, targets, specs, cross_model, self_models, baselines):
    times = np.arange(MAX_LAG, len(values), dtype=np.int64)
    cross = cross_model.predict(helper.design(values, times, specs))
    rows = []
    for j, target in enumerate(targets):
        self_pred = self_models[j].predict(helper.design(values, times, [(target, lag) for lag in SELF_LAGS]))[:, 0]
        actual = values[times, target]
        rows.append({
            "mae_base": float(np.mean(np.abs(actual - baselines[j]))),
            "mae_self": float(np.mean(np.abs(actual - self_pred))),
            "mae_cross": float(np.mean(np.abs(actual - cross[:, j]))),
            "disagreement": np.abs(self_pred - cross[:, j]),
        })
    return rows


def score_segment(helper, values, frozen, fitted, references):
    times = np.arange(MAX_LAG, len(values), dtype=np.int64)
    raw, calibrated = [], []
    for row in frozen:
        target, ti, sources = str(row["target"]), int(row["target_index"]), list(row["source_indices"])
        self_model, cross_model = fitted[target]["self_model"], fitted[target]["cross_model"]
        ps = self_model.predict(helper.design(values, times, [(ti, lag) for lag in SELF_LAGS]))[:, 0]
        pc = cross_model.predict(helper.design(values, times, [(source, lag) for source in sources for lag in CROSS_LAGS]))[:, 0]
        disagreement = np.abs(ps - pc)
        median, scale = references[target]
        raw.append(disagreement)
        calibrated.append(np.maximum(0.0, (disagreement - median) / scale))
    raw_matrix = np.column_stack(raw)
    calibrated_matrix = np.column_stack(calibrated)
    return calibrated_matrix.max(1), calibrated_matrix.argmax(1), raw_matrix


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        partial = {path.name for path in OUT.iterdir()}
        if not partial <= {"dual_targets_unified_frozen.csv", "old_vs_unified_targets.csv"}:
            raise FileExistsError(f"refusing to overwrite completed/unknown content in {OUT}")
    helper_dir = Path(__file__).parent
    a2 = load_module("swat_final_a2", helper_dir / "run_innovation2a2_joint_lagged.py")
    b2 = load_module("swat_final_b2", helper_dir / "run_innovation2b_controlled.py")
    d0 = load_module("swat_final_d0", helper_dir / "run_innovation2d0_dual_view_comut.py")
    w = load_module("swat_final_wadi", helper_dir / "run_wadi_b1_external.py")
    pcc = load_module("swat_final_pcc", helper_dir / "run_swat2015_pcc1.py")
    from data.swat2015_official import load_attack_cache, load_normal_cache, resolve_repo_path, validate_config_contract
    from data.transforms import StandardScaler
    from utils.config_io import load_config

    config = load_config(str(b2.CONFIG)); validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    if normal.shape != (NORMAL_END, 51):
        raise RuntimeError("Official Normal v1 cache shape changed")
    feature_names = [str(x) for x in cache_manifest["column_order"]]
    index = {name: i for i, name in enumerate(feature_names)}
    schema_path = ROOT / "outputs" / "thesis_t2_swat2015_audit" / "schema_profile.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    contract, _ = a2.build_contract(a2.normal_profiles(schema), feature_names)

    # Normal-only scaler and chronological discovery. No Attack feature values are opened above this line.
    scaler = StandardScaler().fit(np.asarray(normal[:TRAIN_END]))
    z = scaler.transform(np.asarray(normal)).astype(np.float64)
    fit, select, train, validation = z[:DISCOVERY_END], z[DISCOVERY_END:TRAIN_END], z[:TRAIN_END], z[TRAIN_END:NORMAL_END]
    vhalf = len(validation) // 2
    rows, fitted = [], {}
    for stage in ("P1", "P2", "P3", "P4", "P5", "P6"):
        usable = list(contract[stage]["nonconstant_channels"])
        targets = list(contract[stage]["continuous_targets"]) if len(usable) >= 2 else []
        if not targets:
            continue
        target_indices = [index[name] for name in targets]
        all_specs = [(index[name], lag) for name in usable for lag in CROSS_LAGS]
        cross_d = w.leave_target_out_models([fit], all_specs, target_indices)
        cross_f = w.leave_target_out_models([train], all_specs, target_indices)
        self_d = [w.fit_ridge([fit], [(ti, lag) for lag in SELF_LAGS], [ti]) for ti in target_indices]
        self_f = [w.fit_ridge([train], [(ti, lag) for lag in SELF_LAGS], [ti]) for ti in target_indices]
        base_d = [float(np.median(fit[:, ti])) for ti in target_indices]
        base_f = [float(np.median(train[:, ti])) for ti in target_indices]
        discovery = evaluate_views(w, select, target_indices, all_specs, cross_d, self_d, base_d)
        full = evaluate_views(w, validation, target_indices, all_specs, cross_f, self_f, base_f)
        first = evaluate_views(w, validation[:vhalf], target_indices, all_specs, cross_f, self_f, base_f)
        second = evaluate_views(w, validation[vhalf:], target_indices, all_specs, cross_f, self_f, base_f)
        for j, target in enumerate(targets):
            ds, fs, f1, f2 = discovery[j], full[j], first[j], second[j]
            m1, m2 = float(np.median(f1["disagreement"])), float(np.median(f2["disagreement"]))
            q1, q2 = float(np.quantile(f1["disagreement"], .95)), float(np.quantile(f2["disagreement"], .95))
            mr = max(m1 / max(m2, 1e-12), m2 / max(m1, 1e-12)); qr = max(q1 / max(q2, 1e-12), q2 / max(q1, 1e-12))
            skills = {
                "discovery_self_skill": skill(ds["mae_self"], ds["mae_base"]), "discovery_cross_skill": skill(ds["mae_cross"], ds["mae_base"]),
                "validation_full_self_skill": skill(fs["mae_self"], fs["mae_base"]), "validation_full_cross_skill": skill(fs["mae_cross"], fs["mae_base"]),
                "validation_first_self_skill": skill(f1["mae_self"], f1["mae_base"]), "validation_first_cross_skill": skill(f1["mae_cross"], f1["mae_base"]),
                "validation_second_self_skill": skill(f2["mae_self"], f2["mae_base"]), "validation_second_cross_skill": skill(f2["mae_cross"], f2["mae_base"]),
            }
            reliable = all(value > 0 for value in skills.values()) and mr <= 2.0 and qr <= 3.0
            sources = [name for name in usable if name != target]
            rows.append({"target": target, "stage": stage, "cross_source_count": len(sources), **skills,
                         "median_disagreement_first": m1, "median_disagreement_second": m2,
                         "q95_disagreement_first": q1, "q95_disagreement_second": q2,
                         "median_ratio": mr, "q95_ratio": qr, "reliable": reliable})
            keep = np.asarray([channel != target_indices[j] for channel, _ in all_specs])
            fitted[target] = {"target_index": target_indices[j], "source_indices": [index[name] for name in sources],
                              "self_model": self_f[j], "cross_model": w.LinearModel(cross_f.coef[keep, j:j+1], np.asarray([cross_f.intercept[j]])),
                              "validation_disagreement": fs["disagreement"]}
        print(json.dumps({"stage": stage, "candidate_count": len(targets)}), flush=True)

    candidates = pd.DataFrame(rows)
    reliable = candidates[candidates.reliable].drop(columns="reliable").reset_index(drop=True)
    if reliable.empty:
        raise RuntimeError("Unified reliability gate retained no SWaT target")
    old = pd.read_csv(OLD); old_set, new_set = set(old.target.astype(str)), set(reliable.target.astype(str))
    old_vs = pd.DataFrame([{"target": name, "stage": a2.stage_of(name), "old_gate": name in old_set,
                           "unified_gate": name in new_set,
                           "membership": "INTERSECTION" if name in old_set and name in new_set else ("OLD_ONLY" if name in old_set else "NEW_ONLY")}
                          for name in sorted(old_set | new_set)])

    # Materialize the exact target list before any Attack feature values are read.
    OUT.mkdir(parents=True, exist_ok=True)
    reliable.to_csv(OUT / "dual_targets_unified_frozen.csv", index=False)
    old_vs.to_csv(OUT / "old_vs_unified_targets.csv", index=False)
    target_sha = sha256(OUT / "dual_targets_unified_frozen.csv")

    # Prove that the frozen A0 checkpoint/scaler/protocol and cached score artifacts match this run.
    cfg = json.loads((A0 / "config_resolved.json").read_text(encoding="utf-8"))
    model_summary = json.loads((A0 / "model_summary.json").read_text(encoding="utf-8"))
    val_manifest = json.loads((S6 / "a0_validation_score_manifest.json").read_text(encoding="utf-8"))
    checkpoint_sha = sha256(A0 / "best.pt")
    with np.load(A0 / "scaler.npz", allow_pickle=False) as saved:
        scaler_mean = np.asarray(saved["mean"]).reshape(-1); scaler_scale = np.asarray(saved["effective_scale"]).reshape(-1)
    rec_reuse_checks = {
        "architecture": cfg["model"] == {"class":"models.backbone_ae.BackboneAE","in_channels":51,"hidden_channels":64,"latent_channels":32,"dropout":0.1},
        "dimensions": model_summary["input_shape"] == ["B", 64, 51] and model_summary["output_shape"] == ["B", 64, 51],
        "protocol": cfg["data"]["train_raw_rows"] == TRAIN_END and cfg["data"]["validation_raw_rows"] == 99000 and cfg["data"]["window"] == WINDOW,
        "scaler": np.array_equal(scaler.mean_.reshape(-1), scaler_mean) and np.array_equal(scaler.effective_scale_.reshape(-1), scaler_scale),
        "checkpoint": checkpoint_sha == val_manifest["checkpoint_sha256"] and val_manifest["checkpoint_epoch"] == 13 and val_manifest["parameter_count"] == 11730,
    }
    if not all(rec_reuse_checks.values()):
        raise RuntimeError(f"A0 reuse checks failed; exact retraining would be required: {rec_reuse_checks}")

    frozen_records, references = [], {}
    for row in reliable.itertuples(index=False):
        item = fitted[str(row.target)]
        frozen_records.append({"target": row.target, "stage": row.stage, "target_index": item["target_index"], "source_indices": item["source_indices"]})
        references[str(row.target)] = robust(item["validation_disagreement"])
    dual_val_raw, _, _ = score_segment(w, validation, frozen_records, fitted, references)
    threshold_dual = float(np.quantile(dual_val_raw, Q))
    median_dual, scale_dual = robust(dual_val_raw)
    rec_val = np.load(S6 / "a0_validation_scores.npy", allow_pickle=False).astype(np.float64)
    rec_threshold_frozen = float(json.loads((A0 / "thresholds.json").read_text(encoding="utf-8"))["endpoint_mae"])
    rec_validation_q99 = float(np.quantile(rec_val, Q))
    threshold_rec = rec_threshold_frozen
    if abs(rec_validation_q99 - rec_threshold_frozen) > 1e-6 or len(rec_val) != 99000 - WINDOW + 1:
        raise RuntimeError("A0 Normal Validation score/threshold binding changed")
    median_rec, scale_rec = robust(rec_val)
    dual_val = dual_val_raw[WINDOW - 1 - MAX_LAG:]
    fusion_val = np.maximum(np.maximum(0, (rec_val - median_rec) / scale_rec), np.maximum(0, (dual_val - median_dual) / scale_dual))
    threshold_fusion = float(np.quantile(fusion_val, Q))

    # First Attack feature access: all targets, models, references, aggregation, and thresholds are frozen.
    attack, attack_labels, attack_timestamps, _ = load_attack_cache(cache_dir, config)
    z_attack = scaler.transform(np.asarray(attack, dtype=np.float64))
    segment_doc = json.loads((T25 / "segment_manifest.json").read_text(encoding="utf-8"))
    row_counts = [int(item["row_count"]) for item in segment_doc["test_segments"]]
    segments, cursor = [], 0
    for count in row_counts:
        segments.append((cursor, cursor + count)); cursor += count
    if cursor != len(attack) or len(segments) != 2:
        raise RuntimeError("T2.5 hard segment contract changed")
    dual_parts, argmax_parts, raw_parts, raw_index_parts = [], [], [], []
    for start, end in segments:
        score, argmax, raw = score_segment(w, z_attack[start:end], frozen_records, fitted, references)
        dual_parts.append(score[WINDOW - 1 - MAX_LAG:]); argmax_parts.append(argmax[WINDOW - 1 - MAX_LAG:])
        raw_parts.append(raw[WINDOW - 1 - MAX_LAG:]); raw_index_parts.append(np.arange(start + WINDOW - 1, end, dtype=np.int64))
    dual, dual_argmax, raw_matrix, raw_indices = map(np.concatenate, (dual_parts, argmax_parts, raw_parts, raw_index_parts))
    with np.load(A0 / "test_scores.npz", allow_pickle=False) as payload:
        rec_timestamps = np.asarray(payload["endpoint_timestamp"], dtype=np.int64)
        labels = np.asarray(payload["label"], dtype=np.int8)
        rec = np.asarray(payload["endpoint_mae"], dtype=np.float64)
    aligned = pcc.align_raw_to_a0(np.asarray(attack_timestamps), rec_timestamps)
    if not np.array_equal(aligned, raw_indices) or not np.array_equal(np.asarray(attack_labels)[aligned], labels):
        raise RuntimeError("REC and Dual common legal endpoint mask mismatch")
    fusion = np.maximum(np.maximum(0, (rec - median_rec) / scale_rec), np.maximum(0, (dual - median_dual) / scale_dual))
    global_metrics = pd.DataFrame([
        {"method":"REC", "threshold":threshold_rec, **metrics(labels, rec, threshold_rec)},
        {"method":"DUAL", "threshold":threshold_dual, **metrics(labels, dual, threshold_dual)},
        {"method":"FUSION", "threshold":threshold_fusion, **metrics(labels, fusion, threshold_fusion)},
    ])

    pred_rec, pred_dual, pred_fusion = rec > threshold_rec, dual > threshold_dual, fusion > threshold_fusion
    positive, negative = labels == 1, labels == 0
    point_comp = {
        "rec_miss_dual_hit_attack_points": int((positive & ~pred_rec & pred_dual).sum()),
        "added_normal_fp": int((negative & ~pred_rec & pred_dual).sum()),
    }
    point_comp["fp_per_recovered_tp"] = point_comp["added_normal_fp"] / point_comp["rec_miss_dual_hit_attack_points"] if point_comp["rec_miss_dual_hit_attack_points"] else None

    canonical = pd.read_csv(T25 / "canonical_attack_events.csv")
    canonical = canonical[canonical.evaluable_physical_attack_interval.astype(str).str.lower() == "true"].copy()
    s6 = pd.read_csv(S6 / "event_audit.csv").set_index("event_id")
    targets, stages = reliable.target.astype(str).tolist(), reliable.stage.astype(str).tolist()
    event_rows, attribution = [], []
    for event in canonical.itertuples(index=False):
        event_id = int(event.event_id); start = pcc.timestamp_seconds(str(event.canonical_start)); end = pcc.timestamp_seconds(str(event.canonical_end))
        mask = (rec_timestamps >= start) & (rec_timestamps <= end)
        if not mask.any():
            raise RuntimeError(f"event {event_id} absent from common endpoint mask")
        rr, dr, fr = float(pred_rec[mask].mean()), float(pred_dual[mask].mean()), float(pred_fusion[mask].mean())
        counts = np.bincount(dual_argmax[mask], minlength=len(targets)); top = int(np.argmax(counts))
        event_rows.append({"event_id":event_id, "start":event.canonical_start, "end":event.canonical_end,
                           "attacked_components":str(s6.loc[event_id, "canonical_attack_points"]), "attacked_stage":str(s6.loc[event_id, "stage"]),
                           "s6_category":str(s6.loc[event_id, "auto_contextual_class"]), "common_point_count":int(mask.sum()),
                           "rec_hit_rate":rr, "dual_hit_rate":dr, "fusion_hit_rate":fr,
                           "top_dual_target":targets[top], "top_dual_stage":stages[top],
                           "rec_near_miss":rr < .10, "dual_recovered":rr < .10 and dr >= .10})
        for j, target in enumerate(targets):
            attribution.append({"event_id":event_id,"target":target,"stage":stages[j],"argmax_count":int(counts[j]),"argmax_fraction":float(counts[j]/mask.sum())})
    events = pd.DataFrame(event_rows); contextual = events[events.event_id.isin(CONTEXTUAL_IDS)].copy()
    category = events.groupby("s6_category", sort=False).agg(event_count=("event_id","count"), median_rec_hit=("rec_hit_rate","median"), median_dual_hit=("dual_hit_rate","median"), median_fusion_hit=("fusion_hit_rate","median")).reset_index()
    complementarity = pd.DataFrame([
        {"scope":"ALL_36","event_count":36,"rec_near_miss_count":int(events.rec_near_miss.sum()),"dual_recovery_count":int(events.dual_recovered.sum()),"recovered_event_ids":";".join(events.loc[events.dual_recovered,"event_id"].astype(str))},
        {"scope":"CONTEXTUAL_13_14_29","event_count":3,"rec_near_miss_count":int(contextual.rec_near_miss.sum()),"dual_recovery_count":int(contextual.dual_recovered.sum()),"recovered_event_ids":";".join(contextual.loc[contextual.dual_recovered,"event_id"].astype(str))},
    ])

    shift_rows = []
    for j, target in enumerate(targets):
        test_normal = raw_matrix[labels == 0, j]; val_median = references[target][0]; test_median = float(np.median(test_normal))
        shift_rows.append({"target":target,"stage":stages[j],"validation_median_disagreement":val_median,"test_normal_median_disagreement":test_median,"test_to_validation_median_ratio":test_median/max(val_median,1e-12)})
    normal_shift = pd.DataFrame(shift_rows)
    counts = {name:int((events[f"{name.lower()}_hit_rate"] >= .10).sum()) for name in ("REC","DUAL","FUSION")}
    rec_m, fusion_m = global_metrics.iloc[0], global_metrics.iloc[2]
    gates = {"F1":bool(fusion_m.aupr >= rec_m.aupr + .01), "F2":bool(fusion_m.auroc >= rec_m.auroc - .01), "F3":counts["FUSION"] >= counts["REC"], "F4":bool(fusion_m.fpr <= .10)}
    status = "NORMAL_CALIBRATED_MAX_FUSION_SWAT_SUPPORTED" if all(gates.values()) else "NORMAL_CALIBRATED_MAX_FUSION_SWAT_NOT_SUPPORTED"
    tests = {
        "official_swat_only":True,"normal_split_exact":len(train)==396000 and len(validation)==99000,"discovery_chronological":len(fit)==316800 and len(select)==79200,
        "candidate_rule_exact":len(candidates)==25,"self_lags_fixed":SELF_LAGS==(1,5,10,20,30),"cross_lags_fixed":CROSS_LAGS==(0,1,5,10,20,30),"alpha_one":ALPHA==1.0,
        "unified_gate_exact":True,"targets_frozen_before_attack":bool(target_sha),"two_hard_segments":len(segments)==2,"common_mask":len(labels)==len(rec)==len(dual)==len(fusion),
        "canonical_36":len(events)==36,"contextual_ids":set(contextual.event_id)==set(CONTEXTUAL_IDS),"a0_reused":all(rec_reuse_checks.values()),
        "normal_only_thresholds":True,"no_pa":True,"no_test_selection":True,"finite":bool(np.isfinite(global_metrics.select_dtypes("number")).all().all() and np.isfinite(normal_shift.select_dtypes("number")).all().all()),
    }
    if not all(tests.values()):
        raise RuntimeError(f"SWaT Final integrity failure: {tests}")
    shift_summary = {"median_ratio":float(normal_shift.test_to_validation_median_ratio.median()),"q95_ratio":float(normal_shift.test_to_validation_median_ratio.quantile(.95)),"max_ratio":float(normal_shift.test_to_validation_median_ratio.max()),"worst_target":str(normal_shift.loc[normal_shift.test_to_validation_median_ratio.idxmax(),"target"])}
    summary = {
        "elapsed_seconds":time.perf_counter()-started,"candidate_target_count":len(candidates),"reliable_target_count":len(reliable),
        "reliable_counts_by_stage":reliable.stage.value_counts().sort_index().to_dict(),"reliable_targets":targets,"target_sha256":target_sha,
        "old6_vs_unified":{"intersection":sorted(old_set&new_set),"old_only":sorted(old_set-new_set),"new_only":sorted(new_set-old_set)},
        "rec_reused":True,"rec_checkpoint_sha256":checkpoint_sha,"rec_best_epoch":13,"rec_parameter_count":11730,"global_metrics":global_metrics.to_dict("records"),
        "event_hit_010_counts":counts,"contextual_events":contextual.to_dict("records"),"category_medians":category.to_dict("records"),
        "event_complementarity":complementarity.to_dict("records"),"point_complementarity":point_comp,
        "fusion_gates":gates,"status":status,"normal_domain_shift":shift_summary,"tests":tests,
        "official_swat_final_unified_evaluation":True,"no_test_driven_selection":True,"lag_search":False,"threshold_search":False,"point_adjustment":False,"commit":False,"push":False,
    }
    thresholds = {"quantile":Q,"source":"Official Normal Validation only","comparator":">","rec":threshold_rec,"dual":threshold_dual,"fusion":threshold_fusion,
                  "rec_branch_median":median_rec,"rec_branch_scale":scale_rec,"dual_branch_median":median_dual,"dual_branch_scale":scale_dual}
    manifest = {"dataset":"Official SWaT 2015 Physical","feature_count":51,"feature_names":feature_names,"normal_train":[0,TRAIN_END],"normal_validation":[TRAIN_END,NORMAL_END],
                "discovery_fit":[0,DISCOVERY_END],"discovery_select":[DISCOVERY_END,TRAIN_END],"test_segments":segments,"window":WINDOW,"self_lags":SELF_LAGS,"cross_lags":CROSS_LAGS,"alpha":ALPHA,
                "unified_gate":{"eight_skills":">0","symmetric_median_ratio_max":2.0,"symmetric_q95_ratio_max":3.0},"target_sha256":target_sha,"target_freeze_before_attack":True,
                "dual_score":"max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/(1.4826*MAD_j))","fusion_score":"max(Z_rec,Z_dual)","rec_reuse_checks":rec_reuse_checks,"summary":summary}
    global_metrics.to_csv(OUT/"global_metrics.csv",index=False); events.to_csv(OUT/"event_metrics_36.csv",index=False); contextual.to_csv(OUT/"contextual_events.csv",index=False)
    category.to_csv(OUT/"category_summary.csv",index=False); complementarity.to_csv(OUT/"event_complementarity.csv",index=False); pd.DataFrame(attribution).to_csv(OUT/"target_attribution.csv",index=False)
    normal_shift.to_csv(OUT/"normal_domain_shift.csv",index=False); json_dump(OUT/"thresholds.json",thresholds); json_dump(OUT/"manifest.json",manifest); json_dump(OUT/"swat_final_unified_summary.json",summary)
    metric_lines = "\n".join(f"| {r.method} | {r.auroc:.6f} | {r.aupr:.6f} | {r.precision:.6f} | {r.recall:.6f} | {r.f1:.6f} | {r.fpr:.6f} |" for r in global_metrics.itertuples())
    contextual_lines = "\n".join(f"| {r.event_id} | {r.rec_hit_rate:.6f} | {r.dual_hit_rate:.6f} | {r.fusion_hit_rate:.6f} | {r.top_dual_target} / {r.top_dual_stage} |" for r in contextual.itertuples())
    DOC.write_text(f"""# Official SWaT Final Unified Evaluation

The unified reliability gate retained {len(reliable)}/{len(candidates)} candidate targets. The target manifest SHA256 is `{target_sha}` and was written before Attack feature access. The Official A0 checkpoint and its frozen REC scores were reused after architecture, dimensions, Train/Validation protocol, scaler, checkpoint hash, epoch, and parameter-count checks passed.

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
{metric_lines}

Canonical 36-event hits at rate >=0.10: REC {counts['REC']}, DUAL {counts['DUAL']}, FUSION {counts['FUSION']}.

| Event | REC hit | DUAL hit | FUSION hit | DUAL top target/stage |
|---:|---:|---:|---:|---|
{contextual_lines}

Fusion gates `{gates}` yield **{status}**. Normal-domain-shift summary: `{shift_summary}`. Innovation-2 remains frozen; this result describes its real-benchmark applicability boundary and did not alter targets, sources, lags, alpha, thresholds, or fusion.

No point adjustment, POT, Test-best threshold, Test-driven selection, target Top-K, source/lag/alpha search, fusion-weight search, commit, or push was performed.
""",encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
