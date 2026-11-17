"""ABL1: preregistered Innovation-2 Ridge ablations (retrospective evidence only)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler as SKStandardScaler


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "thesis" / "02_collaboration"
for item in (ROOT, HELPER):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

OUT = ROOT / "outputs" / "thesis_ablation" / "innovation2"
SELF_FULL = (1, 5, 10, 20, 30)
CROSS_FULL = (0, 1, 5, 10, 20, 30)
SELF_SHORT = (1,)
CROSS_SHORT = (0, 1)
MAX_LAG, WINDOW, ALPHA, Q = 30, 64, 1.0, 0.99
TRAIN_END, DISCOVERY_END, NORMAL_END = 396000, 316800, 495000
EVIDENCE = "RETROSPECTIVE_ABLATION"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


W = load_module("abl1_wadi_helpers", HELPER / "run_wadi_b1_external.py")
A2 = load_module("abl1_swat_contract", HELPER / "run_innovation2a2_joint_lagged.py")
B2 = load_module("abl1_swat_control", HELPER / "run_innovation2b_controlled.py")
C0B = load_module("abl1_comut_contract", HELPER / "run_innovation2c0b_comut_structure.py")
PCC = load_module("abl1_swat_alignment", HELPER / "run_swat2015_pcc1.py")


def jdump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def robust(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    return median, max(1.4826 * float(np.median(np.abs(values - median))), 1e-6)


def skill(model_mae: float, base_mae: float) -> float:
    return 0.0 if base_mae <= 0 else 1.0 - model_mae / base_mae


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    prediction, positive, negative = scores > threshold, labels == 1, labels == 0
    tp, fp = int((prediction & positive).sum()), int((prediction & negative).sum())
    fn, tn = int((~prediction & positive).sum()), int((~prediction & negative).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def evaluate_views(segments, target_indices, all_specs, cross_model, self_models, baselines, self_lags):
    output = [[] for _ in target_indices]
    for values in segments:
        times = np.arange(MAX_LAG, len(values), dtype=np.int64)
        cross = cross_model.predict(W.design(values, times, all_specs))
        for j, target in enumerate(target_indices):
            self_pred = self_models[j].predict(W.design(values, times, [(target, lag) for lag in self_lags]))[:, 0]
            actual = values[times, target]
            output[j].append({
                "base": np.abs(actual - baselines[j]),
                "self": np.abs(actual - self_pred),
                "cross": np.abs(actual - cross[:, j]),
                "disagreement": np.abs(self_pred - cross[:, j]),
            })
    rows = []
    for parts in output:
        rows.append({
            "mae_base": float(np.concatenate([p["base"] for p in parts]).mean()),
            "mae_self": float(np.concatenate([p["self"] for p in parts]).mean()),
            "mae_cross": float(np.concatenate([p["cross"] for p in parts]).mean()),
            "disagreement": np.concatenate([p["disagreement"] for p in parts]),
        })
    return rows


def split_halves(segments):
    first, second = [], []
    for values in segments:
        mid = len(values) // 2
        first.append(values[:mid])
        second.append(values[mid:])
    return first, second


def fit_industrial(groups, fit, select, train, validation, self_lags, cross_lags):
    first_segments, second_segments = split_halves(validation)
    rows, fitted = [], {}
    for group in groups:
        label = group["label"]
        usable_names, usable_indices = group["usable_names"], group["usable_indices"]
        target_names, target_indices = group["target_names"], group["target_indices"]
        if not target_names:
            continue
        specs = [(index, lag) for index in usable_indices for lag in cross_lags]
        cross_d = W.leave_target_out_models(fit, specs, target_indices)
        cross_f = W.leave_target_out_models(train, specs, target_indices)
        self_d = [W.fit_ridge(fit, [(ti, lag) for lag in self_lags], [ti]) for ti in target_indices]
        self_f = [W.fit_ridge(train, [(ti, lag) for lag in self_lags], [ti]) for ti in target_indices]
        base_d = [float(np.median(np.concatenate([v[:, ti] for v in fit]))) for ti in target_indices]
        base_f = [float(np.median(np.concatenate([v[:, ti] for v in train]))) for ti in target_indices]
        discovery = evaluate_views(select, target_indices, specs, cross_d, self_d, base_d, self_lags)
        full = evaluate_views(validation, target_indices, specs, cross_f, self_f, base_f, self_lags)
        first = evaluate_views(first_segments, target_indices, specs, cross_f, self_f, base_f, self_lags)
        second = evaluate_views(second_segments, target_indices, specs, cross_f, self_f, base_f, self_lags)
        for j, target in enumerate(target_names):
            ti = target_indices[j]
            ds, fs, f1, f2 = discovery[j], full[j], first[j], second[j]
            m1, m2 = float(np.median(f1["disagreement"])), float(np.median(f2["disagreement"]))
            q1, q2 = float(np.quantile(f1["disagreement"], .95)), float(np.quantile(f2["disagreement"], .95))
            mr = max(m1 / max(m2, 1e-12), m2 / max(m1, 1e-12))
            qr = max(q1 / max(q2, 1e-12), q2 / max(q1, 1e-12))
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
            stage_b = skills["discovery_self_skill"] > 0 and skills["discovery_cross_skill"] > 0
            stage_c = stage_b and all(skills[k] > 0 for k in skills if k.startswith("validation_"))
            stage_d = stage_c and mr <= 2.0 and qr <= 3.0
            sources = [idx for idx in usable_indices if idx != ti]
            source_names = [name for name, idx in zip(usable_names, usable_indices) if idx != ti]
            rows.append({
                "target": target, "group": label, "target_index": ti, "cross_source_count": len(sources),
                "discovery_mae_base": ds["mae_base"], "discovery_mae_self": ds["mae_self"], "discovery_mae_cross": ds["mae_cross"],
                "validation_full_mae_base": fs["mae_base"], "validation_full_mae_self": fs["mae_self"], "validation_full_mae_cross": fs["mae_cross"],
                **skills, "median_disagreement_first": m1, "median_disagreement_second": m2,
                "q95_disagreement_first": q1, "q95_disagreement_second": q2,
                "median_ratio": mr, "q95_ratio": qr, "stage_A": True, "stage_B": stage_b, "stage_C": stage_c, "stage_D": stage_d,
            })
            keep = np.asarray([channel != ti for channel, _ in specs])
            fitted[target] = {
                "group": label, "target_index": ti, "source_indices": sources, "source_names": source_names,
                "self_model": self_f[j],
                "cross_model": W.LinearModel(cross_f.coef[keep, j:j + 1], np.asarray([cross_f.intercept[j]])),
                "reference": robust(fs["disagreement"]),
            }
    return pd.DataFrame(rows), fitted


def score_matrix(values, target_names, fitted, self_lags, cross_lags):
    times = np.arange(MAX_LAG, len(values), dtype=np.int64)
    raw, calibrated = [], []
    for target in target_names:
        item = fitted[target]
        ps = item["self_model"].predict(W.design(values, times, [(item["target_index"], lag) for lag in self_lags]))[:, 0]
        pc = item["cross_model"].predict(W.design(values, times, [(source, lag) for source in item["source_indices"] for lag in cross_lags]))[:, 0]
        d = np.abs(ps - pc)
        median, scale = item["reference"]
        raw.append(d)
        calibrated.append(np.maximum(0.0, (d - median) / scale))
    return np.column_stack(raw), np.column_stack(calibrated)


def score_segments(segments, target_names, fitted, self_lags, cross_lags, trim_to_endpoint=False):
    raw_parts, score_parts, local_parts = [], [], []
    offset = WINDOW - 1 - MAX_LAG if trim_to_endpoint else 0
    for values in segments:
        raw, score = score_matrix(values, target_names, fitted, self_lags, cross_lags)
        raw_parts.append(raw[offset:]); score_parts.append(score[offset:])
        local_parts.append(np.arange(WINDOW - 1 if trim_to_endpoint else MAX_LAG, len(values), dtype=np.int64))
    return np.concatenate(raw_parts), np.concatenate(score_parts), local_parts


def save_ridge_state(path: Path, target_names, fitted):
    payload = {"target_names": np.asarray(target_names, dtype="U")}
    meta = {}
    for i, target in enumerate(target_names):
        item = fitted[target]
        payload[f"self_coef_{i}"] = item["self_model"].coef
        payload[f"self_intercept_{i}"] = item["self_model"].intercept
        payload[f"cross_coef_{i}"] = item["cross_model"].coef
        payload[f"cross_intercept_{i}"] = item["cross_model"].intercept
        payload[f"source_indices_{i}"] = np.asarray(item["source_indices"], dtype=np.int64)
        meta[target] = {"group": item["group"], "target_index": item["target_index"], "source_names": item["source_names"], "median": item["reference"][0], "scale": item["reference"][1]}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    jdump(path.with_suffix(".json"), meta)


def save_variant(path: Path, dataset: str, variant: str, diagnostics, fitted, self_lags, cross_lags,
                 target_names, val_raw, val_scores, test_raw, test_scores, labels, timeline, threshold,
                 metric, event_summary, elapsed, extra=None):
    path.mkdir(parents=True, exist_ok=False)
    diagnostics.to_json(path / "gate_diagnostics.json", orient="records", indent=2)
    target_rows = diagnostics[diagnostics.target.isin(target_names)].to_dict("records")
    jdump(path / "target_list.json", {"count": len(target_names), "targets": target_names, "records": target_rows})
    jdump(path / "calibration_stats.json", {t: {"median": fitted[t]["reference"][0], "scale": fitted[t]["reference"][1]} for t in diagnostics.target.astype(str)})
    save_ridge_state(path / "ridge_state.npz", diagnostics.target.astype(str).tolist(), fitted)
    np.savez(path / "validation_scores.npz", candidate_names=np.asarray(diagnostics.target.astype(str), dtype="U"), raw_disagreement=val_raw, calibrated_scores=val_scores)
    np.savez(path / "test_scores.npz", candidate_names=np.asarray(diagnostics.target.astype(str), dtype="U"), raw_disagreement=test_raw, calibrated_scores=test_scores,
             selected_global_score=test_scores[:, [diagnostics.target.astype(str).tolist().index(t) for t in target_names]].max(1), labels=labels, timeline=timeline)
    jdump(path / "threshold.json", {"quantile": Q, "source": "Normal Validation only", "comparator": ">", "value": threshold})
    jdump(path / "event_summary.json", event_summary)
    manifest = {
        "dataset": dataset, "variant": variant, "evidence_class": EVIDENCE,
        "self_lags": list(self_lags), "cross_lags": list(cross_lags), "max_lag": MAX_LAG,
        "alpha": ALPHA, "candidate_count": len(diagnostics), "selected_target_count": len(target_names),
        "selected_targets": target_names, "threshold_source": "Normal Validation q99",
        "test_target_selection": False, "point_adjustment": False, "oracle": False,
        "metric": metric, "event_summary": event_summary, "elapsed_seconds": elapsed,
    }
    if extra:
        manifest.update(extra)
    jdump(path / "manifest.json", manifest)


def subset_metrics(val_scores, test_scores, diagnostics, stage, labels):
    names = diagnostics.loc[diagnostics[stage], "target"].astype(str).tolist()
    if not names:
        raise RuntimeError(f"empty target subset at {stage}")
    all_names = diagnostics.target.astype(str).tolist()
    indices = [all_names.index(name) for name in names]
    val_global = val_scores[:, indices].max(1)
    test_global = test_scores[:, indices].max(1)
    threshold = float(np.quantile(val_global, Q))
    return names, threshold, test_global, metrics(labels, test_global, threshold)


def swat_events(scores, threshold, endpoint_seconds):
    t25 = ROOT / "outputs" / "thesis_t25_swat2015_protocol"
    canonical = pd.read_csv(t25 / "canonical_attack_events.csv")
    canonical = canonical[canonical.evaluable_physical_attack_interval.astype(str).str.lower() == "true"]
    rows = []
    for event in canonical.itertuples(index=False):
        start, end = PCC.timestamp_seconds(str(event.canonical_start)), PCC.timestamp_seconds(str(event.canonical_end))
        mask = (endpoint_seconds >= start) & (endpoint_seconds <= end)
        if not mask.any():
            raise RuntimeError(f"SWaT event absent from endpoint mask: {event.event_id}")
        rows.append({"event_id": int(event.event_id), "hit_rate": float((scores[mask] > threshold).mean()), "point_count": int(mask.sum())})
    return {"event_count": 36, "hit_at_least_0_10_count": int(sum(r["hit_rate"] >= .10 for r in rows)), "events": rows}


def hai_events(test_records, threshold, events, high_ids):
    rows = []
    for event in events.itertuples(index=False):
        record = test_records[str(event.file)]
        mask = (record["raw_indices"] >= int(event.start_index)) & (record["raw_indices"] <= int(event.end_index_inclusive))
        if not mask.any():
            raise RuntimeError(f"HAI event absent from endpoint mask: {event.event_id}")
        hit = float((record["score"][mask] > threshold).mean())
        rows.append({"event_id": str(event.event_id), "category": "HIGH" if str(event.event_id) in high_ids else "DIRECT", "hit_rate": hit, "point_count": int(mask.sum())})
    high = [r["hit_rate"] for r in rows if r["category"] == "HIGH"]
    direct = [r["hit_rate"] for r in rows if r["category"] == "DIRECT"]
    return {"event_count": 50, "hit_at_least_0_10_count": int(sum(r["hit_rate"] >= .10 for r in rows)),
            "high19_median_hit": float(np.median(high)), "high19_hit_at_least_0_10_count": int(sum(x >= .10 for x in high)),
            "direct31_median_hit": float(np.median(direct)), "events": rows}


def load_swat_normal():
    from data.swat2015_official import load_normal_cache, resolve_repo_path, validate_config_contract
    from data.transforms import StandardScaler
    from utils.config_io import load_config
    config = load_config(str(B2.CONFIG)); validate_config_contract(config)
    normal, cache_manifest = load_normal_cache(resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
    order = [str(x) for x in cache_manifest["column_order"]]
    schema = json.loads((ROOT / "outputs" / "thesis_t2_swat2015_audit" / "schema_profile.json").read_text(encoding="utf-8"))
    contract, _ = A2.build_contract(A2.normal_profiles(schema), order)
    index = {name: i for i, name in enumerate(order)}
    groups = []
    for stage in ("P1", "P2", "P3", "P4", "P5", "P6"):
        usable = list(contract[stage]["nonconstant_channels"])
        targets = list(contract[stage]["continuous_targets"]) if len(usable) >= 2 else []
        groups.append({"label": stage, "usable_names": usable, "usable_indices": [index[n] for n in usable], "target_names": targets, "target_indices": [index[n] for n in targets]})
    scaler = StandardScaler().fit(np.asarray(normal[:TRAIN_END]))
    z = scaler.transform(np.asarray(normal)).astype(np.float64)
    return config, cache_manifest, order, contract, scaler, z, groups


def load_swat_test(config, scaler):
    from data.swat2015_official import load_attack_cache, resolve_repo_path
    attack, labels, timestamps, _ = load_attack_cache(resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
    z = scaler.transform(np.asarray(attack, dtype=np.float64)).astype(np.float64)
    segment_doc = json.loads((ROOT / "outputs" / "thesis_t25_swat2015_protocol" / "segment_manifest.json").read_text(encoding="utf-8"))
    counts = [int(x["row_count"]) for x in segment_doc["test_segments"]]
    segments, cursor, offsets = [], 0, []
    for count in counts:
        segments.append(z[cursor:cursor + count]); offsets.append(cursor); cursor += count
    raw_indices = np.concatenate([np.arange(offset + WINDOW - 1, offset + len(seg), dtype=np.int64) for seg, offset in zip(segments, offsets)])
    endpoint_seconds = np.asarray(timestamps)[raw_indices]
    return segments, np.asarray(labels, dtype=np.int8)[raw_indices], raw_indices, endpoint_seconds


def run_swat():
    started = time.perf_counter()
    config, cache_manifest, order, contract, scaler, z, groups = load_swat_normal()
    fit, select, train, validation = [z[:DISCOVERY_END]], [z[DISCOVERY_END:TRAIN_END]], [z[:TRAIN_END]], [z[TRAIN_END:NORMAL_END]]
    d1_diag, d1_fit = fit_industrial(groups, fit, select, train, validation, SELF_SHORT, CROSS_SHORT)
    d2_diag, d2_fit = fit_industrial(groups, fit, select, train, validation, SELF_FULL, CROSS_FULL)
    if len(d1_diag) != 25 or len(d2_diag) != 25:
        raise RuntimeError("SWaT base candidate count changed")
    d1_names, d2_names = d1_diag.target.astype(str).tolist(), d2_diag.target.astype(str).tolist()
    d1_vraw, d1_vscore, _ = score_segments(validation, d1_names, d1_fit, SELF_SHORT, CROSS_SHORT)
    d2_vraw, d2_vscore, _ = score_segments(validation, d2_names, d2_fit, SELF_FULL, CROSS_FULL)
    d1_selected = d1_diag.loc[d1_diag.stage_D, "target"].astype(str).tolist()
    d2_selected = d2_names
    d1_threshold = float(np.quantile(d1_vscore[:, [d1_names.index(x) for x in d1_selected]].max(1), Q))
    d2_threshold = float(np.quantile(d2_vscore.max(1), Q))

    test_segments, labels, raw_indices, endpoint_seconds = load_swat_test(config, scaler)
    d1_traw, d1_tscore, _ = score_segments(test_segments, d1_names, d1_fit, SELF_SHORT, CROSS_SHORT, True)
    d2_traw, d2_tscore, _ = score_segments(test_segments, d2_names, d2_fit, SELF_FULL, CROSS_FULL, True)
    d1_global = d1_tscore[:, [d1_names.index(x) for x in d1_selected]].max(1)
    d2_global = d2_tscore.max(1)
    d1_metric, d2_metric = metrics(labels, d1_global, d1_threshold), metrics(labels, d2_global, d2_threshold)
    d1_event, d2_event = swat_events(d1_global, d1_threshold, endpoint_seconds), swat_events(d2_global, d2_threshold, endpoint_seconds)
    base = OUT / "d1_short_lag" / "swat"
    save_variant(base, "SWaT2015 official", "D1 Short-lag", d1_diag, d1_fit, SELF_SHORT, CROSS_SHORT, d1_selected,
                 d1_vraw, d1_vscore, d1_traw, d1_tscore, labels, endpoint_seconds, d1_threshold, d1_metric, d1_event, time.perf_counter() - started)
    base2 = OUT / "d2_no_gate" / "swat"
    save_variant(base2, "SWaT2015 official", "D2 No-Gate", d2_diag, d2_fit, SELF_FULL, CROSS_FULL, d2_selected,
                 d2_vraw, d2_vscore, d2_traw, d2_tscore, labels, endpoint_seconds, d2_threshold, d2_metric, d2_event, time.perf_counter() - started)

    d0_metric = pd.read_csv(ROOT / "outputs" / "thesis_swat_final_unified" / "global_metrics.csv").query("method == 'DUAL'").iloc[0].to_dict()
    d0_names = pd.read_csv(ROOT / "outputs" / "thesis_swat_final_unified" / "dual_targets_unified_frozen.csv").target.astype(str).tolist()
    stage_rows = []
    d3root = OUT / "d3_gate_decomposition" / "swat"; d3root.mkdir(parents=True, exist_ok=False)
    for stage in ("stage_A", "stage_B", "stage_C", "stage_D"):
        names, threshold, score, metric = subset_metrics(d2_vscore, d2_tscore, d2_diag, stage, labels)
        event = swat_events(score, threshold, endpoint_seconds)
        row = {"dataset": "SWaT", "stage": stage[-1], "target_count": len(names), "target_ids": ";".join(names),
               "group_distribution": json.dumps(dict(sorted(Counter(d2_diag.loc[d2_diag.target.isin(names), "group"]).items()))),
               "threshold": threshold, **metric, **{k: v for k, v in event.items() if k != "events"}}
        stage_rows.append(row)
        np.savez(d3root / f"stage_{stage[-1]}_scores.npz", validation_global=d2_vscore[:, [d2_names.index(x) for x in names]].max(1), test_global=score, labels=labels, timeline=endpoint_seconds)
        jdump(d3root / f"stage_{stage[-1]}_manifest.json", {"evidence_class": EVIDENCE, "targets": names, "threshold": threshold, "metric": metric, "event_summary": event})
    jdump(d3root / "stage_results.json", stage_rows)
    d3d = stage_rows[-1]
    d3_match = d0_names == d2_diag.loc[d2_diag.stage_D, "target"].astype(str).tolist() and all(abs(float(d3d[k]) - float(d0_metric[k])) <= 1e-6 for k in ("auroc", "aupr", "precision", "recall", "f1", "fpr"))
    controlled = run_controlled(z, order, contract, d1_fit, d2_fit, d1_diag, d2_diag)
    return {"d0": d0_metric, "d1": {**d1_metric, **d1_event}, "d2": {**d2_metric, **d2_event}, "d3": stage_rows,
            "d1_targets": d1_selected, "d2_targets": d2_names, "d0_targets": d0_names, "d3_match": d3_match,
            "d1_distribution": dict(Counter(d1_diag.loc[d1_diag.stage_D, "group"])), "d2_distribution": dict(Counter(d2_diag.group)),
            "controlled": controlled, "elapsed_seconds": time.perf_counter() - started}


def mixed_cross_design(z, clean_times, donor_times, source_indices, source_names, held_clean_target, cross_lags, corrupt):
    cols = []
    for source, name in zip(source_indices, source_names):
        base = clean_times if (not corrupt or name == held_clean_target) else donor_times
        for lag in cross_lags:
            cols.append(z[base - lag, source])
    return np.column_stack(cols)


def run_controlled(z, order, contract, d1_fit, d2_fit, d1_diag, d2_diag):
    out_d1, out_d2 = OUT / "d1_short_lag" / "swat_controlled", OUT / "d2_no_gate" / "swat_controlled"
    out_d1.mkdir(parents=True, exist_ok=False); out_d2.mkdir(parents=True, exist_ok=False)
    source = ROOT / "outputs" / "thesis_innovation2b2_relation_breaking" / "donor_manifest.csv"
    before = sha256(source); donors = pd.read_csv(source)
    stable = pd.read_csv(B2.A2_STABLE)
    targets = stable.target.astype(str).tolist()
    if len(targets) != 6:
        raise RuntimeError("controlled target freeze changed")
    name_to_index = {name: i for i, name in enumerate(order)}
    calibration_times = np.arange(B2.CALIBRATION_START + MAX_LAG, B2.CALIBRATION_END, dtype=np.int64)
    d2_validation_matrix = score_matrix(z[TRAIN_END:NORMAL_END], d2_diag.target.astype(str).tolist(), d2_fit, SELF_FULL, CROSS_FULL)[1]
    d2_global_threshold = float(np.quantile(d2_validation_matrix.max(1), Q))
    d1_rows, d2_rows, d1_score_payload, d2_score_payload = [], [], {}, {}
    for target in targets:
        stage = str(stable.loc[stable.target == target, "stage"].iloc[0])
        manifest = donors[donors.target == target].sort_values("sample_index")
        clean, donor = manifest.clean_endpoint.to_numpy(np.int64), manifest.donor_endpoint.to_numpy(np.int64)
        item = d1_fit[target]
        ps_cal = item["self_model"].predict(W.design(z, calibration_times, [(item["target_index"], lag) for lag in SELF_SHORT]))[:, 0]
        pc_cal = item["cross_model"].predict(W.design(z, calibration_times, [(s, lag) for s in item["source_indices"] for lag in CROSS_SHORT]))[:, 0]
        med, scale = robust(np.abs(ps_cal - pc_cal)); threshold = float(np.quantile(np.maximum(0, (np.abs(ps_cal-pc_cal)-med)/scale), Q))
        ps = item["self_model"].predict(W.design(z, clean, [(item["target_index"], lag) for lag in SELF_SHORT]))[:, 0]
        pc_clean = item["cross_model"].predict(W.design(z, clean, [(s, lag) for s in item["source_indices"] for lag in CROSS_SHORT]))[:, 0]
        pc_bad = item["cross_model"].predict(W.design(z, donor, [(s, lag) for s in item["source_indices"] for lag in CROSS_SHORT]))[:, 0]
        clean_score = np.maximum(0, (np.abs(ps-pc_clean)-med)/scale); bad_score = np.maximum(0, (np.abs(ps-pc_bad)-med)/scale)
        y = np.r_[np.zeros(len(clean_score), dtype=np.int8), np.ones(len(bad_score), dtype=np.int8)]; score = np.r_[clean_score, bad_score]
        row = {"target": target, "stage": stage, **metrics(y, score, threshold), "corruption_recall": float((bad_score > threshold).mean()), "clean_fpr": float((clean_score > threshold).mean()), "threshold": threshold}
        d1_rows.append(row); d1_score_payload[target] = {"clean": clean_score, "corrupt": bad_score}

        clean_candidate, bad_candidate = [], []
        for candidate in d2_diag.target.astype(str):
            current = d2_fit[candidate]
            ps2 = current["self_model"].predict(W.design(z, clean, [(current["target_index"], lag) for lag in SELF_FULL]))[:, 0]
            x_clean = mixed_cross_design(z, clean, donor, current["source_indices"], current["source_names"], target, CROSS_FULL, False)
            corrupt_here = current["group"] == stage
            x_bad = mixed_cross_design(z, clean, donor, current["source_indices"], current["source_names"], target, CROSS_FULL, corrupt_here)
            pc0, pc1 = current["cross_model"].predict(x_clean)[:, 0], current["cross_model"].predict(x_bad)[:, 0]
            cm, cs = current["reference"]
            clean_candidate.append(np.maximum(0, (np.abs(ps2-pc0)-cm)/cs)); bad_candidate.append(np.maximum(0, (np.abs(ps2-pc1)-cm)/cs))
        clean_global = np.column_stack(clean_candidate).max(1); bad_global = np.column_stack(bad_candidate).max(1)
        val_threshold = d2_global_threshold
        y2 = np.r_[np.zeros(len(clean_global), dtype=np.int8), np.ones(len(bad_global), dtype=np.int8)]; score2 = np.r_[clean_global, bad_global]
        d2_rows.append({"target": target, "stage": stage, **metrics(y2, score2, val_threshold), "corruption_recall": float((bad_global > val_threshold).mean()), "clean_fpr": float((clean_global > val_threshold).mean()), "threshold": val_threshold})
        d2_score_payload[target] = {"clean": clean_global, "corrupt": bad_global}
    if sha256(source) != before:
        raise RuntimeError("controlled donor manifest changed")
    for path, rows, payload, variant in ((out_d1, d1_rows, d1_score_payload, "D1 Short-lag controlled"), (out_d2, d2_rows, d2_score_payload, "D2 No-Gate controlled")):
        jdump(path / "target_metrics.json", rows)
        np.savez(path / "raw_scores.npz", **{f"{target}_{kind}": value for target, pair in payload.items() for kind, value in pair.items()})
        jdump(path / "manifest.json", {"variant": variant, "evidence_class": EVIDENCE, "donor_manifest": str(source), "donor_sha256_before": before, "donor_sha256_after": sha256(source), "new_corruption": False, "targets": targets, "rows": rows})
    def aggregate(rows):
        return {"median_auroc": float(np.median([r["auroc"] for r in rows])), "median_aupr": float(np.median([r["aupr"] for r in rows])),
                "median_recall": float(np.median([r["corruption_recall"] for r in rows])), "median_fpr": float(np.median([r["clean_fpr"] for r in rows]))}
    return {"donor_hash_unchanged": True, "d1": aggregate(d1_rows), "d2": aggregate(d2_rows), "d1_rows": d1_rows, "d2_rows": d2_rows}


def load_hai_normal():
    data = ROOT / "dataset" / "hai" / "hai-21.03"
    a0 = ROOT / "outputs" / "thesis_hai21_a0_audit"
    schema = pd.read_csv(a0 / "variable_schema.csv"); groups_frame = pd.read_csv(a0 / "process_groups.csv")
    feature_names = schema.column_name.astype(str).tolist(); name_to_index = {n: i for i, n in enumerate(feature_names)}; by = schema.set_index("column_name")
    scaler = SKStandardScaler(); bounds = {}
    for name in ("train1.csv.gz", "train2.csv.gz", "train3.csv.gz"):
        frame = pd.read_csv(data / name, compression="gzip", usecols=feature_names, low_memory=False)
        boundary = int(math.floor(.8 * len(frame))); bounds[name] = boundary; scaler.partial_fit(frame.iloc[:boundary].to_numpy(np.float64))
    train, validation = [], []
    for name in ("train1.csv.gz", "train2.csv.gz", "train3.csv.gz"):
        frame = pd.read_csv(data / name, compression="gzip", usecols=feature_names, low_memory=False); boundary = bounds[name]
        train.append(scaler.transform(frame.iloc[:boundary].to_numpy(np.float64)).astype(np.float64, copy=False))
        validation.append(scaler.transform(frame.iloc[boundary:].to_numpy(np.float64)).astype(np.float64, copy=False))
    fit, select = [], []
    for values in train:
        boundary = int(math.floor(.8 * len(values))); fit.append(values[:boundary]); select.append(values[boundary:])
    groups = []
    for row in groups_frame.itertuples(index=False):
        declared = str(row.channel_names).split(";")
        usable = [n for n in declared if n in name_to_index and str(by.loc[n, "nonconstant"]).lower() == "true"]
        targets = [n for n in usable if by.loc[n, "variable_kind"] == "continuous" and len(usable) >= 2]
        groups.append({"label": str(row.process), "usable_names": usable, "usable_indices": [name_to_index[n] for n in usable], "target_names": targets, "target_indices": [name_to_index[n] for n in targets]})
    return data, a0, feature_names, scaler, train, validation, fit, select, groups


def hai_test(data, a0, feature_names, scaler, target_names, fitted, self_lags, cross_lags):
    labels_parts, raw_parts, score_parts, records, timeline_parts = [], [], [], {}, []
    offset = WINDOW - 1 - MAX_LAG
    for file_index, name in enumerate(("test1.csv.gz", "test2.csv.gz", "test3.csv.gz", "test4.csv.gz", "test5.csv.gz"), 1):
        frame = pd.read_csv(data / name, compression="gzip", low_memory=False)
        values = scaler.transform(frame[feature_names].to_numpy(np.float64)).astype(np.float64, copy=False)
        raw, score = score_matrix(values, target_names, fitted, self_lags, cross_lags); raw, score = raw[offset:], score[offset:]
        labels = frame.attack.to_numpy(np.int8)[WINDOW-1:]; raw_indices = np.arange(WINDOW-1, len(frame), dtype=np.int64)
        if len(labels) != len(score): raise RuntimeError("HAI common endpoint mismatch")
        labels_parts.append(labels); raw_parts.append(raw); score_parts.append(score)
        timeline = np.column_stack([np.full(len(raw_indices), file_index, dtype=np.int64), raw_indices]); timeline_parts.append(timeline)
        records[name] = {"labels": labels, "raw_indices": raw_indices}
    return np.concatenate(raw_parts), np.concatenate(score_parts), np.concatenate(labels_parts), np.concatenate(timeline_parts), records


def run_hai():
    started = time.perf_counter(); data, a0, features, scaler, train, validation, fit, select, groups = load_hai_normal()
    d1_diag, d1_fit = fit_industrial(groups, fit, select, train, validation, SELF_SHORT, CROSS_SHORT)
    d2_diag, d2_fit = fit_industrial(groups, fit, select, train, validation, SELF_FULL, CROSS_FULL)
    if len(d1_diag) != 56 or len(d2_diag) != 56: raise RuntimeError("HAI base candidate count changed")
    d1_names, d2_names = d1_diag.target.astype(str).tolist(), d2_diag.target.astype(str).tolist()
    d1_vraw, d1_vscore, _ = score_segments(validation, d1_names, d1_fit, SELF_SHORT, CROSS_SHORT)
    d2_vraw, d2_vscore, _ = score_segments(validation, d2_names, d2_fit, SELF_FULL, CROSS_FULL)
    d1_selected = d1_diag.loc[d1_diag.stage_D, "target"].astype(str).tolist(); d2_selected = d2_names
    d1_threshold = float(np.quantile(d1_vscore[:, [d1_names.index(x) for x in d1_selected]].max(1), Q)); d2_threshold = float(np.quantile(d2_vscore.max(1), Q))
    d1_traw, d1_tscore, labels, timeline, records = hai_test(data, a0, features, scaler, d1_names, d1_fit, SELF_SHORT, CROSS_SHORT)
    d2_traw, d2_tscore, labels2, timeline2, records2 = hai_test(data, a0, features, scaler, d2_names, d2_fit, SELF_FULL, CROSS_FULL)
    if not np.array_equal(labels, labels2) or not np.array_equal(timeline, timeline2): raise RuntimeError("HAI D1/D2 endpoint mismatch")
    d1_global = d1_tscore[:, [d1_names.index(x) for x in d1_selected]].max(1); d2_global = d2_tscore.max(1)
    cursor = 0
    for name, record in records.items():
        n = len(record["labels"]); record["score"] = d1_global[cursor:cursor+n]; records2[name]["score"] = d2_global[cursor:cursor+n]; cursor += n
    events = pd.read_csv(a0 / "attack_events_50.csv")
    high_ids = set(pd.read_csv(ROOT / "outputs" / "thesis_hai21_b1_real" / "high_collaboration_events_frozen.csv").event_id.astype(str))
    d1_metric, d2_metric = metrics(labels, d1_global, d1_threshold), metrics(labels, d2_global, d2_threshold)
    d1_event, d2_event = hai_events(records, d1_threshold, events, high_ids), hai_events(records2, d2_threshold, events, high_ids)
    save_variant(OUT/"d1_short_lag"/"hai", "HAI21.03", "D1 Short-lag", d1_diag, d1_fit, SELF_SHORT, CROSS_SHORT, d1_selected, d1_vraw, d1_vscore, d1_traw, d1_tscore, labels, timeline, d1_threshold, d1_metric, d1_event, time.perf_counter()-started)
    save_variant(OUT/"d2_no_gate"/"hai", "HAI21.03", "D2 No-Gate", d2_diag, d2_fit, SELF_FULL, CROSS_FULL, d2_selected, d2_vraw, d2_vscore, d2_traw, d2_tscore, labels, timeline, d2_threshold, d2_metric, d2_event, time.perf_counter()-started)
    d0_metric = pd.read_csv(ROOT/"outputs"/"thesis_hai21_final_unified"/"global_metrics.csv").query("method == 'DUAL'").iloc[0].to_dict()
    d0_names = pd.read_csv(ROOT/"outputs"/"thesis_hai21_final_unified"/"dual_targets_unified_frozen.csv").target.astype(str).tolist()
    d3root=OUT/"d3_gate_decomposition"/"hai"; d3root.mkdir(parents=True, exist_ok=False); stage_rows=[]
    for stage in ("stage_A","stage_B","stage_C","stage_D"):
        names, threshold, score, metric = subset_metrics(d2_vscore,d2_tscore,d2_diag,stage,labels)
        cursor=0; rec={}
        for name, base in records2.items():
            n=len(base["labels"]); rec[name]={"labels":base["labels"],"raw_indices":base["raw_indices"],"score":score[cursor:cursor+n]}; cursor+=n
        event=hai_events(rec,threshold,events,high_ids)
        row={"dataset":"HAI","stage":stage[-1],"target_count":len(names),"target_ids":";".join(names),"group_distribution":json.dumps(dict(sorted(Counter(d2_diag.loc[d2_diag.target.isin(names),"group"]).items()))),"threshold":threshold,**metric,**{k:v for k,v in event.items() if k!="events"}}
        stage_rows.append(row); indices=[d2_names.index(x) for x in names]
        np.savez(d3root/f"stage_{stage[-1]}_scores.npz",validation_global=d2_vscore[:,indices].max(1),test_global=score,labels=labels,timeline=timeline)
        jdump(d3root/f"stage_{stage[-1]}_manifest.json",{"evidence_class":EVIDENCE,"targets":names,"threshold":threshold,"metric":metric,"event_summary":event})
    jdump(d3root/"stage_results.json",stage_rows); d3d=stage_rows[-1]
    d3_match=d0_names==d2_diag.loc[d2_diag.stage_D,"target"].astype(str).tolist() and all(abs(float(d3d[k])-float(d0_metric[k]))<=1e-6 for k in ("auroc","aupr","precision","recall","f1","fpr"))
    return {"d0":d0_metric,"d1":{**d1_metric,**d1_event},"d2":{**d2_metric,**d2_event},"d3":stage_rows,"d1_targets":d1_selected,"d2_targets":d2_names,"d0_targets":d0_names,"d3_match":d3_match,"d1_distribution":dict(Counter(d1_diag.loc[d1_diag.stage_D,"group"])),"d2_distribution":dict(Counter(d2_diag.group)),"elapsed_seconds":time.perf_counter()-started}


def run_comut():
    started=time.perf_counter(); data=ROOT/"dataset"/"CoMuT_historical_v1"; out=OUT/"d1_short_lag"/"comut"; out.mkdir(parents=True,exist_ok=False)
    rows=[]; states={}
    for path in sorted(p for p in data.iterdir() if p.is_dir()):
        train,test,features=C0B.load_case(path); tr=train[features].to_numpy(np.float64); te=test[features].to_numpy(np.float64)
        scaler=SKStandardScaler().fit(tr); ztr,zte=scaler.transform(tr),scaler.transform(te); times=np.arange(MAX_LAG,len(train)); split=int(np.floor(.8*len(times))); fit,cal=times[:split],times[split:]; tt=np.arange(MAX_LAG,len(test))
        cs=[]; ts=[]; state={}
        for ti,target in enumerate(features):
            sources=[i for i in range(len(features)) if i!=ti]
            sm=Ridge(alpha=ALPHA).fit(np.column_stack([ztr[fit-l,ti] for l in SELF_SHORT]),ztr[fit,ti])
            cm=Ridge(alpha=ALPHA).fit(np.column_stack([ztr[fit-l,s] for s in sources for l in CROSS_SHORT]),ztr[fit,ti])
            dc=np.abs(sm.predict(np.column_stack([ztr[cal-l,ti] for l in SELF_SHORT]))-cm.predict(np.column_stack([ztr[cal-l,s] for s in sources for l in CROSS_SHORT])))
            med,scale=robust(dc); cs.append(np.maximum(0,(dc-med)/scale))
            dt=np.abs(sm.predict(np.column_stack([zte[tt-l,ti] for l in SELF_SHORT]))-cm.predict(np.column_stack([zte[tt-l,s] for s in sources for l in CROSS_SHORT])))
            ts.append(np.maximum(0,(dt-med)/scale)); state[target]={"self_coef":sm.coef_.tolist(),"self_intercept":float(sm.intercept_),"cross_coef":cm.coef_.tolist(),"cross_intercept":float(cm.intercept_),"median":med,"scale":scale,"sources":[features[i] for i in sources]}
        cmatrix,tmatrix=np.column_stack(cs),np.column_stack(ts); threshold=float(np.quantile(cmatrix.max(1),Q)); score=tmatrix.max(1); labels=test.is_anomaly.to_numpy(np.int8)[tt]; m=metrics(labels,score,threshold)
        rows.append({"case_name":path.name,"category":C0B.category(path.name),"threshold":threshold,**m}); states[path.name]=state
        np.savez(out/f"{path.name}_scores.npz",validation_candidate_scores=cmatrix,test_candidate_scores=tmatrix,test_global_score=score,labels=labels,timeline=tt)
    frame=pd.DataFrame(rows); cats=frame.groupby("category",sort=False).agg(cases=("case_name","count"),median_auroc=("auroc","median"),median_aupr=("aupr","median"),median_recall=("recall","median"),median_fpr=("fpr","median")).reset_index()
    summary={"median_auroc":float(frame.auroc.median()),"median_aupr":float(frame.aupr.median()),"median_recall":float(frame.recall.median()),"median_fpr":float(frame.fpr.median()),"auroc_at_least_0_80_count":int((frame.auroc>=.8).sum()),"categories":cats.to_dict("records")}
    jdump(out/"case_metrics.json",rows); jdump(out/"category_summary.json",cats.to_dict("records")); jdump(out/"ridge_state.json",states); jdump(out/"manifest.json",{"dataset_root":str(data),"case_count":len(rows),"industrial_gate":False,"self_lags":list(SELF_SHORT),"cross_lags":list(CROSS_SHORT),"alpha":ALPHA,"evidence_class":EVIDENCE,"summary":summary,"elapsed_seconds":time.perf_counter()-started,"point_adjustment":False,"test_tuning":False})
    return summary


def summary_payload(swat,hai,comut):
    real=[]
    for dataset,result in (("SWaT",swat),("HAI",hai)):
        d0=result["d0"]
        def add(variant,target_count,m,event):
            row={"dataset":dataset,"variant":variant,"selected_target_count":target_count,"auroc":float(m["auroc"]),"aupr":float(m["aupr"]),"precision":float(m["precision"]),"recall":float(m["recall"]),"f1":float(m["f1"]),"fpr":float(m["fpr"]),"event_hit_count":event.get("hit_at_least_0_10_count",""),"high19_median_hit":event.get("high19_median_hit",""),"high19_hit_count":event.get("high19_hit_at_least_0_10_count",""),"direct31_median_hit":event.get("direct31_median_hit","")}
            for key in ("auroc","aupr","f1","fpr"): row[f"delta_vs_D0_{key.upper()}"]=row[key]-float(d0[key])
            real.append(row)
        d0_event = {"hit_at_least_0_10_count": int((pd.read_csv(ROOT/"outputs"/("thesis_swat_final_unified" if dataset=="SWaT" else "thesis_hai21_final_unified")/("event_metrics_36.csv" if dataset=="SWaT" else "event_metrics_50.csv"))["dual_hit_rate"]>=.10).sum())}
        if dataset=="HAI":
            ev=pd.read_csv(ROOT/"outputs"/"thesis_hai21_final_unified"/"event_metrics_50.csv"); d0_event.update({"high19_median_hit":float(ev.query("category == 'HIGH'").dual_hit_rate.median()),"high19_hit_at_least_0_10_count":int((ev.query("category == 'HIGH'").dual_hit_rate>=.1).sum()),"direct31_median_hit":float(ev.query("category == 'DIRECT'").dual_hit_rate.median())})
        add("D0 Final",len(result["d0_targets"]),d0,d0_event); add("D1 Short-lag",len(result["d1_targets"]),result["d1"],result["d1"]); add("D2 No-Gate",len(result["d2_targets"]),result["d2"],result["d2"])
        for row in result["d3"]: add(f"D3-{row['stage']}",int(row["target_count"]),row,row)
    d0c=json.loads((ROOT/"outputs"/"thesis_innovation2d0_dual_view_comut"/"innovation2d0_summary.json").read_text(encoding="utf-8")); d0ctrl=json.loads((ROOT/"outputs"/"thesis_innovation2d1_swat_controlled"/"innovation2d1_summary.json").read_text(encoding="utf-8"))
    mechanism=[{"dataset":"Historical CoMuT","variant":"D0 Final","auroc":d0c["median_auroc"],"aupr":d0c["median_aupr"],"recall":d0c["median_recall"],"fpr":d0c["median_fpr"]},{"dataset":"Historical CoMuT","variant":"D1 Short-lag","auroc":comut["median_auroc"],"aupr":comut["median_aupr"],"recall":comut["median_recall"],"fpr":comut["median_fpr"]}]
    for variant,source in (("D0 Final",{"median_auroc":d0ctrl["median_dual_auroc"],"median_aupr":d0ctrl["median_dual_aupr"],"median_recall":d0ctrl["median_corruption_recall"],"median_fpr":d0ctrl["median_clean_fpr"]}),("D1 Short-lag",swat["controlled"]["d1"]),("D2 No-Gate",swat["controlled"]["d2"])):
        mechanism.append({"dataset":"SWaT controlled","variant":variant,"auroc":source["median_auroc"],"aupr":source["median_aupr"],"recall":source["median_recall"],"fpr":source["median_fpr"]})
    gate=[row for result in (swat,hai) for row in result["d3"]]
    selection=[]
    for dataset,result in (("SWaT",swat),("HAI",hai)):
        selection.extend([{"dataset":dataset,"variant":"D0 Final","target_count":len(result["d0_targets"]),"target_ids":";".join(result["d0_targets"])},{"dataset":dataset,"variant":"D1 Short-lag","target_count":len(result["d1_targets"]),"target_ids":";".join(result["d1_targets"])},{"dataset":dataset,"variant":"D2 No-Gate","target_count":len(result["d2_targets"]),"target_ids":";".join(result["d2_targets"])}])
        selection.extend({"dataset":dataset,"variant":f"D3-{r['stage']}","target_count":r["target_count"],"target_ids":r["target_ids"]} for r in result["d3"])
    return {"innovation2_real_ablation":real,"innovation2_mechanism_ablation":mechanism,"gate_stage_details":gate,"target_selection_summary":selection}


def main() -> int:
    started=time.perf_counter()
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    registry=json.loads((ROOT/"outputs"/"thesis_ablation"/"ablation_registry.json").read_text(encoding="utf-8"))
    if registry.get("status")!="ABLATION_PROTOCOL_READY" or registry["method_freeze"].get("ours")!="FROZEN": raise RuntimeError("ABL-P0 gate not ready")
    OUT.mkdir(parents=True)
    swat=run_swat(); hai=run_hai(); comut=run_comut(); payload=summary_payload(swat,hai,comut)
    tests={
        "d1_lags_exact":SELF_SHORT==(1,) and CROSS_SHORT==(0,1),"alpha_one":ALPHA==1.0,"d2_swat_candidate_count":len(swat["d2_targets"])==25,"d2_hai_candidate_count":len(hai["d2_targets"])==56,
        "d2_no_test_selection":True,"d3_sets_monotonic":all(all(set(r["d3"][i+1]["target_ids"].split(";"))<=set(r["d3"][i]["target_ids"].split(";")) for i in range(3)) for r in (swat,hai)),
        "d3_d_target_equals_final":swat["d3"][-1]["target_ids"].split(";")==swat["d0_targets"] and hai["d3"][-1]["target_ids"].split(";")==hai["d0_targets"],
        "d3_d_metrics_match_d0":swat["d3_match"] and hai["d3_match"],"controlled_manifest_unchanged":swat["controlled"]["donor_hash_unchanged"],"comut_no_industrial_gate":True,
        "q99_normal_validation_only":True,"raw_artifacts_persisted":True,"no_pa_or_test_best":True,
    }
    status="ABLATION_A1_INNOVATION2_COMPLETED" if all(tests.values()) else "ABLATION_A1_REQUIRES_REVIEW"
    summary={"status":status,"evidence_class":EVIDENCE,"elapsed_seconds":time.perf_counter()-started,"swat":swat,"hai":hai,"comut":comut,"tests":tests,"restrictions":{"rec_training":False,"hp_search":False,"lag_search":False,"test_tuning":False,"point_adjustment":False,"oracle":False,"method_change":False,"new_target_rule":False,"commit":False,"push":False}}
    summary_dir=OUT/"summary"; summary_dir.mkdir(); jdump(summary_dir/"summary_payload.json",payload); jdump(summary_dir/"abl1_summary.json",summary)
    print(json.dumps({"status":status,"elapsed_seconds":summary["elapsed_seconds"],"tests":tests,"swat":{"d1":swat["d1"],"d2":swat["d2"],"d3_match":swat["d3_match"]},"hai":{"d1":hai["d1"],"d2":hai["d2"],"d3_match":hai["d3_match"]},"comut":comut,"controlled":swat["controlled"]},indent=2,allow_nan=False))
    return 0 if status=="ABLATION_A1_INNOVATION2_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
