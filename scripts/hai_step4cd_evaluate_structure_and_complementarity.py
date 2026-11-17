from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "hai_step4cd_structure_real_and_complementarity"
STEP1 = ROOT / "outputs" / "hai_step1_dataset_audit"
STEP2 = ROOT / "outputs" / "hai_step2_dataloader_validation"
STEP3D = ROOT / "outputs" / "hai_step3d_rec_channel_calibration"
STEP3G = ROOT / "outputs" / "hai_step3g_rec_capacity_ablation"
STEP4A = ROOT / "outputs" / "hai_step4a_normal_relation_diagnosis"
STEP4B = ROOT / "outputs" / "hai_step4b_structure_controlled_validation"
DATA_ROOT = ROOT / "dataset" / "dataset" / "hai-22.04"
TEST_FILES = tuple(f"test{i}.csv" for i in range(1, 5))
WINDOW = 128


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def frozen_config() -> dict[str, Any]:
    return {"relation_pool": {"source": "Step 4A diversity pool", "pair_count": 10, "channel_count": 15},
            "window": 128, "relation_representation": "pearson_fisher_z",
            "reference": "train_only_global_median_mad", "aggregation": "original_distance_max",
            "threshold": "existing_clean_validation_q99", "selection_status": "frozen_before_hai_test",
            "evaluation_status": "exploratory_not_confirmatory", "tiny_rec": {"seed": 1,
            "score": "endpoint_raw_top5", "alarm": "pointwise", "threshold": "existing_normal_validation_q99"}}


def write_frozen_config() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "frozen_exploratory_structure_config.yaml"
    expected = yaml.safe_dump(frozen_config(), sort_keys=False)
    if path.exists() and path.read_text(encoding="utf-8") != expected:
        raise RuntimeError("Frozen exploratory structure config changed")
    if not path.exists(): path.write_text(expected, encoding="utf-8")
    return path


def load_registry():
    pool = pd.read_csv(STEP4B / "structure_relation_pool_registry.csv").reset_index(drop=True)
    ref = pd.read_csv(STEP4B / "structure_reference_fallback_audit.csv")
    ref = ref[ref.window == 128].set_index("pair_id").loc[pool.pair_id]
    threshold = pd.read_csv(STEP4B / "structure_controlled_thresholds.csv")
    threshold = float(threshold.loc[threshold.candidate_id == "diversity_global|w128|max", "threshold"].iloc[0])
    scaler = pd.read_csv(STEP2 / "hai_scaler_stats.csv").set_index("feature")
    if len(pool) != 10 or len(set(pool.feature_i) | set(pool.feature_j)) != 15 or threshold <= 0:
        raise RuntimeError("Frozen structure registry mismatch")
    return pool, ref.global_median.to_numpy(float), ref.global_scale.to_numpy(float), threshold, scaler


def rolling_fisher(values: np.ndarray, pairs: np.ndarray, window: int = 128) -> np.ndarray:
    n = len(values) - window + 1
    starts = np.arange(n); ends = starts + window
    result = np.empty((n, len(pairs)), np.float64)
    for edge, (left, right) in enumerate(pairs):
        x, y = values[:, left].astype(np.float64), values[:, right].astype(np.float64)
        def sums(v): return np.r_[0.0, np.cumsum(v)]
        sx, sy, sxx, syy, sxy = sums(x), sums(y), sums(x*x), sums(y*y), sums(x*y)
        ax, ay = sx[ends]-sx[starts], sy[ends]-sy[starts]
        vx = np.maximum(sxx[ends]-sxx[starts]-ax*ax/window, 0)
        vy = np.maximum(syy[ends]-syy[starts]-ay*ay/window, 0)
        cov = sxy[ends]-sxy[starts]-ax*ay/window
        den = np.sqrt(vx*vy)
        corr = np.divide(cov, den, out=np.full(n, np.nan), where=den > 1e-12)
        result[:, edge] = np.arctanh(np.clip(corr, -.999999, .999999))
    return result


def score_numeric_test(values: np.ndarray, pairs: np.ndarray, median: np.ndarray, scale: np.ndarray):
    z = rolling_fisher(values, pairs, WINDOW)
    z = np.where(np.isfinite(z), z, median)
    distance = np.abs(z - median) / scale
    return distance.astype(np.float32), distance.max(axis=1).astype(np.float32)


def dependency_hash(pool_path: Path, ref_path: Path) -> str:
    inventory = pd.read_csv(STEP1 / "hai_file_inventory.csv").set_index("file_name")
    items = [(name, str(inventory.loc[name, "sha256"])) for name in TEST_FILES]
    items += [("pool", sha256(pool_path)), ("reference", sha256(ref_path)), ("window", "128"),
              ("representation", "pearson_fisher_z"), ("aggregation", "original_distance_max")]
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def generate_score_cache(pool, median, scale, scaler) -> tuple[Path, bool]:
    cache = OUTPUT / "structure_test_scores.npz"; meta = OUTPUT / "structure_test_scores.meta.json"
    dep = dependency_hash(STEP4B / "structure_relation_pool_registry.csv", STEP4B / "structure_reference_fallback_audit.csv")
    if cache.exists() and meta.exists() and json.loads(meta.read_text()).get("dependency_hash") == dep:
        return cache, True
    features = list(dict.fromkeys(pool.feature_i.tolist() + pool.feature_j.tolist()))
    index = {name: i for i, name in enumerate(features)}
    pairs = np.asarray([[index[a], index[b]] for a, b in zip(pool.feature_i, pool.feature_j)], int)
    means = scaler.loc[features, "mean"].to_numpy(float); scales = scaler.loc[features, "effective_scale"].to_numpy(float)
    payload: dict[str, np.ndarray] = {"relation_pair_ids": pool.pair_id.to_numpy(np.int32)}
    for file_name in TEST_FILES:
        seq = Path(file_name).stem; path = DATA_ROOT / file_name
        header = pd.read_csv(path, nrows=0)
        timestamp = next(c for c in header if c.strip().lower() == "timestamp")
        label = next(c for c in header if c.strip().lower() == "attack")
        frame = pd.read_csv(path, usecols=[timestamp, label, *features], low_memory=False)
        numeric = frame[features].to_numpy(np.float64)
        normalized = ((numeric - means) / scales).astype(np.float32)
        per_relation, score = score_numeric_test(normalized, pairs, median, scale)
        endpoints = np.arange(WINDOW - 1, len(frame), dtype=np.int64)
        payload[f"{seq}_sequence_id"] = np.full(len(score), seq, dtype="U5")
        payload[f"{seq}_timestamp"] = frame[timestamp].astype(str).to_numpy()[endpoints].astype("U19")
        payload[f"{seq}_endpoint_row"] = endpoints
        payload[f"{seq}_label"] = pd.to_numeric(frame[label], errors="raise").to_numpy(np.int8)[endpoints]
        payload[f"{seq}_per_relation_distance"] = per_relation
        payload[f"{seq}_structure_score_original_max"] = score
    np.savez_compressed(cache, **payload)
    meta.write_text(json.dumps({"dependency_hash": dep, "score_generated_before_label_evaluation": True,
                                "test_labels_used_for_score": False, "metadata_complete": True}, indent=2))
    return cache, False


def point_metrics(labels, scores, alarms):
    p, r, f, _ = precision_recall_fscore_support(labels, alarms, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "aupr": average_precision_score(labels, scores),
            "auroc": roc_auc_score(labels, scores), "tp": int(((alarms==1)&(labels==1)).sum()),
            "fp": int(((alarms==1)&(labels==0)).sum()), "fn": int(((alarms==0)&(labels==1)).sum())}


def event_table(events, sequences, rows, alarms, prefix):
    records=[]
    for e in events.itertuples():
        mask=(sequences==e.sequence_id)&(rows>=e.start_row)&(rows<=e.end_row)
        hit=np.flatnonzero(mask&alarms)
        records.append({"global_event_id":int(e.global_event_id),"sequence_id":e.sequence_id,"start_row":int(e.start_row),
                        "end_row":int(e.end_row),f"{prefix}_detected":bool(len(hit)),
                        f"{prefix}_delay":int(rows[hit[0]]-e.start_row) if len(hit) else -1})
    return pd.DataFrame(records)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--refresh-report",action="store_true",help="Refresh metrics from valid score/Tiny caches without test score recomputation"); args=parser.parse_args()
    config_path=write_frozen_config()
    if args.dry_run:
        print(f"FROZEN_CONFIG={config_path}",flush=True); print("NEW_COMPUTE=test-only 10-pair w128 original max once",flush=True)
        print("REUSE=Step2 scaler/events, Step3D Tiny NPZ/q99, Step4A/B pool/reference/threshold",flush=True); return 0
    if not args.refresh_report and (OUTPUT/"structure_real_evaluation_summary.json").exists(): print(f"REUSE completed Step 4C+D: {OUTPUT}",flush=True); return 0
    historical=[STEP2/"hai_scaler_stats.csv",STEP2/"hai_attack_events.csv",STEP3D/"rec_channel_scores_seed1.npz",
                STEP3D/"rec_channel_thresholds.csv",STEP3G/"stage_status.json",STEP4A/"diversity_candidate_pool.csv",
                STEP4B/"structure_reference_fallback_audit.csv",STEP4B/"structure_controlled_thresholds.csv"]
    before={str(p):sha256(p) for p in historical}
    pool,median,scale,struct_thr,scaler=load_registry()
    cache_path,reused=generate_score_cache(pool,median,scale,scaler)
    z=np.load(cache_path,allow_pickle=False); rec=np.load(STEP3D/"rec_channel_scores_seed1.npz",allow_pickle=False)
    rec_thr_frame=pd.read_csv(STEP3D/"rec_channel_thresholds.csv")
    rec_thr=float(rec_thr_frame[(rec_thr_frame.seed==1)&(rec_thr_frame.score_variant=="endpoint_raw_top5")&
                                (rec_thr_frame.threshold_method=="normal_validation_q99")].threshold_value.iloc[0])
    arrays={key:[] for key in ("sequence","timestamp","row","label","distance","structure","rec")}
    file_rows=[]
    for file_name in TEST_FILES:
        seq=Path(file_name).stem; srow=z[f"{seq}_endpoint_row"]; rrow=rec[f"test_{seq}_end_row"]
        _,si,ri=np.intersect1d(srow,rrow,return_indices=True)
        if not (np.array_equal(z[f"{seq}_timestamp"][si],rec[f"test_{seq}_timestamp"][ri]) and
                np.array_equal(z[f"{seq}_label"][si],rec[f"test_{seq}_label"][ri])): raise RuntimeError("NPZ alignment error")
        arrays["sequence"].append(z[f"{seq}_sequence_id"][si]); arrays["timestamp"].append(z[f"{seq}_timestamp"][si]); arrays["row"].append(srow[si])
        arrays["label"].append(z[f"{seq}_label"][si]); arrays["distance"].append(z[f"{seq}_per_relation_distance"][si])
        arrays["structure"].append(z[f"{seq}_structure_score_original_max"][si]); arrays["rec"].append(rec[f"test_{seq}_endpoint_raw_top5"][ri])
    arrays={k:np.concatenate(v) for k,v in arrays.items()}; labels=arrays["label"].astype(int)
    salarm=arrays["structure"]>struct_thr; ralarm=arrays["rec"]>rec_thr; oralarm=salarm|ralarm; andalarm=salarm&ralarm
    for seq in [f"test{i}" for i in range(1,5)]:
        m=arrays["sequence"]==seq; file_rows.append({"sequence_id":seq,"method":"structure",**point_metrics(labels[m],arrays["structure"][m],salarm[m])})
    structure_point=pd.DataFrame([{"scope":"all","threshold":struct_thr,**point_metrics(labels,arrays["structure"],salarm)}])
    structure_point.to_csv(OUTPUT/"structure_only_point_metrics.csv",index=False); pd.DataFrame(file_rows).to_csv(OUTPUT/"structure_metrics_by_test_file.csv",index=False)
    events=pd.read_csv(STEP2/"hai_attack_events.csv"); se=event_table(events,arrays["sequence"],arrays["row"],salarm,"structure")
    re=event_table(events,arrays["sequence"],arrays["row"],ralarm,"rec"); oe=event_table(events,arrays["sequence"],arrays["row"],oralarm,"or")
    event=re.merge(se,on=["global_event_id","sequence_id","start_row","end_row"]).merge(oe,on=["global_event_id","sequence_id","start_row","end_row"])
    detected=int(se.structure_detected.sum()); delays=se.loc[se.structure_detected,"structure_delay"]
    event_metric_rows=[{"scope":"all","detected_events":detected,"total_events":len(events),"event_recall":detected/len(events),
                   "median_delay":float(delays.median()) if len(delays) else -1,
                   "undetected_event_ids":";".join(map(str,se.loc[~se.structure_detected,"global_event_id"]))}]
    for seq in [f"test{i}" for i in range(1,5)]:
        current=se[se.sequence_id==seq]; current_delays=current.loc[current.structure_detected,"structure_delay"]
        event_metric_rows.append({"scope":seq,"detected_events":int(current.structure_detected.sum()),"total_events":len(current),
            "event_recall":float(current.structure_detected.mean()),"median_delay":float(current_delays.median()) if len(current_delays) else -1,
            "undetected_event_ids":";".join(map(str,current.loc[~current.structure_detected,"global_event_id"]))})
    pd.DataFrame(event_metric_rows).to_csv(OUTPUT/"structure_only_event_metrics.csv",index=False)
    event["joint_detected"]=event.rec_detected&event.structure_detected; event["rec_only_event"]=event.rec_detected&~event.structure_detected
    event["structure_only_event"]=~event.rec_detected&event.structure_detected; event["both_missed"]=~event.rec_detected&~event.structure_detected
    event.to_csv(OUTPUT/"rec_structure_event_overlap.csv",index=False)
    unique=event[event.structure_only_event].copy(); unique.to_csv(OUTPUT/"structure_unique_recovered_events.csv",index=False)
    overlap={"rec_tp":int((ralarm& (labels==1)).sum()),"rec_fp":int((ralarm&(labels==0)).sum()),"rec_fn":int((~ralarm&(labels==1)).sum()),
             "structure_tp":int((salarm&(labels==1)).sum()),"structure_fp":int((salarm&(labels==0)).sum()),"structure_fn":int((~salarm&(labels==1)).sum()),
             "or_tp":int((oralarm&(labels==1)).sum()),"or_fp":int((oralarm&(labels==0)).sum()),"or_fn":int((~oralarm&(labels==1)).sum()),
             "and_tp":int((andalarm&(labels==1)).sum()),"and_fp":int((andalarm&(labels==0)).sum()),"and_fn":int((~andalarm&(labels==1)).sum()),
             "alarm_jaccard":float((salarm&ralarm).sum()/max((salarm|ralarm).sum(),1)),
             "normal_alarm_overlap":int((salarm&ralarm&(labels==0)).sum()),"anomaly_alarm_overlap":int((salarm&ralarm&(labels==1)).sum()),
             "structure_added_tp":int((salarm&~ralarm&(labels==1)).sum()),"structure_added_fp":int((salarm&~ralarm&(labels==0)).sum())}
    overlap["added_fp_per_added_tp"]=overlap["structure_added_fp"]/max(overlap["structure_added_tp"],1)
    pd.DataFrame([overlap]).to_csv(OUTPUT/"rec_structure_point_overlap.csv",index=False)
    rec_events=int(event.rec_detected.sum()); struct_events=int(event.structure_detected.sum()); or_events=int(event.or_detected.sum())
    union={"rec_detected_events":rec_events,"structure_detected_events":struct_events,"joint_events":int(event.joint_detected.sum()),
           "rec_only_events":int(event.rec_only_event.sum()),"structure_only_events":int(event.structure_only_event.sum()),
           "both_missed_events":int(event.both_missed.sum()),"or_detected_events":or_events,"or_event_recall":or_events/58,
           "or_recall_improvement":(or_events-rec_events)/58}
    pd.DataFrame([union]).to_csv(OUTPUT/"rec_structure_union_metrics.csv",index=False)
    maxedge=np.argmax(arrays["distance"],axis=1); alarm_idx=np.flatnonzero(salarm)
    attr=pd.DataFrame({"sequence_id":arrays["sequence"][alarm_idx],"endpoint_row":arrays["row"][alarm_idx],"timestamp":arrays["timestamp"][alarm_idx],
                       "max_pair_id":pool.pair_id.to_numpy()[maxedge[alarm_idx]],"max_distance":arrays["structure"][alarm_idx],
                       "rec_alarm":ralarm[alarm_idx],"attack":labels[alarm_idx].astype(bool)})
    attr.to_csv(OUTPUT/"structure_relation_alarm_attribution.csv",index=False)
    fp=attr[~attr.attack].sort_values(["sequence_id","endpoint_row"]).copy()
    fp["new_segment"]=(fp.sequence_id.ne(fp.sequence_id.shift())|fp.endpoint_row.sub(fp.endpoint_row.shift()).ne(1))
    fp["segment_id"]=fp.new_segment.cumsum()
    fp_rows=[]
    for (seq,pair),count in fp.groupby(["sequence_id","max_pair_id"]).size().items():
        fp_rows.append({"record_type":"relation_total","sequence_id":seq,"max_pair_id":int(pair),"false_positive_count":int(count),
                        "segment_start_row":-1,"segment_end_row":-1,"segment_length":-1})
    for _,segment in fp.groupby("segment_id"):
        if len(segment)>=10:
            fp_rows.append({"record_type":"long_false_positive_segment","sequence_id":segment.sequence_id.iloc[0],
                "max_pair_id":int(segment.max_pair_id.mode().iloc[0]),"false_positive_count":len(segment),
                "segment_start_row":int(segment.endpoint_row.min()),"segment_end_row":int(segment.endpoint_row.max()),"segment_length":len(segment)})
    pd.DataFrame(fp_rows).to_csv(OUTPUT/"structure_false_positive_relation_attribution.csv",index=False)
    recovered_files=unique.sequence_id.nunique(); added_fp_ok=overlap["structure_added_fp"]<=overlap["rec_fp"]
    dominant=attr[attr.attack].max_pair_id.value_counts(normalize=True).max() if len(attr[attr.attack]) else 1
    if len(unique)>=3 and union["or_recall_improvement"]>=.05 and recovered_files>=2 and added_fp_ok and dominant<.8:
        classification="structure_provides_practical_complementarity"
    else:
        sm=structure_point.iloc[0]
        if sm.auroc>.55 or sm.aupr>labels.mean() or detected>0:
            classification="structure_repeats_rec_without_independent_value" if len(unique)==0 and event.joint_detected.sum()>=.9*max(struct_events,1) else "structure_signal_exists_but_complementarity_is_limited"
        else: classification="controlled_structure_signal_does_not_transfer_to_real_attacks"
    cache_audit=pd.DataFrame([{"cache_path":str(cache_path),"reused":reused,"computed_once":not reused,
                               "dependency_hash":json.loads((OUTPUT/"structure_test_scores.meta.json").read_text())["dependency_hash"],"valid":True}])
    cache_audit.to_csv(OUTPUT/"structure_test_score_cache_audit.csv",index=False)
    after={str(p):sha256(p) for p in historical}; pd.DataFrame([{"artifact_id":Path(p).name,"source_path":p,"sha256_before":h,
        "sha256_after":after[p],"hash_match":h==after[p],"reused":True,"recomputed":False} for p,h in before.items()]).to_csv(OUTPUT/"artifact_reuse_audit.csv",index=False)
    summary={"classification":classification,"exploratory_not_confirmatory":True,"structure_threshold":struct_thr,"rec_threshold":rec_thr,
             "structure_point":structure_point.iloc[0].to_dict(),"structure_events":detected,"rec_events":rec_events,
             "or_events":or_events,"unique_recovered_events":len(unique),"unique_recovered_event_ids":unique.global_event_id.astype(int).tolist(),"recovered_test_files":recovered_files,
             "point_overlap":overlap,"event_union":union,"historical_hashes_unchanged":before==after,
             "test_used_for_configuration":False,"tiny_reinference":False,"point_adjustment":False}
    (OUTPUT/"structure_real_evaluation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    report=f"""# HAI Step 4C+D structure-only and Tiny-rec complementarity

- Frozen before test: 10 relations, window 128, train-only global median/MAD, original max, clean-validation q99.
- Structure point AUROC/AUPR/F1: {structure_point.iloc[0].auroc:.4f}/{structure_point.iloc[0].aupr:.4f}/{structure_point.iloc[0].f1:.4f}.
- Structure events: {detected}/58; Tiny rec events: {rec_events}/58; OR events: {or_events}/58.
- Structure-only recovered rec-missed events: {len(unique)} across {recovered_files} test files.
- Structure-added TP/FP: {overlap['structure_added_tp']}/{overlap['structure_added_fp']} ({overlap['added_fp_per_added_tp']:.3f} FP per TP).
- Classification: **{classification}**.

This is one-shot exploratory evidence, not confirmatory evaluation. No configuration, threshold, relation, Tiny score, or event definition was changed after test access.
"""
    (OUTPUT/"final_hai_structure_real_and_rec_complementarity.md").write_text(report,encoding="utf-8")
    print(json.dumps({"classification":classification,"structure_events":detected,"rec_events":rec_events,"or_events":or_events,
                      "unique_recovered_events":len(unique),"historical_hashes_unchanged":before==after},indent=2),flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
