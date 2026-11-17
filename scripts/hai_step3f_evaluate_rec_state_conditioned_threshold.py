from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hai import build_hai_dataloaders  # noqa: E402
from scripts.hai_step3e_evaluate_rec_causal_alarm_policy import (  # noqa: E402
    extract_events, load_seed_scores, metric_record, sha256_file,
)
from utils.causal_alarm_policy import binary_segments  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.normal_operating_state import (  # noqa: E402
    active_feature_indices, clustering_agreement, endpoint_rows, normalized_entropy,
    pointwise_state_alarm, robust_location_scale, robust_z, run_lengths,
)

OUTPUT_FILES = (
    "config_resolved.json", "state_threshold_input_consistency_audit.csv", "normal_state_feature_audit.csv",
    "normal_state_pca_audit.csv", "normal_state_pca_loadings.csv", "normal_state_k_selection.csv",
    "normal_state_clustering_stability.csv", "normal_state_assignment_summary.csv", "rec_state_thresholds.csv",
    "rec_state_validation_diagnostics.csv", "rec_state_point_metrics_by_seed.csv",
    "rec_state_point_metrics_by_file.csv", "rec_state_point_metrics_aggregate.csv",
    "rec_state_event_diagnostics.csv", "rec_state_event_summary.csv", "rec_state_event_transition.csv",
    "rec_state_false_positive_segments.csv", "rec_state_false_positive_segment_summary.csv",
    "rec_state_baseline_comparison.csv", "rec_state_multiseed_summary.csv", "rec_state_summary.json",
    "final_hai_rec_state_conditioned_threshold_diagnosis.md",
)
METHODS = (
    "pooled_q99_baseline", "pooled_q995_baseline", "state_validation_q99",
    "state_validation_q995", "state_train_robust_z_q99", "state_train_robust_z_q995",
)


def digest_tree(path: Path) -> dict[str, str]:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p)
            for p in sorted(path.glob("*")) if p.is_file()}


def ensure_fresh(output: Path) -> None:
    existing = [name for name in OUTPUT_FILES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite HAI Step 3F outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.replace([np.inf, -np.inf], np.nan).fillna("not_applicable").to_csv(path, index=False)


def sequence_map(bundle) -> dict[str, dict[str, Any]]:
    return {
        "train": {s.file_name: s for s in bundle.train_sequences},
        "validation": {s.file_name: s for s in bundle.val_sequences},
        "test": {s.file_name: s for s in bundle.test_sequences},
    }


def aligned_features(scores: dict[str, Any], bundle, active: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sequences = sequence_map(bundle); result: dict[str, Any] = {split: {} for split in sequences}; audit = []
    for split, files in scores.items():
        for name, values in files.items():
            seq = sequences[split][name]
            x = endpoint_rows(seq, values["end_row"])[:, active].astype(np.float32, copy=False)
            expected_ts = seq.timestamps[np.asarray(values["end_row"], np.int64) - seq.original_row_offset]
            ok = np.array_equal(expected_ts.astype(str), values["timestamp"].astype(str))
            if not ok:
                raise ValueError(f"Timestamp/end_row alignment failed for {split}/{name}")
            result[split][name] = x
            audit.append({"split": split, "file_name": name, "point_count": len(x),
                          "endpoint_alignment_exact": ok, "uses_current_endpoint_only": True,
                          "uses_label_or_score_as_feature": False, "uses_file_identity_as_feature": False})
    return result, audit


def fit_pca(train_x: np.ndarray, cfg: dict) -> tuple[PCA, int, pd.DataFrame, pd.DataFrame]:
    maximum = min(int(cfg["pca_max_components"]), train_x.shape[1])
    full = PCA(n_components=maximum, whiten=False, svd_solver="randomized", random_state=1).fit(train_x)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    hits = np.flatnonzero(cumulative >= float(cfg["pca_variance_target"]))
    count = int(hits[0] + 1) if len(hits) else maximum
    count = max(int(cfg["pca_min_components"]), min(count, maximum))
    pca = PCA(n_components=count, whiten=False, svd_solver="randomized", random_state=1).fit(train_x)
    audit = pd.DataFrame([{"component_count": count, "cumulative_explained_variance": float(
        pca.explained_variance_ratio_.sum()), "target": cfg["pca_variance_target"], "target_met": bool(
        pca.explained_variance_ratio_.sum() >= float(cfg["pca_variance_target"]) - 1e-6),
        "maximum_components": maximum, "whiten": False, "fit_scope": "normal_train_only",
        "validation_used_for_dimension_selection": False, "test_used_for_dimension_selection": False,
        **{f"pc{i+1}_explained_variance_ratio": float(v) for i, v in enumerate(pca.explained_variance_ratio_)}}])
    return pca, count, audit, pd.DataFrame(pca.components_)


def deterministic_sample(n: int, maximum: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, maximum), dtype=np.int64))


def fit_cluster(x: np.ndarray, k: int, seed: int, cfg: dict) -> MiniBatchKMeans:
    return MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=int(cfg["kmeans_n_init"]),
                           max_iter=int(cfg["kmeans_max_iter"]), batch_size=int(cfg["kmeans_batch_size"]),
                           reassignment_ratio=0.0).fit(x)


def k_diagnostics(train_z: np.ndarray, train_files: np.ndarray, cfg: dict
                  ) -> tuple[MiniBatchKMeans, pd.DataFrame, pd.DataFrame]:
    sample = deterministic_sample(len(train_z), int(cfg["silhouette_sample_count"])); rows = []; models = {}
    for k in cfg["k_candidates"]:
        model = fit_cluster(train_z, int(k), 1, cfg); labels = model.labels_; models[int(k)] = model
        counts = np.bincount(labels, minlength=int(k)); coverage = [] ; dominated = []
        for state in range(int(k)):
            file_counts = pd.Series(train_files[labels == state]).value_counts()
            coverage.append(int(len(file_counts)))
            dominated.append(float(file_counts.iloc[0] / counts[state]) if counts[state] else 1.0)
        lengths = [run_lengths(labels[train_files == name]) for name in np.unique(train_files)]
        valid = bool(counts.min() >= int(cfg["minimum_cluster_count"]) and
                     counts.min() / len(labels) >= float(cfg["minimum_cluster_ratio"]) and
                     min(coverage) >= 2 and np.isfinite(model.cluster_centers_).all())
        rows.append({"k": int(k), "train_silhouette": float(silhouette_score(train_z[sample], labels[sample])),
                     "calinski_harabasz": float(calinski_harabasz_score(train_z[sample], labels[sample])),
                     "davies_bouldin": float(davies_bouldin_score(train_z[sample], labels[sample])),
                     "diagnostic_sample_count": len(sample), "minimum_cluster_count": int(counts.min()),
                     "minimum_cluster_ratio": float(counts.min()/len(labels)), "cluster_entropy": normalized_entropy(counts),
                     "minimum_train_sequence_coverage": min(coverage), "maximum_single_file_domination": max(dominated),
                     "single_file_fully_dominates": bool(max(dominated) >= .999),
                     "minimum_observed_segment_length": int(min(np.min(v) for v in lengths if len(v))),
                     "valid_k": valid, "selection_uses_test": False})
    frame = pd.DataFrame(rows); valid = frame[frame.valid_k]
    if valid.empty:
        raise ValueError("No valid K among the pre-registered candidates")
    chosen = int(valid.sort_values("train_silhouette", ascending=False).iloc[0].k)
    frame["selected"] = frame.k == chosen
    formal = models[chosen]; stability = []
    for seed in cfg["stability_random_states"]:
        model = formal if int(seed) == 1 else fit_cluster(train_z, chosen, int(seed), cfg)
        agreement = clustering_agreement(formal.labels_, formal.cluster_centers_, model.labels_, model.cluster_centers_)
        stability.append({"selected_k": chosen, "formal_random_state": 1, "comparison_random_state": int(seed), **agreement})
    return formal, frame, pd.DataFrame(stability)


def assign_all(features: dict[str, Any], pca: PCA, model: MiniBatchKMeans,
               train_distances: np.ndarray) -> dict[str, Any]:
    sorted_train = np.sort(train_distances); result = {split: {} for split in features}
    for split, files in features.items():
        for name, x in files.items():
            z = pca.transform(x); distances = model.transform(z); order = np.argsort(distances, axis=1)
            nearest = order[:, 0].astype(np.int16); d1 = distances[np.arange(len(x)), nearest]
            d2 = distances[np.arange(len(x)), order[:, 1]]
            result[split][name] = {"state": nearest, "distance": d1.astype(np.float32),
                "second_distance": d2.astype(np.float32), "margin": (d2-d1).astype(np.float32),
                "train_distance_percentile": (np.searchsorted(sorted_train, d1, side="right") /
                                                len(sorted_train)).astype(np.float32)}
    return result


def assignment_summary(assignments: dict[str, Any], k: int, train_q99: float) -> pd.DataFrame:
    rows = []
    for split, files in assignments.items():
        for name, values in files.items():
            states = values["state"]
            for state in range(k):
                mask = states == state; lengths = run_lengths(states)[()] if False else run_lengths(states)
                rows.append({"split": split, "file_name": name, "state_id": state, "sample_count": int(mask.sum()),
                             "state_ratio": float(mask.mean()), "state_switch_count": int(np.sum(states[1:] != states[:-1])),
                             "median_state_duration": float(np.median(lengths)),
                             "distance_q95": float(np.quantile(values["distance"][mask], .95)) if mask.any() else np.nan,
                             "distance_q99": float(np.quantile(values["distance"][mask], .99)) if mask.any() else np.nan,
                             "ratio_above_train_distance_q99": float(np.mean(values["distance"] > train_q99)),
                             "test_used_for_fit": False})
    return pd.DataFrame(rows)


def pooled_threshold(step3d: pd.DataFrame, seed: int, q: str) -> float:
    method = f"normal_validation_{q}"
    row = step3d[(step3d.seed == seed) & (step3d.score_variant == "endpoint_raw_top5") &
                 (step3d.threshold_method == method)]
    if len(row) != 1 or row.iloc[0].status != "ok":
        raise ValueError(f"Missing Step 3D threshold seed={seed}/{method}")
    return float(row.iloc[0].threshold_value)


def fit_thresholds(seed_scores: dict[str, Any], assignments: dict[str, Any], step3d: pd.DataFrame,
                   seed: int, k: int, cfg: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_score = np.concatenate([seed_scores["train"][n]["score"] for n in sorted(seed_scores["train"])])
    val_score = np.concatenate([seed_scores["validation"][n]["score"] for n in sorted(seed_scores["validation"])])
    train_state = np.concatenate([assignments["train"][n]["state"] for n in sorted(assignments["train"])])
    val_state = np.concatenate([assignments["validation"][n]["state"] for n in sorted(assignments["validation"])])
    pooled99, pooled995 = pooled_threshold(step3d, seed, "q99"), pooled_threshold(step3d, seed, "q995")
    state99 = np.zeros(k); state995 = np.zeros(k); medians = np.zeros(k); scales = np.zeros(k); rows=[]
    for state in range(k):
        tr = train_score[train_state == state]; va = val_score[val_state == state]
        median, scale, source, mad, iqr, std = robust_location_scale(tr); medians[state]=median; scales[state]=scale
        fallback = len(va) < int(cfg["minimum_validation_state_count"])
        state99[state] = pooled99 if fallback else np.quantile(va, .99)
        state995[state] = pooled995 if fallback else np.quantile(va, .995)
        common = {"seed": seed, "state_id": state, "train_count": len(tr), "validation_count": len(va),
                  "score_mean": float(tr.mean()), "score_median": median, "score_std": std, "score_mad": mad,
                  "score_iqr": iqr, "effective_robust_scale": scale, "robust_scale_source": source,
                  "q95": float(np.quantile(va, .95)) if len(va) else np.nan,
                  "q99": float(np.quantile(va, .99)) if len(va) else np.nan,
                  "q995": float(np.quantile(va, .995)) if len(va) else np.nan,
                  "fallback_used": fallback, "fallback_reason": "validation_count_below_2000" if fallback else "none"}
        rows.extend([{**common, "threshold_method": "state_validation_q99", "effective_threshold": state99[state]},
                     {**common, "threshold_method": "state_validation_q995", "effective_threshold": state995[state]}])
    val_z = np.concatenate([robust_z(seed_scores["validation"][n]["score"], assignments["validation"][n]["state"],
                                    medians, scales) for n in sorted(seed_scores["validation"])])
    z99, z995 = float(np.quantile(val_z, .99)), float(np.quantile(val_z, .995))
    for state in range(k):
        common = {"seed": seed, "state_id": state, "train_count": int((train_state==state).sum()),
                  "validation_count": int((val_state==state).sum()), "score_mean": float(train_score[train_state==state].mean()),
                  "score_median": medians[state], "score_std": float(train_score[train_state==state].std()),
                  "score_mad": float(np.median(np.abs(train_score[train_state==state]-medians[state]))),
                  "effective_robust_scale": scales[state], "fallback_used": False, "fallback_reason": "none"}
        rows.extend([{**common, "threshold_method": "state_train_robust_z_q99", "effective_threshold": z99},
                     {**common, "threshold_method": "state_train_robust_z_q995", "effective_threshold": z995}])
    fitted = {"pooled_q99_baseline": pooled99, "pooled_q995_baseline": pooled995,
              "state_validation_q99": state99, "state_validation_q995": state995,
              "state_train_robust_z_q99": (medians, scales, z99),
              "state_train_robust_z_q995": (medians, scales, z995)}
    return pd.DataFrame(rows), fitted


def alarms_for(scores: dict[str, Any], assignments: dict[str, Any], fitted: dict[str, Any]) -> dict[str, Any]:
    out = {method: {split: {} for split in scores} for method in METHODS}
    for method in METHODS:
        for split, files in scores.items():
            for name, values in files.items():
                states = assignments[split][name]["state"]
                if method.startswith("pooled"):
                    alarm = (values["score"] > float(fitted[method])).astype(np.int8)
                elif method.startswith("state_validation"):
                    alarm = pointwise_state_alarm(values["score"], states, fitted[method])
                else:
                    medians, scales, threshold = fitted[method]
                    alarm = (robust_z(values["score"], states, medians, scales) > threshold).astype(np.int8)
                out[method][split][name] = alarm
    return out


def fp_stats(labels: np.ndarray, alarm: np.ndarray) -> dict[str, Any]:
    fp = ((labels == 0) & (alarm == 1)).astype(np.int8); starts, ends = binary_segments(fp); lengths=ends-starts+1
    return {"longest_false_positive_segment": int(lengths.max()) if len(lengths) else 0,
            "continuous_fp_segments_ge_30": int((lengths >= 30).sum()), "false_positive_segment_count": len(starts)}


def point_metrics(all_scores: dict[int, Any], all_alarms: dict[int, Any]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    files=[]; seeds=[]
    for seed, scores in all_scores.items():
        for method in METHODS:
            current=[]
            for name, values in scores["test"].items():
                alarm=all_alarms[seed][method]["test"][name]
                row={"seed":seed,"threshold_method":method,"file_name":name,
                     **metric_record(values["label"],alarm,values["score"]),**fp_stats(values["label"],alarm)}
                files.append(row); current.append(row)
            labels=np.concatenate([scores["test"][n]["label"] for n in sorted(scores["test"])])
            alarm=np.concatenate([all_alarms[seed][method]["test"][n] for n in sorted(scores["test"])])
            score=np.concatenate([scores["test"][n]["score"] for n in sorted(scores["test"])])
            seeds.append({"seed":seed,"threshold_method":method,"scope":"micro",**metric_record(labels,alarm,score),
                          **fp_stats(labels,alarm)})
            seeds.append({"seed":seed,"threshold_method":method,"scope":"macro",
                          **{m:float(np.mean([r[m] for r in current])) for m in ("precision","recall","f1","normal_fpr","reference_aupr","reference_auroc")},
                          **{m:int(sum(r[m] for r in current)) for m in ("true_positive_count","false_positive_count","false_negative_count","continuous_fp_segments_ge_30")},
                          "ranking_score_unchanged":True})
    sf=pd.DataFrame(seeds); aggregate=[]; multi=[]
    for (method,scope),g in sf.groupby(["threshold_method","scope"]):
        row={"threshold_method":method,"scope":scope}
        for m in ("precision","recall","f1","false_positive_count","continuous_fp_segments_ge_30","reference_aupr","reference_auroc"):
            row[f"{m}_mean"]=float(g[m].mean()); row[f"{m}_std"]=float(g[m].std(ddof=0))
            if m in ("precision","recall","f1"):
                aggregate.append({"threshold_method":method,"scope":scope,"metric":m,"mean":row[f"{m}_mean"],"std":row[f"{m}_std"],
                                  **{f"seed{int(r.seed)}":float(getattr(r,m)) for r in g.itertuples()}})
        multi.append(row)
    return pd.DataFrame(files),sf,pd.DataFrame(aggregate),pd.DataFrame(multi)


def validation_metrics(all_scores, all_alarms, assignments) -> pd.DataFrame:
    rows=[]
    for seed,scores in all_scores.items():
        for method in METHODS:
            for name,v in scores["validation"].items():
                alarm=all_alarms[seed][method]["validation"][name]; starts,ends=binary_segments(alarm); lengths=ends-starts+1
                for state in range(int(assignments["validation"][name]["state"].max())+1):
                    mask=assignments["validation"][name]["state"]==state
                    rows.append({"seed":seed,"threshold_method":method,"file_name":name,"state_id":state,
                                 "normal_point_count":int(mask.sum()),"false_positive_count":int(alarm[mask].sum()),
                                 "normal_fpr":float(alarm[mask].mean()) if mask.any() else np.nan,
                                 "alarm_segment_count":len(starts),"longest_alarm_segment":int(lengths.max()) if len(lengths) else 0,
                                 "median_segment_length":float(np.median(lengths)) if len(lengths) else 0.0,
                                 "continuous_segments_ge_30":int((lengths>=30).sum()),"alarms_per_hour":float(alarm.sum()*3600/len(alarm))})
    frame=pd.DataFrame(rows); base=frame[frame.threshold_method=="pooled_q99_baseline"].set_index(["seed","file_name","state_id"]).false_positive_count
    frame["reduction_vs_pooled_q99"]=[float((base.get((r.seed,r.file_name,r.state_id),0)-r.false_positive_count)/max(base.get((r.seed,r.file_name,r.state_id),0),1)) for r in frame.itertuples()]
    frame["worst_state"] = frame.groupby(["seed","threshold_method","file_name"]).normal_fpr.transform("max") == frame.normal_fpr
    frame["worst_sequence"] = frame.groupby(["seed","threshold_method","state_id"]).normal_fpr.transform("max") == frame.normal_fpr
    return frame


def event_outputs(all_scores, all_alarms, assignments, smoke: bool) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rows=[]
    for seed,scores in all_scores.items():
        events=extract_events(scores["test"])
        if not smoke and len(events)!=58: raise ValueError(f"Expected 58 events, got {len(events)}")
        for method in METHODS:
            for event in events:
                v=scores["test"][event["file_name"]]; alarm=all_alarms[seed][method]["test"][event["file_name"]]
                states=assignments["test"][event["file_name"]]["state"]; a,b=event["start_index"],event["end_index"]
                local=alarm[a:b+1]; hit=np.flatnonzero(local); before=states[max(0,a-30):a]
                rows.append({"seed":seed,"threshold_method":method,**event,
                             "dominant_state_before_event":int(np.bincount(before).argmax()) if len(before) else "not_available",
                             "dominant_state_in_event":int(np.bincount(states[a:b+1]).argmax()),"detected":bool(len(hit)),
                             "first_alarm_timestamp":str(v["timestamp"][a+hit[0]]) if len(hit) else "not_detected",
                             "detection_delay_points":int(hit[0]) if len(hit) else "not_detected",
                             "point_recall_in_event":float(local.mean()),"first_point_hit":bool(len(hit) and hit[0]==0),
                             "within5_hit":bool(len(hit) and hit[0]<5),"within10_hit":bool(len(hit) and hit[0]<10),
                             "within30_hit":bool(len(hit) and hit[0]<30)})
    diag=pd.DataFrame(rows); summary=[]
    for (seed,method),g in diag.groupby(["seed","threshold_method"]):
        detected=g[g.detected]; delays=pd.to_numeric(detected.detection_delay_points,errors="coerce")
        summary.append({"seed":seed,"threshold_method":method,"event_count":len(g),"detected_event_count":int(g.detected.sum()),
                        "event_recall":float(g.detected.mean()),"fully_missed_event_count":int((~g.detected).sum()),
                        "median_delay":float(delays.median()) if len(delays) else np.nan,"mean_delay":float(delays.mean()) if len(delays) else np.nan,
                        "within5_hit_rate":float(g.within5_hit.mean()),"within10_hit_rate":float(g.within10_hit.mean()),
                        "within30_hit_rate":float(g.within30_hit.mean())})
    summ=pd.DataFrame(summary); trans=[]
    idx=diag.set_index(["seed","threshold_method","event_id"])
    for seed in all_scores:
        for method in METHODS:
            baseline="pooled_q995_baseline" if "q995" in method else "pooled_q99_baseline"
            for event in extract_events(all_scores[seed]["test"]):
                old=idx.loc[(seed,baseline,event["event_id"])]; new=idx.loc[(seed,method,event["event_id"])]
                trans.append({"seed":seed,"threshold_method":method,"baseline_method":baseline,"event_id":event["event_id"],
                              "file_name":event["file_name"],"baseline_detected":old.detected,"new_detected":new.detected,
                              "recovered":bool(not old.detected and new.detected),"lost":bool(old.detected and not new.detected)})
    return diag,summ,pd.DataFrame(trans)


def fp_segments(all_scores, all_alarms, assignments) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]; summaries=[]
    for seed,scores in all_scores.items():
        for name,v in scores["test"].items():
            base=((v["label"]==0)&(all_alarms[seed]["pooled_q99_baseline"]["test"][name]==1)).astype(np.int8)
            new=((v["label"]==0)&(all_alarms[seed]["state_validation_q99"]["test"][name]==1)).astype(np.int8)
            bs,be=binary_segments(base); ns,ne=binary_segments(new)
            for method,starts,ends in (("pooled_q99_baseline",bs,be),("state_validation_q99",ns,ne)):
                for sid,(a,b) in enumerate(zip(starts,ends),1):
                    st=assignments["test"][name]["state"][a:b+1]; overlap=np.maximum(0,np.minimum(be,b)-np.maximum(bs,a)+1)
                    old_len=int((be-bs+1)[np.argmax(overlap)]) if len(overlap) and overlap.max()>0 else 0
                    rows.append({"seed":seed,"threshold_method":method,"file_name":name,"false_positive_segment_id":sid,
                                 "start_timestamp":v["timestamp"][a],"end_timestamp":v["timestamp"][b],"length":int(b-a+1),
                                 "dominant_state":int(np.bincount(st).argmax()),"state_switch_count":int(np.sum(st[1:]!=st[:-1])),
                                 "peak_score":float(v["score"][a:b+1].max()),"assigned_threshold":"state_specific_or_pooled",
                                 "eliminated_by_state_threshold":bool(method=="pooled_q99_baseline" and overlap.max(initial=0)==0),
                                 "shortened_length":max(0,old_len-(b-a+1)) if method=="state_validation_q99" else 0,
                                 "unchanged_or_remaining":method=="state_validation_q99"})
            summaries.extend([{"seed":seed,"file_name":name,"threshold_method":"pooled_q99_baseline","segment_count":len(bs),
                               "segments_ge_30":int(((be-bs+1)>=30).sum()),"longest_segment":int((be-bs+1).max()) if len(bs) else 0},
                              {"seed":seed,"file_name":name,"threshold_method":"state_validation_q99","segment_count":len(ns),
                               "segments_ge_30":int(((ne-ns+1)>=30).sum()),"longest_segment":int((ne-ns+1).max()) if len(ns) else 0}])
    return pd.DataFrame(rows),pd.DataFrame(summaries)


def comparisons(seed_metrics, event_summary, fp_summary) -> pd.DataFrame:
    rows=[]; micro=seed_metrics[seed_metrics.scope=="micro"].set_index(["seed","threshold_method"])
    events=event_summary.set_index(["seed","threshold_method"])
    for seed in sorted(seed_metrics.seed.unique()):
        for method in ("state_validation_q99","state_validation_q995","state_train_robust_z_q99","state_train_robust_z_q995"):
            base="pooled_q995_baseline" if "q995" in method else "pooled_q99_baseline"; a=micro.loc[(seed,base)]; b=micro.loc[(seed,method)]
            ea=events.loc[(seed,base)]; eb=events.loc[(seed,method)]
            la=fp_summary[(fp_summary.seed==seed)&(fp_summary.threshold_method==base)].segments_ge_30.sum()
            lb=fp_summary[(fp_summary.seed==seed)&(fp_summary.threshold_method==method)].segments_ge_30.sum()
            rows.append({"seed":seed,"threshold_method":method,"comparison_baseline":base,
                         "precision_delta":b.precision-a.precision,"precision_relative_change":(b.precision-a.precision)/max(a.precision,1e-12),
                         "recall_delta":b.recall-a.recall,"f1_delta":b.f1-a.f1,
                         "false_positive_delta":b.false_positive_count-a.false_positive_count,
                         "false_positive_relative_change":(b.false_positive_count-a.false_positive_count)/max(a.false_positive_count,1),
                         "event_recall_delta":eb.event_recall-ea.event_recall,"median_delay_delta":eb.median_delay-ea.median_delay,
                         "long_fp_segment_delta":int(lb-la),"robust_changes_normalization_and_threshold_space":method.startswith("state_train_robust")})
    return pd.DataFrame(rows)


def classify(comparison: pd.DataFrame, stability: pd.DataFrame, seed_metrics: pd.DataFrame,
             event_summary: pd.DataFrame, fp_summary: pd.DataFrame) -> tuple[str,dict[str,bool]]:
    p=comparison[comparison.threshold_method=="state_validation_q99"]
    stable=bool((stability.adjusted_rand_index.iloc[1:]>=.8).all())
    gates={"precision":float(p.precision_relative_change.mean())>=.2 and int((p.precision_delta>0).sum())>=2,
           "f1":float(p.f1_delta.mean())>=.03 and int((p.f1_delta>0).sum())>=2,
           "false_positive":float(p.false_positive_relative_change.mean())<=-.25,
           "long_false_positive":float(p.long_fp_segment_delta.mean())<=-0.3*max(1,float((p.long_fp_segment_delta-p.long_fp_segment_delta).mean()+1)),
           "event":float(p.event_recall_delta.mean())>=-.05 and float(p.median_delay_delta.mean())<=10,
           "clustering_stable":stable}
    if not stable: return "state_model_is_unstable",gates
    robust=comparison[comparison.threshold_method=="state_train_robust_z_q99"]
    if robust.f1_delta.mean()>p.f1_delta.mean()+.03 and robust.event_recall_delta.mean()>=p.event_recall_delta.mean():
        return "robust_state_normalization_is_better_than_state_thresholds",gates
    if all(gates.values()): return "state_conditioned_threshold_makes_rec_viable",gates
    if p.false_positive_relative_change.mean()<0: return "state_conditioning_reduces_false_positives_but_rec_still_weak",gates
    return "normal_state_heterogeneity_is_not_main_rec_problem",gates


def run(cfg: dict, output: Path, smoke: bool) -> dict[str, Any]:
    ensure_fresh(output); before={**digest_tree(ROOT/cfg["step3d_output_dir"]),**digest_tree(ROOT/cfg["step3e_output_dir"])}
    maximum=int(cfg["smoke_max_points_per_sequence"]) if smoke else None
    all_scores={seed:load_seed_scores(ROOT/cfg["score_npz_pattern"].format(seed=seed),maximum) for seed in cfg["seeds"]}
    reference=all_scores[cfg["seeds"][0]]
    bundle=build_hai_dataloaders(cfg["data_root"],window_size=int(cfg["window_size"]),validation_ratio=float(cfg["validation_ratio"]),
                                 batch_size=128,num_workers=0,return_metadata=True,max_rows_per_file=None)
    active=active_feature_indices(bundle.scaler.raw_std_,bundle.scaler.eps)
    features,audit=aligned_features(reference,bundle,active)
    for seed in cfg["seeds"][1:]:
        for split in reference:
            for name in reference[split]:
                for field in ("label","timestamp","end_row","sequence_id"):
                    if not np.array_equal(reference[split][name][field],all_scores[seed][split][name][field]):
                        raise ValueError(f"Cross-seed metadata mismatch {seed}/{split}/{name}/{field}")
    train_names=sorted(features["train"]); train_x=np.concatenate([features["train"][n] for n in train_names])
    train_files=np.concatenate([np.repeat(n,len(features["train"][n])) for n in train_names])
    pca,ncomp,pca_audit,raw_loadings=fit_pca(train_x,cfg); train_z=pca.transform(train_x)
    model,k_frame,stability=k_diagnostics(train_z,train_files,cfg); k=model.n_clusters
    train_dist=model.transform(train_z).min(axis=1); assignments=assign_all(features,pca,model,train_dist)
    all_thresholds=[]; all_alarms={}
    step3d=pd.read_csv(ROOT/cfg["step3d_threshold_file"])
    for seed in cfg["seeds"]:
        frame,fitted=fit_thresholds(all_scores[seed],assignments,step3d,seed,k,cfg); all_thresholds.append(frame)
        all_alarms[seed]=alarms_for(all_scores[seed],assignments,fitted)
    val=validation_metrics(all_scores,all_alarms,assignments)
    by_file,by_seed,aggregate,multiseed=point_metrics(all_scores,all_alarms)
    event_diag,event_summary,event_transition=event_outputs(all_scores,all_alarms,assignments,smoke)
    fp,fp_summary=fp_segments(all_scores,all_alarms,assignments)
    comparison=comparisons(by_seed,event_summary,fp_summary)
    classification,gates=classify(comparison,stability,by_seed,event_summary,fp_summary)
    excluded=[bundle.feature_names[i] for i in range(len(bundle.feature_names)) if i not in set(active)]
    feature_audit=pd.DataFrame([{"original_feature_count":len(bundle.feature_names),"active_state_feature_count":len(active),
        "excluded_zero_variance_feature_count":len(excluded),"excluded_zero_variance_features":"|".join(excluded),
        "scaler_source":"data.hai.build_hai_dataloaders / normal train prefixes only",
        "train_row_count":sum(len(v) for v in features["train"].values()),"validation_row_count":sum(len(v) for v in features["validation"].values()),
        "test_row_count":sum(len(v) for v in features["test"].values()),"test_used_for_fit":False}])
    load=[]
    for pc,row in enumerate(pca.components_,1):
        for rank,idx in enumerate(np.argsort(np.abs(row))[::-1],1):
            load.append({"component":pc,"feature":bundle.feature_names[active[idx]],"loading":float(row[idx]),"absolute_loading":float(abs(row[idx])),"loading_rank":rank})
    after={**digest_tree(ROOT/cfg["step3d_output_dir"]),**digest_tree(ROOT/cfg["step3e_output_dir"])}
    hash_rows=[{"source_file":name,"sha256_before":digest,"sha256_after":after.get(name),"unchanged":after.get(name)==digest} for name,digest in before.items()]
    input_audit=pd.concat([pd.DataFrame(audit),pd.DataFrame(hash_rows)],ignore_index=True,sort=False)
    outputs={"state_threshold_input_consistency_audit.csv":input_audit,"normal_state_feature_audit.csv":feature_audit,
             "normal_state_pca_audit.csv":pca_audit,"normal_state_pca_loadings.csv":pd.DataFrame(load),
             "normal_state_k_selection.csv":k_frame,"normal_state_clustering_stability.csv":stability,
             "normal_state_assignment_summary.csv":assignment_summary(assignments,k,float(np.quantile(train_dist,.99))),
             "rec_state_thresholds.csv":pd.concat(all_thresholds,ignore_index=True),"rec_state_validation_diagnostics.csv":val,
             "rec_state_point_metrics_by_seed.csv":by_seed,"rec_state_point_metrics_by_file.csv":by_file,
             "rec_state_point_metrics_aggregate.csv":aggregate,"rec_state_event_diagnostics.csv":event_diag,
             "rec_state_event_summary.csv":event_summary,"rec_state_event_transition.csv":event_transition,
             "rec_state_false_positive_segments.csv":fp,"rec_state_false_positive_segment_summary.csv":fp_summary,
             "rec_state_baseline_comparison.csv":comparison,"rec_state_multiseed_summary.csv":multiseed}
    for name,frame in outputs.items(): write_csv(frame,output/name)
    resolved={**cfg,"mode":"smoke" if smoke else "formal","selected_k":k,"pca_component_count":ncomp,
              "model_or_checkpoint_loaded":False,"inference_rerun":False,"test_used_for_fit":False}
    (output/"config_resolved.json").write_text(json.dumps(resolved,indent=2,ensure_ascii=True),encoding="utf-8")
    summary={"classification":classification,"gates":gates,"selected_k":k,"pca_component_count":ncomp,
             "active_state_feature_count":len(active),"excluded_zero_variance_feature_count":len(excluded),
             "event_count_per_seed":len(extract_events(reference["test"])),"ranking_score_unchanged":True,
             "model_or_checkpoint_loaded":False,"inference_rerun":False,"source_hashes_unchanged":before==after}
    (output/"rec_state_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=True),encoding="utf-8")
    primary=comparison[comparison.threshold_method=="state_validation_q99"]
    md=f"""# HAI Step 3F: State-conditioned reconstruction threshold diagnosis

## Scope and validity

- Fixed score: `endpoint_raw_top5`; pointwise current-state threshold only.
- No model/checkpoint load, retraining, inference, score modification, future information, or test fitting.
- PCA and K selection use only six normal training prefixes. Test labels are evaluation-only.

## State model

- Active features: {len(active)}/86; excluded global zero-variance features: {len(excluded)}.
- PCA components: {ncomp}; retained variance: {pca.explained_variance_ratio_.sum():.6f}.
- Selected K: {k}, highest valid train-only silhouette among K={{2,4,6}}.
- Stability ARI minimum: {stability.adjusted_rand_index.min():.6f}.

## Primary comparison

- Mean precision relative change: {primary.precision_relative_change.mean():.6f}.
- Mean F1 delta: {primary.f1_delta.mean():.6f}.
- Mean false-positive relative change: {primary.false_positive_relative_change.mean():.6f}.
- Mean event-recall delta: {primary.event_recall_delta.mean():.6f}.

## Decision

**{classification}**

Gate results: `{json.dumps(gates, ensure_ascii=True)}`.
"""
    (output/"final_hai_rec_state_conditioned_threshold_diagnosis.md").write_text(md,encoding="utf-8")
    return summary


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/hai_rec_state_conditioned_threshold.yaml")
    parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    cfg=load_config(ROOT/args.config)
    required=[Path(cfg["data_root"]),ROOT/cfg["step3d_threshold_file"],ROOT/cfg["step3d_output_dir"],ROOT/cfg["step3e_output_dir"]]
    required += [ROOT/cfg["score_npz_pattern"].format(seed=s) for s in cfg["seeds"]]
    if args.dry_run:
        print("HAI Step 3F dry-run: no files will be written")
        for path in required: print(f"{'OK' if path.exists() else 'MISSING'}: {path}")
        print("fit=Step2 scaler -> active endpoint features -> train-only PCA -> train-only K in {2,4,6}")
        print("evaluation=pointwise endpoint_raw_top5 only; no model/checkpoint/inference")
        return 0 if all(p.exists() for p in required) else 2
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError(missing)
    output=ROOT/(cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
    summary=run(cfg,output,args.smoke); print(json.dumps(summary,indent=2)); print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
