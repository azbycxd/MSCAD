"""RETROSPECTIVE_VALIDITY_AUDIT: post-freeze residual controls on replayed endpoints."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import real_helper

TAG = "RETROSPECTIVE_VALIDITY_AUDIT"


def _reference(values):
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, max(1.4826 * mad, 1e-6)


def _normalized(values, ref):
    result = np.maximum(0., (values - ref[0]) / ref[2])
    if not np.isfinite(result).all():
        raise RuntimeError("NONFINITE_CONTROL_SCORE")
    return result


def _write_frame(frame, stem, outputs, fallbacks):
    if importlib.util.find_spec("pyarrow") is not None:
        path = stem.with_suffix(".parquet")
        frame.to_parquet(path, index=False, engine="pyarrow")
    else:
        path = stem.with_suffix(".csv.gz")
        frame.to_csv(path, index=False, compression="gzip")
        fallbacks.append({"artifact": path.name, "reason": "pyarrow unavailable; used explicitly reported gzip CSV with identical endpoint bindings"})
    outputs.append(str(path))


def evaluate(prepared, replay, normal_freeze_sha, output_dir, stats_state=None):
    """Called only after exact S3 replay and recorded Normal-only SHA freeze."""
    if not replay.get("passed"):
        raise RuntimeError("STOP_B: real S3 replay did not pass")
    if not isinstance(normal_freeze_sha, str) or len(normal_freeze_sha) != 64:
        raise RuntimeError("NORMAL_STATISTICS_NOT_HASH_FROZEN")
    out = Path(output_dir)
    if not out.is_dir():
        raise FileNotFoundError(out)
    dataset = prepared["dataset"]
    all_targets = prepared["all_targets"]
    main_targets = prepared["targets"]
    skill_targets = prepared["skill_targets"]
    relevant = [target for target in all_targets if target in set(main_targets) | set(skill_targets)]
    prediction = replay["prediction"]
    binding = replay["binding"].copy()
    binding["audit_label"] = TAG
    binding["valid_mask"] = binding["valid_endpoint"].astype(bool)
    if not binding["valid_mask"].all():
        raise RuntimeError("Unexpected invalid endpoint in formally bound replay")
    labels = binding["raw_label"].to_numpy(np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("REAL_LABEL_CONTRACT_MISMATCH")
    n = len(binding)
    ncal = sum(len(v["local_index"]) for v in prepared["val_predictions"])
    global_test = {score: np.zeros(n, dtype=np.float64) for score in ("S1", "S2")}
    global_cal = {score: np.zeros(ncal, dtype=np.float64) for score in ("S1", "S2")}
    skill_test = {score: np.zeros(n, dtype=np.float64) for score in ("S2", "S3")}
    skill_cal = {score: np.zeros(ncal, dtype=np.float64) for score in ("S2", "S3")}
    refs, pertarget, wide = [], [], {column: binding[column].to_numpy() for column in binding}
    score_notes = "Per-target AUROC uses prescribed target-specific Normal median/MAD calibration and nonnegative truncation; raw scores are separately persisted."
    for target in relevant:
        j = all_targets.index(target)
        yval = np.concatenate([v["z"][:, j] for v in prepared["val_predictions"]])
        psval = np.concatenate([v["ps"][:, j] for v in prepared["val_predictions"]])
        pcval = np.concatenate([v["pc"][:, j] for v in prepared["val_predictions"]])
        rsval, rcval = yval - psval, yval - pcval
        covariance = float(np.mean((rcval - rcval.mean()) * (rsval - rsval.mean())))
        kappa = covariance - .5 * float(np.var(rsval))
        calraw = {"S1": np.abs(rsval), "S2": np.abs(rcval),
                  "S3": prepared["full_disagreements"][target]}
        test_raw = {"S1": np.abs(prediction["z"][:, j] - prediction["ps"][:, j]),
                    "S2": np.abs(prediction["z"][:, j] - prediction["pc"][:, j]),
                    "S3": np.abs(prediction["ps"][:, j] - prediction["pc"][:, j])}
        target_row = {"audit_label": TAG, "dataset": dataset, "target": target, "kappa": kappa,
                      "main_frozen_target": target in main_targets, "skill_only_target": target in skill_targets,
                      "normal_stats_sha256": normal_freeze_sha, "score_semantics": score_notes}
        for score in ("S1", "S2", "S3"):
            ref = _reference(calraw[score])
            if score == "S3" and target in main_targets:
                official_median, official_scale = prepared["references"][target]
                ref = (float(official_median), ref[1], float(official_scale))
            val_normalized = _normalized(calraw[score], ref)
            test_normalized = _normalized(test_raw[score], ref)
            wide[f"{target}__{score}_raw"] = test_raw[score]
            target_row[f"{score}_auroc"] = float(roc_auc_score(labels, test_normalized))
            refs.append({"audit_label": TAG, "dataset": dataset, "target": target, "score": score,
                         "normal_median": ref[0], "normal_mad": ref[1], "normal_scaled_mad": 1.4826 * ref[1],
                         "normal_scale": ref[2], "main_frozen_target": target in main_targets,
                         "skill_only_target": target in skill_targets, "normal_stats_sha256": normal_freeze_sha})
            if target in main_targets and score in global_test:
                np.maximum(global_test[score], test_normalized, out=global_test[score])
                np.maximum(global_cal[score], val_normalized, out=global_cal[score])
            if target in skill_targets and score in skill_test:
                np.maximum(skill_test[score], test_normalized, out=skill_test[score])
                np.maximum(skill_cal[score], val_normalized, out=skill_cal[score])
        target_row["S3_minus_S2_auroc"] = target_row["S3_auroc"] - target_row["S2_auroc"]
        pertarget.append(target_row)
    global_test["S3"] = replay["S3"]
    thresholds = {score: float(np.quantile(values, .99)) for score, values in global_cal.items()}
    thresholds["S3"] = float(prepared["threshold"])
    outputs, fallbacks, metrics, event_tables, sensitivities = [], [], [], {}, []
    metric_fn = replay["metric_fn"]
    for score in ("S1", "S2", "S3"):
        values, threshold = global_test[score], thresholds[score]
        measured = metric_fn(labels, values, threshold)
        events = real_helper.event_rates(prepared, replay["binding"], values, threshold)
        event_tables[score] = events
        row = {"audit_label": TAG, "dataset": dataset, "score": score, **measured,
               "threshold": threshold, "target_count": len(main_targets), "endpoint_count": len(labels),
               "event_hit_count": int((events.hit_rate >= .10).sum()), "event_count": len(events),
               "normal_stats_sha256": normal_freeze_sha}
        if dataset == "hai":
            row["high19_hit_count"] = int((events.loc[events.high, "hit_rate"] >= .10).sum())
            row["high19_median_hit"] = float(events.loc[events.high, "hit_rate"].median())
        metrics.append(row)
        frame = binding.copy()
        frame["score_id"], frame["score"], frame["prediction"] = score, values, values > threshold
        frame["normal_stats_sha256"] = normal_freeze_sha
        _write_frame(frame, out / f"scores_{dataset}_{score}", outputs, fallbacks)
    reference_frame = pd.DataFrame(refs)
    reference_frame["aggregate_q99_threshold"] = reference_frame.score.map(thresholds)
    reference_frame.to_csv(out / f"normal_reference_{dataset}.csv", index=False)
    outputs.append(str(out / f"normal_reference_{dataset}.csv"))
    pertarget_frame = pd.DataFrame(pertarget)
    pertarget_frame.to_csv(out / f"per_target_{dataset}.csv", index=False)
    outputs.append(str(out / f"per_target_{dataset}.csv"))
    _write_frame(pd.DataFrame(wide, copy=False), out / f"per_target_scores_{dataset}", outputs, fallbacks)
    merged_events = event_tables["S3"].drop(columns="hit_rate").copy()
    merged_events["audit_label"], merged_events["dataset"] = TAG, dataset
    for score, events in event_tables.items():
        if not events.event_id.equals(event_tables["S3"].event_id):
            raise RuntimeError("EVENT_ID_ALIGNMENT_MISMATCH")
        merged_events[f"{score}_hit_rate"] = events.hit_rate.to_numpy()
        merged_events[f"{score}_hit_ge_0_10"] = events.hit_rate.to_numpy() >= .10
    merged_events["S3_minus_S2_hit_rate"] = merged_events.S3_hit_rate - merged_events.S2_hit_rate
    merged_events.to_csv(out / f"events_{dataset}.csv", index=False)
    outputs.append(str(out / f"events_{dataset}.csv"))
    event_differences = {"S3_greater_than_S2": int((merged_events.S3_minus_S2_hit_rate > 0).sum()),
                         "S3_less_than_S2": int((merged_events.S3_minus_S2_hit_rate < 0).sum()),
                         "equal": int((merged_events.S3_minus_S2_hit_rate == 0).sum())}
    for score in ("S2", "S3"):
        if not skill_targets:
            continue
        threshold = float(np.quantile(skill_cal[score], .99))
        measured = metric_fn(labels, skill_test[score], threshold)
        sensitivities.append({"audit_label": TAG, "dataset": dataset, "score": score,
                              "target_set": "SKILL_ONLY_SUPPLEMENTARY", "target_count": len(skill_targets),
                              "auroc": float(measured["auroc"]), "aupr": float(measured["aupr"]),
                              "normal_stats_sha256": normal_freeze_sha, "included_in_decision": False})
        frame = binding.copy()
        frame["score_id"], frame["score"] = score, skill_test[score]
        frame["target_set"] = "SKILL_ONLY_SUPPLEMENTARY"
        _write_frame(frame, out / f"scores_{dataset}_skill_only_{score}", outputs, fallbacks)
    main_per = pertarget_frame[pertarget_frame.main_frozen_target]
    positive, nonpositive = main_per[main_per.kappa > 0], main_per[main_per.kappa <= 0]
    ppos = float((positive.S3_minus_S2_auroc > 0).mean()) if len(positive) else None
    pnon = float((nonpositive.S3_minus_S2_auroc > 0).mean()) if len(nonpositive) else None
    hk = {"status": "NOT_TESTABLE" if not len(positive) or not len(nonpositive) else "TESTABLE",
          "positive_count": len(positive), "nonpositive_count": len(nonpositive),
          "positive_group_S3_gt_S2_fraction": ppos, "nonpositive_group_S3_gt_S2_fraction": pnon,
          "trend_present": None if ppos is None or pnon is None else ppos > pnon,
          "interpretation": "descriptive association only; no causal or AUROC/AUPR guarantee"}
    target_differences = {"S3_greater_than_S2": int((main_per.S3_minus_S2_auroc > 0).sum()),
                          "S3_less_than_S2": int((main_per.S3_minus_S2_auroc < 0).sum()),
                          "equal": int((main_per.S3_minus_S2_auroc == 0).sum())}
    return {"audit_label": TAG, "dataset": dataset, "metrics": metrics, "metrics_rows": metrics,
            "per_target": pertarget, "hk": hk, "H_kappa": hk, "event_differences": event_differences,
            "target_differences": target_differences, "sensitivity": sensitivities,
            "outputs": outputs, "format_fallbacks": fallbacks, "thresholds": thresholds}
