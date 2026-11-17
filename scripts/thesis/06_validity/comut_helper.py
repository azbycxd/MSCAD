"""RETROSPECTIVE_VALIDITY_AUDIT: optional frozen Historical CoMuT controls.

Normal preparation is separate from all Test reads. No official main is called.
"""
from __future__ import annotations
import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAG = "RETROSPECTIVE_VALIDITY_AUDIT"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    root = Path(root).resolve()
    for path in (root, root / "scripts/thesis/02_collaboration"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    d0 = importlib.import_module("run_innovation2d0_dual_view_comut")
    if d0.ROOT.resolve() != root:
        raise RuntimeError("COMUT_SOURCE_ROOT_MISMATCH")
    metrics_path, manifest_path = d0.OUT / "case_metrics.csv", d0.OUT / "manifest.json"
    if not d0.DATA.is_dir() or not metrics_path.is_file() or not manifest_path.is_file():
        return {"status": "SKIPPED", "reason": "Historical data or original D0 metrics/manifest unavailable", "normal_stats": [], "input_paths": []}
    cases = sorted(path for path in d0.DATA.iterdir() if path.is_dir())
    if len(cases) != 13 or not all((p / "train_no_anomaly.csv").is_file() and (p / "test.csv").is_file() for p in cases):
        return {"status": "SKIPPED", "reason": "Frozen 13-case train/test file inventory incomplete", "normal_stats": [], "input_paths": []}
    frozen = pd.read_csv(metrics_path).set_index("case_name")
    if list(frozen.index) != [p.name for p in cases]:
        return {"status": "SKIPPED", "reason": "Frozen case identity/order mismatch", "normal_stats": [], "input_paths": []}
    states, normal_stats = {}, []
    inputs = [str(metrics_path), str(manifest_path)]
    for path in cases:
        # The official loader reads train and test together. Defer it until all
        # predictors are frozen; use the identical train CSV and column rule here.
        train = pd.read_csv(path / "train_no_anomaly.csv")
        features = [name for name in train.columns if name not in {"timestamp", "label", "is_anomaly"}]
        if len(train) != 10000 or not (train.is_anomaly.to_numpy() == 0).all():
            raise RuntimeError(f"COMUT_NORMAL_CONTRACT_MISMATCH: {path.name}")
        scaler = d0.StandardScaler().fit(train[features].to_numpy(np.float64))
        z = scaler.transform(train[features].to_numpy(np.float64))
        legal = np.arange(d0.MAX_LAG, len(train), dtype=np.int64)
        split = int(np.floor(0.8 * len(legal)))
        fit_times, cal_times = legal[:split], legal[split:]
        models, references = {}, {}
        cal_matrices = {s: [] for s in ("S1", "S2", "S3")}
        for j, target in enumerate(features):
            cross_indices = [k for k in range(len(features)) if k != j]
            self_model = d0.Ridge(alpha=d0.ALPHA).fit(d0.design(z, fit_times, [j], d0.SELF_LAGS), z[fit_times, j])
            cross_model = d0.Ridge(alpha=d0.ALPHA).fit(d0.design(z, fit_times, cross_indices, d0.CROSS_LAGS), z[fit_times, j])
            p_self = self_model.predict(d0.design(z, cal_times, [j], d0.SELF_LAGS))
            p_cross = cross_model.predict(d0.design(z, cal_times, cross_indices, d0.CROSS_LAGS))
            r_self, r_cross = z[cal_times, j] - p_self, z[cal_times, j] - p_cross
            cov = float(np.mean((r_self - r_self.mean()) * (r_cross - r_cross.mean())))
            raw_scores = {"S1": np.abs(r_self), "S2": np.abs(r_cross), "S3": np.abs(p_self - p_cross)}
            references[target] = {}
            for score_name, values in raw_scores.items():
                median = float(np.median(values))
                mad = float(np.median(np.abs(values - median)))
                scale = max(1.4826 * mad, 1e-6)
                ref = {"median": median, "mad": mad, "scaled_mad": 1.4826 * mad, "scale": scale,
                       "q95": float(np.quantile(values, .95)), "q99": float(np.quantile(values, .99))}
                ref["q99_median_ratio"] = ref["q99"] / median if median != 0 else None
                references[target][score_name] = ref
                cal_matrices[score_name].append(d0.calibrated(values, median, scale))
                normal_stats.append({"audit_label": TAG, "dataset": "Historical CoMuT", "case_name": path.name,
                                     "target": target, "score": score_name, "scope": "D0_NORMAL_TRAIN_LAST_20_PERCENT_LEGAL",
                                     "n": len(cal_times), "var_self": float(np.var(r_self)), "var_cross": float(np.var(r_cross)),
                                     "cov_cross_self": cov, "kappa": cov - .5 * float(np.var(r_self)), **ref})
            models[target] = {"self": self_model, "cross": cross_model, "target_index": j, "cross_indices": cross_indices}
        thresholds = {score: float(np.quantile(np.column_stack(arrays).max(axis=1), .99)) for score, arrays in cal_matrices.items()}
        states[path.name] = {"path": path, "features": features, "models": models, "references": references,
                             "thresholds": thresholds, "scaler": scaler, "train_sha256": sha(path / "train_no_anomaly.csv")}
        inputs.extend([str(path / "train_no_anomaly.csv"), str(path / "test.csv")])
    return {"status": "PREPARED", "d0": d0, "cases": states, "normal_stats": normal_stats,
            "input_paths": inputs, "frozen_metrics": frozen, "training_test_reads": False}


def replay_and_evaluate(state, normal_freeze_sha):
    if state["status"] == "SKIPPED":
        return {"status": "SKIPPED", "reason": state["reason"], "rows": [], "score_frames": [], "per_target_frames": []}
    if not isinstance(normal_freeze_sha, str) or len(normal_freeze_sha) != 64:
        raise RuntimeError("NORMAL_STATISTICS_NOT_HASH_FROZEN")
    d0 = state["d0"]
    loaded, replay = {}, []
    for name, case in state["cases"].items():
        train, test, features = d0.c0b.load_case(case["path"])
        if sha(case["path"] / "train_no_anomaly.csv") != case["train_sha256"] or features != case["features"]:
            raise RuntimeError(f"COMUT_INPUT_CHANGED: {name}")
        z = case["scaler"].transform(test[features].to_numpy(np.float64))
        times = np.arange(d0.MAX_LAG, len(test), dtype=np.int64)
        labels = test.is_anomaly.to_numpy(np.int8)[times]
        predictions, s3_targets = {}, []
        for target in features:
            model = case["models"][target]
            p_self = model["self"].predict(d0.design(z, times, [model["target_index"]], d0.SELF_LAGS))
            p_cross = model["cross"].predict(d0.design(z, times, model["cross_indices"], d0.CROSS_LAGS))
            predictions[target] = (z[times, model["target_index"]], p_self, p_cross)
            ref = case["references"][target]["S3"]
            s3_targets.append(d0.calibrated(np.abs(p_self - p_cross), ref["median"], ref["scale"]))
        score = np.column_stack(s3_targets).max(axis=1)
        pred = score > case["thresholds"]["S3"]
        precision, recall, f1, _ = d0.precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
        observed = {"auroc": float(d0.roc_auc_score(labels, score)), "aupr": float(d0.average_precision_score(labels, score)),
                    "precision_q99": float(precision), "recall_q99": float(recall), "f1_q99": float(f1),
                    "fpr_q99": float(np.mean(pred[labels == 0]))}
        frozen = state["frozen_metrics"].loc[name]
        checks = {metric: round(value, 6) == round(float(frozen[metric]), 6) for metric, value in observed.items()}
        replay.append({"case_name": name, "passed": all(checks.values()), "checks": checks,
                       "differences": {metric: value - float(frozen[metric]) for metric, value in observed.items()}})
        loaded[name] = {"test": test, "times": times, "labels": labels, "predictions": predictions}
    if not all(row["passed"] for row in replay):
        return {"status": "SKIPPED_S3_REPRODUCTION_MISMATCH", "replay": replay,
                "reason": "Original D0 S3 metrics did not match frozen results; no new CoMuT controls evaluated",
                "rows": [], "score_frames": [], "per_target_frames": []}
    rows, frames, target_frames = [], [], []
    for name, data in loaded.items():
        case = state["cases"][name]
        test, times, labels = data["test"], data["times"], data["labels"]
        source_hash = sha(case["path"] / "test.csv")
        base = pd.DataFrame({"audit_label": TAG, "dataset": "Historical CoMuT", "case_name": name,
                             "file": str(case["path"] / "test.csv"), "local_index": times,
                             "raw_label": labels, "valid_mask": True, "source_hash": source_hash})
        if "timestamp" in test:
            base["timestamp"] = test.timestamp.to_numpy()[times]
        all_targets = {score: [] for score in ("S1", "S2", "S3")}
        for target, (y, p_self, p_cross) in data["predictions"].items():
            raw = {"S1": np.abs(y - p_self), "S2": np.abs(y - p_cross), "S3": np.abs(p_self - p_cross)}
            target_frame = base.copy()
            target_frame["target"] = target
            for score_name, values in raw.items():
                ref = case["references"][target][score_name]
                target_frame[score_name + "_raw"] = values
                all_targets[score_name].append(d0.calibrated(values, ref["median"], ref["scale"]))
            target_frames.append(target_frame)
        for score_name, arrays in all_targets.items():
            score = np.column_stack(arrays).max(axis=1)
            threshold = case["thresholds"][score_name]
            prediction = score > threshold
            precision, recall, f1, _ = d0.precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
            rows.append({"audit_label": TAG, "case_name": name, "category": d0.c0b.category(name), "score": score_name,
                         "auroc": float(d0.roc_auc_score(labels, score)), "aupr": float(d0.average_precision_score(labels, score)),
                         "precision": float(precision), "recall": float(recall), "f1": float(f1),
                         "fpr": float(np.mean(prediction[labels == 0])), "threshold": threshold,
                         "normal_stats_sha256": normal_freeze_sha, "test_point_count": len(times)})
            frame = base.copy()
            frame["score_id"], frame["score"], frame["prediction"] = score_name, score, prediction
            frames.append(frame)
    return {"status": "COMPLETED", "replay": replay, "rows": rows, "score_frames": frames,
            "per_target_frames": target_frames, "normal_stats_sha256": normal_freeze_sha}
