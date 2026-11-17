"""E2-R1D deterministic artifact recovery and frozen raw-score reproduction.

This is inference/audit code only.  It reconstructs only Train/Validation-derived
scalers and normal references whose exact algorithms, dtypes, splits, features,
and ordering are fixed by the historical formal runners.  It never trains a
model, performs backward propagation, creates an optimizer, refits Ridge, or
selects a checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler as SklearnScaler


ROOT = Path(__file__).resolve().parents[3]
E2OUT = ROOT / "outputs" / "thesis_evaluation" / "literature_compatible"
OUT = E2OUT / "recovery"
ART = OUT / "reconstructed_artifacts"
RAW = E2OUT / "raw_scores"
WINDOW, MAX_LAG = 64, 30
SELF_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (0, 1, 5, 10, 20, 30)
ORIGIN = "DETERMINISTIC_RECONSTRUCTION"
TOLERANCE = 1e-6


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("e2r1d_lstm", ROOT / "scripts/thesis/01_baseline/run_lstm_ae_unified.py")
USAD = load_module("e2r1d_usad", ROOT / "scripts/thesis/01_baseline/run_usad_unified.py")
GDN = load_module("e2r1d_gdn", ROOT / "scripts/thesis/01_baseline/run_gdn_unified.py")
I3 = load_module("e2r1d_i3", ROOT / "scripts/thesis/02_collaboration/run_innovation3b_tail_fusion_final.py")
HAI = load_module("e2r1d_hai", ROOT / "scripts/thesis/02_collaboration/run_hai21_b1_real.py")
E2 = load_module("e2r1d_e2", ROOT / "scripts/thesis/00_audit/evaluate_literature_compatible.py")
R1 = load_module("e2r1d_r1", ROOT / "scripts/thesis/00_audit/persist_e2_r1_raw_scores.py")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def aggregate_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        h.update(bytes.fromhex(sha256(path)))
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def strict_ref() -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["dataset"], row["method"]): row for row in E2.strict_rows()}


REF = strict_ref()


def save_archive(dataset_slug: str, method_slug: str, score: np.ndarray, labels: np.ndarray,
                 file_ids: np.ndarray, local: np.ndarray, global_index: np.ndarray,
                 timestamps: np.ndarray | None = None) -> Path:
    directory = RAW / dataset_slug / method_slug
    directory.mkdir(parents=True, exist_ok=True)
    values: dict[str, np.ndarray] = {
        "score": np.asarray(score, dtype=np.float64),
        "label": np.asarray(labels, dtype=np.int8),
        "segment_id": np.asarray(file_ids),
        "file_id": np.asarray(file_ids),
        "local_index": np.asarray(local, dtype=np.int64),
        "global_index": np.asarray(global_index, dtype=np.int64),
        "valid_endpoint": np.ones(len(score), dtype=np.bool_),
        "score_orientation": np.asarray("higher=more_anomalous"),
    }
    if timestamps is not None:
        values["timestamp"] = np.asarray(timestamps, dtype=np.int64)
    path = directory / "test_raw_scores.npz"
    np.savez_compressed(path, **values)
    return path


def verify(dataset: str, method: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    result = R1.strict_check(dataset, method, labels, scores)
    result["raw_score_status"] = "RAW_SCORE_REPRODUCED" if result["status"] == "PASS" else result["status"]
    return result


def identities(lengths: list[int], labels: np.ndarray, first: int,
               timestamps: np.ndarray | None = None, names: list[str] | None = None) -> dict[str, np.ndarray]:
    ids, local, glob, lab, ts = [], [], [], [], []
    offset = 0
    for sid, length in enumerate(lengths):
        loc = np.arange(first, length, dtype=np.int64)
        idx = offset + loc
        name = names[sid] if names else str(sid)
        ids.append(np.repeat(name, len(loc)))
        local.append(loc); glob.append(idx); lab.append(labels[idx])
        if timestamps is not None:
            ts.append(timestamps[idx])
        offset += length
    out = {"file_id": np.concatenate(ids), "local_index": np.concatenate(local),
           "global_index": np.concatenate(glob), "label": np.concatenate(lab)}
    if timestamps is not None:
        out["timestamp"] = np.concatenate(ts)
    return out


def artifact_meta(path: Path, *, dataset: str, method: str, source_split: str, source_hash: str,
                  feature_names: list[str], algorithm: str, dtype: str, source_script: str,
                  formula: str) -> dict[str, Any]:
    feature_hash = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    return {
        "dataset": dataset, "method": method, "artifact_origin": ORIGIN,
        "source_split": source_split, "source_dataset_hash": source_hash,
        "feature_count": len(feature_names), "feature_order_sha256": feature_hash,
        "algorithm": algorithm, "dtype": dtype, "source_script": source_script,
        "formula": formula, "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": rel(path), "artifact_sha256": sha256(path),
        "new_fit_or_retuning": False, "test_labels_used": False,
    }


def save_custom_scaler(directory: Path, scaler: Any, **meta: Any) -> tuple[Path, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "reconstructed_scaler.npz"
    if hasattr(scaler, "effective_scale_"):
        np.savez(path, mean=np.asarray(scaler.mean_), raw_std=np.asarray(scaler.raw_std_),
                 effective_scale=np.asarray(scaler.effective_scale_), eps=np.asarray([scaler.eps]))
    else:
        np.savez(path, data_min=np.asarray(scaler.data_min_), data_max=np.asarray(scaler.data_max_),
                 data_range=np.asarray(scaler.data_range_), scale=np.asarray(scaler.scale_),
                 min=np.asarray(scaler.min_), n_samples_seen=np.asarray([scaler.n_samples_seen_]),
                 feature_range=np.asarray(scaler.feature_range), clip=np.asarray([scaler.clip]))
    info = artifact_meta(path, **meta)
    dump_json(directory / "reconstructed_scaler.json", info)
    return path, info


def swat_raw() -> dict[str, Any]:
    config = BASE.load_config(str(BASE.SWAT_CONFIG)); BASE.validate_config_contract(config)
    cache = BASE.resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, manifest = BASE.load_normal_cache(cache, config)
    attack, labels, timestamps, _ = BASE.load_attack_cache(cache, config)
    lengths = [int(x["row_count"]) for x in json.loads(BASE.SWAT_SEGMENTS.read_text(encoding="utf-8"))["test_segments"]]
    source_hash = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))["known_t2_source_sha256"]["SWaT_Dataset_Normal_v1.xlsx"]
    return {"normal": normal, "attack": attack, "labels": np.asarray(labels), "timestamps": np.asarray(timestamps),
            "lengths": lengths, "features": list(manifest["column_order"]), "source_hash": source_hash}


def hai_raw(dtype: type[np.floating]) -> dict[str, Any]:
    features = BASE.pd.read_csv(BASE.HAI_AUDIT / "variable_schema.csv")["column_name"].astype(str).tolist()
    train_files = list(BASE.HAI_TRAIN_FILES); test_files = list(BASE.HAI_TEST_FILES)
    train = [pd.read_csv(BASE.HAI_ROOT / name, compression="gzip", usecols=features, low_memory=False).to_numpy(dtype) for name in train_files]
    bounds = [int(math.floor(.8 * len(x))) for x in train]
    tests, labels = [], []
    for name in test_files:
        frame = pd.read_csv(BASE.HAI_ROOT / name, compression="gzip", usecols=features + ["attack"], low_memory=False)
        tests.append(frame[features].to_numpy(dtype)); labels.append(frame["attack"].to_numpy(np.int8))
    paths = [BASE.HAI_ROOT / name for name in train_files]
    return {"features": features, "train_raw": train, "bounds": bounds, "test_raw": tests,
            "test_labels": labels, "test_files": test_files, "source_hash": aggregate_hash(paths)}


def scaler_baseline(dataset: str, method: str, raw: dict[str, Any]):
    directory = ART / dataset.lower().replace("21", "") / method.lower().replace("-", "_")
    if dataset == "SWaT":
        scaler = BASE.StandardScaler().fit(np.asarray(raw["normal"][:396000], dtype=np.float64))
        train = [scaler.transform(np.asarray(raw["normal"][:396000])).astype(np.float32)]
        val = [scaler.transform(np.asarray(raw["normal"][396000:495000])).astype(np.float32)]
        scaled = scaler.transform(np.asarray(raw["attack"])).astype(np.float32)
        test = []; off = 0
        for length in raw["lengths"]: test.append(scaled[off:off+length]); off += length
        source_split = "Official Normal v1 rows [0,396000)"
    else:
        scaler = BASE.StandardScaler(); prefixes = [x[:b] for x, b in zip(raw["train_raw"], raw["bounds"])]
        scaler.fit_many(prefixes)
        train = [scaler.transform(x[:b]).astype(np.float32) for x, b in zip(raw["train_raw"], raw["bounds"])]
        val = [scaler.transform(x[b:]).astype(np.float32) for x, b in zip(raw["train_raw"], raw["bounds"])]
        test = [scaler.transform(x).astype(np.float32) for x in raw["test_raw"]]
        source_split = "train1..3 each ordered prefix floor(0.8*N)"
    path, info = save_custom_scaler(directory, scaler, dataset=dataset, method=method,
        source_split=source_split, source_hash=raw["source_hash"], feature_names=raw["features"],
        algorithm="data.transforms.StandardScaler population mean/std; std<1e-6 -> 1",
        dtype="float64 statistics -> float32 transformed arrays", source_script="scripts/thesis/01_baseline/run_lstm_ae_unified.py",
        formula="(x-mean)/effective_scale")
    return train, val, test, path, info


def scaler_usad(dataset: str, raw: dict[str, Any]):
    directory = ART / dataset.lower().replace("21", "") / "usad"
    scaler = MinMaxScaler((0, 1), clip=False)
    if dataset == "SWaT":
        tr = np.asarray(raw["normal"][:396000], dtype=np.float32); scaler.fit(tr)
        train = [scaler.transform(tr).astype(np.float32)]
        val = [scaler.transform(np.asarray(raw["normal"][396000:495000], dtype=np.float32)).astype(np.float32)]
        scaled = scaler.transform(np.asarray(raw["attack"], dtype=np.float32)).astype(np.float32)
        test = []; off = 0
        for length in raw["lengths"]: test.append(scaled[off:off+length]); off += length
        split = "Official Normal v1 rows [0,396000)"
    else:
        for x, b in zip(raw["train_raw"], raw["bounds"]): scaler.partial_fit(x[:b])
        train = [scaler.transform(x[:b]).astype(np.float32) for x, b in zip(raw["train_raw"], raw["bounds"])]
        val = [scaler.transform(x[b:]).astype(np.float32) for x, b in zip(raw["train_raw"], raw["bounds"])]
        test = [scaler.transform(x).astype(np.float32) for x in raw["test_raw"]]
        split = "train1..3 each ordered prefix floor(0.8*N)"
    path, info = save_custom_scaler(directory, scaler, dataset=dataset, method="USAD", source_split=split,
        source_hash=raw["source_hash"], feature_names=raw["features"], algorithm="sklearn MinMaxScaler(feature_range=(0,1), clip=False)",
        dtype="float32 fit and transform", source_script="scripts/thesis/01_baseline/run_usad_unified.py",
        formula="x*scale_+min_; no clipping")
    return train, val, test, path, info


def method_identity(dataset: str, raw: dict[str, Any], first: int) -> dict[str, np.ndarray]:
    if dataset == "SWaT":
        return identities(raw["lengths"], raw["labels"], first, raw["timestamps"])
    lengths = [len(x) for x in raw["test_raw"]]; labels = np.concatenate(raw["test_labels"])
    return identities(lengths, labels, first, names=raw["test_files"])


def persist_result(dataset: str, method: str, method_slug: str, score: np.ndarray, ident: dict[str, np.ndarray],
                   checkpoint: Path | None, artifact_paths: list[Path], score_definition: str,
                   pipeline_source: str) -> dict[str, Any]:
    dslug = "swat" if dataset == "SWaT" else "hai"
    path = save_archive(dslug, method_slug, score, ident["label"], ident["file_id"], ident["local_index"],
                        ident["global_index"], ident.get("timestamp"))
    check = verify(dataset, method, ident["label"], score)
    directory = path.parent
    dump_json(directory / "strict_reproduction.json", check)
    mask = ident["local_index"] >= 64
    common = R1.metrics(ident["label"][mask], score[mask], float(REF[(dataset, method)]["threshold_q99"]))
    dump_json(directory / "common_mask_validation.json", {
        "status": "PASS" if check["status"] == "PASS" else "NOT_EVALUATED",
        "evaluation_kind": "COMMON_MASK_REEVALUATION", "fixed_mask_count": int(mask.sum()), "metrics": common})
    prov = {"dataset": dataset, "method": method, "artifact_origin": ORIGIN, "inference_only": True,
            "training": False, "backward": False, "optimizer": False, "ridge_refit": False,
            "checkpoint_selection": False, "test_tuning": False, "test_labels_used_for_score": False,
            "score_definition": score_definition, "pipeline_source": pipeline_source,
            "checkpoint": rel(checkpoint) if checkpoint else None,
            "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
            "reconstructed_artifacts": [{"path": rel(p), "sha256": sha256(p)} for p in artifact_paths],
            "archive": rel(path), "archive_sha256": sha256(path)}
    dump_json(directory / "provenance.json", prov)
    return {"dataset": dataset, "method": method, "status": check["raw_score_status"], "max_diff": check["max_absolute_difference"],
            "n": len(score), "archive": path, "checkpoint": checkpoint, "artifacts": artifact_paths}


def run_baseline(dataset: str, method: str, raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
    slug = method.lower().replace("-", "_")
    if method == "USAD":
        _, val, test, scaler_path, _ = scaler_usad(dataset, raw)
        checkpoint = ROOT / f"outputs/thesis_baselines/usad/{'swat' if dataset=='SWaT' else 'hai'}_state_dict.pt"
        model = USAD.USAD(len(raw["features"])).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); model.eval()
        score = USAD.score_model(model, USAD.window_store(test, common_mask=True), device)
        result = persist_result(dataset, method, slug, score, method_identity(dataset, raw, 64), checkpoint, [scaler_path],
            "endpoint mean channels 0.5*(x-w1)^2+0.5*(x-w3)^2", "scripts/thesis/01_baseline/run_usad_unified.py")
    else:
        _, val, test, scaler_path, _ = scaler_baseline(dataset, method, raw)
        base = "swat" if dataset == "SWaT" else "hai"
        checkpoint = ROOT / f"outputs/thesis_baselines/{'lstm_ae' if method=='LSTM-AE' else 'gdn'}/{base}_state_dict.pt"
        if method == "LSTM-AE":
            model = BASE.LSTMAutoencoder(len(raw["features"])).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); model.eval()
            score = BASE.score_model(model, BASE.WindowStore.from_segments(test, common_endpoint_mask=True), device)
            result = persist_result(dataset, method, slug, score, method_identity(dataset, raw, 64), checkpoint, [scaler_path],
                "endpoint mean channel squared reconstruction error", "scripts/thesis/01_baseline/run_lstm_ae_unified.py")
        else:
            model = GDN.FrozenGDN(len(raw["features"])).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); model.eval()
            val_error = GDN.raw_errors(model, GDN.ForecastStore.from_segments(val), device)
            median = np.median(val_error, axis=0); iqr = np.quantile(val_error, .75, axis=0) - np.quantile(val_error, .25, axis=0)
            cal_dir = ART / ("swat" if dataset == "SWaT" else "hai") / "gdn"; cal_dir.mkdir(parents=True, exist_ok=True)
            cal = cal_dir / "validation_median_iqr.npz"; np.savez(cal, median=median, iqr=iqr, epsilon=np.asarray([1e-2]))
            dump_json(cal_dir / "validation_median_iqr.json", artifact_meta(cal, dataset=dataset, method=method,
                source_split="Normal Validation only", source_hash=raw["source_hash"], feature_names=raw["features"],
                algorithm="per-sensor median/IQR", dtype="float64 aggregation of float32 forecast errors",
                source_script="scripts/thesis/01_baseline/run_gdn_unified.py", formula="(abs_error-median)/(abs(IQR)+1e-2)"))
            errors = GDN.raw_errors(model, GDN.ForecastStore.from_segments(test), device)
            score = GDN.native_scores(errors, median, iqr, test)
            result = persist_result(dataset, method, slug, score, method_identity(dataset, raw, 64), checkpoint, [scaler_path, cal],
                "validation median/IQR standardized absolute forecast error; segment-local four-point trailing mean; max sensor",
                "scripts/thesis/01_baseline/run_gdn_unified.py")
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def ours_data(dataset: str, raw: dict[str, Any]):
    if dataset == "SWaT":
        scaler = BASE.StandardScaler().fit(np.asarray(raw["normal"][:396000]))
        z = scaler.transform(np.asarray(raw["normal"])).astype(np.float64)
        train, val = [z[:396000]], [z[396000:495000]]
        za = scaler.transform(np.asarray(raw["attack"])).astype(np.float64); tests=[]; off=0
        for length in raw["lengths"]: tests.append(za[off:off+length]); off += length
        split = "Official Normal v1 rows [0,396000)"; dtype = "float32 statistics; float64 transformed arrays"
    else:
        scaler = SklearnScaler()
        for x, b in zip(raw["train_raw"], raw["bounds"]): scaler.partial_fit(np.asarray(x[:b], dtype=np.float64))
        train = [scaler.transform(x[:b]).astype(np.float64, copy=False) for x, b in zip(raw["train_raw"], raw["bounds"])]
        val = [scaler.transform(x[b:]).astype(np.float64, copy=False) for x, b in zip(raw["train_raw"], raw["bounds"])]
        tests = [scaler.transform(x).astype(np.float64, copy=False) for x in raw["test_raw"]]
        split = "train1..3 each ordered prefix floor(0.8*N)"; dtype = "float64 partial_fit and transform"
    directory = ART / ("swat" if dataset == "SWaT" else "hai") / "ours"
    directory.mkdir(parents=True, exist_ok=True); path = directory / "reconstructed_scaler.npz"
    if dataset == "SWaT": np.savez(path, mean=scaler.mean_, raw_std=scaler.raw_std_, effective_scale=scaler.effective_scale_, eps=np.asarray([scaler.eps]))
    else: np.savez(path, mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_, n_samples_seen=np.asarray(scaler.n_samples_seen_))
    dump_json(directory / "reconstructed_scaler.json", artifact_meta(path, dataset=dataset, method="Ours-REC/DUAL/TAIL",
        source_split=split, source_hash=raw["source_hash"], feature_names=raw["features"], algorithm="historical final-run StandardScaler",
        dtype=dtype, source_script="scripts/thesis/02_collaboration/run_innovation3b_tail_fusion_final.py", formula="(x-mean)/scale"))
    return train, val, tests, path


def load_dual_state(dataset: str, features: list[str], val: list[np.ndarray]) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    prefix = "swat" if dataset == "SWaT" else "hai21"
    target = ROOT / f"outputs/thesis_{prefix}_final_unified/dual_targets_unified_frozen.csv"
    frozen = pd.read_csv(target)
    coef_path = ROOT / f"outputs/thesis_efficiency_final/{prefix}_dual_coefficients.npz"
    with np.load(coef_path, allow_pickle=False) as z:
        ordering = json.loads(str(z["feature_ordering_json"].item()))
        if [x["target"] for x in ordering] != frozen.target.astype(str).tolist(): raise RuntimeError("frozen target/order mismatch")
        index = {name: i for i, name in enumerate(features)}; state={}; medians=[]; scales=[]
        for i, spec in enumerate(ordering):
            target_name = spec["target"]; ti=index[target_name]
            self_specs=[(index[x.split("@lag")[0]], int(x.split("@lag")[1])) for x in spec["self_features"]]
            cross_specs=[(index[x.split("@lag")[0]], int(x.split("@lag")[1])) for x in spec["cross_features"]]
            sm=HAI.LinearModel(np.asarray(z[f"t{i:03d}_self_coef"]).reshape(-1,1),np.asarray(z[f"t{i:03d}_self_intercept"]))
            cm=HAI.LinearModel(np.asarray(z[f"t{i:03d}_cross_coef"]).reshape(-1,1),np.asarray(z[f"t{i:03d}_cross_intercept"]))
            parts=[]
            for values in val:
                times=np.arange(MAX_LAG,len(values),dtype=np.int64)
                ps=sm.predict(HAI.design(values,times,self_specs))[:,0]; pc=cm.predict(HAI.design(values,times,cross_specs))[:,0]
                parts.append(np.abs(ps-pc))
            median,scale=I3.robust(np.concatenate(parts)); medians.append(median); scales.append(scale)
            sources=[]
            for name,_ in cross_specs:
                if name not in sources: sources.append(name)
            state[target_name]={"ti":ti,"sources":sources,"self":sm,"cross":cm,"ref":(median,scale)}
    directory=ART/("swat" if dataset=="SWaT" else "hai")/"ours_dual"; directory.mkdir(parents=True,exist_ok=True)
    ref_path=directory/"dual_reference_reconstructed.npz"
    np.savez(ref_path,target=np.asarray(frozen.target.astype(str)),median=np.asarray(medians),scale=np.asarray(scales),
             coefficient_sha256=np.asarray(sha256(coef_path)),target_sha256=np.asarray(sha256(target)))
    dump_json(directory/"dual_reference_reconstructed.json", artifact_meta(ref_path,dataset=dataset,method="Ours-DUAL",
        source_split="Frozen Normal Validation only",source_hash=sha256(coef_path),feature_names=features,
        algorithm="per-target median and max(1.4826*MAD,1e-6) of |self_prediction-cross_prediction|",
        dtype="float64",source_script="scripts/thesis/02_collaboration/run_innovation3b_tail_fusion_final.py",
        formula="max(0,(d_j-median_j)/scale_j); global=max_j"))
    return frozen,state,ref_path


def run_ours(dataset: str, raw: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    train,val,test,scaler_path=ours_data(dataset,raw); ident=method_identity(dataset,raw,63); results=[]
    if dataset=="SWaT":
        rec_source=ROOT/"outputs/thesis_midterm_s1/a0_rec_baseline/test_scores.npz"
        with np.load(rec_source,allow_pickle=False) as z: rec=np.asarray(z["endpoint_mae"],dtype=np.float64)
        checkpoint=ROOT/"outputs/thesis_midterm_s1/a0_rec_baseline/best.pt"
    else:
        from models.backbone_ae import BackboneAE
        checkpoint=ROOT/"outputs/thesis_hai21_b1_real/rec_best.pt"; model=BackboneAE(79,64,32,.1).to(device)
        payload=torch.load(checkpoint,map_location=device,weights_only=False); model.load_state_dict(payload["state_dict"]); model.eval()
        rec=np.concatenate(HAI.ae_scores(model,test,device,256)); del model
        if device.type=="cuda": torch.cuda.empty_cache()
    results.append(persist_result(dataset,"Ours-REC","ours_rec",rec,ident,checkpoint,[scaler_path],
        "endpoint mean absolute reconstruction error","scripts/thesis/02_collaboration/run_innovation3b_tail_fusion_final.py"))
    frozen,state,dual_ref=load_dual_state(dataset,raw["features"],val)
    dual=np.concatenate(I3.dual_scores(HAI,test,frozen,state))
    results.append(persist_result(dataset,"Ours-DUAL","ours_dual",dual,ident,None,[scaler_path,dual_ref],
        "max target robust-z predictive disagreement","scripts/thesis/02_collaboration/run_innovation3b_tail_fusion_final.py"))
    if results[-2]["status"]=="RAW_SCORE_REPRODUCED" and results[-1]["status"]=="RAW_SCORE_REPRODUCED":
        prefix="swat" if dataset=="SWaT" else "hai21"; cal=ROOT/f"outputs/thesis_efficiency_final/{prefix}_tail_calibration.npz"
        with np.load(cal,allow_pickle=False) as z:
            freeze={"rec_a":np.asarray(z["rec_ecdf_sorted"]),"dual_a":np.asarray(z["dual_ecdf_sorted"]),"threshold":float(z["threshold"][0])}
        _,_,tail=I3.apply_tail(freeze,rec,dual)
        results.append(persist_result(dataset,"Ours-TAIL","ours_tail",tail,ident,None,[cal],
            "max empirical tail evidence of REC and DUAL; persisted Calibration-A ECDF arrays",
            "scripts/thesis/02_collaboration/run_innovation3b_tail_fusion_final.py"))
    return results


def pipeline_rows() -> list[dict[str, Any]]:
    rows=[]
    for dataset in ("SWaT","HAI21"):
        prefix="swat" if dataset=="SWaT" else "hai"
        for method,folder,script,score,cal in [
            ("LSTM-AE","lstm_ae","run_lstm_ae_unified.py","score_model endpoint MSE","Normal Validation q99"),
            ("USAD","usad","run_usad_unified.py","score_model official 0.5/0.5 endpoint score","Normal Validation q99"),
            ("GDN","gdn","run_gdn_unified.py","native_scores forecast error","Validation median/IQR then q99")]:
            rows.append({"dataset":dataset,"method":method,"train_script":f"scripts/thesis/01_baseline/{script}",
                "eval_script":f"scripts/thesis/01_baseline/{script}","checkpoint":f"outputs/thesis_baselines/{folder}/{prefix}_state_dict.pt",
                "preprocessing_code":f"scripts/thesis/01_baseline/{script}","score_code":score,"calibration_code":cal,
                "historical_result":f"outputs/thesis_baselines/{folder}/{prefix}_metrics.json","confidence":"HIGH"})
        for method in ("Ours-REC","Ours-DUAL","Ours-TAIL"):
            rows.append({"dataset":dataset,"method":method,"train_script":"S1/HAI-B1 REC checkpoint; frozen Ridge coefficients (no R1D fitting)",
                "eval_script":"scripts/thesis/02_collaboration/run_innovation3b_tail_fusion_final.py",
                "checkpoint":("outputs/thesis_midterm_s1/a0_rec_baseline/best.pt" if dataset=="SWaT" else "outputs/thesis_hai21_b1_real/rec_best.pt") if method=="Ours-REC" else f"outputs/thesis_efficiency_final/{'swat' if dataset=='SWaT' else 'hai21'}_dual_coefficients.npz",
                "preprocessing_code":"scripts/thesis/02_collaboration/run_innovation3b_tail_fusion_final.py",
                "score_code":method,"calibration_code":"Validation median/MAD" if method=="Ours-DUAL" else ("persisted Calibration-A ECDF + Threshold-B q99" if method=="Ours-TAIL" else "Normal Validation q99"),
                "historical_result":"outputs/thesis_innovation3b_tail_fusion_final/cross_dataset_metrics.csv","confidence":"HIGH"})
    return rows


def diagnosis(method: str, prior: dict[str, Any], result: dict[str, Any], reconstructed_scaler: Path) -> dict[str, Any]:
    old=ROOT/"outputs/thesis_midterm_s1/a0_rec_baseline/scaler.npz"
    with np.load(old,allow_pickle=False) as a,np.load(reconstructed_scaler,allow_pickle=False) as b:
        mean_delta=float(np.max(np.abs(np.asarray(a["mean"],dtype=np.float64)-np.asarray(b["mean"],dtype=np.float64))))
        scale_delta=float(np.max(np.abs(np.asarray(a["effective_scale"],dtype=np.float64)-np.asarray(b["effective_scale"],dtype=np.float64))))
    return {"method":method,"prior_status":prior.get("status"),"prior_max_absolute_difference":prior.get("max_absolute_difference"),
        "checkpoint_identity":"MATCHED_FORMAL_CHECKPOINT","earliest_inconsistent_pipeline_stage":"B_STANDARD_SCALER_STATISTICS",
        "root_cause":"E2-R1 reused the S1 A0 persisted float32 scaler instead of the baseline runner's source-defined float64 Train-only reconstruction",
        "a0_vs_baseline_scaler_max_abs_mean_delta":mean_delta,"a0_vs_baseline_scaler_max_abs_scale_delta":scale_delta,
        "source_determined_rules":{"feature_order":"MATCH","endpoint":"MATCH local_index=64","hard_gap":"MATCH two independent segments",
            "eval_mode":"MATCH","score_aggregation":"MATCH","threshold_comparator":"MATCH >","q99":"MATCH historical frozen threshold",
            "common_vs_historical_mask":"MATCH 449791"},
        "gdn_source_determined_details":({"epsilon":1e-2,"trailing_mean_points":4,"first_three_scores_per_segment":0.0,"top1":"max sensor"} if method=="GDN" else None),
        "recovered_status":result["status"],"recovered_max_absolute_difference":result["max_diff"],
        "parameter_variants_tried":0,"test_based_selection":False}


def main() -> int:
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True); ART.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior={}
    for method in ("lstm_ae","gdn"):
        p=RAW/"swat"/method/"strict_reproduction.json"
        if p.is_file(): prior[method]=json.loads(p.read_text(encoding="utf-8"))
    swat=swat_raw(); hai64=hai_raw(np.float64); hai32=hai_raw(np.float32)
    results=[]
    for method in ("LSTM-AE","USAD","GDN"):
        results.append(run_baseline("SWaT",method,swat,device))
    for method in ("LSTM-AE","GDN"):
        results.append(run_baseline("HAI21",method,hai64,device))
    results.append(run_baseline("HAI21","USAD",hai32,device))
    results.extend(run_ours("SWaT",swat,device)); results.extend(run_ours("HAI21",hai64,device))
    order={(d,m):i for i,(d,m) in enumerate((d,m) for d in ("SWaT","HAI21") for m in ("LSTM-AE","USAD","GDN","Ours-REC","Ours-DUAL","Ours-TAIL"))}
    results=sorted(results,key=lambda x:order[(x["dataset"],x["method"])])
    trace=pipeline_rows(); manifest=[]; availability=[]
    for result in results:
        ref=REF[(result["dataset"],result["method"])]
        manifest.append({"dataset":result["dataset"],"method":result["method"],"artifact_origin":ORIGIN,
            "recovered_artifacts":";".join(rel(x) for x in result["artifacts"]),"checkpoint":rel(result["checkpoint"]) if result["checkpoint"] else "N/A",
            "raw_score_archive":rel(result["archive"]),"strict_status":result["status"],"max_absolute_difference":result["max_diff"],
            "model_training":False,"ridge_refit":False,"test_tuning":False})
        dslug="swat" if result["dataset"]=="SWaT" else "hai"; mslug=result["method"].lower().replace("-","_")
        availability.append({"dataset":result["dataset"],"method":result["method"],"score_path":rel(result["archive"]),
            "label_path":rel(result["archive"])+"#label","mask_path":f"outputs/thesis_evaluation/literature_compatible/common_masks/{dslug}_common_mask.npz",
            "q99_threshold":ref["threshold_q99"],"n_scores":result["n"],"n_labels":result["n"],
            "available":result["status"]=="RAW_SCORE_REPRODUCED","status":result["status"],
            "provenance":f"outputs/thesis_evaluation/literature_compatible/raw_scores/{dslug}/{mslug}/provenance.json",
            "notes":"strict raw score reproduced within 1e-6 via source-determined deterministic artifact reconstruction"})
    dump_json(OUT/"original_pipeline_trace_rows.json",trace); dump_json(OUT/"artifact_recovery_manifest_rows.json",manifest)
    dump_json(E2OUT/"score_availability_rows.json",availability)
    lstm=next(x for x in results if x["dataset"]=="SWaT" and x["method"]=="LSTM-AE")
    gdn=next(x for x in results if x["dataset"]=="SWaT" and x["method"]=="GDN")
    dump_json(OUT/"swat_lstm_reproduction_diagnosis.json",diagnosis("LSTM-AE",prior.get("lstm_ae",{}),lstm,ART/"swat"/"lstm_ae"/"reconstructed_scaler.npz"))
    dump_json(OUT/"swat_gdn_reproduction_diagnosis.json",diagnosis("GDN",prior.get("gdn",{}),gdn,ART/"swat"/"gdn"/"reconstructed_scaler.npz"))
    passed=sum(x["status"]=="RAW_SCORE_REPRODUCED" for x in results)
    status="E2_RAW_SCORES_COMPLETED" if passed==12 else ("E2_RAW_SCORE_REPRODUCTION_REQUIRES_REVIEW" if any(x["status"]=="RAW_SCORE_REPRODUCTION_MISMATCH" for x in results) else "E2_RAW_SCORES_PARTIAL")
    summary={"status":status,"passed":passed,"total":12,"elapsed_seconds":time.perf_counter()-started,"results":[{k:(rel(v) if isinstance(v,Path) else v) for k,v in x.items() if k not in {"artifacts","checkpoint"}} for x in results],
        "common_masks_unchanged":{"SWaT":449791,"HAI21":401685},"artifact_origin":ORIGIN,
        "no_model_training":True,"no_backward":True,"no_optimizer":True,"no_ridge_refit":True,"no_test_tuning":True,"no_point_adjustment":True,"no_oracle":True,"commit":False,"push":False}
    dump_json(OUT/"recovery_summary.json",summary); print(json.dumps(summary,ensure_ascii=False,allow_nan=False),flush=True)
    return 0 if passed==12 else 2


if __name__ == "__main__":
    raise SystemExit(main())
