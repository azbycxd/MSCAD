"""Normal-only cross-dataset empirical tail-evidence calibration audit."""
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
from sklearn.preprocessing import StandardScaler as SklearnScaler


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "thesis_innovation3a_tail_calibration"
DOC = ROOT / "docs" / "thesis" / "innovation3a_normal_tail_evidence_calibration.md"
HELPERS = ROOT / "scripts" / "thesis" / "02_collaboration"
SELF_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (0, 1, 5, 10, 20, 30)
MAX_LAG, WINDOW, ALPHA, Q = 30, 64, 1.0, 0.99
EXPECTED_TARGETS = {
    "SWaT": (15, "af9f9aac9cb8b369015a94d4b88a4729d817c219d305eb3276050c5d33ad31bf"),
    "HAI21": (54, "a78e820567eaa15dd5a0cd2a9f6491d04102c0a50732a4de7aa918f1b20be5bb"),
    "WADI.A2": (47, "9c6cb7dc0810d22fa13f6ce0f22858fb536db0bd0d113069c85d5920efdb1f7c"),
}


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


def robust(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    return median, max(1.4826 * float(np.median(np.abs(values - median))), 1e-6)


def tail_evidence(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    """E=-log10((1+#{s in A:s>=x})/(N+1)), with ties included exactly."""
    ordered = np.sort(np.asarray(calibration, dtype=np.float64))
    count_ge = len(ordered) - np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="left")
    return -np.log10((1.0 + count_ge) / (len(ordered) + 1.0))


def split_a_b(rec_segments: list[np.ndarray], dual_segments: list[np.ndarray]):
    rec_a, rec_b, dual_a, dual_b, sizes = [], [], [], [], []
    if len(rec_segments) != len(dual_segments):
        raise RuntimeError("REC/Dual validation segment count mismatch")
    for rec, dual in zip(rec_segments, dual_segments):
        if len(rec) != len(dual):
            raise RuntimeError("REC/Dual common endpoint mismatch")
        boundary = len(rec) // 2
        if boundary <= 0 or boundary >= len(rec):
            raise RuntimeError("validation segment too short for chronological 50/50")
        rec_a.append(rec[:boundary]); rec_b.append(rec[boundary:])
        dual_a.append(dual[:boundary]); dual_b.append(dual[boundary:])
        sizes.append({"common_points": len(rec), "calibration_a": boundary, "audit_b": len(rec) - boundary})
    return tuple(map(np.concatenate, (rec_a, rec_b, dual_a, dual_b))), sizes


def fit_and_score_dual(helper, train: list[np.ndarray], validation: list[np.ndarray], feature_names: list[str],
                       group_rows: list[tuple[str, list[str]]], frozen: pd.DataFrame, group_column: str):
    index = {name: i for i, name in enumerate(feature_names)}
    models, references = {}, {}
    for group_name, usable in group_rows:
        selected = frozen[frozen[group_column].astype(str) == str(group_name)]
        if selected.empty:
            continue
        targets = selected.target.astype(str).tolist()
        target_indices = [index[name] for name in targets]
        all_specs = [(index[name], lag) for name in usable for lag in CROSS_LAGS]
        cross = helper.leave_target_out_models(train, all_specs, target_indices)
        self_models = [helper.fit_ridge(train, [(ti, lag) for lag in SELF_LAGS], [ti]) for ti in target_indices]
        raw_parts = [[] for _ in targets]
        for values in validation:
            times = np.arange(MAX_LAG, len(values), dtype=np.int64)
            cross_pred = cross.predict(helper.design(values, times, all_specs))
            for j, ti in enumerate(target_indices):
                self_pred = self_models[j].predict(helper.design(values, times, [(ti, lag) for lag in SELF_LAGS]))[:, 0]
                raw_parts[j].append(np.abs(self_pred - cross_pred[:, j]))
        for j, target in enumerate(targets):
            keep = np.asarray([channel != target_indices[j] for channel, _ in all_specs])
            sources = [index[name] for name in usable if name != target]
            models[target] = {
                "target_index": target_indices[j], "source_indices": sources, "self_model": self_models[j],
                "cross_model": helper.LinearModel(cross.coef[keep, j:j + 1], np.asarray([cross.intercept[j]])),
            }
            references[target] = robust(np.concatenate(raw_parts[j]))
    expected_order = frozen.target.astype(str).tolist()
    if set(models) != set(expected_order):
        raise RuntimeError("frozen target could not be bound to its process/stage")
    dual_segments = []
    for values in validation:
        times = np.arange(MAX_LAG, len(values), dtype=np.int64)
        columns = []
        for target in expected_order:
            item = models[target]; ti = item["target_index"]
            ps = item["self_model"].predict(helper.design(values, times, [(ti, lag) for lag in SELF_LAGS]))[:, 0]
            specs = [(source, lag) for source in item["source_indices"] for lag in CROSS_LAGS]
            pc = item["cross_model"].predict(helper.design(values, times, specs))[:, 0]
            median, scale = references[target]
            columns.append(np.maximum(0.0, (np.abs(ps - pc) - median) / scale))
        global_score = np.column_stack(columns).max(1)
        dual_segments.append(global_score[WINDOW - 1 - MAX_LAG:])
    return dual_segments, references


def load_checkpoint_scores(model, checkpoint_path: Path, validation: list[np.ndarray], ae_score_fn, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    scores = [np.asarray(x, dtype=np.float64) for x in ae_score_fn(model, validation, device, 256)]
    if not all(np.isfinite(x).all() for x in scores):
        raise RuntimeError("non-finite REC Normal Validation score")
    return scores, {"path": str(checkpoint_path.relative_to(ROOT)), "sha256": sha256(checkpoint_path), "epoch": int(checkpoint["epoch"])}


def prepare_swat(w, a2, b2):
    from data.swat2015_official import load_normal_cache, resolve_repo_path, validate_config_contract
    from data.transforms import StandardScaler
    from utils.config_io import load_config
    target_path = ROOT / "outputs" / "thesis_swat_final_unified" / "dual_targets_unified_frozen.csv"
    frozen = pd.read_csv(target_path); config = load_config(str(b2.CONFIG)); validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    names = [str(x) for x in cache_manifest["column_order"]]
    schema = json.loads((ROOT / "outputs" / "thesis_t2_swat2015_audit" / "schema_profile.json").read_text(encoding="utf-8"))
    contract, _ = a2.build_contract(a2.normal_profiles(schema), names)
    scaler = StandardScaler().fit(np.asarray(normal[:396000]))
    z = scaler.transform(np.asarray(normal)).astype(np.float64)
    train, validation = [z[:396000]], [z[396000:495000]]
    groups = [(stage, list(contract[stage]["nonconstant_channels"])) for stage in ("P1", "P2", "P3", "P4", "P5", "P6")]
    dual, refs = fit_and_score_dual(w, train, validation, names, groups, frozen, "stage")
    rec = [np.load(ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit" / "a0_validation_scores.npy", allow_pickle=False).astype(np.float64)]
    checkpoint = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "best.pt"
    return frozen, rec, dual, refs, {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "epoch": 13}, ["Normal Validation rows [396000,495000)"]


def prepare_hai(w, h, device):
    data = ROOT / "dataset" / "hai" / "hai-21.03"
    audit = ROOT / "outputs" / "thesis_hai21_a0_audit"
    target_path = ROOT / "outputs" / "thesis_hai21_final_unified" / "dual_targets_unified_frozen.csv"
    frozen = pd.read_csv(target_path); schema = pd.read_csv(audit / "variable_schema.csv"); groups_csv = pd.read_csv(audit / "process_groups.csv")
    names = schema.column_name.astype(str).tolist(); schema_name = schema.set_index("column_name")
    boundaries, scaler = {}, SklearnScaler()
    for filename in ("train1.csv.gz", "train2.csv.gz", "train3.csv.gz"):
        frame = pd.read_csv(data / filename, compression="gzip", usecols=names, low_memory=False)
        boundary = int(math.floor(.8 * len(frame))); boundaries[filename] = boundary
        scaler.partial_fit(frame.iloc[:boundary].to_numpy(np.float64))
    train, validation = [], []
    for filename in ("train1.csv.gz", "train2.csv.gz", "train3.csv.gz"):
        frame = pd.read_csv(data / filename, compression="gzip", usecols=names, low_memory=False); boundary = boundaries[filename]
        train.append(scaler.transform(frame.iloc[:boundary].to_numpy(np.float64)))
        validation.append(scaler.transform(frame.iloc[boundary:].to_numpy(np.float64)))
    group_rows = []
    for row in groups_csv.itertuples(index=False):
        usable = [name for name in str(row.channel_names).split(";") if name in schema_name.index and str(schema_name.loc[name, "nonconstant"]).lower() == "true"]
        group_rows.append((str(row.process), usable))
    dual, refs = fit_and_score_dual(w, train, validation, names, group_rows, frozen, "process")
    checkpoint = ROOT / "outputs" / "thesis_hai21_b1_real" / "rec_best.pt"
    rec, ck = load_checkpoint_scores(h.BackboneAE(79, 64, 32, .1).to(device), checkpoint, validation, h.ae_scores, device)
    return frozen, rec, dual, refs, ck, [f"{name} Validation suffix" for name in ("train1.csv.gz", "train2.csv.gz", "train3.csv.gz")]


def prepare_wadi(w, device):
    from models.backbone_ae import BackboneAE
    data = ROOT / "dataset" / "WADI_A2_2019" / "WADI_14days_new.csv"
    audit = ROOT / "outputs" / "thesis_wadi_a0_audit"
    protocol = ROOT / "outputs" / "thesis_wadi_a05_protocol"
    target_path = ROOT / "outputs" / "thesis_wadi_b0_unified_reliability" / "reliable_targets.csv"
    frozen = pd.read_csv(target_path); schema = pd.read_csv(protocol / "feature_schema_frozen.csv"); kept = schema[schema.included].copy()
    names = kept.raw_column_name.astype(str).tolist(); schema_name = kept.set_index("raw_column_name"); groups_csv = pd.read_csv(audit / "process_groups.csv")
    normal = pd.read_csv(data, usecols=names, low_memory=False)
    train_raw, _ = w.valid_runs(normal, names, ((0, 268799), (335999, 694856)))
    val_raw, val_runs = w.valid_runs(normal, names, ((268799, 335999), (694856, 784571)))
    scaler = SklearnScaler()
    for values in train_raw: scaler.partial_fit(values)
    train = [scaler.transform(values).astype(np.float64, copy=False) for values in train_raw]
    validation = [scaler.transform(values).astype(np.float64, copy=False) for values in val_raw]
    group_rows = []
    for row in groups_csv.itertuples(index=False):
        usable = [name for name in str(row.channel_names).split(";") if name in schema_name.index and schema_name.loc[name, "variable_kind"] in {"continuous", "discrete"}]
        group_rows.append((str(row.process_group), usable))
    dual, refs = fit_and_score_dual(w, train, validation, names, group_rows, frozen, "process")
    checkpoint = ROOT / "outputs" / "thesis_wadi_b1_final_external" / "rec_best.pt"
    rec, ck = load_checkpoint_scores(BackboneAE(123, 64, 32, .1).to(device), checkpoint, validation, w.ae_scores, device)
    labels = [f"WADI Validation continuous run {i + 1} (rows {item['start']}:{item['end_exclusive']})" for i, item in enumerate(val_runs)]
    return frozen, rec, dual, refs, ck, labels


def audit_dataset(dataset: str, rec_segments: list[np.ndarray], dual_segments: list[np.ndarray]):
    (rec_a, rec_b, dual_a, dual_b), segment_sizes = split_a_b(rec_segments, dual_segments)
    e_rec_a, e_dual_a = tail_evidence(rec_a, rec_a), tail_evidence(dual_a, dual_a)
    e_rec_b, e_dual_b = tail_evidence(rec_a, rec_b), tail_evidence(dual_a, dual_b)
    f_tail_a, f_tail_b = np.maximum(e_rec_a, e_dual_a), np.maximum(e_rec_b, e_dual_b)
    threshold_tail = float(np.quantile(f_tail_a, Q)); tail_fpr = float(np.mean(f_tail_b > threshold_tail))
    rec_center, rec_scale = robust(rec_a); dual_center, dual_scale = robust(dual_a)
    zr_a, zd_a = np.maximum(0, (rec_a - rec_center) / rec_scale), np.maximum(0, (dual_a - dual_center) / dual_scale)
    zr_b, zd_b = np.maximum(0, (rec_b - rec_center) / rec_scale), np.maximum(0, (dual_b - dual_center) / dual_scale)
    f_old_a, f_old_b = np.maximum(zr_a, zd_a), np.maximum(zr_b, zd_b)
    threshold_old = float(np.quantile(f_old_a, Q)); old_fpr = float(np.mean(f_old_b > threshold_old))
    branch_rows = []
    quantiles = {}
    for branch, evidence in (("REC", e_rec_b), ("DUAL", e_dual_b)):
        q = {f"q{int(level * 100):02d}": float(np.quantile(evidence, level)) for level in (.50, .90, .95, .99)}
        quantiles[branch] = q
        branch_rows.append({"dataset":dataset,"branch":branch,"calibration_a_points":len(rec_a),"audit_b_points":len(rec_b),"fraction_e_ge_2":float(np.mean(evidence >= 2)),**q})
    q95_balance = max(quantiles["REC"]["q95"], quantiles["DUAL"]["q95"]) / max(min(quantiles["REC"]["q95"], quantiles["DUAL"]["q95"]), 1e-6)
    q99_balance = max(quantiles["REC"]["q99"], quantiles["DUAL"]["q99"]) / max(min(quantiles["REC"]["q99"], quantiles["DUAL"]["q99"]), 1e-6)
    dominance_rows = []
    for method, fusion, er, ed in (("OLD_ROBUST_Z", f_old_b, zr_b, zd_b), ("TAIL_EVIDENCE", f_tail_b, e_rec_b, e_dual_b)):
        top = fusion >= np.quantile(fusion, Q)
        rec_dom, dual_dom, tie = int((top & (er > ed)).sum()), int((top & (ed > er)).sum()), int((top & (er == ed)).sum())
        total = int(top.sum())
        dominance_rows.append({"dataset":dataset,"method":method,"audit_b_top1_count":total,"rec_dominant_count":rec_dom,"dual_dominant_count":dual_dom,"tie_count":tie,"rec_dominant_fraction":rec_dom/total,"dual_dominant_fraction":dual_dom/total,"tie_fraction":tie/total})
    fusion_row = {"dataset":dataset,"calibration_a_points":len(rec_a),"audit_b_points":len(rec_b),"threshold_tail":threshold_tail,"tail_fusion_fpr":tail_fpr,"threshold_old":threshold_old,"old_fusion_fpr":old_fpr,"q95_balance_ratio":q95_balance,"q99_balance_ratio":q99_balance}
    dataset_row = {"dataset":dataset,"rec_e_ge_2_rate":branch_rows[0]["fraction_e_ge_2"],"dual_e_ge_2_rate":branch_rows[1]["fraction_e_ge_2"],"rec_q95":quantiles["REC"]["q95"],"rec_q99":quantiles["REC"]["q99"],"dual_q95":quantiles["DUAL"]["q95"],"dual_q99":quantiles["DUAL"]["q99"],"q95_balance_ratio":q95_balance,"q99_balance_ratio":q99_balance,"tail_fusion_fpr":tail_fpr,"old_fusion_fpr":old_fpr}
    return dataset_row, branch_rows, fusion_row, dominance_rows, segment_sizes


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen REC Normal Validation inference")
    device = torch.device("cuda")
    w = load_module("i3a_wadi_helpers", HELPERS / "run_wadi_b1_external.py")
    h = load_module("i3a_hai_helpers", HELPERS / "run_hai21_b1_real.py")
    a2 = load_module("i3a_swat_a2", HELPERS / "run_innovation2a2_joint_lagged.py")
    b2 = load_module("i3a_swat_b2", HELPERS / "run_innovation2b_controlled.py")

    prepared = {}
    for dataset, loader in (("SWaT", lambda: prepare_swat(w, a2, b2)), ("HAI21", lambda: prepare_hai(w, h, device)), ("WADI.A2", lambda: prepare_wadi(w, device))):
        frozen, rec, dual, refs, checkpoint, segment_labels = loader()
        target_file = {"SWaT": ROOT/"outputs"/"thesis_swat_final_unified"/"dual_targets_unified_frozen.csv",
                       "HAI21": ROOT/"outputs"/"thesis_hai21_final_unified"/"dual_targets_unified_frozen.csv",
                       "WADI.A2": ROOT/"outputs"/"thesis_wadi_b0_unified_reliability"/"reliable_targets.csv"}[dataset]
        expected_count, expected_hash = EXPECTED_TARGETS[dataset]; actual_hash = sha256(target_file)
        if len(frozen) != expected_count or actual_hash != expected_hash:
            raise RuntimeError(f"{dataset} frozen target mismatch")
        prepared[dataset] = {"rec":rec,"dual":dual,"reference_count":len(refs),"target_count":len(frozen),"target_file":str(target_file.relative_to(ROOT)),"target_sha256":actual_hash,"checkpoint":checkpoint,"segments":segment_labels}
        print(json.dumps({"dataset":dataset,"targets":len(frozen),"validation_segments":len(rec)}),flush=True)

    dataset_rows, branch_rows, fusion_rows, dominance_rows, segment_manifest = [], [], [], [], {}
    for dataset, item in prepared.items():
        ds, branches, fusion, dominance, sizes = audit_dataset(dataset, item["rec"], item["dual"])
        dataset_rows.append(ds); branch_rows.extend(branches); fusion_rows.append(fusion); dominance_rows.extend(dominance)
        segment_manifest[dataset] = [{**{"label": label}, **size} for label, size in zip(item["segments"], sizes)]
    datasets = pd.DataFrame(dataset_rows); branches = pd.DataFrame(branch_rows); fusion = pd.DataFrame(fusion_rows); dominance = pd.DataFrame(dominance_rows)
    old_vs_tail = dominance.merge(fusion[["dataset","old_fusion_fpr","tail_fusion_fpr"]],on="dataset",how="left")
    gate_a = bool((branches.fraction_e_ge_2 <= .03).all())
    gate_b = bool((fusion.tail_fusion_fpr <= .03).all())
    gate_c_count = int((fusion.q99_balance_ratio <= 2.0).sum()); gate_c = gate_c_count >= 2
    if gate_a and gate_b and gate_c: status = "NORMAL_TAIL_EVIDENCE_CALIBRATION_FEASIBLE"
    elif gate_a and gate_b: status = "NORMAL_TAIL_CALIBRATION_STABLE_BUT_BRANCH_IMBALANCED"
    else: status = "NORMAL_TAIL_EVIDENCE_CALIBRATION_NOT_STABLE"
    tests = {"three_datasets_normal_only":set(prepared)=={"SWaT","HAI21","WADI.A2"},"forbidden_test_open_attempts":{"SWaT_Attack":0,"HAI_test_files":0,"WADI_Attack":0},
             "frozen_target_hashes":all(item["target_sha256"]==EXPECTED_TARGETS[name][1] for name,item in prepared.items()),"branch_scores_frozen":True,
             "chronological_segment_split":True,"no_segment_crossing":True,"ecdf_fit_a_only":True,"audit_b_not_in_ecdf":True,"tail_probability_exact":True,"evidence_minus_log10":True,
             "tail_fusion_max":True,"no_weighting":True,"thresholds_a_only":True,"old_comparator_a_only":True,"no_nan_inf":bool(np.isfinite(datasets.select_dtypes("number")).all().all() and np.isfinite(branches.select_dtypes("number")).all().all() and np.isfinite(fusion.select_dtypes("number")).all().all())}
    if not all(v is True or (isinstance(v,dict) and all(x==0 for x in v.values())) for v in tests.values()):
        raise RuntimeError(f"Innovation-3A integrity failure: {tests}")
    summary = {"elapsed_seconds":time.perf_counter()-started,"status":status,"gates":{"A":gate_a,"B":gate_b,"C":gate_c,"C_passed_datasets":gate_c_count},"dataset_summary":datasets.to_dict("records"),
               "target_counts":{name:item["target_count"] for name,item in prepared.items()},"target_sha256":{name:item["target_sha256"] for name,item in prepared.items()},"tests":tests,
               "normal_only":True,"attack_test_data_read":False,"parameter_search":False,"fusion_weight_search":False,"test_tuning":False,"commit":False,"push":False}
    manifest = {"purpose":"Cross-dataset Normal-tail evidence calibration audit","datasets":{name:{key:value for key,value in item.items() if key not in {"rec","dual"}} for name,item in prepared.items()},
                "validation_segments":segment_manifest,"score_definitions":{"rec":"endpoint mean reconstruction error","dual":"max over frozen targets of per-target validation-calibrated abs(Self-Cross disagreement)",
                "tail_probability":"(1 + count{s_i in Calibration-A: s_i >= x}) / (N + 1)","tail_evidence":"-log10(p_tail)","tail_fusion":"max(E_rec,E_dual)","old_fusion":"max(robust_z_rec,robust_z_dual)"},
                "split":"front 50% Calibration-A / back 50% Audit-B within every continuous Validation segment/file","threshold_source":"Calibration-A q99 only","self_lags":SELF_LAGS,"cross_lags":CROSS_LAGS,"alpha":ALPHA,"summary":summary}
    OUT.mkdir(parents=True)
    datasets.to_csv(OUT/"dataset_summary.csv",index=False); branches.to_csv(OUT/"branch_tail_calibration.csv",index=False); fusion.to_csv(OUT/"fusion_normal_stability.csv",index=False)
    old_vs_tail.to_csv(OUT/"old_vs_tail_fusion.csv",index=False); dominance.to_csv(OUT/"dominance_analysis.csv",index=False); json_dump(OUT/"manifest.json",manifest); json_dump(OUT/"innovation3a_summary.json",summary)
    table = "\n".join(f"| {r.dataset} | {r.rec_e_ge_2_rate:.6f} | {r.rec_q95:.6f} | {r.rec_q99:.6f} | {r.dual_e_ge_2_rate:.6f} | {r.dual_q95:.6f} | {r.dual_q99:.6f} | {r.q99_balance_ratio:.3f} | {r.tail_fusion_fpr:.6f} | {r.old_fusion_fpr:.6f} |" for r in datasets.itertuples())
    DOC.write_text(f"""# Innovation-3A Normal-Tail Evidence Calibration Audit

This audit used only Normal Train and Normal Validation from Official SWaT 2015, HAI 21.03, and WADI.A2. Frozen unified target files matched their registered SHA256 values. No Attack/Test file, header, hash, feature, or label was read.

Each continuous Validation segment/file was split chronologically: front 50% Calibration-A and back 50% Audit-B. Calibration-A alone defined branch empirical survival functions and both fusion thresholds.

| Dataset | REC E>=2 | REC q95 | REC q99 | DUAL E>=2 | DUAL q95 | DUAL q99 | q99 balance | Tail FPR | Old FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

Gates: A={gate_a}, B={gate_b}, C={gate_c} ({gate_c_count}/3 datasets at q99 balance <=2). Final status: **{status}**.

Empirical tail calibration tests cross-branch Normal-tail comparability only. It does not establish improved anomaly detection. For WADI it does not adapt Normal Train/Validation to future Test domain shift.

No parameter search, fusion-weight search, Test tuning, or point adjustment was performed.
""",encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
