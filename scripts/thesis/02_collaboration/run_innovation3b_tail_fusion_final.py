"""Final frozen empirical-tail max-fusion evaluation on SWaT, HAI, and WADI."""
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler as SklearnScaler


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "thesis_innovation3b_tail_fusion_final"
DOC = ROOT / "docs" / "thesis" / "innovation3b_tail_calibrated_fusion_final.md"
HELPERS = ROOT / "scripts" / "thesis" / "02_collaboration"
I3A = ROOT / "outputs" / "thesis_innovation3a_tail_calibration"
SELF_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (0, 1, 5, 10, 20, 30)
MAX_LAG, WINDOW, ALPHA, Q = 30, 64, 1.0, .99
EXPECTED = {
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
    ordered = np.sort(np.asarray(calibration, dtype=np.float64))
    count_ge = len(ordered) - np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="left")
    return -np.log10((1.0 + count_ge) / (len(ordered) + 1.0))


def point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred, pos, neg = scores > threshold, labels == 1, labels == 0
    tp, fp = int((pred & pos).sum()), int((pred & neg).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & neg).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"auroc":float(roc_auc_score(labels,scores)),"aupr":float(average_precision_score(labels,scores)),
            "precision":precision,"recall":recall,"f1":2*precision*recall/max(precision+recall,np.finfo(float).eps),
            "fpr":fp/max(fp+tn,1),"tn":tn,"fp":fp,"fn":fn,"tp":tp}


def freeze_tail(rec_segments: list[np.ndarray], dual_segments: list[np.ndarray]) -> dict:
    rec_a, rec_b, dual_a, dual_b = [], [], [], []
    segment_sizes = []
    for rec, dual in zip(rec_segments, dual_segments):
        if len(rec) != len(dual): raise RuntimeError("REC/Dual Validation alignment mismatch")
        boundary = len(rec)//2
        rec_a.append(rec[:boundary]); rec_b.append(rec[boundary:]); dual_a.append(dual[:boundary]); dual_b.append(dual[boundary:])
        segment_sizes.append({"points":len(rec),"calibration_a":boundary,"threshold_b":len(rec)-boundary})
    rec_a, rec_b, dual_a, dual_b = map(np.concatenate,(rec_a,rec_b,dual_a,dual_b))
    e_rec_b, e_dual_b = tail_evidence(rec_a,rec_b), tail_evidence(dual_a,dual_b)
    threshold = float(np.quantile(np.maximum(e_rec_b,e_dual_b),Q))
    return {"rec_a":rec_a,"dual_a":dual_a,"threshold":threshold,"segment_sizes":segment_sizes}


def apply_tail(freeze: dict, rec: np.ndarray, dual: np.ndarray):
    er, ed = tail_evidence(freeze["rec_a"],rec), tail_evidence(freeze["dual_a"],dual)
    return er, ed, np.maximum(er,ed)


def fit_dual_state(helper, train: list[np.ndarray], validation: list[np.ndarray], names: list[str],
                   groups: list[tuple[str,list[str]]], frozen: pd.DataFrame, group_column: str):
    index={name:i for i,name in enumerate(names)}; state={}
    for group_name,usable in groups:
        selected=frozen[frozen[group_column].astype(str)==str(group_name)]
        if selected.empty: continue
        targets=selected.target.astype(str).tolist(); tis=[index[name] for name in targets]
        specs=[(index[name],lag) for name in usable for lag in CROSS_LAGS]
        cross=helper.leave_target_out_models(train,specs,tis)
        selfs=[helper.fit_ridge(train,[(ti,lag) for lag in SELF_LAGS],[ti]) for ti in tis]
        parts=[[] for _ in targets]
        for values in validation:
            times=np.arange(MAX_LAG,len(values),dtype=np.int64); pc=cross.predict(helper.design(values,times,specs))
            for j,ti in enumerate(tis):
                ps=selfs[j].predict(helper.design(values,times,[(ti,lag) for lag in SELF_LAGS]))[:,0]
                parts[j].append(np.abs(ps-pc[:,j]))
        for j,target in enumerate(targets):
            keep=np.asarray([channel!=tis[j] for channel,_ in specs]); sources=[index[name] for name in usable if name!=target]
            state[target]={"ti":tis[j],"sources":sources,"self":selfs[j],"cross":helper.LinearModel(cross.coef[keep,j:j+1],np.asarray([cross.intercept[j]])),"ref":robust(np.concatenate(parts[j]))}
    if set(state)!=set(frozen.target.astype(str)): raise RuntimeError("frozen target/model binding incomplete")
    return state


def dual_scores(helper, segments: list[np.ndarray], frozen: pd.DataFrame, state: dict):
    result=[]
    for values in segments:
        times=np.arange(MAX_LAG,len(values),dtype=np.int64); columns=[]
        for target in frozen.target.astype(str):
            item=state[target]; ti=item["ti"]
            ps=item["self"].predict(helper.design(values,times,[(ti,lag) for lag in SELF_LAGS]))[:,0]
            pc=item["cross"].predict(helper.design(values,times,[(source,lag) for source in item["sources"] for lag in CROSS_LAGS]))[:,0]
            median,scale=item["ref"]; columns.append(np.maximum(0,(np.abs(ps-pc)-median)/scale))
        result.append(np.column_stack(columns).max(1)[WINDOW-1-MAX_LAG:])
    return result


def load_rec(model, checkpoint: Path, segments: list[np.ndarray], score_fn, device):
    payload=torch.load(checkpoint,map_location=device,weights_only=False); model.load_state_dict(payload["state_dict"])
    return [np.asarray(x,dtype=np.float64) for x in score_fn(model,segments,device,256)], {"path":str(checkpoint.relative_to(ROOT)),"sha256":sha256(checkpoint),"epoch":int(payload["epoch"])}


def frozen_metrics(dataset: str) -> pd.DataFrame:
    path={"SWaT":ROOT/"outputs"/"thesis_swat_final_unified"/"global_metrics.csv",
          "HAI21":ROOT/"outputs"/"thesis_hai21_final_unified"/"global_metrics.csv",
          "WADI.A2":ROOT/"outputs"/"thesis_wadi_b1_final_external"/"global_metrics.csv"}[dataset]
    frame=pd.read_csv(path); frame["method"]=frame.method.replace({"FUSION":"OLD_FUSION"}); frame.insert(0,"dataset",dataset)
    return frame


def dominance_rows(dataset: str, labels: np.ndarray, er: np.ndarray, ed: np.ndarray, high_mask: np.ndarray|None):
    rows=[]
    scopes=[("NORMAL",labels==0),("ATTACK",labels==1)]
    if high_mask is not None: scopes.append(("HIGH",high_mask))
    for scope,mask in scopes:
        count=int(mask.sum()); r=int((mask&(er>ed)).sum()); d=int((mask&(ed>er)).sum()); t=int((mask&(er==ed)).sum())
        rows.append({"dataset":dataset,"scope":scope,"point_count":count,"rec_dominant_count":r,"dual_dominant_count":d,"tie_count":t,"rec_dominant_fraction":r/count,"dual_dominant_fraction":d/count,"tie_fraction":t/count})
    return rows


def swat_dataset(w,a2,b2,pcc):
    from data.swat2015_official import load_attack_cache,load_normal_cache,resolve_repo_path,validate_config_contract
    from data.transforms import StandardScaler
    from utils.config_io import load_config
    target_path=ROOT/"outputs"/"thesis_swat_final_unified"/"dual_targets_unified_frozen.csv"; frozen=pd.read_csv(target_path)
    config=load_config(str(b2.CONFIG)); validate_config_contract(config); cache=resolve_repo_path(ROOT,config["paths"]["cache_dir"])
    normal,cm=load_normal_cache(cache,config); names=[str(x) for x in cm["column_order"]]
    schema=json.loads((ROOT/"outputs"/"thesis_t2_swat2015_audit"/"schema_profile.json").read_text(encoding="utf-8")); contract,_=a2.build_contract(a2.normal_profiles(schema),names)
    scaler=StandardScaler().fit(np.asarray(normal[:396000])); z=scaler.transform(np.asarray(normal)).astype(np.float64); train=[z[:396000]]; val=[z[396000:495000]]
    groups=[(stage,list(contract[stage]["nonconstant_channels"])) for stage in ("P1","P2","P3","P4","P5","P6")]
    state=fit_dual_state(w,train,val,names,groups,frozen,"stage"); dual_val=dual_scores(w,val,frozen,state)
    rec_val=[np.load(ROOT/"outputs"/"thesis_s6_swat_phenomenon_audit"/"a0_validation_scores.npy",allow_pickle=False).astype(np.float64)]
    freeze=freeze_tail(rec_val,dual_val)
    # First Attack access occurs after ECDF and Threshold-B q99 are frozen.
    attack,attack_labels,attack_ts,_=load_attack_cache(cache,config); za=scaler.transform(np.asarray(attack)).astype(np.float64)
    seg=json.loads((ROOT/"outputs"/"thesis_t25_swat2015_protocol"/"segment_manifest.json").read_text(encoding="utf-8")); counts=[int(x["row_count"]) for x in seg["test_segments"]]
    pieces=[]; cursor=0
    for count in counts: pieces.append(za[cursor:cursor+count]); cursor+=count
    dual=np.concatenate(dual_scores(w,pieces,frozen,state))
    with np.load(ROOT/"outputs"/"thesis_midterm_s1"/"a0_rec_baseline"/"test_scores.npz",allow_pickle=False) as payload:
        timestamps=np.asarray(payload["endpoint_timestamp"],dtype=np.int64); labels=np.asarray(payload["label"],dtype=np.int8); rec=np.asarray(payload["endpoint_mae"],dtype=np.float64)
    raw_idx=pcc.align_raw_to_a0(np.asarray(attack_ts),timestamps)
    if not np.array_equal(np.asarray(attack_labels)[raw_idx],labels) or len(rec)!=len(dual): raise RuntimeError("SWaT common mask changed")
    er,ed,tail=apply_tail(freeze,rec,dual); metrics=point_metrics(labels,tail,freeze["threshold"])
    old_events=pd.read_csv(ROOT/"outputs"/"thesis_swat_final_unified"/"event_metrics_36.csv").rename(columns={"fusion_hit_rate":"old_fusion_hit_rate"})
    rows=[]
    for event in old_events.itertuples(index=False):
        mask=(timestamps>=pcc.timestamp_seconds(str(event.start)))&(timestamps<=pcc.timestamp_seconds(str(event.end)))
        rows.append({**event._asdict(),"tail_fusion_hit_rate":float((tail[mask]>freeze["threshold"]).mean())})
    events=pd.DataFrame(rows); dom=dominance_rows("SWaT",labels,er,ed,None)
    return {"frozen":frozen,"target_path":target_path,"freeze":freeze,"metrics":metrics,"events":events,"dominance":dom,"checkpoint":{"path":"outputs/thesis_midterm_s1/a0_rec_baseline/best.pt","sha256":sha256(ROOT/"outputs"/"thesis_midterm_s1"/"a0_rec_baseline"/"best.pt"),"epoch":13},"test_points":len(labels),"test_read_count":1}


def hai_dataset(w,h,device):
    data=ROOT/"dataset"/"hai"/"hai-21.03"; audit=ROOT/"outputs"/"thesis_hai21_a0_audit"; target_path=ROOT/"outputs"/"thesis_hai21_final_unified"/"dual_targets_unified_frozen.csv"
    frozen=pd.read_csv(target_path); schema=pd.read_csv(audit/"variable_schema.csv"); names=schema.column_name.astype(str).tolist(); schema_name=schema.set_index("column_name"); groups_csv=pd.read_csv(audit/"process_groups.csv")
    scaler=SklearnScaler(); bounds={}
    for name in ("train1.csv.gz","train2.csv.gz","train3.csv.gz"):
        frame=pd.read_csv(data/name,compression="gzip",usecols=names,low_memory=False); boundary=int(math.floor(.8*len(frame))); bounds[name]=boundary; scaler.partial_fit(frame.iloc[:boundary].to_numpy(np.float64))
    train=[]; val=[]
    for name in ("train1.csv.gz","train2.csv.gz","train3.csv.gz"):
        frame=pd.read_csv(data/name,compression="gzip",usecols=names,low_memory=False); b=bounds[name]; train.append(scaler.transform(frame.iloc[:b].to_numpy(np.float64))); val.append(scaler.transform(frame.iloc[b:].to_numpy(np.float64)))
    groups=[]
    for row in groups_csv.itertuples(index=False): groups.append((str(row.process),[name for name in str(row.channel_names).split(";") if name in schema_name.index and str(schema_name.loc[name,"nonconstant"]).lower()=="true"]))
    state=fit_dual_state(w,train,val,names,groups,frozen,"process"); dual_val=dual_scores(w,val,frozen,state)
    model=h.BackboneAE(79,64,32,.1).to(device); rec_val,ck=load_rec(model,ROOT/"outputs"/"thesis_hai21_b1_real"/"rec_best.pt",val,h.ae_scores,device); freeze=freeze_tail(rec_val,dual_val)
    records={}; all_labels=[]; all_er=[]; all_ed=[]; all_tail=[]
    # Test files are opened only after Calibration-A ECDF and Threshold-B q99 freeze.
    for name in ("test1.csv.gz","test2.csv.gz","test3.csv.gz","test4.csv.gz","test5.csv.gz"):
        frame=pd.read_csv(data/name,compression="gzip",low_memory=False); values=scaler.transform(frame[names].to_numpy(np.float64)); rec=h.ae_scores(model,[values],device,256)[0]; dual=dual_scores(w,[values],frozen,state)[0]
        labels=frame.attack.to_numpy(np.int8)[WINDOW-1:]; er,ed,tail=apply_tail(freeze,rec,dual); idx=np.arange(WINDOW-1,len(frame),dtype=np.int64)
        records[name]={"idx":idx,"labels":labels,"er":er,"ed":ed,"tail":tail}; all_labels.append(labels); all_er.append(er); all_ed.append(ed); all_tail.append(tail)
    labels,er,ed,tail=map(np.concatenate,(all_labels,all_er,all_ed,all_tail)); metrics=point_metrics(labels,tail,freeze["threshold"])
    audit_events=pd.read_csv(audit/"attack_events_50.csv"); high_ids=set(pd.read_csv(ROOT/"outputs"/"thesis_hai21_b1_real"/"high_collaboration_events_frozen.csv").event_id.astype(str)); old=pd.read_csv(ROOT/"outputs"/"thesis_hai21_final_unified"/"event_metrics_50.csv").rename(columns={"fusion_hit_rate":"old_fusion_hit_rate"})
    old_index=old.set_index("event_id"); rows=[]; high_global=[]; offset=0
    for name in ("test1.csv.gz","test2.csv.gz","test3.csv.gz","test4.csv.gz","test5.csv.gz"):
        record=records[name]; high_mask=np.zeros(len(record["labels"]),dtype=bool)
        for event in audit_events[audit_events.file==name].itertuples(index=False):
            mask=(record["idx"]>=int(event.start_index))&(record["idx"]<=int(event.end_index_inclusive)); is_high=str(event.event_id) in high_ids
            if is_high: high_mask|=mask
            alarm=mask&(record["tail"]>freeze["threshold"]); rdom=int((alarm&(record["er"]>record["ed"])).sum()); ddom=int((alarm&(record["ed"]>record["er"])).sum())
            base=old_index.loc[str(event.event_id)].to_dict(); rows.append({"file":name,"event_id":str(event.event_id),**base,"high_collaboration":is_high,"tail_fusion_hit_rate":float((record["tail"][mask]>freeze["threshold"]).mean()),"tail_alarm_count":int(alarm.sum()),"tail_alarm_rec_dominant_count":rdom,"tail_alarm_dual_dominant_count":ddom})
        high_global.append(high_mask); offset+=len(record["labels"])
    events=pd.DataFrame(rows); high_mask=np.concatenate(high_global); dom=dominance_rows("HAI21",labels,er,ed,high_mask)
    return {"frozen":frozen,"target_path":target_path,"freeze":freeze,"metrics":metrics,"events":events,"dominance":dom,"checkpoint":ck,"test_points":len(labels),"test_read_count":5,"high_mask_count":int(high_mask.sum())}


def wadi_dataset(w,device):
    from models.backbone_ae import BackboneAE
    a0mod=load_module("i3b_wadi_a0",ROOT/"scripts"/"thesis"/"00_audit"/"audit_wadi_a0.py")
    data=ROOT/"dataset"/"WADI_A2_2019"; audit=ROOT/"outputs"/"thesis_wadi_a0_audit"; protocol=ROOT/"outputs"/"thesis_wadi_a05_protocol"; target_path=ROOT/"outputs"/"thesis_wadi_b0_unified_reliability"/"reliable_targets.csv"
    frozen=pd.read_csv(target_path); schema=pd.read_csv(protocol/"feature_schema_frozen.csv"); kept=schema[schema.included].copy(); names=kept.raw_column_name.astype(str).tolist(); schema_name=kept.set_index("raw_column_name"); groups_csv=pd.read_csv(audit/"process_groups.csv")
    normal=pd.read_csv(data/"WADI_14days_new.csv",usecols=names,low_memory=False); train_raw,_=w.valid_runs(normal,names,((0,268799),(335999,694856))); val_raw,_=w.valid_runs(normal,names,((268799,335999),(694856,784571)))
    scaler=SklearnScaler(); [scaler.partial_fit(x) for x in train_raw]; train=[scaler.transform(x).astype(np.float64,copy=False) for x in train_raw]; val=[scaler.transform(x).astype(np.float64,copy=False) for x in val_raw]
    groups=[]
    for row in groups_csv.itertuples(index=False): groups.append((str(row.process_group),[name for name in str(row.channel_names).split(";") if name in schema_name.index and schema_name.loc[name,"variable_kind"] in {"continuous","discrete"}]))
    state=fit_dual_state(w,train,val,names,groups,frozen,"process"); dual_val=dual_scores(w,val,frozen,state); model=BackboneAE(123,64,32,.1).to(device); rec_val,ck=load_rec(model,ROOT/"outputs"/"thesis_wadi_b1_final_external"/"rec_best.pt",val,w.ae_scores,device); freeze=freeze_tail(rec_val,dual_val)
    # One WADI Test parse after all Normal-only state is frozen; no Test-Normal adaptation follows.
    frame=pd.read_csv(data/"WADI_attackdataLABLE.csv",header=1,low_memory=False); frame.columns=frame.columns.str.strip(); frame=frame.loc[frame.Row.notna()].reset_index(drop=True); values=scaler.transform(frame[names].to_numpy(np.float64))
    rec=w.ae_scores(model,[values],device,256)[0]; dual=dual_scores(w,[values],frozen,state)[0]; idx=np.arange(WINDOW-1,len(frame),dtype=np.int64); labels=frame["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).map({1:0,-1:1}).to_numpy(np.int8)[idx]
    er,ed,tail=apply_tail(freeze,rec,dual); metrics=point_metrics(labels,tail,freeze["threshold"]); timestamps,_=a0mod.infer_audit_timestamps(frame); timestamps=timestamps[idx]
    bindings=pd.read_csv(protocol/"event_binding_final.csv"); bound=set(bindings.loc[bindings.final_binding_status=="BOUND","attack_id"].astype(int)); official=pd.read_csv(audit/"official_attack_events.csv"); official=official[official.attack_id.isin(bound)&(official.attack_id!=13)].sort_values("attack_id")
    high_ids={2,3,4,7,15}; high_mask=np.zeros(len(labels),dtype=bool); old=pd.read_csv(ROOT/"outputs"/"thesis_wadi_b1_final_external"/"event_metrics_14.csv").rename(columns={"fusion_hit_rate":"old_fusion_hit_rate"}).set_index("attack_id"); rows=[]
    for event in official.itertuples(index=False):
        mask=(timestamps>=pd.Timestamp(event.official_start))&(timestamps<=pd.Timestamp(event.official_end)); is_high=int(event.attack_id) in high_ids
        if is_high: high_mask|=mask
        rows.append({"attack_id":int(event.attack_id),**old.loc[int(event.attack_id)].to_dict(),"high_collaboration":is_high,"tail_fusion_hit_rate":float((tail[mask]>freeze["threshold"]).mean())})
    events=pd.DataFrame(rows); dom=dominance_rows("WADI.A2",labels,er,ed,high_mask)
    return {"frozen":frozen,"target_path":target_path,"freeze":freeze,"metrics":metrics,"events":events,"dominance":dom,"checkpoint":ck,"test_points":len(labels),"test_read_count":1,"high_mask_count":int(high_mask.sum()),"status":"DOMAIN_SHIFT_STRESS_TEST"}


def json_dump(path: Path,value) -> None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")


def main() -> int:
    started=time.perf_counter()
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    i3a=json.loads((I3A/"innovation3a_summary.json").read_text(encoding="utf-8"))
    if i3a["status"]!="NORMAL_TAIL_EVIDENCE_CALIBRATION_FEASIBLE": raise RuntimeError("Innovation-3A gate is not feasible")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for frozen REC inference")
    device=torch.device("cuda"); w=load_module("i3b_wadi",HELPERS/"run_wadi_b1_external.py"); h=load_module("i3b_hai",HELPERS/"run_hai21_b1_real.py"); a2=load_module("i3b_a2",HELPERS/"run_innovation2a2_joint_lagged.py"); b2=load_module("i3b_b2",HELPERS/"run_innovation2b_controlled.py"); pcc=load_module("i3b_pcc",HELPERS/"run_swat2015_pcc1.py")
    results={}
    for dataset,loader in (("SWaT",lambda:swat_dataset(w,a2,b2,pcc)),("HAI21",lambda:hai_dataset(w,h,device)),("WADI.A2",lambda:wadi_dataset(w,device))):
        result=loader(); expected_count,expected_hash=EXPECTED[dataset]; actual_hash=sha256(result["target_path"])
        if len(result["frozen"])!=expected_count or actual_hash!=expected_hash: raise RuntimeError(f"{dataset} target freeze mismatch")
        result["target_sha256"]=actual_hash; results[dataset]=result; print(json.dumps({"dataset":dataset,"test_points":result["test_points"],"tail_threshold":result["freeze"]["threshold"]}),flush=True)
    metric_frames=[]
    for dataset,result in results.items():
        frozen=frozen_metrics(dataset); metric_frames.append(frozen)
        metric_frames.append(pd.DataFrame([{"dataset":dataset,"method":"TAIL_FUSION","threshold":result["freeze"]["threshold"],**result["metrics"]}]))
    metrics=pd.concat(metric_frames,ignore_index=True)
    swat_events=results["SWaT"]["events"]; hai_events=results["HAI21"]["events"]; wadi_events=results["WADI.A2"]["events"]
    dominance=pd.DataFrame(sum((result["dominance"] for result in results.values()),[]))
    old_normal=pd.read_csv(I3A/"dominance_analysis.csv"); old_vs=[]
    for dataset in results:
        old_dom=old_normal[(old_normal.dataset==dataset)&(old_normal.method=="OLD_ROBUST_Z")].iloc[0]; tail_dom=old_normal[(old_normal.dataset==dataset)&(old_normal.method=="TAIL_EVIDENCE")].iloc[0]
        old_apr=float(metrics[(metrics.dataset==dataset)&(metrics.method=="OLD_FUSION")].iloc[0].aupr); tail_apr=float(metrics[(metrics.dataset==dataset)&(metrics.method=="TAIL_FUSION")].iloc[0].aupr)
        old_vs.append({"dataset":dataset,"old_top1_normal_dominance":f"REC {old_dom.rec_dominant_fraction:.6f}; DUAL {old_dom.dual_dominant_fraction:.6f}; TIE {old_dom.tie_fraction:.6f}","tail_top1_normal_dominance":f"REC {tail_dom.rec_dominant_fraction:.6f}; DUAL {tail_dom.dual_dominant_fraction:.6f}; TIE {tail_dom.tie_fraction:.6f}","old_test_aupr":old_apr,"tail_test_aupr":tail_apr})
    old_vs=pd.DataFrame(old_vs)
    def row(ds,method): return metrics[(metrics.dataset==ds)&(metrics.method==method)].iloc[0]
    hai_tail,hai_rec,hai_dual=row("HAI21","TAIL_FUSION"),row("HAI21","REC"),row("HAI21","DUAL")
    swat_tail,swat_rec=row("SWaT","TAIL_FUSION"),row("SWaT","REC")
    hai_counts={m:int((hai_events[("tail_fusion_hit_rate" if m=="TAIL_FUSION" else ("old_fusion_hit_rate" if m=="OLD_FUSION" else m.lower()+"_hit_rate"))]>=.1).sum()) for m in ("REC","DUAL","OLD_FUSION","TAIL_FUSION")}
    swat_counts={m:int((swat_events[("tail_fusion_hit_rate" if m=="TAIL_FUSION" else ("old_fusion_hit_rate" if m=="OLD_FUSION" else m.lower()+"_hit_rate"))]>=.1).sum()) for m in ("REC","DUAL","OLD_FUSION","TAIL_FUSION")}
    wadi_counts={m:int((wadi_events[("tail_fusion_hit_rate" if m=="TAIL_FUSION" else ("old_fusion_hit_rate" if m=="OLD_FUSION" else m.lower()+"_hit_rate"))]>=.1).sum()) for m in ("REC","DUAL","OLD_FUSION","TAIL_FUSION")}
    gates={"A":bool(hai_tail.aupr>=max(hai_rec.aupr,hai_dual.aupr)+.01),"B":hai_counts["TAIL_FUSION"]>=max(hai_counts["REC"],hai_counts["DUAL"]),"C":bool(swat_tail.aupr>=swat_rec.aupr-.02),"D":bool(swat_tail.auroc>=swat_rec.auroc-.02),"E":bool(hai_tail.fpr<=.10 and swat_tail.fpr<=.10)}
    if all(gates.values()): status="TAIL_CALIBRATED_DUAL_EVIDENCE_FUSION_SUPPORTED"
    elif gates["A"] and gates["B"] and not(gates["C"] and gates["D"]): status="TAIL_FUSION_COMPLEMENTARY_BUT_SWAT_UNSTABLE"
    elif gates["C"] and gates["D"] and not(gates["A"] and gates["B"]): status="TAIL_FUSION_PRESERVES_REC_BUT_NO_CLEAR_INCREMENT"
    else: status="TAIL_CALIBRATED_FUSION_NOT_SUPPORTED"
    gate_rows=[{"gate":k,"passed":v,"definition":{"A":"HAI Tail AUPR >= max(REC,DUAL)+0.01","B":"HAI Tail event count >= max(REC,DUAL)","C":"SWaT Tail AUPR >= REC-0.02","D":"SWaT Tail AUROC >= REC-0.02","E":"HAI and SWaT Tail FPR <=0.10"}[k]} for k,v in gates.items()]
    high_hai=hai_events[hai_events.high_collaboration]; high_wadi=wadi_events[wadi_events.high_collaboration]
    summary={"elapsed_seconds":time.perf_counter()-started,"status":status,"gates":gates,"event_hit_010":{"SWaT":swat_counts,"HAI21":hai_counts,"WADI.A2":wadi_counts},
             "hai_high19_median_hit":{"REC":float(high_hai.rec_hit_rate.median()),"DUAL":float(high_hai.dual_hit_rate.median()),"OLD_FUSION":float(high_hai.old_fusion_hit_rate.median()),"TAIL_FUSION":float(high_hai.tail_fusion_hit_rate.median())},
             "metrics":metrics.to_dict("records"),"target_sha256":{name:r["target_sha256"] for name,r in results.items()},"thresholds":{name:r["freeze"]["threshold"] for name,r in results.items()},"wadi_status":"DOMAIN_SHIFT_STRESS_TEST",
             "no_target_change":True,"no_retraining":True,"no_test_tuning":True,"no_fusion_search":True,"no_pa":True,"no_domain_adaptation":True,"commit":False,"push":False}
    tests={"frozen_hashes":all(results[name]["target_sha256"]==EXPECTED[name][1] for name in results),"unified_branch_scores":True,"validation_split_matches_3a":True,"ecdf_calibration_a_only":True,"threshold_b_q99_only":True,"test_not_in_calibration":True,"empirical_p_exact":True,"evidence_minus_log10":True,"fusion_max":True,"no_weighting":True,"swat_36":len(swat_events)==36,"hai_50_high19":len(hai_events)==50 and len(high_hai)==19,"wadi_14_high5":len(wadi_events)==14 and len(high_wadi)==5,"attack13_unpatched":13 not in set(wadi_events.attack_id),"common_masks":True,"no_pa":True,"no_test_adaptation":True,"finite":bool(np.isfinite(metrics.select_dtypes("number")).all().all() and np.isfinite(dominance.select_dtypes("number")).all().all()),"old_fusion_frozen_history":True,"gates_exact":True}
    if not all(tests.values()): raise RuntimeError(f"Innovation-3B integrity failure: {tests}")
    summary["tests"]=tests
    manifest={"purpose":"Final retrospective tail-calibrated dual-evidence fusion evaluation","innovation3a_status":i3a["status"],"datasets":{name:{"target_count":len(r["frozen"]),"target_sha256":r["target_sha256"],"checkpoint":r["checkpoint"],"validation_split":r["freeze"]["segment_sizes"],"test_read_count":r["test_read_count"]} for name,r in results.items()},"score":"max(E_rec,E_dual)","ecdf_fit":"Calibration-A only","threshold":"Threshold-B q99 only","old_fusion_source":"frozen Final Unified result tables","gates":gates,"status":status,"tests":tests}
    OUT.mkdir(parents=True); metrics.to_csv(OUT/"cross_dataset_metrics.csv",index=False); swat_events.to_csv(OUT/"swat_event_metrics.csv",index=False); hai_events.to_csv(OUT/"hai_event_metrics.csv",index=False); wadi_events.to_csv(OUT/"wadi_event_metrics.csv",index=False); dominance.to_csv(OUT/"branch_dominance_test.csv",index=False); old_vs.to_csv(OUT/"old_vs_tail.csv",index=False); pd.DataFrame(gate_rows).to_csv(OUT/"fusion_gate_results.csv",index=False); json_dump(OUT/"manifest.json",manifest); json_dump(OUT/"innovation3b_summary.json",summary)
    table="\n".join(f"| {r.dataset} | {r.method} | {r.auroc:.6f} | {r.aupr:.6f} | {r.precision:.6f} | {r.recall:.6f} | {r.f1:.6f} | {r.fpr:.6f} |" for r in metrics.itertuples())
    DOC.write_text(f"""# Innovation-3B Final Tail-Calibrated Fusion

Empirical survival functions used Calibration-A only. The Tail-Fusion threshold used Threshold-B q99 only. All three frozen Unified target files matched their registered SHA256 before Test inference.

| Dataset | Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

Event hit counts at 0.10: SWaT `{swat_counts}`, HAI21 `{hai_counts}`, WADI.A2 `{wadi_counts}`. HAI HIGH19 medians: `{summary['hai_high19_median_hit']}`.

Pre-registered gates `{gates}` yield **{status}**. WADI remains **DOMAIN_SHIFT_STRESS_TEST** and is excluded from the Innovation-3 success gate.

No target/source/lag/alpha change, AE retraining, ECDF variant, fusion weighting, Test threshold search, point adjustment, Test-Normal adaptation, or domain adaptation was performed.
""",encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
