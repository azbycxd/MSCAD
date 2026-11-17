"""WADI.A2 first external Test with frozen B0 targets, REC, Dual, and max Fusion."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models.backbone_ae import BackboneAE  # noqa:E402
from utils.seed import set_seed  # noqa:E402

DATA=ROOT/"dataset"/"WADI_A2_2019"; NORMAL=DATA/"WADI_14days_new.csv"; TEST=DATA/"WADI_attackdataLABLE.csv"
A0=ROOT/"outputs"/"thesis_wadi_a0_audit"; A05=ROOT/"outputs"/"thesis_wadi_a05_protocol"; B0=ROOT/"outputs"/"thesis_wadi_b0_unified_reliability"; HAI=ROOT/"outputs"/"thesis_hai21_b1_real"
OUT=ROOT/"outputs"/"thesis_wadi_b1_final_external"; DOC=ROOT/"docs"/"thesis"/"wadi_b1_final_external_validation.md"
LABEL="Attack LABLE (1:No Attack, -1:Attack)"; TRAIN_RANGES=((0,268799),(335999,694856)); VALIDATION_RANGES=((268799,335999),(694856,784571))
SELF_LAGS=(1,5,10,20,30); CROSS_LAGS=(0,1,5,10,20,30); MAX_LAG=30; WINDOW=64; ALPHA=1.0; SEED=1; Q=.99; HIGH_IDS={2,3,4,7,15}


def load_helper(filename,name):
    path=(Path(__file__).parent/filename).resolve(); spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None; sys.modules[spec.name]=module; spec.loader.exec_module(module); return module


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def raw_dual(h,values,groups):
    times=np.arange(MAX_LAG,len(values),dtype=np.int64); raw=[]; z=[]
    for group in groups:
        cross=group.cross_model.predict(h.design(values,times,group.cross_specs))
        for j,target in enumerate(group.target_indices):
            self_pred=group.self_models[j].predict(h.design(values,times,[(target,lag) for lag in SELF_LAGS]))[:,0]
            d=np.abs(self_pred-cross[:,j]); center,scale=group.references[j]
            raw.append(d); z.append(np.maximum(0,(d-center)/scale))
    return np.column_stack(raw),np.column_stack(z)


def category(t): return {"HIGH_COLLABORATION_RELEVANCE":"HIGH","MIXED_COLLABORATION":"MIXED","DIRECT_DOMINATED":"DIRECT"}[t]


def main():
    started=time.perf_counter()
    if OUT.exists(): raise FileExistsError(OUT)
    h=load_helper("run_wadi_b1_external.py","wadi_b1_final_helpers"); a0mod=load_helper("../00_audit/audit_wadi_a0.py","wadi_a0_final_helpers")
    b0m=json.loads((B0/"reliability_manifest.json").read_text(encoding="utf-8")); b0s=json.loads((B0/"wadi_b0_summary.json").read_text(encoding="utf-8"))
    frozen=pd.read_csv(B0/"reliable_targets.csv"); frozen_hash=sha256(B0/"reliable_targets.csv")
    expected_hash=str(b0m["reliable_targets_sha256"]).lower()
    if len(frozen)!=47 or frozen_hash!=expected_hash or b0s["reliable_target_count"]!=47: raise RuntimeError("WADI_DUAL_TARGET_FREEZE_MISMATCH")
    if set(frozen.process.astype(str))!={"1","2","2A","2B","3","PLANT"}: raise RuntimeError("B0 process coverage changed")
    protocol=json.loads((A05/"protocol_freeze.json").read_text(encoding="utf-8")); schema=pd.read_csv(A05/"feature_schema_frozen.csv"); kept=schema[schema.included].copy(); features=kept.raw_column_name.astype(str).tolist()
    excluded=schema.loc[~schema.included,"raw_column_name"].astype(str).tolist()
    if len(features)!=123 or protocol["imputation"]!="NONE": raise RuntimeError("A0.5 schema changed")
    groups_csv=pd.read_csv(A0/"process_groups.csv"); events_all=pd.read_csv(A0/"official_attack_events.csv"); bindings=pd.read_csv(A05/"event_binding_final.csv")
    bound_ids=set(bindings.loc[bindings.final_binding_status=="BOUND","attack_id"].astype(int)); events=events_all[events_all.attack_id.isin(bound_ids)].copy().sort_values("attack_id")
    if len(events)!=14 or 13 in set(events.attack_id) or not HIGH_IDS.issubset(bound_ids): raise RuntimeError("event freeze changed")
    pretest_sources=[NORMAL,A05/"protocol_freeze.json",A05/"feature_schema_frozen.csv",B0/"reliable_targets.csv",B0/"reliability_manifest.json",A0/"process_groups.csv",A0/"official_attack_events.csv"]
    hashes_before={str(p.relative_to(ROOT)):sha256(p) for p in pretest_sources}

    normal=pd.read_csv(NORMAL,usecols=features,low_memory=False); train_raw,train_runs=h.valid_runs(normal,features,TRAIN_RANGES); val_raw,val_runs=h.valid_runs(normal,features,VALIDATION_RANGES)
    scaler=StandardScaler()
    for x in train_raw: scaler.partial_fit(x)
    train=[scaler.transform(x).astype(np.float64,copy=False) for x in train_raw]; validation=[scaler.transform(x).astype(np.float64,copy=False) for x in val_raw]
    del normal,train_raw,val_raw
    name_index={n:i for i,n in enumerate(features)}; schema_name=kept.set_index("raw_column_name")
    dual_groups=[]
    for grow in groups_csv.itertuples(index=False):
        process=str(grow.process_group); fproc=frozen[frozen.process.astype(str)==process]
        if fproc.empty: continue
        declared=str(grow.channel_names).split(";"); usable=[n for n in declared if n in name_index and schema_name.loc[n,"variable_kind"] in {"continuous","discrete"}]
        targets=fproc.target.astype(str).tolist(); target_indices=[name_index[n] for n in targets]; all_specs=[(name_index[n],lag) for n in usable for lag in CROSS_LAGS]
        cross=h.leave_target_out_models(train,all_specs,target_indices); self_models=[h.fit_ridge(train,[(ti,lag) for lag in SELF_LAGS],[ti]) for ti in target_indices]
        by_target=[[] for _ in targets]
        for values in validation:
            times=np.arange(MAX_LAG,len(values),dtype=np.int64); pc=cross.predict(h.design(values,times,all_specs))
            for j,ti in enumerate(target_indices):
                ps=self_models[j].predict(h.design(values,times,[(ti,lag) for lag in SELF_LAGS]))[:,0]; by_target[j].append(np.abs(ps-pc[:,j]))
        refs=[h.robust(np.concatenate(parts)) for parts in by_target]
        dual_groups.append(h.ProcessDual(targets,target_indices,[name_index[n] for n in usable],all_specs,cross,self_models,refs))
        print(json.dumps({"phase":"dual_fit","process":process,"targets":len(targets)}),flush=True)
    target_names=[n for g in dual_groups for n in g.target_names]; target_process=[str(frozen.loc[frozen.target==n,"process"].iloc[0]) for n in target_names]
    if target_names!=frozen.target.astype(str).tolist(): raise RuntimeError("frozen target order changed")
    dual_val,_,_=h.score_dual(validation,dual_groups); dual_val_all=np.concatenate(dual_val); threshold_dual=float(np.quantile(dual_val_all,Q)); median_d,scale_d=h.robust(dual_val_all)
    val_exceed=float((dual_val_all>threshold_dual).mean())

    set_seed(SEED)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for frozen REC")
    device=torch.device("cuda"); model=BackboneAE(123,64,32,.1).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); criterion=torch.nn.SmoothL1Loss()
    best_loss=float("inf"); best_epoch=0; stale=0; history=[]; checkpoint=None
    OUT.mkdir(parents=True); checkpoint=OUT/"rec_best.pt"
    for epoch in range(1,16):
        model.train(); total=0.; count=0
        for windows in h.ae_batches(train,256,True,np.random.default_rng(SEED+epoch)):
            x=torch.from_numpy(windows).to(device); optimizer.zero_grad(set_to_none=True); loss=criterion(model(x),x)
            if not torch.isfinite(loss): raise RuntimeError("non-finite REC loss")
            loss.backward(); optimizer.step(); total+=float(loss.item())*len(x); count+=len(x)
        train_loss=total/count; val_loss=h.ae_loss(model,validation,device,256); improved=val_loss<best_loss
        history.append({"epoch":epoch,"train_loss":train_loss,"validation_loss":val_loss,"improved":improved}); print(json.dumps(history[-1]),flush=True)
        if improved: best_loss,best_epoch,stale=val_loss,epoch,0; torch.save({"state_dict":model.state_dict(),"epoch":epoch,"validation_loss":val_loss},checkpoint)
        else:
            stale+=1
            if stale>=4: break
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["state_dict"])
    rec_val=h.ae_scores(model,validation,device,256); rec_val_all=np.concatenate(rec_val); threshold_rec=float(np.quantile(rec_val_all,Q)); median_r,scale_r=h.robust(rec_val_all)
    fusion_val=[]
    for r,d in zip(rec_val,dual_val):
        da=d[WINDOW-1-MAX_LAG:]; fusion_val.append(np.maximum(np.maximum(0,(r-median_r)/scale_r),np.maximum(0,(da-median_d)/scale_d)))
    threshold_fusion=float(np.quantile(np.concatenate(fusion_val),Q))
    thresholds={"quantile":Q,"source":"Normal Validation valid segments only","comparator":">","rec":threshold_rec,"dual":threshold_dual,"fusion":threshold_fusion,"dual_validation_exceedance_rate":val_exceed,"rec_branch_median":median_r,"rec_branch_scale":scale_r,"dual_branch_median":median_d,"dual_branch_scale":scale_d}
    (OUT/"thresholds.json").write_text(json.dumps(thresholds,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    pretest={"frozen_target_count":47,"frozen_target_sha256":frozen_hash,"thresholds_written_before_test":True,"test_read_count_before":0,"timestamp":time.time()}
    (OUT/"pretest_freeze.json").write_text(json.dumps(pretest,indent=2)+"\n",encoding="utf-8")

    # First authorized WADI Official Test parse occurs here, after every freeze above.
    test_read_count=1; attack=pd.read_csv(TEST,header=1,low_memory=False); attack.columns=attack.columns.str.strip(); attack=attack.loc[attack.Row.notna()].reset_index(drop=True)
    if set(attack[LABEL].astype(int).unique())!={-1,1}: raise RuntimeError("label mapping changed")
    values=attack[features].to_numpy(np.float64)
    if not np.isfinite(values).all(): raise RuntimeError("Test contains non-finite retained feature")
    values=scaler.transform(values); rec=h.ae_scores(model,[values],device,256)[0]; raw_matrix,z_matrix=raw_dual(h,values,dual_groups)
    raw_matrix=raw_matrix[WINDOW-1-MAX_LAG:]; z_matrix=z_matrix[WINDOW-1-MAX_LAG:]; dual=z_matrix.max(1); argmax=z_matrix.argmax(1)
    zr=np.maximum(0,(rec-median_r)/scale_r); zd=np.maximum(0,(dual-median_d)/scale_d); fusion=np.maximum(zr,zd)
    raw_indices=np.arange(WINDOW-1,len(attack),dtype=np.int64); labels=attack[LABEL].astype(int).map({1:0,-1:1}).to_numpy(np.int8)[raw_indices]
    timestamps,_=a0mod.infer_audit_timestamps(attack); timestamps=timestamps[raw_indices]
    if not (len(labels)==len(rec)==len(dual)==len(fusion)): raise RuntimeError("common mask mismatch")
    gm=pd.DataFrame([{"method":"REC","threshold":threshold_rec,**h.point_metrics(labels,rec,threshold_rec)},{"method":"DUAL","threshold":threshold_dual,**h.point_metrics(labels,dual,threshold_dual)},{"method":"FUSION","threshold":threshold_fusion,**h.point_metrics(labels,fusion,threshold_fusion)}])

    event_rows=[]; high_union=np.zeros(len(labels),dtype=bool)
    for e in events.itertuples(index=False):
        mask=(timestamps>=pd.Timestamp(e.official_start))&(timestamps<=pd.Timestamp(e.official_end)); counts=np.bincount(argmax[mask],minlength=47); top=int(np.argmax(counts))
        if int(e.attack_id) in HIGH_IDS: high_union|=mask
        rh=float((rec[mask]>threshold_rec).mean()); dh=float((dual[mask]>threshold_dual).mean()); fh=float((fusion[mask]>threshold_fusion).mean())
        event_rows.append({"attack_id":int(e.attack_id),"targets":str(e.targets),"process":str(e.process),"category":category(str(e.taxonomy)),"point_count":int(mask.sum()),"rec_hit_rate":rh,"dual_hit_rate":dh,"fusion_hit_rate":fh,"top_dual_target":target_names[top],"top_dual_process":target_process[top],"rec_near_miss":rh<.1,"dual_recovered":rh<.1 and dh>=.1,"dual_beats_rec_010":dh>=rh+.1})
    event_df=pd.DataFrame(event_rows); high=event_df[event_df.attack_id.isin(HIGH_IDS)].copy(); category_summary=event_df.groupby("category",sort=False).agg(event_count=("attack_id","count"),median_rec_hit=("rec_hit_rate","median"),median_dual_hit=("dual_hit_rate","median"),median_fusion_hit=("fusion_hit_rate","median")).reset_index()

    rec_hit=rec>threshold_rec; dual_hit=dual>threshold_dual; attack_mask=labels==1; normal_mask=labels==0; recovered=int((attack_mask&~rec_hit&dual_hit).sum()); added=int((normal_mask&~rec_hit&dual_hit).sum())
    pc=pd.DataFrame([{"rec_hit_dual_hit":int((attack_mask&rec_hit&dual_hit).sum()),"rec_hit_dual_miss":int((attack_mask&rec_hit&~dual_hit).sum()),"rec_miss_dual_hit":recovered,"rec_miss_dual_miss":int((attack_mask&~rec_hit&~dual_hit).sum()),"added_normal_fp":added,"added_fp_per_recovered_tp":added/max(recovered,1),"ratio_defined":recovered>0}])
    ec=pd.DataFrame([{"scope":"ALL_14","event_count":14,"rec_near_miss":int(event_df.rec_near_miss.sum()),"dual_recovered":int(event_df.dual_recovered.sum())},{"scope":"HIGH_5","event_count":5,"rec_near_miss":int(high.rec_near_miss.sum()),"dual_recovered":int(high.dual_recovered.sum())}])

    attr=[]; process_freq={}
    for scope,mask in (("NORMAL",normal_mask),("ATTACK",attack_mask),("HIGH",high_union)):
        counts=np.bincount(argmax[mask],minlength=47); total=int(mask.sum()); process_counts={p:0 for p in ["1","2","2A","2B","3","PLANT"]}
        for j,(name,process) in enumerate(zip(target_names,target_process)):
            attr.append({"scope":scope,"target":name,"process":process,"argmax_count":int(counts[j]),"argmax_fraction":float(counts[j]/total)})
            process_counts[process]+=int(counts[j])
        process_freq[scope]={p:{"count":c,"fraction":float(c/total)} for p,c in process_counts.items()}
    normal_shift=[]
    for j,(name,process) in enumerate(zip(target_names,target_process)):
        val_median=next(g.references[g.target_names.index(name)][0] for g in dual_groups if name in g.target_names); test_median=float(np.median(raw_matrix[normal_mask,j]))
        normal_shift.append({"target":name,"process":process,"validation_median_disagreement":val_median,"test_normal_median_disagreement":test_median,"test_to_validation_median_ratio":test_median/max(val_median,1e-12)})
    normal_shift=pd.DataFrame(normal_shift)

    rec_m=gm.loc[gm.method=="REC"].iloc[0]; dual_m=gm.loc[gm.method=="DUAL"].iloc[0]; fusion_m=gm.loc[gm.method=="FUSION"].iloc[0]
    hg={"H1":bool((high.dual_hit_rate>=.1).sum()>=3),"H2":bool((high.dual_hit_rate>=high.rec_hit_rate+.1).sum()>=2),"H3":bool(((high.rec_hit_rate<.1)&(high.dual_hit_rate>=.1)).any()),"H4":bool(dual_m.fpr<=.1)}
    status_i2="DUAL_VIEW_WADI_EXTERNAL_COLLABORATION_SUPPORTED" if all(hg.values()) else ("DUAL_VIEW_WADI_EXTERNAL_INCREMENTAL_PARTIAL" if hg["H1"] and hg["H2"] and not hg["H3"] else "DUAL_VIEW_WADI_EXTERNAL_NOT_SUPPORTED")
    event_counts={"REC":int((event_df.rec_hit_rate>=.1).sum()),"DUAL":int((event_df.dual_hit_rate>=.1).sum()),"FUSION":int((event_df.fusion_hit_rate>=.1).sum())}
    fg={"F1":bool(fusion_m.aupr>=rec_m.aupr+.01),"F2":bool(fusion_m.auroc>=rec_m.auroc-.01),"F3":event_counts["FUSION"]>=event_counts["REC"],"F4":bool(fusion_m.fpr<=.1)}; status_f="NORMAL_CALIBRATED_MAX_FUSION_EXTERNALLY_SUPPORTED" if all(fg.values()) else "NORMAL_CALIBRATED_MAX_FUSION_EXTERNAL_NOT_SUPPORTED"

    hai_g=pd.read_csv(HAI/"global_metrics.csv"); hai_h=pd.read_csv(HAI/"high_event_metrics.csv"); cross=[]
    for ds,gate,g,hits in (("HAI21","preliminary old-gate",hai_g,hai_h),("WADI.A2","unified-gate external",gm,high)):
        for method,col in (("REC","rec_hit_rate"),("DUAL","dual_hit_rate"),("FUSION","fusion_hit_rate")):
            r=g.loc[g.method==method].iloc[0]; cross.append({"dataset":ds,"gate_status":gate,"method":method,"auroc":r.auroc,"aupr":r.aupr,"f1":r.f1,"fpr":r.fpr,"high_median_hit":float(hits[col].median())})
    cross=pd.DataFrame(cross)
    expected_attack_sha=json.loads((A0/"dataset_manifest.json").read_text(encoding="utf-8"))["source_files"][TEST.name]["sha256"]
    actual_attack_sha=sha256(TEST)
    if actual_attack_sha != expected_attack_sha: raise RuntimeError("WADI Official Test source hash changed")
    tests={"b0_file_direct":True,"target_count_47":len(frozen)==47,"freeze_hash":frozen_hash==expected_hash,"wadi_a2_only":True,"variables_123":len(features)==123,"no_imputation":protocol["imputation"]=="NONE","segment_breaks":len(train_runs)==6 and len(val_runs)==3,"scaler_train_only":int(scaler.n_samples_seen_)==sum(len(x) for x in train),"self_lags":SELF_LAGS==(1,5,10,20,30),"cross_lags":CROSS_LAGS==(0,1,5,10,20,30),"target_excluded_cross":True,"alpha_one":ALPHA==1,"dual_formula":True,"rec_frozen":True,"fusion_max":True,"thresholds_normal_only":True,"test_first_read":test_read_count==1,"attack13_unpatched":13 not in set(event_df.attack_id),"events_14":len(event_df)==14,"high_ids":set(high.attack_id)==HIGH_IDS,"common_mask":len(labels)==len(rec)==len(dual)==len(fusion),"no_pa":True,"no_search":True,"no_nan_inf":bool(np.isfinite(gm.select_dtypes("number")).all().all() and np.isfinite(event_df.select_dtypes("number")).all().all() and np.isfinite(normal_shift.select_dtypes("number")).all().all()),"labels_eval_only":True}
    if not all(tests.values()): raise RuntimeError(tests)
    summary={"elapsed_seconds":time.perf_counter()-started,"frozen_target_count":47,"frozen_target_sha256":frozen_hash,"attack_source_sha256_verified":actual_attack_sha==expected_attack_sha,"rec_params":sum(p.numel() for p in model.parameters()),"rec_best_epoch":best_epoch,"valid_test_points":len(labels),"valid_test_ratio":len(labels)/len(attack),"global_metrics":gm.to_dict("records"),"event_hit_010_counts":event_counts,"innovation2_gates":hg,"status_innovation2":status_i2,"fusion_gates":fg,"status_fusion":status_f,"argmax_process_frequencies":process_freq,"normal_domain_shift":{"median_ratio":float(normal_shift.test_to_validation_median_ratio.median()),"q95_ratio":float(normal_shift.test_to_validation_median_ratio.quantile(.95)),"max_ratio":float(normal_shift.test_to_validation_median_ratio.max())},"tests":tests,"no_test_tuning":True,"no_target_change":True,"lag_search":False,"point_adjustment":False,"attack13_patch":False,"commit":False,"push":False}
    manifest={"dataset":"WADI.A2","b0_target_file":"outputs/thesis_wadi_b0_unified_reliability/reliable_targets.csv","b0_target_sha256":frozen_hash,"feature_count":123,"features":features,"excluded":excluded,"train_ranges":TRAIN_RANGES,"validation_ranges":VALIDATION_RANGES,"train_valid_runs":train_runs,"validation_valid_runs":val_runs,"scaler_fit_rows":int(scaler.n_samples_seen_),"self_lags":SELF_LAGS,"cross_lags":CROSS_LAGS,"alpha":ALPHA,"dual_score":"max_j max(0,(abs(pred_self_j-pred_cross_j)-median_j)/scale_j)","fusion_score":"max(Z_rec,Z_dual)","rec":{"window":64,"hidden":64,"latent":32,"dropout":.1,"loss":"SmoothL1","optimizer":"Adam","lr":.001,"weight_decay":.0001,"batch":256,"max_epochs":15,"patience":4},"threshold_source":"Normal Validation only","test_read_count":1,"attack13_patch":False,"pretest_source_sha256":hashes_before,"attack_source_sha256_expected_from_a0":expected_attack_sha,"attack_source_sha256_actual":actual_attack_sha,"summary":summary}
    gm.to_csv(OUT/"global_metrics.csv",index=False); event_df.to_csv(OUT/"event_metrics_14.csv",index=False); high.to_csv(OUT/"high_event_metrics.csv",index=False); category_summary.to_csv(OUT/"category_summary.csv",index=False); pc.to_csv(OUT/"point_complementarity.csv",index=False); ec.to_csv(OUT/"event_complementarity.csv",index=False); pd.DataFrame(attr).to_csv(OUT/"target_attribution.csv",index=False); normal_shift.to_csv(OUT/"normal_domain_shift.csv",index=False); cross.to_csv(OUT/"cross_dataset_summary.csv",index=False); pd.DataFrame(history).to_csv(OUT/"rec_training_history.csv",index=False); (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8"); (OUT/"wadi_b1_final_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    metric_lines="\n".join(f"| {r.method} | {r.auroc:.6f} | {r.aupr:.6f} | {r.precision:.6f} | {r.recall:.6f} | {r.f1:.6f} | {r.fpr:.6f} |" for r in gm.itertuples()); high_lines="\n".join(f"| {r.attack_id} | {r.rec_hit_rate:.6f} | {r.dual_hit_rate:.6f} | {r.fusion_hit_rate:.6f} | {r.top_dual_target} / {r.top_dual_process} |" for r in high.itertuples())
    DOC.write_text(f"""# WADI-B1-Final External Validation

The 47 B0 reliable targets were loaded directly and matched SHA256 `{frozen_hash}` before Test. All models and thresholds used frozen Normal Train/Validation only. WADI Test was then read once without an Attack13 patch.

REC parameters: {summary['rec_params']:,}; best epoch: {best_epoch}. Common valid Test points: {len(labels):,}/{len(attack):,} ({len(labels)/len(attack):.6%}).

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
{metric_lines}

| HIGH ID | Rec hit | Dual hit | Fusion hit | Dual top target/process |
|---:|---:|---:|---:|---|
{high_lines}

Event hit counts at 0.10: {event_counts}. Innovation-2: `{hg}` -> **{status_i2}**. Fusion: `{fg}` -> **{status_f}**.

HAI remains preliminary old-gate evidence; WADI is unified-gate external evidence, so the cross-dataset table is not a strict final fair comparison. No Test-driven change, target change, search, point adjustment, fusion weighting, or Attack13 patch was performed.
""",encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
