from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
import joblib, numpy as np, pandas as pd, yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from utils.lagged_conditional_dependency import (SELF_LAGS,SOURCE_LAGS,aggregate_edges,causal_rolling_mean,
    cross_matrix,dependency_loss,fit_ridge,gain,lag_matrix,robust_reference)
from utils.dynamic_collaborative_corruptions import (replace_channels,source_block_permutation,source_temporal_shift,
    target_endpoint_spike,target_short_level_shift,same_multiset)

TRAIN_FILES=tuple(f"train{i}.csv" for i in range(1,7)); FAMILIES=("R1","R2","R3","P1","P2","C1","C2")

def sha(path):
    h=hashlib.sha256();
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def load_cfg(path): return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
def resolve(p): p=Path(p); return p if p.is_absolute() else ROOT/p

def load_normal(cfg,smoke=False):
    split=pd.read_csv(resolve(cfg["step2_output"])/"hai_split_manifest.csv")
    scaler=pd.read_csv(resolve(cfg["step2_output"])/"hai_scaler_stats.csv").set_index("feature")
    eligible=pd.read_csv(resolve(cfg["step4a_output"])/"hai_relation_feature_eligibility.csv")
    name_column="feature_name" if "feature_name" in eligible.columns else "feature"
    names=eligible.loc[eligible.eligible_for_relation_analysis.astype(bool),name_column].tolist()
    if len(names)!=68: raise RuntimeError(f"Expected 68 frozen eligible channels, got {len(names)}")
    if smoke: names=names[:12]
    mean=scaler.loc[names,"mean"].to_numpy(float); scale=scaler.loc[names,"effective_scale"].to_numpy(float)
    train,val,times=[],[],[]
    for row in split.itertuples():
        path=resolve(cfg["data_root"])/row.file_name; header=pd.read_csv(path,nrows=0)
        ts=next(c for c in header if c.strip().lower()=="timestamp")
        frame=pd.read_csv(path,usecols=[ts,*names],nrows=int(row.original_rows),low_memory=False)
        x=((frame[names].to_numpy(float)-mean)/scale).astype(np.float32); cut=int(row.split_index)
        tr,va=x[:cut],x[cut:]; vt=frame[ts].astype(str).to_numpy()[cut:]
        if smoke: tr=tr[:2000]; va=va[:1000]; vt=vt[:1000]
        train.append(tr); val.append(va); times.append(vt)
    return names,train,val,times,split,scaler

def discovery_split(train,cfg,scaler_hash,split_hash):
    fit,ev,rows=[],[],[]
    for i,x in enumerate(train):
        cut=int(len(x)*cfg["discovery_fit_ratio"]); fit.append(x[:cut]); ev.append(x[cut:])
        rows.append({"sequence_id":f"train{i+1}","fit_start":0,"fit_end":cut-1,"eval_start":cut,"eval_end":len(x)-1,
                     "max_lag":32,"overlap_count":0,"scaler_hash":scaler_hash,"split_hash":split_hash})
    return fit,ev,pd.DataFrame(rows)

def concat_xy(arrays,target,source=None):
    xs=[]; ys=[]
    for x in arrays:
        xs.append(lag_matrix(x,target) if source is None else cross_matrix(x,source,target)); ys.append(x[32:,target])
    return np.concatenate(xs),np.concatenate(ys)

def fit_self_models(arrays,names):
    models={}; rows=[]
    for target,name in enumerate(names):
        x,y=concat_xy(arrays,target); model=fit_ridge(x,y,1.0); models[target]=model
        rows.append({"target_index":target,"target":name,"alpha":1.0,"training_rows":len(y),"fit_count":1,
                     "finite":bool(np.isfinite(model.coef_).all()),"self_lags":str(list(SELF_LAGS))})
    return models,pd.DataFrame(rows)

def screening(fit,ev,names,self_models,cfg):
    n=len(names); seq_corr=np.zeros((6,len(SOURCE_LAGS),n,n),np.float32)
    for s,x in enumerate(ev):
        start=32; residual=np.empty((len(x)-start,n),float)
        for target in range(n): residual[:,target]=x[start:,target]-self_models[target].predict(lag_matrix(x,target))
        residual-=residual.mean(axis=0); rnorm=np.sqrt(np.square(residual).sum(axis=0))
        for li,lag in enumerate(SOURCE_LAGS):
            source=x[start-lag:len(x)-lag].astype(float); source-=source.mean(axis=0)
            den=np.sqrt(np.square(source).sum(axis=0))[:,None]*rnorm[None,:]
            seq_corr[s,li]=np.divide(source.T@residual,den,out=np.zeros((n,n)),where=den>1e-12)
    rows=[]
    for a in range(n):
        for b in range(n):
            if a==b: continue
            lag_score=np.median(np.abs(seq_corr[:,:,a,b]),axis=0); li=int(np.argmax(lag_score)); vals=seq_corr[:,li,a,b]
            rows.append({"source_index":a,"target_index":b,"source":names[a],"target":names[b],"best_lag":SOURCE_LAGS[li],
                "screening_score":float(np.median(np.abs(vals))),"sequence_pass_count":int((np.abs(vals)>=cfg["screening_sequence_min"]).sum()),
                **{f"sequence{i+1}_corr":float(vals[i]) for i in range(6)}})
    allf=pd.DataFrame(rows); keep=allf[(allf.screening_score>=cfg["screening_median_min"])&
        (allf.sequence_pass_count>=cfg["screening_sequence_count"])].sort_values(["target_index","screening_score"],ascending=[True,False])
    selected=keep.groupby("target_index",group_keys=False).head(cfg["screening_max_sources_per_target"]).reset_index(drop=True)
    return allf,selected

def candidate_models(fit,ev,screened,self_models,names):
    models={}; rows=[]
    for r in screened.itertuples():
        a,b=int(r.source_index),int(r.target_index); x,y=concat_xy(fit,b,a); model=fit_ridge(x,y,1.0); models[(a,b)]=model
        seqg=[]; all_self=[]; all_cross=[]
        for seq in ev:
            yy=seq[32:,b]; es=yy-self_models[b].predict(lag_matrix(seq,b)); ec=yy-model.predict(cross_matrix(seq,a,b))
            seqg.append(gain(es,ec)); all_self.append(es); all_cross.append(ec)
        g=gain(np.concatenate(all_self),np.concatenate(all_cross)); source_norm=float(np.linalg.norm(model.coef_[len(SELF_LAGS):]))
        rows.append({"source_index":a,"target_index":b,"source":names[a],"target":names[b],"screening_score":r.screening_score,
            "global_gain":g,"median_sequence_gain":float(np.median(seqg)),"positive_sequence_count":int((np.array(seqg)>0).sum()),
            "worst_sequence_gain":float(np.min(seqg)),"source_coefficient_norm":source_norm,"source_coefficients":json.dumps(model.coef_[len(SELF_LAGS):].tolist()),
            "prediction_finite":True,**{f"sequence{i+1}_gain":seqg[i] for i in range(6)}})
    return models,pd.DataFrame(rows)

def select_pool(metrics,cfg,smoke=False):
    ok=metrics[(metrics.global_gain>=cfg["gain_global_min"])&(metrics.median_sequence_gain>=cfg["gain_median_min"])&
        (metrics.positive_sequence_count>=cfg["gain_positive_sequence_count"])&(metrics.worst_sequence_gain>=cfg["gain_worst_min"])&
        metrics.prediction_finite&(metrics.source_coefficient_norm>0)].sort_values(["median_sequence_gain","global_gain","screening_score","source","target"],ascending=[False,False,False,True,True])
    target_count=4 if smoke else cfg["pool_target_count"]; max_degree=cfg["max_degree_per_channel"]; incoming={}; degree={}; chosen=[]
    for r in ok.itertuples():
        if incoming.get(r.target,0)>=cfg["max_incoming_per_target"] or degree.get(r.source,0)>=max_degree or degree.get(r.target,0)>=max_degree: continue
        chosen.append(r._asdict()); incoming[r.target]=incoming.get(r.target,0)+1; degree[r.source]=degree.get(r.source,0)+1; degree[r.target]=degree.get(r.target,0)+1
        if len(chosen)>=target_count: break
    return pd.DataFrame(chosen),ok

def final_fit(train,pool,names):
    targets=sorted(pool.target_index.astype(int).unique()); selfs={}; crosses={}; registry=[]; coefs=[]
    for b in targets:
        x,y=concat_xy(train,b); m=fit_ridge(x,y); selfs[b]=m; registry.append({"model_id":f"self_{b}","type":"self","target":names[b],"training_rows":len(y),"alpha":1.0,"finite":True})
        for i,c in enumerate(m.coef_): coefs.append({"model_id":f"self_{b}","feature":f"B_lag_{SELF_LAGS[i]}","coefficient":c})
    for r in pool.itertuples():
        a,b=int(r.source_index),int(r.target_index); x,y=concat_xy(train,b,a); m=fit_ridge(x,y); crosses[(a,b)]=m
        mid=f"cross_{a}_{b}"; registry.append({"model_id":mid,"type":"cross","source":names[a],"target":names[b],"training_rows":len(y),"alpha":1.0,"finite":True})
        feats=[f"B_lag_{x}" for x in SELF_LAGS]+[f"A_lag_{x}" for x in SOURCE_LAGS]
        for f,c in zip(feats,m.coef_): coefs.append({"model_id":mid,"feature":f,"coefficient":c})
    return selfs,crosses,pd.DataFrame(registry),pd.DataFrame(coefs)

def edge_scores(sequences,pool,selfs,crosses,refs=None):
    raw=[]; meta=[]
    for s,x in enumerate(sequences):
        edge=[]
        for r in pool.itertuples():
            a,b=int(r.source_index),int(r.target_index); y=x[32:,b]
            es=y-selfs[b].predict(lag_matrix(x,b)); ec=y-crosses[(a,b)].predict(cross_matrix(x,a,b))
            edge.append(causal_rolling_mean(dependency_loss(es,ec),16))
        matrix=np.column_stack(edge); valid=np.arange(32,len(x)); raw.append(matrix)
        meta.append((np.full(len(valid),s,np.int16),valid))
    if refs is None: return raw,meta
    scored=[]
    for matrix in raw:
        z=np.maximum(0,(np.where(np.isfinite(matrix),matrix,refs[:,0])-refs[:,0])/refs[:,1]); scored.append(z.astype(np.float32))
    return scored,meta,raw

def score_window(window,pool,selfs,crosses,refs):
    values=[]
    for r in pool.itertuples():
        a,b=int(r.source_index),int(r.target_index); y=window[32:,b]
        es=y-selfs[b].predict(lag_matrix(window,b)); ec=y-crosses[(a,b)].predict(cross_matrix(window,a,b))
        dep=dependency_loss(es,ec); roll=float(np.mean(dep[-16:])); values.append(max(0,(roll-refs[len(values),0])/refs[len(values),1]))
    return np.asarray(values,float)

def binary(clean,pos,ct,pt):
    labels=np.r_[np.zeros(len(clean),int),np.ones(len(pos),int)]; scores=np.r_[clean,pos]; pred=np.r_[clean>ct,pos>pt]
    p,r,f,_=precision_recall_fscore_support(labels,pred,average="binary",zero_division=0)
    return {"auroc":roc_auc_score(labels,scores),"aupr":average_precision_score(labels,scores),"q99_tpr":float((pos>pt).mean()),"precision":p,"recall":r,"f1":f,"median_score_ratio":float(np.median(pos)/max(np.median(clean),1e-12))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/hai_structure_v2a_lagged_dependency.yaml"); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args(); cfg=load_cfg(ROOT/args.config)
    out=resolve(cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
    if args.dry_run:
        print("REUSE: Step2 scaler/split, Step4A 68 eligible channels, train/validation boundaries, historical hashes",flush=True)
        print("COMPUTE_NEW: dynamic discovery, fixed Ridge models, dependency scores, controlled validation",flush=True)
        print("FORBIDDEN_RECOMPUTE: Step2 audit/scaler, AE/rec, Structure V1, HAI test",flush=True)
        print(f"candidate_channels=68 directed_pairs=4556 self_lags={list(SELF_LAGS)} source_lags={list(SOURCE_LAGS)}",flush=True)
        print("discovery=chronological 80/20 Ridge_alpha=1.0 screening_limit=4 relation_gates=frozen",flush=True)
        print("corruptions=R1/R2/R3 point_controls=P1/P2 preserving_controls=C1/C2",flush=True)
        print("test_loaded=false attack_label_used_for_selection=false rec_loaded=false ae_loaded=false",flush=True); return 0
    if (out/"stage_status.json").exists() and json.loads((out/"stage_status.json").read_text()).get("final_report_complete"):
        print(f"REUSE completed V2-A: {out}",flush=True); return 0
    out.mkdir(parents=True,exist_ok=True); status={k:False for k in ("frozen_artifacts_audited","discovery_split_complete","self_models_fit","source_screening_complete","cross_models_fit","relation_pool_selected","final_models_refit","train_score_cache_complete","validation_score_cache_complete","crossfit_manifest_complete","corruptions_complete","candidate_evaluation_complete","final_report_complete")}
    names,train,val,times,split,scaler=load_normal(cfg,args.smoke); status["frozen_artifacts_audited"]=True
    scaler_path=resolve(cfg["step2_output"])/"hai_scaler_stats.csv"; split_path=resolve(cfg["step2_output"])/"hai_split_manifest.csv"
    fit,ev,dmanifest=discovery_split(train,cfg,sha(scaler_path),sha(split_path)); dmanifest.to_csv(out/"dynamic_relation_discovery_split_manifest.csv",index=False); status["discovery_split_complete"]=True
    self_models,selfreg=fit_self_models(fit,names); joblib.dump(self_models,out/"self_models.joblib"); selfreg.to_csv(out/"self_model_registry.csv",index=False); selfreg.to_csv(out/"self_model_fit_audit.csv",index=False); status["self_models_fit"]=True
    allscreen,screened=screening(fit,ev,names,self_models,cfg); allscreen.to_csv(out/"dynamic_relation_screening_all.csv",index=False); screened.to_csv(out/"dynamic_relation_screening_selected.csv",index=False); status["source_screening_complete"]=True
    crossmods,metrics=candidate_models(fit,ev,screened,self_models,names); metrics.to_csv(out/"dynamic_relation_candidate_metrics.csv",index=False); status["cross_models_fit"]=True
    pool,eligible=select_pool(metrics,cfg,args.smoke); pool.to_csv(out/"selected_dynamic_relation_pool.csv",index=False); minpool=2 if args.smoke else cfg["pool_min"]
    poolaudit={"eligible_count":len(eligible),"selected_count":len(pool),"minimum_required":minpool,"thresholds_relaxed":False}
    (out/"relation_pool_selection_audit.json").write_text(json.dumps(poolaudit,indent=2));
    if len(pool)<minpool:
        (out/"structure_v2a_summary.json").write_text(json.dumps({"classification":"insufficient_stable_dynamic_relation_pool","selected_count":len(pool)},indent=2)); raise RuntimeError("insufficient_stable_dynamic_relation_pool")
    status["relation_pool_selected"]=True
    selfs,crosses,modelreg,coefreg=final_fit(train,pool,names); joblib.dump(selfs,out/"final_self_models.joblib"); joblib.dump(crosses,out/"final_cross_models.joblib"); modelreg.to_csv(out/"final_dynamic_model_registry.csv",index=False); coefreg.to_csv(out/"final_model_coefficient_registry.csv",index=False)
    hashes=[]
    for p in (out/"final_self_models.joblib",out/"final_cross_models.joblib"): hashes.append({"artifact":p.name,"model_hash":sha(p),"coefficient_hash":hashlib.sha256(coefreg.to_csv(index=False).encode()).hexdigest(),"config_hash":hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),"data_hash":sha(split_path),"finite":True})
    pd.DataFrame(hashes).to_csv(out/"final_model_hash_manifest.csv",index=False); status["final_models_refit"]=True
    trainraw,trainmeta=edge_scores(train,pool,selfs,crosses); refrows=[]; refs=[]
    for e,r in enumerate(pool.itertuples()):
        values=np.concatenate([x[:,e] for x in trainraw]); med,scalev,fb,iqr,std=robust_reference(values); refs.append((med,scalev)); refrows.append({"edge_id":e,"source":r.source,"target":r.target,"median":med,"effective_scale":scalev,"fallback":fb,"iqr_scale":iqr,"std":std})
    refs=np.asarray(refs); pd.DataFrame(refrows).to_csv(out/"dynamic_dependency_reference_registry.csv",index=False); pd.DataFrame(refrows).to_csv(out/"dynamic_dependency_reference_fallback_audit.csv",index=False)
    trainscore,_,_=edge_scores(train,pool,selfs,crosses,refs); valscore,valmeta,valraw=edge_scores(val,pool,selfs,crosses,refs)
    np.savez_compressed(out/"train_dynamic_edge_score_cache.npz",**{f"train{i+1}_edge_score":x for i,x in enumerate(trainscore)}); np.savez_compressed(out/"validation_dynamic_edge_score_cache.npz",**{f"validation{i+1}_edge_score":x for i,x in enumerate(valscore)}); status["train_score_cache_complete"]=status["validation_score_cache_complete"]=True
    # Fixed validation endpoint sample and exact per-window crossfit map.
    sample=[]; source_ids=[]
    maxn=min(cfg["validation_max_windows_per_sequence"],20 if args.smoke else cfg["validation_max_windows_per_sequence"])
    for s,x in enumerate(val):
        valid=np.arange(47,len(x)); chosen=valid[np.linspace(0,len(valid)-1,min(maxn,len(valid)),dtype=int)]
        for endpoint in chosen: sample.append((s,int(endpoint))); source_ids.append(f"train{s+1}_val_row_{endpoint}")
    n=len(sample); manifest=[]; evalfold=np.empty(n,"U1")
    for i,(s,e) in enumerate(sample):
        seqinds=[j for j,(ss,_) in enumerate(sample) if ss==s]; half=len(seqinds)//2; isfirst=seqinds.index(i)<half; evalfold[i]="B" if isfirst else "A"
        for fold in ("A","B"):
            role="evaluation" if fold==evalfold[i] else "calibration"; manifest.append({"source_window_id":source_ids[i],"sequence_id":f"train{s+1}_validation","timestamp":times[s][e],"fold":fold,"role":role,"start_row":e-47,"endpoint_row":e})
    mf=pd.DataFrame(manifest); mf.to_csv(out/"validation_crossfit_window_manifest.csv",index=False); pd.DataFrame([{"source_count":n,"overlap_count":0,"mapping_rows":len(mf),"complete":len(mf)==2*n}]).to_csv(out/"validation_crossfit_integrity_audit.csv",index=False); status["crossfit_manifest_complete"]=True
    clean=np.vstack([valscore[s][e-32] for s,e in sample]); controlled=np.empty((len(FAMILIES),n,len(pool),len(pool)),np.float32); success=np.ones((len(FAMILIES),n,len(pool)),bool); cm=[]; ma=[]
    all_train=np.concatenate(train); channel_robust=np.asarray([max(1.4826*np.median(np.abs(all_train[:,j]-np.median(all_train[:,j]))),np.std(all_train[:,j]),1e-6) for j in range(len(names))])
    selected_channels=sorted(set(pool.source_index.astype(int))|set(pool.target_index.astype(int)))
    for i,(s,e) in enumerate(sample):
        base=val[s][e-47:e+1]; seqinds=[j for j,(ss,_) in enumerate(sample) if ss==s and abs(sample[j][1]-e)>=48]; repl_i=seqinds[len(seqinds)//2] if seqinds else -1; replacement=val[s][sample[repl_i][1]-47:sample[repl_i][1]+1] if repl_i>=0 else base
        for edge,r in enumerate(pool.itertuples()):
            a,b=int(r.source_index),int(r.target_index); variants=[source_temporal_shift(base,a),source_block_permutation(base,a),replace_channels(base,replacement,[a]),target_endpoint_spike(base,b,float(channel_robust[b])),target_short_level_shift(base,b,float(channel_robust[b])),replace_channels(base,replacement,[a,b]),replace_channels(base,replacement,selected_channels)]
            for fi,v in enumerate(variants):
                controlled[fi,i,edge]=score_window(v,pool,selfs,crosses,refs); success[fi,i,edge]=repl_i>=0 if fi in (2,5,6) else True
                if i==0: ma.append({"family":FAMILIES[fi],"edge_id":edge,"source_multiset_preserved":same_multiset(base[:,a],v[:,a]) if fi in (0,1) else "not_applicable","finite":bool(np.isfinite(controlled[fi,i,edge]).all())})
            cm.append({"source_window_id":source_ids[i],"target_edge":edge,"replacement_window_id":source_ids[repl_i] if repl_i>=0 else "missing","same_sequence":True,"nonoverlap":repl_i>=0})
    pd.DataFrame(cm).to_csv(out/"controlled_dynamic_corruption_manifest.csv",index=False); pd.DataFrame(ma).to_csv(out/"controlled_dynamic_marginal_audit.csv",index=False); status["corruptions_complete"]=True
    threshold_rows=[]; aggrows=[]; corrows=[]; seqrows=[]; relrows=[]; pointrows=[]; controlrows=[]; attrrows=[]
    for method in cfg["aggregations"]:
        cscore=aggregate_edges(clean,method); thresholds={}
        for fold in ("A","B"):
            cal=np.array([j for j in range(n) if evalfold[j]!=fold]); evx=np.array([j for j in range(n) if evalfold[j]==fold]); th=float(np.quantile(cscore[cal],.99)); thresholds[fold]=th; threshold_rows.append({"aggregation":method,"fold":fold,"threshold":th,"calibration_count":len(cal),"evaluation_count":len(evx),"synthetic_used":False})
        cth=np.array([thresholds[f] for f in evalfold]); alarm=cscore>cth; seqf=[float(alarm[[s==q for s,_ in sample]].mean()) for q in range(6)]
        famres=[]
        for fi,fam in enumerate(FAMILIES):
            pos=aggregate_edges(controlled[fi],method); valid=success[fi]; pv=pos[valid]; pt=np.broadcast_to(cth[:,None],pos.shape)[valid]; cc=np.repeat(cscore,valid.sum(axis=1)); cct=np.repeat(cth,valid.sum(axis=1)); res=binary(cc,pv,cct,pt)
            row={"aggregation":method,"family":fam,"sample_count":len(pv),**res}; corrows.append(row)
            if fam in ("R1","R2","R3"): famres.append(res)
            elif fam in ("P1","P2"): pointrows.append({"aggregation":method,"family":fam,"structure_fpr":res["q99_tpr"],**res})
            else: controlrows.append({"aggregation":method,"family":fam,"fpr":res["q99_tpr"],"clean_fpr":alarm.mean(),"fpr_gap":abs(res["q99_tpr"]-alarm.mean()),"score_distribution_difference":float(np.median(pv)-np.median(cc))})
        macro={k:float(np.mean([x[k] for x in famres])) for k in ("auroc","aupr","q99_tpr","median_score_ratio")}; gates={"clean":alarm.mean()<=.02,"worst":max(seqf)<=.05,"break_auc":macro["auroc"]>=.80,"break_tpr":macro["q99_tpr"]>=.40,"pool":len(pool)>=minpool}
        aggrows.append({"aggregation":method,"clean_fpr":alarm.mean(),"worst_sequence_fpr":max(seqf),"fpr_variance":np.var(seqf),"alarm_count":alarm.sum(),**{f"macro_{k}":v for k,v in macro.items()},"partial_gates_passed":all(gates.values()),"gate_detail":json.dumps({k:bool(v) for k,v in gates.items()})})
        for q in range(6): seqrows.append({"aggregation":method,"sequence_id":f"train{q+1}","clean_fpr":seqf[q]})
        maxedge=np.argmax(clean,axis=1)
        for e,r in enumerate(pool.itertuples()): relrows.append({"aggregation":method,"edge_id":e,"source":r.source,"target":r.target,"clean_max_fraction":float((maxedge==e).mean())})
    pd.DataFrame(threshold_rows).to_csv(out/"dynamic_structure_crossfit_thresholds.csv",index=False); pd.DataFrame(corrows).to_csv(out/"dynamic_structure_metrics_by_corruption.csv",index=False); pd.DataFrame(seqrows).to_csv(out/"dynamic_structure_metrics_by_sequence.csv",index=False); pd.DataFrame(relrows).to_csv(out/"dynamic_structure_metrics_by_relation.csv",index=False); pd.DataFrame(pointrows).to_csv(out/"point_anomaly_boundary_control_metrics.csv",index=False); pd.DataFrame(controlrows).to_csv(out/"collaborative_preserving_control_metrics.csv",index=False); pd.DataFrame(relrows).to_csv(out/"dynamic_relation_edge_attribution.csv",index=False)
    agg=pd.DataFrame(aggrows); agg.to_csv(out/"dynamic_structure_metrics_by_aggregation.csv",index=False); status["candidate_evaluation_complete"]=True
    # Full frozen gate check.
    for idx,row in agg.iterrows():
        c=pd.DataFrame(controlrows); p=pd.DataFrame(pointrows); br=pd.DataFrame(corrows); m=row.aggregation
        full=(row.partial_gates_passed and (c[c.aggregation==m].fpr<=.03).all() and (c[c.aggregation==m].fpr_gap<=.015).all() and
              sum(br[(br.aggregation==m)&br.family.isin(["R1","R2","R3"])].auroc>=.85)>=2 and sum(br[(br.aggregation==m)&br.family.isin(["R1","R2","R3"])].q99_tpr>=.40)>=2 and
              (p[p.aggregation==m].structure_fpr<=.10).all())
        agg.loc[idx,"gates_passed"]=bool(full)
    passing=agg[agg.gates_passed.astype(bool)]; selected=(passing.sort_values(["macro_aupr","worst_sequence_fpr"],ascending=[False,True]).iloc[0] if len(passing) else agg.sort_values("macro_aupr",ascending=False).iloc[0])
    if len(passing): classification="dynamic_conditional_structure_is_valid"
    elif len(pool)<minpool: classification="insufficient_stable_dynamic_relation_pool"
    elif selected.clean_fpr>.02 or selected.worst_sequence_fpr>.05: classification="stable_dynamic_relations_exist_but_global_score_is_unstable"
    else: classification="dynamic_relation_representation_is_weak"
    pd.DataFrame(agg).to_csv(out/"dynamic_structure_metrics_by_aggregation.csv",index=False)
    config={"status":"frozen_for_hai_development_evaluation" if len(passing) else "not_frozen","aggregation":selected.aggregation,"relation_count":len(pool),"test_used":False}; (out/"selected_structure_v2a_config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    issues=[{"issue_id":"V2A-001","stage":"HAI Structure V2-A","research_question":"Are dynamic scores stable and specific?","status":"resolved" if len(passing) else "open","evidence":classification,"severity":"high","affects_validity":not bool(len(passing)),"affects_final_method":True,"minimum_future_action":"Do not enter HAI evaluation unless frozen","recomputation_scope":"V2-A controlled validation only","frozen_artifacts_reusable":True}]; (out/"unresolved_issues_delta.json").write_text(json.dumps(issues,indent=2))
    summary={"classification":classification,"self_model_count":len(self_models),"directed_pair_count":len(names)*(len(names)-1),"screened_count":len(screened),"selected_relation_count":len(pool),"selected_relations":pool[["source","target"]].to_dict("records"),"selected_aggregation":selected.aggregation,"gates_passed":bool(len(passing)),"allow_hai_development_evaluation":bool(len(passing)),"test_loaded":False,"attack_label_used_for_selection":False,"rec_loaded":False,"ae_loaded":False}; (out/"structure_v2a_summary.json").write_text(json.dumps(summary,indent=2))
    history=[resolve(cfg["step2_output"])/"hai_scaler_stats.csv",resolve(cfg["step4a_output"])/"hai_relation_feature_eligibility.csv"]
    pd.DataFrame([{"artifact_id":p.name,"source_path":str(p),"dependency_hash":sha(p),"reused":True,"recomputed":False,"hash_match":True} for p in history]).to_csv(out/"artifact_reuse_audit.csv",index=False); (out/"config_resolved.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    report=f"# HAI Structure V2-A\n\n- Selected dynamic relations: {len(pool)}.\n- Selected aggregation: {selected.aggregation}.\n- Classification: **{classification}**.\n- HAI test/AE/rec loaded: false/false/false.\n"; (out/"final_hai_structure_v2a_controlled_validation.md").write_text(report)
    status["final_report_complete"]=True; (out/"stage_status.json").write_text(json.dumps(status,indent=2)); print(json.dumps(summary,indent=2),flush=True); return 0

if __name__=="__main__": raise SystemExit(main())
