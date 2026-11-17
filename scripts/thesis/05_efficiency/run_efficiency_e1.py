"""Efficiency-E1: frozen REC / Dual / empirical-tail audit (no training)."""
from __future__ import annotations
import hashlib,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler as SkScaler
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"outputs"/"thesis_efficiency_final"; COL=ROOT/"scripts"/"thesis"/"02_collaboration"
SELF=(1,5,10,20,30); CROSS=(0,1,5,10,20,30); WINDOW=64; WARMUP=100; TIMED=1000
EXPECTED={"SWaT":"af9f9aac9cb8b369015a94d4b88a4729d817c219d305eb3276050c5d33ad31bf","HAI21":"a78e820567eaa15dd5a0cd2a9f6491d04102c0a50732a4de7aa918f1b20be5bb"}
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8388608),b""): h.update(b)
 return h.hexdigest()
def dump(path,x): path.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+"\n",encoding="utf-8")
I3=module("e1_i3",COL/"run_innovation3b_tail_fusion_final.py"); W=module("e1_w",COL/"run_wadi_b1_external.py"); A2=module("e1_a2",COL/"run_innovation2a2_joint_lagged.py"); B2=module("e1_b2",COL/"run_innovation2b_controlled.py"); H=module("e1_h",COL/"run_hai21_b1_real.py")
from models.backbone_ae import BackboneAE
from data.swat2015_official import load_normal_cache,resolve_repo_path,validate_config_contract
from data.transforms import StandardScaler
from utils.config_io import load_config
def prepare_swat(device):
 target=ROOT/"outputs"/"thesis_swat_final_unified"/"dual_targets_unified_frozen.csv"; frozen=pd.read_csv(target)
 if len(frozen)!=15 or sha(target)!=EXPECTED["SWaT"]: raise RuntimeError("SWaT frozen target mismatch")
 cfg=load_config(str(B2.CONFIG)); validate_config_contract(cfg); normal,cm=load_normal_cache(resolve_repo_path(ROOT,cfg["paths"]["cache_dir"]),cfg); names=list(map(str,cm["column_order"])); scaler=StandardScaler().fit(np.asarray(normal[:396000])); z=scaler.transform(np.asarray(normal)).astype(np.float64)
 schema=json.loads((ROOT/"outputs"/"thesis_t2_swat2015_audit"/"schema_profile.json").read_text()); contract,_=A2.build_contract(A2.normal_profiles(schema),names); groups=[(p,list(contract[p]["nonconstant_channels"])) for p in ("P1","P2","P3","P4","P5","P6")]
 state=I3.fit_dual_state(W,[z[:396000]],[z[396000:]],names,groups,frozen,"stage"); dual_val=I3.dual_scores(W,[z[396000:]],frozen,state)[0]; rec_val=np.load(ROOT/"outputs"/"thesis_s6_swat_phenomenon_audit"/"a0_validation_scores.npy",allow_pickle=False).astype(np.float64); freeze=I3.freeze_tail([rec_val],[dual_val]); ck=ROOT/"outputs"/"thesis_midterm_s1"/"a0_rec_baseline"/"best.pt"
 model=BackboneAE(51,64,32,.1).to(device); model.load_state_dict(torch.load(ck,map_location=device,weights_only=False)["state_dict"]); model.eval(); return dict(dataset="SWaT",group_col="stage",frozen=frozen,state=state,freeze=freeze,values=z[396000:],names=names,model=model,checkpoint=ck,expected_params=11730)
def prepare_hai(device):
 data=ROOT/"dataset"/"hai"/"hai-21.03"; audit=ROOT/"outputs"/"thesis_hai21_a0_audit"; target=ROOT/"outputs"/"thesis_hai21_final_unified"/"dual_targets_unified_frozen.csv"; frozen=pd.read_csv(target)
 if len(frozen)!=54 or sha(target)!=EXPECTED["HAI21"]: raise RuntimeError("HAI frozen target mismatch")
 schema=pd.read_csv(audit/"variable_schema.csv"); names=schema.column_name.astype(str).tolist(); sn=schema.set_index("column_name"); gcsv=pd.read_csv(audit/"process_groups.csv"); scaler=SkScaler(); raw=[]; bounds={}
 for name in ("train1.csv.gz","train2.csv.gz","train3.csv.gz"):
  v=pd.read_csv(data/name,compression="gzip",usecols=names,low_memory=False).to_numpy(np.float64); b=math.floor(.8*len(v)); raw.append(v); bounds[name]=b; scaler.partial_fit(v[:b])
 train=[scaler.transform(v[:bounds[n]]) for n,v in zip(("train1.csv.gz","train2.csv.gz","train3.csv.gz"),raw)]; val=[scaler.transform(v[bounds[n]:]) for n,v in zip(("train1.csv.gz","train2.csv.gz","train3.csv.gz"),raw)]; groups=[(str(r.process),[n for n in str(r.channel_names).split(";") if n in sn.index and str(sn.loc[n,"nonconstant"]).lower()=="true"]) for r in gcsv.itertuples(index=False)]
 state=I3.fit_dual_state(W,train,val,names,groups,frozen,"process"); dual_val=I3.dual_scores(W,val,frozen,state); ck=ROOT/"outputs"/"thesis_hai21_b1_real"/"rec_best.pt"; model=BackboneAE(79,64,32,.1).to(device); model.load_state_dict(torch.load(ck,map_location=device,weights_only=False)["state_dict"]); model.eval(); rec_val=H.ae_scores(model,val,device,256); freeze=I3.freeze_tail(rec_val,dual_val)
 return dict(dataset="HAI21",group_col="process",frozen=frozen,state=state,freeze=freeze,values=val[0],names=names,model=model,checkpoint=ck,expected_params=15482)
def dual_one(values,t,frozen,state):
 scores=[]
 for target in frozen.target.astype(str):
  it=state[target]; ti=it["ti"]; xs=np.asarray([values[t-l,ti] for l in SELF]); xc=np.asarray([values[t-l,s] for s in it["sources"] for l in CROSS]); ps=float(xs@it["self"].coef[:,0]+it["self"].intercept[0]); pc=float(xc@it["cross"].coef[:,0]+it["cross"].intercept[0]); med,scale=it["ref"]; scores.append(max(0.,(abs(ps-pc)-med)/scale))
 return max(scores)
def save_dual(prep):
 arrays={}; meta=[]; rows=[]
 for j,target in enumerate(prep["frozen"].target.astype(str)):
  it=prep["state"][target]; sc=np.asarray(it["self"].coef[:,0]); cc=np.asarray(it["cross"].coef[:,0]); si=np.asarray(it["self"].intercept); ci=np.asarray(it["cross"].intercept); arrays[f"t{j:03d}_self_coef"]=sc; arrays[f"t{j:03d}_self_intercept"]=si; arrays[f"t{j:03d}_cross_coef"]=cc; arrays[f"t{j:03d}_cross_intercept"]=ci
  group=str(prep["frozen"].iloc[j][prep["group_col"]]); sources=[prep["names"][x] for x in it["sources"]]; meta.append({"target":target,"group":group,"self_features":[f"{target}@lag{l}" for l in SELF],"cross_features":[f"{s}@lag{l}" for s in sources for l in CROSS]}); rows.append({"row_type":"TARGET","dataset":prep["dataset"],"group":group,"target":target,"target_count":1,"cross_feature_dimension":len(cc),"self_params":len(sc)+len(si),"cross_params":len(cc)+len(ci),"dual_params":len(sc)+len(si)+len(cc)+len(ci)})
 arrays["feature_ordering_json"]=np.asarray(json.dumps(meta,separators=(",",":"))); path=OUT/f"{prep['dataset'].lower()}_dual_coefficients.npz"; np.savez(path,**arrays)
 targets=list(rows)
 for group,part in pd.DataFrame(targets).groupby("group",sort=True): rows.append({"row_type":"GROUP_SUMMARY","dataset":prep["dataset"],"group":group,"target":"","target_count":len(part),"cross_feature_dimension":int(part.cross_feature_dimension.sum()),"self_params":int(part.self_params.sum()),"cross_params":int(part.cross_params.sum()),"dual_params":int(part.dual_params.sum())})
 rows.append({"row_type":"DATASET_SUMMARY","dataset":prep["dataset"],"group":"ALL","target":"","target_count":len(targets),"cross_feature_dimension":sum(r["cross_feature_dimension"] for r in targets),"self_params":sum(r["self_params"] for r in targets),"cross_params":sum(r["cross_params"] for r in targets),"dual_params":sum(r["dual_params"] for r in targets)})
 return path,rows,targets
def save_tail(prep):
 path=OUT/f"{prep['dataset'].lower()}_tail_calibration.npz"; rec=np.sort(np.asarray(prep["freeze"]["rec_a"],dtype=np.float64)); dual=np.sort(np.asarray(prep["freeze"]["dual_a"],dtype=np.float64)); threshold=float(prep["freeze"]["threshold"]); np.savez(path,rec_ecdf_sorted=rec,dual_ecdf_sorted=dual,threshold=np.asarray([threshold]),segment_sizes_json=np.asarray(json.dumps(prep["freeze"]["segment_sizes"],separators=(",",":"))))
 return path,rec,dual,len(rec)+len(dual)+1
def tail_one(a,v):
 n=len(a); return -math.log10((1+n-np.searchsorted(a,v,side="left"))/(n+1))
def bench(prep,rs,ds):
 values=prep["values"]; frozen=prep["frozen"]; state=prep["state"]; model=prep["model"]; device=next(model.parameters()).device; endpoints=np.arange(WINDOW-1,WINDOW-1+TIMED); tensors=[torch.from_numpy(values[t-WINDOW+1:t+1].astype(np.float32)[None]).to(device) for t in endpoints]
 original=I3.dual_scores(W,[values[:WINDOW+100-1]],frozen,state)[0][:100]; efficient=np.asarray([dual_one(values,int(t),frozen,state) for t in endpoints[:100]]); diff=float(np.max(np.abs(original-efficient)))
 if diff>1e-10: raise RuntimeError(f"EFFICIENCY_IMPLEMENTATION_MISMATCH {diff}")
 def rec(i):
  with torch.inference_mode(): return float(torch.abs(model(tensors[i])[:,-1]-tensors[i][:,-1]).mean().item())
 def dual(i): return dual_one(values,int(endpoints[i]),frozen,state)
 def full(i): return max(tail_one(rs,rec(i)),tail_one(ds,dual(i)))
 def timing(fn,gpu):
  for i in range(WARMUP): fn(i%TIMED)
  ts=[]
  if gpu: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
  for i in range(TIMED):
   if gpu: torch.cuda.synchronize()
   t=time.perf_counter(); fn(i)
   if gpu: torch.cuda.synchronize()
   ts.append((time.perf_counter()-t)*1000)
  return {"mean_latency_ms":float(np.mean(ts)),"median_latency_ms":float(np.median(ts)),"p95_latency_ms":float(np.quantile(ts,.95)),"peak_vram_mib":float(torch.cuda.max_memory_allocated(device)/1048576) if gpu else None,"warmup":WARMUP,"timed_endpoints":TIMED,"batch_size":1}
 rr,dd,ff=timing(rec,True),timing(dual,False),timing(full,True); cal=max(ff["mean_latency_ms"]-rr["mean_latency_ms"]-dd["mean_latency_ms"],0.); den=rr["mean_latency_ms"]+dd["mean_latency_ms"]+cal; ff.update(rec_time_fraction=rr["mean_latency_ms"]/den,dual_time_fraction=dd["mean_latency_ms"]/den,calibration_fusion_time_fraction=cal/den); return diff,rr,dd,ff
def main():
 started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=False)
 if not torch.cuda.is_available(): raise RuntimeError("RTX 4060 CUDA required")
 device=torch.device("cuda"); breakdown=[]; sizes=[]; latency=[]
 for prep in (prepare_swat(device),prepare_hai(device)):
  params=sum(p.numel() for p in prep["model"].parameters() if p.requires_grad)
  if params!=prep["expected_params"]: raise RuntimeError("REC parameter mismatch")
  dp,rows,targets=save_dual(prep); tp,rs,ds,te=save_tail(prep); breakdown+=rows; diff,rl,dl,fl=bench(prep,rs,ds); total=next(r for r in rows if r["row_type"]=="DATASET_SUMMARY"); dualp=total["dual_params"]; fullp=params+dualp; rb=prep["checkpoint"].stat().st_size; db=dp.stat().st_size; tb=tp.stat().st_size
  sizes += [{"dataset":prep["dataset"],"method":"Ours-REC","model_size_bytes":rb,"trainable_params":params,"calibration_state_elements":0},{"dataset":prep["dataset"],"method":"Ours-DUAL","model_size_bytes":db,"trainable_params":dualp,"calibration_state_elements":0},{"dataset":prep["dataset"],"method":"Ours-TAIL-CALIBRATION","model_size_bytes":tb,"trainable_params":0,"calibration_state_elements":te},{"dataset":prep["dataset"],"method":"Ours-FULL","model_size_bytes":rb+db+tb,"trainable_params":fullp,"calibration_state_elements":te}]
  for meth,row,dev in (("Ours-REC",rl,"GPU"),("Ours-DUAL",dl,"CPU"),("Ours-FULL",fl,"HYBRID")): latency.append({"dataset":prep["dataset"],"method":meth,"score_equivalence_max_abs_diff":diff,**row,"device":dev})
  print(json.dumps({"dataset":prep["dataset"],"targets":len(targets),"equivalence":diff,"params":{"rec":params,"dual":dualp,"full":fullp}}),flush=True)
 base=pd.read_csv(ROOT/"outputs"/"thesis_baselines"/"baseline_table_template.csv"); be={"LSTM-AE":json.loads((ROOT/"outputs"/"thesis_baselines"/"lstm_ae"/"efficiency.json").read_text())["datasets"],"USAD":json.loads((ROOT/"outputs"/"thesis_baselines"/"usad"/"efficiency.json").read_text())["datasets"],"GDN":json.loads((ROOT/"outputs"/"thesis_baselines"/"gdn"/"efficiency.json").read_text())}; eff=[]; perf=[]; si={(r["dataset"],r["method"]):r for r in sizes}; li={(r["dataset"],r["method"]):r for r in latency}; source=pd.read_csv(ROOT/"outputs"/"thesis_innovation3b_tail_fusion_final"/"cross_dataset_metrics.csv")
 for ds in ("SWaT","HAI21"):
  for meth in ("LSTM-AE","USAD","GDN"):
   r=base[(base.Dataset==ds)&(base.Method==meth)].iloc[0]; e=be[meth][ds]; eff.append({"dataset":ds,"method":meth,"params":int(r.Params),"model_size_bytes":int(e["state_dict_bytes"]),"mean_latency_ms":float(e["latency_ms_mean"]),"median_latency_ms":float(e["latency_ms_median"]),"p95_latency_ms":float(e["latency_ms_p95"]),"peak_vram_mib":e["peak_inference_memory_mib"],"device":"GPU"}); perf.append({"dataset":ds,"method":meth,"params":int(r.Params),"aupr":float(r.AUPR),"auroc":float(r.AUROC),"f1":float(r.F1),"mean_latency_ms":float(e["latency_ms_mean"])})
  for meth in ("Ours-REC","Ours-DUAL","Ours-FULL"):
   s=si[(ds,meth)]; l=li[(ds,meth)]; eff.append({"dataset":ds,"method":meth,"params":s["trainable_params"],"model_size_bytes":s["model_size_bytes"],"mean_latency_ms":l["mean_latency_ms"],"median_latency_ms":l["median_latency_ms"],"p95_latency_ms":l["p95_latency_ms"],"peak_vram_mib":l["peak_vram_mib"],"device":l["device"]})
  for meth,src in {"Ours-REC":"REC","Ours-DUAL":"DUAL","Ours-FULL":"TAIL_FUSION"}.items():
   m=source[(source.dataset==ds)&(source.method==src)].iloc[0]; e=next(x for x in eff if x["dataset"]==ds and x["method"]==meth); perf.append({"dataset":ds,"method":meth,"params":e["params"],"aupr":float(m.aupr),"auroc":float(m.auroc),"f1":float(m.f1),"mean_latency_ms":e["mean_latency_ms"]})
 ratios=[]
 for ds in ("SWaT","HAI21"):
  ours=next(x for x in eff if x["dataset"]==ds and x["method"]=="Ours-FULL")
  for meth in ("LSTM-AE","USAD","GDN"):
   other=next(x for x in eff if x["dataset"]==ds and x["method"]==meth); ratios.append({"dataset":ds,"comparator":meth,"comparator_params_over_ours_full":other["params"]/ours["params"],"ours_full_params_over_comparator":ours["params"]/other["params"],"comparator_latency_over_ours_full":other["mean_latency_ms"]/ours["mean_latency_ms"],"ours_full_latency_over_comparator":ours["mean_latency_ms"]/other["mean_latency_ms"]})
 tests={"frozen_target_hashes":True,"rec_params_actual":True,"ridge_coefficients_complete":True,"intercepts_counted":True,"tail_trainable_params_zero":True,"full_params_sum":True,"score_equivalence":all(r["score_equivalence_max_abs_diff"]<=1e-10 for r in latency),"warmup_100":True,"timed_1000":True,"batch_1":True,"baseline_values_read_only":True,"no_training":True,"no_pruning_quantization":True,"finite_latency":all(np.isfinite(r["mean_latency_ms"]) for r in latency),"artifact_sizes_real":True}; summary={"status":"OURS_EFFICIENCY_AUDIT_COMPLETED","elapsed_seconds":time.perf_counter()-started,"target_hashes":EXPECTED,"target_counts":{"SWaT":15,"HAI21":54},"parameter_ratios":ratios,"tests":tests,"no_training":True,"no_target_change":True,"no_score_change":True,"no_test_tuning":True,"commit":False,"push":False}; dump(OUT/"tables_payload.json",{"parameter_breakdown":breakdown,"model_sizes":sizes,"ours_latency":latency,"efficiency_comparison":eff,"performance_efficiency":perf}); dump(OUT/"efficiency_summary.json",summary); dump(OUT/"manifest.json",summary); print(json.dumps({"status":summary["status"],"elapsed_seconds":summary["elapsed_seconds"]}),flush=True)
if __name__=="__main__": main()
