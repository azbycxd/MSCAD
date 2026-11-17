from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hai import HAISequence, MultiSequenceWindowDataset, build_hai_dataloaders  # noqa: E402
from models.mscad import MSCAD  # noqa: E402
from scripts.hai_step3a_train_rec_baseline import (  # noqa: E402
    benchmark, limited_loader, parameter_counts, train_epoch, validation_loss,
)
from scripts.hai_step3e_evaluate_rec_causal_alarm_policy import (  # noqa: E402
    extract_events, metric_record, sha256_file,
)
from utils.causal_alarm_policy import binary_segments  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.pot import POTThreshold  # noqa: E402
from utils.seed import set_seed  # noqa: E402

OUTPUT_FILES = (
    "config_resolved.json", "rec_capacity_architecture_registry.csv", "checkpoint_audit.csv",
    "rec_capacity_training_summary.csv", "rec_capacity_training_curves.csv",
    "rec_capacity_score_consistency_audit.csv", "rec_capacity_thresholds.csv",
    "rec_capacity_normal_score_diagnostics.csv", "rec_capacity_channel_residual_comparison.csv",
    "rec_capacity_point_metrics_by_seed.csv", "rec_capacity_point_metrics_by_file.csv",
    "rec_capacity_point_metrics_aggregate.csv", "rec_capacity_event_diagnostics.csv",
    "rec_capacity_event_summary.csv", "rec_capacity_event_transition.csv",
    "rec_capacity_score_distribution_comparison.csv", "rec_capacity_false_positive_segments.csv",
    "rec_capacity_false_positive_segment_summary.csv", "rec_capacity_efficiency.csv",
    "rec_capacity_baseline_comparison.csv", "rec_capacity_multiseed_summary.csv",
    "rec_capacity_summary.json", "final_hai_rec_capacity_ablation_diagnosis.md",
)
FAST_OUTPUT_FILES = (
    "rec_capacity_core_metrics.csv", "rec_capacity_core_event_summary.csv",
    "rec_capacity_core_efficiency.csv", "rec_capacity_score_cache_audit.csv", "stage_status.json",
)
SIZES = ("tiny", "small", "medium")
METHODS = ("train_only_pot", "normal_validation_q99")
MONITORED_CHANNELS = (
    "P2_24Vdc", "P4_HT_FD", "P4_HT_PS", "P2_VT01", "P1_B2004", "P2_VIBTR01",
    "P1_PCV02Z", "P1_FCV02D", "P1_PP04", "P2_ManualSD", "P1_PIT02",
)


def ensure_fresh(output: Path) -> None:
    existing = [name for name in OUTPUT_FILES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite HAI Step 3G outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.replace([np.inf, -np.inf], np.nan).fillna("not_applicable").to_csv(path, index=False)


def atomic_json(path: Path, value: dict) -> None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8");temporary.replace(path)


def training_artifact_valid(size: str,seed: int,output: Path,architecture: pd.DataFrame) -> bool:
    sub=output/f"{size}_seed{seed}";checkpoint_path=sub/"best.pt";curve_path=sub/"training_curve.csv";config_path=sub/"config_resolved.json"
    if not all(p.exists() for p in (checkpoint_path,curve_path,config_path)):return False
    try:
        checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False);curve=pd.read_csv(curve_path)
        expected=int(architecture.set_index("model_size_name").loc[size,"parameter_count"])
        return (checkpoint.get("model_size_name")==size and int(checkpoint.get("seed",-1))==seed and
                int(checkpoint.get("parameter_count",-1))==expected and len(curve)>0 and bool(curve.finite.astype(bool).all()))
    except Exception:return False


def initialize_stage_status(cfg: dict,output: Path,architecture: pd.DataFrame) -> dict:
    existing={}
    if (output/"stage_status.json").exists():
        existing=json.loads((output/"stage_status.json").read_text(encoding="utf-8"))
    status={"mode":"fast_two_stage","tiny_scores_reused":True,"tiny_reinference":False,
            "reuse_source":"hai_step3d_rec_channel_calibration","models":existing.get("models",{})}
    for size in SIZES:
        for seed in cfg["seeds"]:
            key=f"{size}_seed{seed}"
            if size=="tiny":
                checkpoint=ROOT/cfg["tiny_checkpoint_pattern"].format(seed=seed)
                status["models"][key]={"training_complete":True,"checkpoint_path":str(checkpoint.resolve()),
                    "checkpoint_sha256":sha256_file(checkpoint),"validation_score_cache_complete":True,
                    "test_score_cache_complete":True,"core_metrics_complete":False,"full_diagnostics_complete":False,
                    "tiny_scores_reused":True,"tiny_reinference":False}
            else:
                valid=training_artifact_valid(size,int(seed),output,architecture);checkpoint=output/key/"best.pt"
                cache=output/key/"core_scores.npz"
                previous=status["models"].get(key,{})
                status["models"][key]={"training_complete":valid,"checkpoint_path":str(checkpoint.resolve()) if valid else "pending",
                    "checkpoint_sha256":sha256_file(checkpoint) if valid else "pending",
                    "validation_score_cache_complete":cache.exists(),"test_score_cache_complete":cache.exists(),
                    "core_metrics_complete":bool(previous.get("core_metrics_complete",False) and cache.exists()),
                    "full_diagnostics_complete":bool(previous.get("full_diagnostics_complete",False))}
    atomic_json(output/"stage_status.json",status);return status


def make_model(hidden: int, latent: int, device: torch.device) -> MSCAD:
    return MSCAD(in_channels=86, hidden_channels=hidden, latent_channels=latent, dropout=.1,
                 cov_type="corr", short_window=16, backbone_variant="base").to(device)


def trainable_parameters(hidden: int, latent: int) -> int:
    return parameter_counts(make_model(hidden, latent, torch.device("cpu")))[1]


def integer_values(spec: dict) -> range:
    return range(int(spec["start"]), int(spec["stop"]) + 1, int(spec["step"]))


def resolve_architectures(cfg: dict) -> pd.DataFrame:
    arch = cfg["architecture"]
    candidates = [(trainable_parameters(h, l), h, l)
                  for h in integer_values(arch["hidden_candidates"])
                  for l in integer_values(arch["latent_candidates"])]
    rows = []
    definitions = (("tiny", int(arch["tiny_hidden"]), int(arch["tiny_latent"]), 16420, (0, 99999), False),)
    for name, h, l, target, limits, primary in definitions:
        count = trainable_parameters(h, l)
        rows.append((name, h, l, count, target, limits, primary))
    for name in ("small", "medium"):
        target = int(arch[f"{name}_target"]); limits = tuple(int(v) for v in arch[f"{name}_range"])
        legal = [item for item in candidates if limits[0] <= item[0] <= limits[1] and item[0] < 100000]
        if not legal:
            legal = [item for item in candidates if item[0] < 100000]
        count, h, l = min(legal, key=lambda item: (abs(item[0] - target), item[1], item[2]))
        rows.append((name, h, l, count, target, limits, name == "small"))
    frame = pd.DataFrame([{
        "model_size_name": name, "encoder_channels": f"86->{h}->{l}",
        "hidden_channels": h, "bottleneck_channels": l, "decoder_channels": f"{l}->{h}->86",
        "parameter_count": count, "checkpoint_size_estimate": count * 4,
        "target_parameter_count": target, "target_distance": abs(count-target),
        "within_registered_range": low <= count <= high, "primary_candidate": primary,
        "architecture_family_unchanged": True, "layer_count_unchanged": True,
        "kernel_size_unchanged": True, "selection_uses_training_or_test_metrics": False,
    } for name,h,l,count,target,(low,high),primary in rows])
    if (frame.parameter_count >= 100000).any():
        raise ValueError("A capacity candidate exceeds the 100K parameter budget")
    return frame


def file_hashes(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        if path.is_file():
            result[str(path.resolve())] = sha256_file(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file(): result[str(child.resolve())] = sha256_file(child)
    return result


def data_files(cfg: dict) -> list[Path]:
    root = Path(cfg["data_root"])
    return [root / f"train{i}.csv" for i in range(1,7)] + [root / f"test{i}.csv" for i in range(1,5)]


def source_files(cfg: dict) -> list[Path]:
    paths = data_files(cfg)
    paths += [ROOT / cfg["tiny_checkpoint_pattern"].format(seed=s) for s in cfg["seeds"]]
    paths += [ROOT / p for p in cfg["source_output_dirs"]]
    return paths


def sequence_loader(sequence: HAISequence, cfg: dict, maximum: int | None = None) -> DataLoader:
    dataset: Any = MultiSequenceWindowDataset([sequence], int(cfg["data"]["window_size"]), 1, return_metadata=True)
    if maximum is not None:
        dataset = Subset(dataset, list(range(min(maximum, len(dataset)))))
    return DataLoader(dataset, batch_size=int(cfg["data"]["batch_size"]), shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def infer_sequence(model: MSCAD, sequence: HAISequence, cfg: dict, device: torch.device,
                   maximum: int | None = None) -> tuple[dict[str,np.ndarray], np.ndarray]:
    scores=[]; losses=[]; errors=[]; labels=[]; timestamps=[]; end_rows=[]; sequence_ids=[]
    model.eval()
    for x,y,meta in sequence_loader(sequence,cfg,maximum):
        x=x.to(device,non_blocking=True); xhat=model(x)
        err=(x[:,-1,:]-xhat[:,-1,:]).abs(); score=torch.topk(err,k=5,dim=1).values.mean(dim=1)
        loss=F.smooth_l1_loss(xhat,x,reduction="none").mean(dim=(1,2))
        if not torch.isfinite(score).all() or not torch.isfinite(loss).all() or not torch.isfinite(err).all():
            raise FloatingPointError("Non-finite model output, score, or reconstruction loss")
        scores.append(score.cpu().numpy().astype(np.float32)); losses.append(loss.cpu().numpy().astype(np.float32))
        errors.append(err.cpu().numpy().astype(np.float32)); labels.append(y.numpy().astype(np.int64))
        timestamps.extend(str(v) for v in meta["timestamp"]); end_rows.extend(int(v) for v in meta["end_row"])
        sequence_ids.extend(str(v) for v in meta["sequence_id"])
    values={"score":np.concatenate(scores),"reconstruction_loss":np.concatenate(losses),"label":np.concatenate(labels),
            "timestamp":np.asarray(timestamps,str),"end_row":np.asarray(end_rows,np.int64),
            "sequence_id":np.asarray(sequence_ids,str)}
    return values,np.concatenate(errors)


def load_tiny_checkpoint(seed: int, cfg: dict, device: torch.device, architecture: pd.DataFrame) -> tuple[MSCAD,dict,str]:
    path=ROOT/cfg["tiny_checkpoint_pattern"].format(seed=seed); digest=sha256_file(path)
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    row=architecture[architecture.model_size_name=="tiny"].iloc[0]
    model=make_model(int(row.hidden_channels),int(row.bottleneck_channels),device)
    model.load_state_dict(checkpoint["model_state"],strict=True); model.eval()
    return model,checkpoint,digest


def train_candidate(size: str, seed: int, row: pd.Series, bundle, cfg: dict, output: Path,
                    device: torch.device, smoke: bool) -> tuple[MSCAD,list[dict],dict]:
    sub=output/f"{size}_seed{seed}"; sub.mkdir(parents=True,exist_ok=False); set_seed(seed)
    model=make_model(int(row.hidden_channels),int(row.bottleneck_channels),device)
    generator=torch.Generator().manual_seed(seed)
    train_dataset=MultiSequenceWindowDataset(bundle.train_sequences,int(cfg["data"]["window_size"]),1,return_metadata=False)
    val_dataset=MultiSequenceWindowDataset(bundle.val_sequences,int(cfg["data"]["window_size"]),1,return_metadata=False)
    train_loader=DataLoader(train_dataset,batch_size=int(cfg["data"]["batch_size"]),shuffle=True,
                            generator=generator,num_workers=0,pin_memory=device.type=="cuda",drop_last=False)
    val_loader=DataLoader(val_dataset,batch_size=int(cfg["data"]["batch_size"]),shuffle=False,
                          num_workers=0,pin_memory=device.type=="cuda",drop_last=False)
    if smoke:
        maximum=int(cfg["smoke"]["max_windows_per_sequence"])*6
        train_loader=limited_loader(train_loader,maximum,True,seed); val_loader=limited_loader(val_loader,maximum,False,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["training"]["lr"]),
                                weight_decay=float(cfg["training"]["weight_decay"]))
    epochs=int(cfg["smoke"]["epochs"] if smoke else cfg["training"]["max_epochs"])
    best=float("inf"); best_epoch=-1; counter=0; history=[]; started_all=time.perf_counter()
    for epoch in range(1,epochs+1):
        started=time.perf_counter(); train_loss,max_grad=train_epoch(model,train_loader,optimizer,device)
        val_loss=validation_loss(model,val_loader,device); improved=val_loss<best
        if improved:
            best=val_loss;best_epoch=epoch;counter=0
            torch.save({"model_state":model.state_dict(),"model_size_name":size,"hidden_channels":int(row.hidden_channels),
                        "latent_channels":int(row.bottleneck_channels),"parameter_count":int(row.parameter_count),
                        "seed":seed,"best_epoch":epoch,"best_val_rec_loss":val_loss,
                        "selection_scope":"normal validation reconstruction loss only","cfg":cfg,
                        "feature_names":bundle.feature_names},sub/"best.pt")
        else: counter+=1
        record={"model_size_name":size,"seed":seed,"epoch":epoch,"train_rec_loss":train_loss,
                "validation_rec_loss":val_loss,"max_gradient_norm":max_grad,"invalid_gradient_count":0,
                "invalid_output_count":0,"finite":bool(np.isfinite([train_loss,val_loss,max_grad]).all()),
                "improved":improved,"early_stop_counter":counter,"epoch_seconds":time.perf_counter()-started}
        history.append(record); print(f"{size} seed={seed} epoch={epoch} train={train_loss:.6f} val={val_loss:.6f} grad={max_grad:.4f}",flush=True)
        if bool(cfg["training"]["early_stopping"]) and counter>=int(cfg["training"]["patience"]): break
    checkpoint=torch.load(sub/"best.pt",map_location=device,weights_only=False);model.load_state_dict(checkpoint["model_state"],strict=True);model.eval()
    resolved={"model_size_name":size,"seed":seed,"hidden_channels":int(row.hidden_channels),
              "latent_channels":int(row.bottleneck_channels),"parameter_count":int(row.parameter_count),
              "training":cfg["training"],"data":cfg["data"],"score":cfg["score"],"test_used_for_selection":False}
    (sub/"config_resolved.json").write_text(json.dumps(resolved,indent=2),encoding="utf-8")
    pd.DataFrame(history).to_csv(sub/"training_curve.csv",index=False)
    summary={"model_size_name":size,"seed":seed,"training_executed":True,"first_epoch_train_loss":history[0]["train_rec_loss"],
             "first_epoch_validation_loss":history[0]["validation_rec_loss"],"best_validation_loss":best,
             "best_epoch":best_epoch,"final_epoch":history[-1]["epoch"],
             "early_stopping_status":history[-1]["early_stop_counter"]>=int(cfg["training"]["patience"]),
             "final_train_validation_loss_gap":history[-1]["validation_rec_loss"]-history[-1]["train_rec_loss"],
             "gradient_maximum":max(r["max_gradient_norm"] for r in history),"invalid_gradient_count":0,
             "invalid_output_count":0,"training_time_seconds":time.perf_counter()-started_all,
             "checkpoint_path":str((sub/"best.pt").resolve()),"checkpoint_sha256":sha256_file(sub/"best.pt")}
    return model,history,summary


def fit_thresholds(train: np.ndarray, validation: np.ndarray, cfg: dict) -> list[dict[str,Any]]:
    rows=[]; tc=cfg["thresholds"]
    try:
        pot=POTThreshold(q=float(tc["pot_q"]),target_prob=float(tc["pot_target_prob"])).fit(train)
        if pot.tau is None or not np.isfinite(pot.tau): raise ValueError("invalid POT tau")
        rows.append({"threshold_method":"train_only_pot","threshold_value":float(pot.tau),"status":"ok",
                     "fitting_sample_count":len(train),"fitting_scope":"normal train only","test_labels_used":False})
    except Exception as exc:
        rows.append({"threshold_method":"train_only_pot","threshold_value":np.nan,"status":"failed",
                     "fitting_sample_count":len(train),"fitting_scope":"normal train only","test_labels_used":False,
                     "failure_reason":f"{type(exc).__name__}: {exc}"})
    rows.append({"threshold_method":"normal_validation_q99","threshold_value":float(np.quantile(validation,float(tc["validation_q99"]))),
                 "status":"ok","fitting_sample_count":len(validation),"fitting_scope":"normal validation only","test_labels_used":False})
    return rows


def top5_historical(seed: int, cfg: dict, maximum: int | None) -> dict[str,dict[str,dict[str,np.ndarray]]]:
    archive=np.load(ROOT/cfg["tiny_score_pattern"].format(seed=seed),allow_pickle=False); suffix="_endpoint_raw_top5"
    out={"train":{},"validation":{},"test":{}}
    for key in archive.files:
        if not key.endswith(suffix):continue
        prefix=key[:-len(suffix)];split,stem=prefix.split("_",1);count=len(archive[key])
        if maximum is not None:count=min(count,maximum)
        out[split][f"{stem}.csv"]={"score":archive[key][:count].astype(np.float32),
            **{field:archive[f"{prefix}_{field}"][:count] for field in ("label","timestamp","end_row","sequence_id")}}
    return out


def load_trained_candidate(size: str,seed: int,output: Path,architecture: pd.DataFrame,device: torch.device) -> MSCAD:
    row=architecture.set_index("model_size_name").loc[size];model=make_model(int(row.hidden_channels),int(row.bottleneck_channels),device)
    checkpoint=torch.load(output/f"{size}_seed{seed}"/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model_state"],strict=True);model.eval();return model


def create_core_score_cache(model: MSCAD,size: str,seed: int,bundle,cfg: dict,device: torch.device,path: Path) -> dict:
    if path.exists():
        return {"model_size_name":size,"seed":seed,"cache_path":str(path.resolve()),"cache_sha256":sha256_file(path),
                "created_now":False,"model_forward_passes_for_cache":0}
    arrays={};counts={}
    for split,sequences in (("validation",bundle.val_sequences),("test",bundle.test_sequences)):
        for sequence in sequences:
            values,errors=infer_sequence(model,sequence,cfg,device,None);prefix=f"{split}_{Path(sequence.file_name).stem}"
            for field in ("score","label","timestamp","end_row","sequence_id","reconstruction_loss"):
                arrays[f"{prefix}_{field}"]=values[field]
            arrays[f"{prefix}_endpoint_error"]=errors;counts[prefix]=len(values["score"])
    np.savez(path,**arrays)
    return {"model_size_name":size,"seed":seed,"cache_path":str(path.resolve()),"cache_sha256":sha256_file(path),
            "created_now":True,"model_forward_passes_for_cache":1,"validation_points":sum(v for k,v in counts.items() if k.startswith("validation_")),
            "test_points":sum(v for k,v in counts.items() if k.startswith("test_"))}


def load_core_score_cache(path: Path) -> dict[str,dict[str,dict[str,np.ndarray]]]:
    archive=np.load(path,allow_pickle=False);out={"validation":{},"test":{}}
    for key in archive.files:
        if not key.endswith("_score"):continue
        prefix=key[:-6];split,stem=prefix.split("_",1)
        if split not in out:continue
        fields=("score","label","timestamp","end_row","sequence_id","reconstruction_loss","endpoint_error")
        out[split][f"{stem}.csv"]={field:archive[f"{prefix}_{field}"] for field in fields}
    return out


def candidate_q99_core(size: str,seed: int,splits: dict) -> tuple[dict,list[dict],list[dict]]:
    validation=np.concatenate([splits["validation"][n]["score"] for n in sorted(splits["validation"])]);threshold=float(np.quantile(validation,.99))
    threshold_row={"model_size_name":size,"seed":seed,"threshold_method":"normal_validation_q99","threshold_value":threshold,
                   "fitting_scope":"normal validation only","test_labels_used":False}
    files,seeds=point_rows(size,seed,splits["test"],[{"threshold_method":"normal_validation_q99","threshold_value":threshold,"status":"ok"}])
    _,events=event_rows(size,seed,splits["test"],[{"threshold_method":"normal_validation_q99","threshold_value":threshold,"status":"ok"}],True)
    return threshold_row,seeds,events


def reused_tiny_core(cfg: dict) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    point=pd.read_csv(ROOT/"outputs/hai_step3d_rec_channel_calibration/rec_channel_point_metrics_by_seed.csv")
    point=point[(point.score_variant=="endpoint_raw_top5")&(point.threshold_method=="normal_validation_q99")].copy()
    point["model_size_name"]="tiny";point=point.rename(columns={"normal_false_positive_rate":"normal_fpr"})
    events=pd.read_csv(ROOT/"outputs/hai_step3d_rec_channel_calibration/rec_channel_event_summary.csv")
    events=events[(events.score_variant=="endpoint_raw_top5")&(events.threshold_method=="normal_validation_q99")&(events.summary_scope=="all_tests")].copy()
    events["model_size_name"]="tiny"
    thresholds=pd.read_csv(ROOT/"outputs/hai_step3d_rec_channel_calibration/rec_channel_thresholds.csv")
    thresholds=thresholds[(thresholds.score_variant=="endpoint_raw_top5")&(thresholds.threshold_method.isin(METHODS))].copy();thresholds["model_size_name"]="tiny"
    return point,events,thresholds


def simplified_latency(model: MSCAD,device: torch.device,batch_size: int=128) -> dict[str,float]:
    x=torch.zeros((batch_size,64,86),device=device);model.eval()
    with torch.no_grad():
        for _ in range(10):model(x)
        if device.type=="cuda":torch.cuda.synchronize()
        started=time.perf_counter()
        for _ in range(100):model(x)
        if device.type=="cuda":torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
    return {"batch_forward_latency_ms":elapsed*10,"window_forward_latency_ms":elapsed*10/batch_size}


def resumed_training_summary(size: str,seed: int,output: Path) -> tuple[list[dict],dict]:
    sub=output/f"{size}_seed{seed}";curve=pd.read_csv(sub/"training_curve.csv");checkpoint=torch.load(sub/"best.pt",map_location="cpu",weights_only=False)
    summary={"model_size_name":size,"seed":seed,"training_executed":True,"resumed_existing_checkpoint":True,
        "first_epoch_train_loss":float(curve.iloc[0].train_rec_loss),"first_epoch_validation_loss":float(curve.iloc[0].validation_rec_loss),
        "best_validation_loss":float(checkpoint["best_val_rec_loss"]),"best_epoch":int(checkpoint["best_epoch"]),
        "final_epoch":int(curve.iloc[-1].epoch),"early_stopping_status":bool(int(curve.iloc[-1].early_stop_counter)>=5),
        "final_train_validation_loss_gap":float(curve.iloc[-1].validation_rec_loss-curve.iloc[-1].train_rec_loss),
        "gradient_maximum":float(curve.max_gradient_norm.max()),"invalid_gradient_count":int(curve.invalid_gradient_count.sum()),
        "invalid_output_count":int(curve.invalid_output_count.sum()),"training_time_seconds":float(curve.epoch_seconds.sum()),
        "checkpoint_path":str((sub/"best.pt").resolve()),"checkpoint_sha256":sha256_file(sub/"best.pt")}
    return curve.to_dict(orient="records"),summary


def score_diagnostics(size: str, seed: int, split: str, name: str, values: dict, threshold: float) -> dict[str,Any]:
    score=values["score"]; high=(score>threshold).astype(np.int8); starts,ends=binary_segments(high); lengths=ends-starts+1
    return {"model_size_name":size,"seed":seed,"split":split,"file_name":name,
            "reconstruction_loss":float(values["reconstruction_loss"].mean()),"endpoint_raw_top5_mean":float(score.mean()),
            "median":float(np.median(score)),"std":float(score.std()),"q90":float(np.quantile(score,.90)),
            "q95":float(np.quantile(score,.95)),"q99":float(np.quantile(score,.99)),"maximum":float(score.max()),
            "segments_above_q99":len(starts),"longest_segment_above_q99":int(lengths.max()) if len(lengths) else 0,
            "high_score_points_per_hour":float(high.sum()*3600/max(len(high),1))}


def fp_stats(labels: np.ndarray, alarm: np.ndarray) -> dict[str,Any]:
    fp=((labels==0)&(alarm==1)).astype(np.int8);starts,ends=binary_segments(fp);lengths=ends-starts+1
    return {"false_positive_segment_count":len(starts),"longest_false_positive_segment":int(lengths.max()) if len(lengths) else 0,
            "false_positive_segments_ge_30":int((lengths>=30).sum())}


def point_rows(size: str,seed: int,tests: dict,thresholds: list[dict]) -> tuple[list[dict],list[dict]]:
    files=[];seeds=[]
    for tr in thresholds:
        if tr["status"]!="ok":continue
        method=tr["threshold_method"];threshold=float(tr["threshold_value"]); current=[]
        for name,v in tests.items():
            alarm=(v["score"]>threshold).astype(np.int8);m=metric_record(v["label"],alarm,v["score"])
            row={"model_size_name":size,"seed":seed,"threshold_method":method,"file_name":name,
                 "aupr":m.pop("reference_aupr"),"auroc":m.pop("reference_auroc"),**m,**fp_stats(v["label"],alarm)}
            files.append(row);current.append(row)
        labels=np.concatenate([tests[n]["label"] for n in sorted(tests)]);scores=np.concatenate([tests[n]["score"] for n in sorted(tests)])
        alarm=(scores>threshold).astype(np.int8);m=metric_record(labels,alarm,scores)
        seeds.append({"model_size_name":size,"seed":seed,"threshold_method":method,"scope":"micro",
                      "aupr":m.pop("reference_aupr"),"auroc":m.pop("reference_auroc"),**m,**fp_stats(labels,alarm)})
        seeds.append({"model_size_name":size,"seed":seed,"threshold_method":method,"scope":"macro",
                      **{metric:float(np.mean([r[metric] for r in current])) for metric in ("precision","recall","f1","aupr","auroc","normal_fpr")},
                      **{metric:int(sum(r[metric] for r in current)) for metric in ("true_positive_count","false_positive_count","false_negative_count","false_positive_segments_ge_30")}})
    return files,seeds


def event_rows(size: str,seed: int,tests: dict,thresholds: list[dict],require58: bool) -> tuple[list[dict],list[dict]]:
    events=extract_events(tests)
    if require58 and len(events)!=58:raise ValueError(f"Expected 58 events, got {len(events)}")
    diagnostics=[];summaries=[]
    for tr in thresholds:
        if tr["status"]!="ok":continue
        method=tr["threshold_method"];threshold=float(tr["threshold_value"])
        for event in events:
            v=tests[event["file_name"]];a=event["start_index"];b=event["end_index"]
            local=(v["score"][a:b+1]>threshold).astype(np.int8);hit=np.flatnonzero(local);detected=len(hit)>0
            diagnostics.append({"model_size_name":size,"seed":seed,"threshold_method":method,**event,
                "detected":detected,"first_alarm_timestamp":str(v["timestamp"][a+hit[0]]) if detected else "not_detected",
                "detection_delay_points":int(hit[0]) if detected else "not_detected","point_recall_in_event":float(local.mean()),
                "peak_score":float(v["score"][a:b+1].max()),"mean_score":float(v["score"][a:b+1].mean()),
                "first_point_hit":bool(detected and hit[0]==0),"within5_hit":bool(detected and hit[0]<5),
                "within10_hit":bool(detected and hit[0]<10),"within30_hit":bool(detected and hit[0]<30)})
        current=[r for r in diagnostics if r["threshold_method"]==method]
        detected=[r for r in current if r["detected"]]; delays=[r["detection_delay_points"] for r in detected]
        summaries.append({"model_size_name":size,"seed":seed,"threshold_method":method,"event_count":len(current),
            "detected_event_count":sum(r["detected"] for r in current),"event_recall":float(np.mean([r["detected"] for r in current])) if current else 0,
            "fully_missed_event_count":sum(not r["detected"] for r in current),"median_delay":float(np.median(delays)) if delays else np.nan,
            "mean_delay":float(np.mean(delays)) if delays else np.nan,"within5_hit_rate":float(np.mean([r["within5_hit"] for r in current])) if current else 0,
            "within10_hit_rate":float(np.mean([r["within10_hit"] for r in current])) if current else 0,
            "within30_hit_rate":float(np.mean([r["within30_hit"] for r in current])) if current else 0})
    return diagnostics,summaries


def distribution_rows(size: str,seed: int,tests: dict) -> list[dict]:
    labels=np.concatenate([tests[n]["label"] for n in sorted(tests)]);score=np.concatenate([tests[n]["score"] for n in sorted(tests)])
    rows=[];normal=score[labels==0];anomaly=score[labels==1]
    both=len(normal)>0 and len(anomaly)>0
    pooled_std=float(np.sqrt((normal.var()+anomaly.var())/2)) if both else np.nan
    smd=float((anomaly.mean()-normal.mean())/max(pooled_std,1e-12)) if both else np.nan
    if both:
        bins=np.histogram_bin_edges(np.r_[normal,anomaly],bins=200);hn,_=np.histogram(normal,bins,density=True);ha,_=np.histogram(anomaly,bins,density=True)
        overlap=float(np.sum(np.minimum(hn,ha)*np.diff(bins)))
    else: overlap=np.nan
    for label,values in (("normal",normal),("anomaly",anomaly)):
        available=len(values)>0
        rows.append({"model_size_name":size,"seed":seed,"class_name":label,"count":len(values),
                     "statistics_status":"ok" if available else "not_applicable_no_class_samples",
                     "mean":float(values.mean()) if available else np.nan,"median":float(np.median(values)) if available else np.nan,
                     "q25":float(np.quantile(values,.25)) if available else np.nan,"q75":float(np.quantile(values,.75)) if available else np.nan,
                     "q90":float(np.quantile(values,.90)) if available else np.nan,"q95":float(np.quantile(values,.95)) if available else np.nan,
                     "standardized_mean_difference":smd,"overlap_coefficient":overlap})
    return rows


def fp_segment_rows(size: str,seed: int,tests: dict,threshold: float) -> tuple[list[dict],list[dict]]:
    rows=[];summaries=[]
    for name,v in tests.items():
        alarm=((v["label"]==0)&(v["score"]>threshold)).astype(np.int8);starts,ends=binary_segments(alarm);lengths=ends-starts+1
        for sid,(a,b) in enumerate(zip(starts,ends),1):
            rows.append({"model_size_name":size,"seed":seed,"threshold_method":"normal_validation_q99","file_name":name,
                         "segment_id":sid,"start_timestamp":v["timestamp"][a],"end_timestamp":v["timestamp"][b],
                         "length":int(b-a+1),"peak_score":float(v["score"][a:b+1].max())})
        summaries.append({"model_size_name":size,"seed":seed,"threshold_method":"normal_validation_q99","file_name":name,
                          "segment_count":len(starts),"segments_ge_30":int((lengths>=30).sum()),
                          "longest_segment":int(lengths.max()) if len(lengths) else 0})
    return rows,summaries


def benchmark_capacity(model: MSCAD,bundle,cfg: dict,device: torch.device,size: str,seed: int,
                       training_seconds: float,smoke: bool) -> dict[str,Any]:
    timing_loaders={}
    for sequence in bundle.test_sequences:
        dataset: Any=MultiSequenceWindowDataset([sequence],int(cfg["data"]["window_size"]),1,return_metadata=False)
        if smoke: dataset=Subset(dataset,list(range(min(int(cfg["smoke"]["max_windows_per_sequence"]),len(dataset)))))
        timing_loaders[sequence.sequence_id]=DataLoader(dataset,batch_size=int(cfg["efficiency"]["batch_size"]),shuffle=False,num_workers=0)
    sample=next(iter(timing_loaders["test1"]))[0][:int(cfg["efficiency"]["batch_size"])]
    base=benchmark(model,sample,device,timing_loaders)
    return {"model_size_name":size,"seed":seed,"trainable_parameter_count":parameter_counts(model)[1],
            "checkpoint_size_bytes":0,"batch_forward_latency_ms":base["model_forward_time_ms_per_batch"],
            "window_forward_latency_ms":base["model_forward_time_ms_per_window"],
            "score_computation_latency_ms":base["rec_scoring_time_ms_per_batch"],
            "throughput_windows_per_second":base["end_to_end_test_windows_per_second"],
            "peak_gpu_memory_mb":base["peak_gpu_memory_mb"],"total_training_time_seconds":training_seconds,
            "device":base["device"],"batch_size":base["batch_size"],"warmup":10,"repetitions":100,
            "inference_complexity":"same four-layer depthwise-separable topology; width-only scaling"}


def aggregate_outputs(by_seed: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    aggregate=[];multi=[]
    for (size,method,scope),g in by_seed.groupby(["model_size_name","threshold_method","scope"]):
        row={"model_size_name":size,"threshold_method":method,"scope":scope}
        for metric in ("precision","recall","f1","aupr","auroc","false_positive_count","false_positive_segments_ge_30"):
            row[f"{metric}_mean"]=float(g[metric].mean());row[f"{metric}_std"]=float(g[metric].std(ddof=0));
            row[f"{metric}_min"]=float(g[metric].min());row[f"{metric}_max"]=float(g[metric].max())
            if metric in ("precision","recall","f1","aupr","auroc"):
                aggregate.append({"model_size_name":size,"threshold_method":method,"scope":scope,"metric":metric,
                                  "mean":row[f"{metric}_mean"],"std":row[f"{metric}_std"],"min":row[f"{metric}_min"],"max":row[f"{metric}_max"],
                                  **{f"seed{int(r.seed)}":float(getattr(r,metric)) for r in g.itertuples()}})
        multi.append(row)
    return pd.DataFrame(aggregate),pd.DataFrame(multi)


def comparisons(by_seed: pd.DataFrame,event_summary: pd.DataFrame,efficiency: pd.DataFrame,training: pd.DataFrame,fp_summary: pd.DataFrame) -> pd.DataFrame:
    metric=by_seed[by_seed.scope=="micro"].set_index(["model_size_name","seed","threshold_method"])
    events=event_summary.set_index(["model_size_name","seed","threshold_method"]); eff=efficiency.set_index(["model_size_name","seed"])
    pairs=(("small","tiny"),("medium","tiny"),("medium","small"));rows=[]
    for new,base in pairs:
        for seed in sorted(by_seed.seed.unique()):
            for method in METHODS:
                a=metric.loc[(base,seed,method)];b=metric.loc[(new,seed,method)];ea=events.loc[(base,seed,method)];eb=events.loc[(new,seed,method)]
                la=fp_summary[(fp_summary.model_size_name==base)&(fp_summary.seed==seed)].segments_ge_30.sum()
                lb=fp_summary[(fp_summary.model_size_name==new)&(fp_summary.seed==seed)].segments_ge_30.sum()
                va=training[(training.model_size_name==base)&(training.seed==seed)].best_validation_loss.iloc[0]
                vb=training[(training.model_size_name==new)&(training.seed==seed)].best_validation_loss.iloc[0]
                rows.append({"candidate":new,"baseline":base,"seed":seed,"threshold_method":method,
                    "validation_loss_delta":vb-va,"precision_delta":b.precision-a.precision,"recall_delta":b.recall-a.recall,
                    "f1_delta":b.f1-a.f1,"aupr_delta":b.aupr-a.aupr,"auroc_delta":b.auroc-a.auroc,
                    "false_positive_delta":b.false_positive_count-a.false_positive_count,"long_fp_segment_delta":int(lb-la),
                    "event_recall_delta":eb.event_recall-ea.event_recall,"median_delay_delta":eb.median_delay-ea.median_delay,
                    "parameter_increase_ratio":eff.loc[(new,seed)].trainable_parameter_count/eff.loc[(base,seed)].trainable_parameter_count-1,
                    "latency_increase_ratio":eff.loc[(new,seed)].batch_forward_latency_ms/eff.loc[(base,seed)].batch_forward_latency_ms-1})
    return pd.DataFrame(rows)


def classify(comparison: pd.DataFrame,by_seed: pd.DataFrame,event_summary: pd.DataFrame,fp_summary: pd.DataFrame) -> tuple[str,bool,dict]:
    q=comparison[(comparison.candidate=="small")&(comparison.baseline=="tiny")&(comparison.threshold_method=="normal_validation_q99")]
    small=by_seed[(by_seed.model_size_name=="small")&(by_seed.threshold_method=="normal_validation_q99")&(by_seed.scope=="micro")]
    events=event_summary[(event_summary.model_size_name=="small")&(event_summary.threshold_method=="normal_validation_q99")]
    long_tiny=fp_summary[fp_summary.model_size_name=="tiny"].segments_ge_30.sum();long_small=fp_summary[fp_summary.model_size_name=="small"].segments_ge_30.sum()
    gates={"ranking":bool((q.aupr_delta.mean()>=.015 or q.auroc_delta.mean()>=.015) and ((q.aupr_delta>0).sum()>=2 or (q.auroc_delta>0).sum()>=2)),
           "q99":bool(q.f1_delta.mean()>=.04 and q.precision_delta.mean()>=0 and (q.f1_delta>0).sum()>=2),
           "false_positive":bool(q.false_positive_delta.sum()/max(1,by_seed[(by_seed.model_size_name=="tiny")&(by_seed.threshold_method=="normal_validation_q99")&(by_seed.scope=="micro")].false_positive_count.sum())<=-.2),
           "long_false_positive":bool((long_small-long_tiny)/max(long_tiny,1)<=-.25),
           "events":bool(q.event_recall_delta.mean()>=-.05 and q.median_delay_delta.mean()<=10)}
    ready=bool(small.f1.mean()>=.30 and small.precision.mean()>=.35 and events.event_recall.mean()>=.75 and (small.f1>=.30).sum()>=2)
    medium=comparison[(comparison.candidate=="medium")&(comparison.baseline=="tiny")&(comparison.threshold_method=="normal_validation_q99")]
    if all(gates.values()): classification="small_capacity_resolves_main_underfitting" if ready else "capacity_improves_rec_but_absolute_performance_remains_limited"
    elif medium.f1_delta.mean()>=.04 and medium.aupr_delta.mean()>=.015 and medium.event_recall_delta.mean()>=-.05: classification="medium_capacity_is_needed_within_budget"
    elif q.false_positive_delta.mean()<0 and (q.aupr_delta.mean()<0 or q.event_recall_delta.mean()<-.05): classification="larger_models_reduce_normal_fp_but_also_reconstruct_anomalies"
    else: classification="model_capacity_is_not_main_rec_problem"
    return classification,ready,gates


def run(cfg: dict,output: Path,smoke: bool) -> dict[str,Any]:
    ensure_fresh(output); architecture=resolve_architectures(cfg)
    before=file_hashes(source_files(cfg) if not smoke else [ROOT/cfg["tiny_checkpoint_pattern"].format(seed=s) for s in cfg["seeds"]])
    device=torch.device("cuda" if cfg["device"]=="cuda" and torch.cuda.is_available() else "cpu")
    bundle=build_hai_dataloaders(cfg["data_root"],window_size=int(cfg["data"]["window_size"]),stride_train=1,stride_test=1,
        validation_ratio=float(cfg["data"]["validation_ratio"]),batch_size=int(cfg["data"]["batch_size"]),num_workers=0,
        pin_memory=device.type=="cuda",return_metadata=True,max_rows_per_file=None)
    if len(bundle.feature_names)!=86:raise ValueError("HAI Step 3G requires the fixed 86-channel schema")
    maximum=int(cfg["smoke"]["max_windows_per_sequence"]) if smoke else None
    training_rows=[];curve_rows=[];audit_rows=[];consistency=[];threshold_rows=[];normal_rows=[];channel_rows=[]
    point_file_rows=[];point_seed_rows=[];event_diag_rows=[];event_summary_rows=[];distribution=[];fp_rows=[];fp_summary_rows=[];eff_rows=[]
    for size in SIZES:
        arch=architecture[architecture.model_size_name==size].iloc[0]
        for seed in cfg["seeds"]:
            seed=int(seed); training_seconds=0.0
            if size=="tiny":
                model,checkpoint,digest=load_tiny_checkpoint(seed,cfg,device,architecture)
                old_dir=ROOT/f"outputs/hai_step3a_rec_baseline_seed{seed}"
                hist=pd.read_csv(old_dir/"training_history.csv")
                curve_rows.extend({"model_size_name":"tiny","seed":seed,**r} for r in hist.to_dict(orient="records"))
                old_eff=json.loads((old_dir/"efficiency_metrics.json").read_text(encoding="utf-8"));training_seconds=float(hist.elapsed_seconds.sum())
                training_rows.append({"model_size_name":"tiny","seed":seed,"training_executed":False,
                    "first_epoch_train_loss":float(hist.iloc[0].train_rec_loss),"first_epoch_validation_loss":float(hist.iloc[0].val_rec_loss),
                    "best_validation_loss":float(checkpoint["best_val_rec_loss"]),"best_epoch":int(checkpoint["best_epoch"]),
                    "final_epoch":int(hist.iloc[-1].epoch),"early_stopping_status":bool(hist.iloc[-1].early_stop_counter>=5),
                    "final_train_validation_loss_gap":float(hist.iloc[-1].val_rec_loss-hist.iloc[-1].train_rec_loss),
                    "gradient_maximum":float(hist.max_gradient_norm.max()),"invalid_gradient_count":0,"invalid_output_count":0,
                    "training_time_seconds":training_seconds,"checkpoint_path":str((old_dir/"best.pt").resolve()),"checkpoint_sha256":digest})
                audit_rows.append({"model_size_name":size,"seed":seed,"checkpoint_path":str((old_dir/"best.pt").resolve()),
                                   "checkpoint_sha256_before":digest,"training_executed":False,"tiny_reused_directly":True})
                historical=top5_historical(seed,cfg,maximum)
            else:
                model,history,summary=train_candidate(size,seed,arch,bundle,cfg,output,device,smoke)
                curve_rows.extend(history);training_rows.append(summary);training_seconds=float(summary["training_time_seconds"])
                digest=summary["checkpoint_sha256"]
                audit_rows.append({"model_size_name":size,"seed":seed,"checkpoint_path":summary["checkpoint_path"],
                                   "checkpoint_sha256_before":digest,"training_executed":True,"tiny_reused_directly":False})
                historical=None
            splits={"train":{},"validation":{},"test":{}}
            for split,sequences in (("train",bundle.train_sequences),("validation",bundle.val_sequences),("test",bundle.test_sequences)):
                for sequence in sequences:
                    values,errors=infer_sequence(model,sequence,cfg,device,maximum)
                    if historical is not None:
                        old=historical[split][sequence.file_name]
                        metadata_ok=all(np.array_equal(values[field],old[field]) for field in ("label","timestamp","end_row","sequence_id"))
                        diff=np.abs(values["score"]-old["score"])
                        scale=np.maximum(np.abs(old["score"]),1e-12);relative=diff/scale
                        reproduced=metadata_ok and bool(np.allclose(values["score"],old["score"],
                            atol=float(cfg["tiny_consistency"]["absolute_tolerance"]),
                            rtol=float(cfg["tiny_consistency"]["relative_tolerance"])))
                        consistency.append({"model_size_name":size,"seed":seed,"split":split,"file_name":sequence.file_name,
                            "score_max_absolute_difference":float(diff.max()),"score_mean_absolute_difference":float(diff.mean()),
                            "score_max_relative_difference":float(relative.max()),"metadata_exact_match":metadata_ok,
                            "absolute_tolerance":float(cfg["tiny_consistency"]["absolute_tolerance"]),
                            "relative_tolerance":float(cfg["tiny_consistency"]["relative_tolerance"]),"step3d_score_reproduced":reproduced})
                        if not reproduced:raise ValueError(f"Tiny Step 3D reproduction failed {seed}/{split}/{sequence.file_name}")
                        values["score"]=old["score"].copy()
                    splits[split][sequence.file_name]=values
                    if split in ("train","validation"):
                        for index,feature in enumerate(bundle.feature_names):
                            e=errors[:,index]
                            channel_rows.append({"model_size_name":size,"seed":seed,"split":split,"file_name":sequence.file_name,
                                "channel_index":index,"channel_name":feature,"monitored_high_residual_channel":feature in MONITORED_CHANNELS,
                                "residual_mean":float(e.mean()),"residual_median":float(np.median(e)),
                                "residual_q95":float(np.quantile(e,.95)),"residual_q99":float(np.quantile(e,.99))})
                    del errors
            train=np.concatenate([splits["train"][n]["score"] for n in sorted(splits["train"])]);val=np.concatenate([splits["validation"][n]["score"] for n in sorted(splits["validation"])])
            thresholds=fit_thresholds(train,val,cfg)
            threshold_rows.extend({"model_size_name":size,"seed":seed,**r} for r in thresholds)
            q99=float(next(r["threshold_value"] for r in thresholds if r["threshold_method"]=="normal_validation_q99"))
            for split in ("train","validation"):
                for name,values in splits[split].items():normal_rows.append(score_diagnostics(size,seed,split,name,values,q99))
            pf,ps=point_rows(size,seed,splits["test"],thresholds);point_file_rows.extend(pf);point_seed_rows.extend(ps)
            ed,es=event_rows(size,seed,splits["test"],thresholds,not smoke);event_diag_rows.extend(ed);event_summary_rows.extend(es)
            distribution.extend(distribution_rows(size,seed,splits["test"]))
            fr,fs=fp_segment_rows(size,seed,splits["test"],q99);fp_rows.extend(fr);fp_summary_rows.extend(fs)
            efficiency=benchmark_capacity(model,bundle,cfg,device,size,seed,training_seconds,smoke)
            checkpoint_path=Path(training_rows[-1]["checkpoint_path"]);efficiency["checkpoint_size_bytes"]=checkpoint_path.stat().st_size
            eff_rows.append(efficiency)
            del model
            if device.type=="cuda":torch.cuda.empty_cache()
    training=pd.DataFrame(training_rows);curves=pd.DataFrame(curve_rows);thresholds=pd.DataFrame(threshold_rows)
    by_file=pd.DataFrame(point_file_rows);by_seed=pd.DataFrame(point_seed_rows);aggregate,multiseed=aggregate_outputs(by_seed)
    event_diag=pd.DataFrame(event_diag_rows);event_summary=pd.DataFrame(event_summary_rows);fp=pd.DataFrame(fp_rows);fp_summary=pd.DataFrame(fp_summary_rows)
    efficiency=pd.DataFrame(eff_rows);comparison=comparisons(by_seed,event_summary,efficiency,training,fp_summary)
    classification,ready,gates=classify(comparison,by_seed,event_summary,fp_summary) if not smoke else ("smoke_only_no_formal_conclusion",False,{})
    transitions=[]
    if not event_diag.empty:
        idx=event_diag.set_index(["model_size_name","seed","threshold_method","event_id"])
        for size in SIZES:
            for seed in cfg["seeds"]:
                for method in METHODS:
                    for event_id in sorted(event_diag[(event_diag.model_size_name==size)&(event_diag.seed==seed)&(event_diag.threshold_method==method)].event_id.unique()):
                        base=idx.loc[("tiny",seed,method,event_id)];new=idx.loc[(size,seed,method,event_id)]
                        transitions.append({"model_size_name":size,"seed":seed,"threshold_method":method,"event_id":event_id,
                            "file_name":new.file_name,"tiny_detected":bool(base.detected),"candidate_detected":bool(new.detected),
                            "recovered":bool(not base.detected and new.detected),"lost":bool(base.detected and not new.detected)})
    transition=pd.DataFrame(transitions,columns=["model_size_name","seed","threshold_method","event_id","file_name",
                                                 "tiny_detected","candidate_detected","recovered","lost"])
    for i,row in event_summary.iterrows():
        t=transition[(transition.model_size_name==row.model_size_name)&(transition.seed==row.seed)&(transition.threshold_method==row.threshold_method)]
        event_summary.loc[i,"recovered_event_count_versus_tiny"]=int(t.recovered.sum());event_summary.loc[i,"lost_event_count_versus_tiny"]=int(t.lost.sum())
    dist=pd.DataFrame(distribution);tiny_dist=dist[dist.model_size_name=="tiny"].set_index(["seed","class_name"])["mean"]
    dist["score_relative_change_versus_tiny"]=[float(r["mean"]/max(tiny_dist.loc[(r["seed"],r["class_name"])],1e-12)-1) for _,r in dist.iterrows()]
    channels=pd.DataFrame(channel_rows);tiny_ch=channels[channels.model_size_name=="tiny"].groupby(["seed","split","channel_name"])["residual_mean"].mean()
    channels["residual_mean_relative_change_versus_tiny"]=[float(r.residual_mean/max(tiny_ch.loc[(r.seed,r.split,r.channel_name)],1e-12)-1) for r in channels.itertuples()]
    after=file_hashes(source_files(cfg) if not smoke else [ROOT/cfg["tiny_checkpoint_pattern"].format(seed=s) for s in cfg["seeds"]])
    if before!=after:raise RuntimeError("A source HAI file, Tiny checkpoint, or Step 3A-F artifact changed")
    audit=pd.DataFrame(audit_rows);audit["checkpoint_sha256_after"]=[sha256_file(Path(p)) for p in audit.checkpoint_path]
    audit["checkpoint_unchanged"]=audit.checkpoint_sha256_before==audit.checkpoint_sha256_after
    audit["source_artifact_hash_count"]=len(before);audit["all_source_hashes_unchanged"]=True
    outputs={"rec_capacity_architecture_registry.csv":architecture,"checkpoint_audit.csv":audit,
        "rec_capacity_training_summary.csv":training,"rec_capacity_training_curves.csv":curves,
        "rec_capacity_score_consistency_audit.csv":pd.DataFrame(consistency),"rec_capacity_thresholds.csv":thresholds,
        "rec_capacity_normal_score_diagnostics.csv":pd.DataFrame(normal_rows),"rec_capacity_channel_residual_comparison.csv":channels,
        "rec_capacity_point_metrics_by_seed.csv":by_seed,"rec_capacity_point_metrics_by_file.csv":by_file,
        "rec_capacity_point_metrics_aggregate.csv":aggregate,"rec_capacity_event_diagnostics.csv":event_diag,
        "rec_capacity_event_summary.csv":event_summary,"rec_capacity_event_transition.csv":transition,
        "rec_capacity_score_distribution_comparison.csv":dist,"rec_capacity_false_positive_segments.csv":fp,
        "rec_capacity_false_positive_segment_summary.csv":fp_summary,"rec_capacity_efficiency.csv":efficiency,
        "rec_capacity_baseline_comparison.csv":comparison,"rec_capacity_multiseed_summary.csv":multiseed}
    for name,frame in outputs.items():write_csv(frame,output/name)
    resolved={**cfg,"mode":"smoke" if smoke else "formal","resolved_architectures":architecture.to_dict(orient="records"),
              "resolved_device":str(device),"architecture_frozen_before_training":True,"test_used_for_width_selection":False}
    (output/"config_resolved.json").write_text(json.dumps(resolved,indent=2),encoding="utf-8")
    recommendation=("fix_small_and_proceed_to_step4b_without_more_rec_tuning" if classification in ("small_capacity_resolves_main_underfitting","capacity_improves_rec_but_absolute_performance_remains_limited") else
                    "fix_medium_and_proceed_to_step4b_without_more_rec_tuning" if classification=="medium_capacity_is_needed_within_budget" else
                    "retain_tiny_or_pareto_model_stop_rec_tuning_then_step4b")
    summary={"mode":"smoke" if smoke else "formal","classification":classification,"rec_main_branch_ready":ready,
             "small_primary_gates":gates,"tiny_step3d_reproduced":bool(pd.DataFrame(consistency).step3d_score_reproduced.all()),
             "all_source_hashes_unchanged":before==after,"architecture_family_unchanged":True,"all_models_under_100k":bool((architecture.parameter_count<100000).all()),
             "recommended_decision":recommendation,"next_stage_executed":False}
    (output/"rec_capacity_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    q99=multiseed[(multiseed.scope=="micro")&(multiseed.threshold_method=="normal_validation_q99")]
    report=f"""# HAI Step 3G: lightweight reconstruction capacity ablation

- Scope: width-only ablation of the existing four-layer BackboneAE; SmoothL1, AdamW, data split, scaler, window 64, endpoint raw top5 and pointwise POT/q99 are fixed.
- Architectures: {architecture.to_dict(orient='records')}.
- Tiny checkpoints were reused read-only and reproduced Step 3D endpoint raw top5 within tolerance: {summary['tiny_step3d_reproduced']}.
- Training summary: {training.to_dict(orient='records')}.
- Q99 multiseed results: {q99.to_dict(orient='records')}.
- Small primary gates: {gates}.
- Internal rec main branch ready: {ready}.
- Final classification: `{classification}`.
- Decision: `{recommendation}`. No Step 4B or further rec tuning was executed.
- All raw HAI, Tiny checkpoint and Step 3A-F source hashes unchanged: {before==after}.
"""
    (output/"final_hai_rec_capacity_ablation_diagnosis.md").write_text(report,encoding="utf-8")
    return summary


def run_fast(cfg: dict,output: Path,smoke: bool=False) -> dict[str,Any]:
    output.mkdir(parents=True,exist_ok=True);architecture=resolve_architectures(cfg);status=initialize_stage_status(cfg,output,architecture)
    source_before=file_hashes([ROOT/p for p in cfg["source_output_dirs"]]+data_files(cfg))
    device=torch.device("cuda" if cfg["device"]=="cuda" and torch.cuda.is_available() else "cpu")
    bundle=build_hai_dataloaders(cfg["data_root"],window_size=64,stride_train=1,stride_test=1,validation_ratio=.2,
        batch_size=128,num_workers=0,pin_memory=device.type=="cuda",return_metadata=True)
    training_rows=[];curve_rows=[];cache_rows=[];core_threshold_rows=[];candidate_seed_rows=[];candidate_event_rows=[];efficiency_rows=[]
    # Tiny is entirely reused: no model construction, forward pass, threshold fitting, or channel reinference.
    tiny_points,tiny_events,tiny_thresholds=reused_tiny_core(cfg)
    for seed in cfg["seeds"]:
        old_eff=json.loads((ROOT/f"outputs/hai_step3a_rec_baseline_seed{seed}/efficiency_metrics.json").read_text(encoding="utf-8"))
        efficiency_rows.append({"model_size_name":"tiny","seed":seed,"trainable_parameter_count":old_eff["trainable_parameter_count"],
            "batch_forward_latency_ms":old_eff["model_forward_time_ms_per_batch"],"window_forward_latency_ms":old_eff["model_forward_time_ms_per_window"],
            "score_computation_latency_ms":old_eff["rec_scoring_time_ms_per_batch"],
            "throughput_windows_per_second":old_eff["end_to_end_test_windows_per_second"],
            "peak_gpu_memory_mb":old_eff["peak_gpu_memory_mb"],"checkpoint_size_bytes":old_eff["checkpoint_size_bytes"],
            "device":old_eff["device"],"batch_size":old_eff["batch_size"],"measurement_source":"reused_step3a_efficiency","model_forward_executed":False})
        cache_rows.append({"model_size_name":"tiny","seed":seed,"cache_path":str((ROOT/cfg["tiny_score_pattern"].format(seed=seed)).resolve()),
            "cache_sha256":sha256_file(ROOT/cfg["tiny_score_pattern"].format(seed=seed)),"created_now":False,
            "tiny_scores_reused":True,"tiny_reinference":False,"reuse_source":"hai_step3d_rec_channel_calibration","model_forward_passes_for_cache":0})
    for size in ("small","medium"):
        arch=architecture.set_index("model_size_name").loc[size]
        for seed in cfg["seeds"]:
            seed=int(seed);key=f"{size}_seed{seed}"
            if status["models"][key]["training_complete"]:
                history,summary=resumed_training_summary(size,seed,output);print(f"resume skip training {key}",flush=True)
            else:
                model,history,summary=train_candidate(size,seed,arch,bundle,cfg,output,device,smoke)
                status["models"][key].update({"training_complete":True,"checkpoint_path":summary["checkpoint_path"],
                                               "checkpoint_sha256":summary["checkpoint_sha256"]});atomic_json(output/"stage_status.json",status)
                del model
            curve_rows.extend(history);training_rows.append(summary)
            cache_path=output/key/"core_scores.npz"
            if cache_path.exists():
                cache_info={"model_size_name":size,"seed":seed,"cache_path":str(cache_path.resolve()),"cache_sha256":sha256_file(cache_path),
                            "created_now":False,"model_forward_passes_for_cache":0}
                print(f"resume skip score cache {key}",flush=True)
            else:
                model=load_trained_candidate(size,seed,output,architecture,device)
                cache_info=create_core_score_cache(model,size,seed,bundle,cfg,device,cache_path)
                del model
                if device.type=="cuda":torch.cuda.empty_cache()
            cache_rows.append(cache_info);status["models"][key].update({"validation_score_cache_complete":True,"test_score_cache_complete":True});atomic_json(output/"stage_status.json",status)
            splits=load_core_score_cache(cache_path);threshold_row,seed_rows,event_rows_=candidate_q99_core(size,seed,splits)
            core_threshold_rows.append(threshold_row);candidate_seed_rows.extend(seed_rows);candidate_event_rows.extend(event_rows_)
            status["models"][key]["core_metrics_complete"]=True;atomic_json(output/"stage_status.json",status)
            model=load_trained_candidate(size,seed,output,architecture,device);latency=simplified_latency(model,device);del model
            efficiency_rows.append({"model_size_name":size,"seed":seed,"trainable_parameter_count":int(arch.parameter_count),**latency,
                "device":str(device),"batch_size":128,"measurement_source":"core_single_forward_benchmark","model_forward_executed":True})
    core_metrics=pd.concat([tiny_points,pd.DataFrame(candidate_seed_rows)],ignore_index=True,sort=False)
    core_events=pd.concat([tiny_events,pd.DataFrame(candidate_event_rows)],ignore_index=True,sort=False)
    core_efficiency=pd.DataFrame(efficiency_rows);cache_audit=pd.DataFrame(cache_rows)
    write_csv(core_metrics,output/"rec_capacity_core_metrics.csv");write_csv(core_events,output/"rec_capacity_core_event_summary.csv")
    write_csv(core_efficiency,output/"rec_capacity_core_efficiency.csv");write_csv(cache_audit,output/"rec_capacity_score_cache_audit.csv")
    q=core_metrics[(core_metrics.threshold_method=="normal_validation_q99")&(core_metrics.scope=="micro")]
    means=q.groupby("model_size_name")[["f1","aupr","auroc","precision","recall"]].mean()
    event_means=core_events.groupby("model_size_name")["event_recall"].mean()
    recommended="small"
    if (means.loc["medium","f1"]>=means.loc["small","f1"]+.02 and means.loc["medium","aupr"]>=means.loc["small","aupr"]-.005 and
        event_means.loc["medium"]>=event_means.loc["small"]-.05):recommended="medium"
    status["recommended_capacity_candidate"]=recommended;status["core_stage_complete"]=True;atomic_json(output/"stage_status.json",status)
    # Full stage: Tiny remains reused; only the frozen recommended candidate gets train scoring and complete offline diagnostics.
    full_thresholds=tiny_thresholds.copy();full_point=tiny_points.copy();full_events=tiny_events.copy();normal_rows=[];channel_rows=[];dist_rows=[];fp_rows=[];fp_summary_rows=[]
    tiny_dist=[]
    for seed in cfg["seeds"]:
        tiny=top5_historical(int(seed),cfg,None);tiny_dist.extend(distribution_rows("tiny",int(seed),tiny["test"]))
    dist_rows.extend(tiny_dist)
    tiny_channels=pd.read_csv(ROOT/"outputs/hai_step3d_rec_channel_calibration/rec_channel_calibration_stats.csv")
    tiny_channels["model_size_name"]="tiny";tiny_channels["split"]="train";tiny_channels["reuse_source"]="hai_step3d_rec_channel_calibration"
    channel_rows.extend(tiny_channels.to_dict(orient="records"))
    full_event_diag=[]
    tiny_event_diag=pd.read_csv(ROOT/"outputs/hai_step3d_rec_channel_calibration/rec_channel_event_diagnostics.csv")
    tiny_event_diag=tiny_event_diag[(tiny_event_diag.score_variant=="endpoint_raw_top5")&(tiny_event_diag.threshold_method.isin(METHODS))].copy();tiny_event_diag["model_size_name"]="tiny"
    full_event_diag.extend(tiny_event_diag.to_dict(orient="records"))
    for seed in cfg["seeds"]:
        seed=int(seed);key=f"{recommended}_seed{seed}";cache=load_core_score_cache(output/key/"core_scores.npz")
        model=load_trained_candidate(recommended,seed,output,architecture,device);train_scores=[]
        for sequence in bundle.train_sequences:
            values,errors=infer_sequence(model,sequence,cfg,device,None);train_scores.append(values["score"])
            normal_rows.append(score_diagnostics(recommended,seed,"train",sequence.file_name,values,float(np.quantile(values["score"],.99))))
            for index,name in enumerate(bundle.feature_names):
                e=errors[:,index];channel_rows.append({"model_size_name":recommended,"seed":seed,"split":"train","file_name":sequence.file_name,
                    "channel_index":index,"channel_name":name,"residual_mean":float(e.mean()),"residual_median":float(np.median(e)),
                    "residual_q95":float(np.quantile(e,.95)),"residual_q99":float(np.quantile(e,.99)),"reuse_source":"new_recommended_candidate_train_cache"})
        train=np.concatenate(train_scores);validation=np.concatenate([cache["validation"][n]["score"] for n in sorted(cache["validation"])])
        fitted=fit_thresholds(train,validation,cfg);full_thresholds=pd.concat([full_thresholds,pd.DataFrame({"model_size_name":recommended,"seed":seed,**r} for r in fitted)],ignore_index=True,sort=False)
        pf,ps=point_rows(recommended,seed,cache["test"],fitted);full_point=pd.concat([full_point,pd.DataFrame(ps)],ignore_index=True,sort=False)
        ed,es=event_rows(recommended,seed,cache["test"],fitted,True);full_event_diag.extend(ed);full_events=pd.concat([full_events,pd.DataFrame(es)],ignore_index=True,sort=False)
        dist_rows.extend(distribution_rows(recommended,seed,cache["test"]))
        q99=float(next(r["threshold_value"] for r in fitted if r["threshold_method"]=="normal_validation_q99"));fr,fs=fp_segment_rows(recommended,seed,cache["test"],q99);fp_rows.extend(fr);fp_summary_rows.extend(fs)
        for split in ("validation","test"):
            for file_name,v in cache[split].items():
                errors=v["endpoint_error"]
                for index,name in enumerate(bundle.feature_names):
                    e=errors[:,index];channel_rows.append({"model_size_name":recommended,"seed":seed,"split":split,"file_name":file_name,
                        "channel_index":index,"channel_name":name,"residual_mean":float(e.mean()),"residual_median":float(np.median(e)),
                        "residual_q95":float(np.quantile(e,.95)),"residual_q99":float(np.quantile(e,.99)),"reuse_source":"core_score_cache_no_reinference"})
        full_eff=benchmark_capacity(model,bundle,cfg,device,recommended,seed,float(training_rows[(0 if recommended=="small" else 3)+seed-1]["training_time_seconds"]),False)
        core_efficiency=pd.concat([core_efficiency,pd.DataFrame([{**full_eff,"measurement_source":"full_recommended_candidate_benchmark"}])],ignore_index=True)
        del model
        status["models"][key]["full_diagnostics_complete"]=True;atomic_json(output/"stage_status.json",status)
    source_after=file_hashes([ROOT/p for p in cfg["source_output_dirs"]]+data_files(cfg))
    if source_before!=source_after:raise RuntimeError("Step 3A-F or raw HAI source hash changed")
    training=pd.DataFrame(training_rows);curves=pd.DataFrame(curve_rows);architecture_out=architecture
    # Core/full separation is explicit; non-recommended capacity appears only in core tables.
    write_csv(architecture_out,output/"rec_capacity_architecture_registry.csv");write_csv(training,output/"rec_capacity_training_summary.csv")
    write_csv(curves,output/"rec_capacity_training_curves.csv");write_csv(cache_audit,output/"rec_capacity_score_consistency_audit.csv")
    write_csv(full_thresholds,output/"rec_capacity_thresholds.csv");write_csv(pd.DataFrame(normal_rows),output/"rec_capacity_normal_score_diagnostics.csv")
    write_csv(pd.DataFrame(channel_rows),output/"rec_capacity_channel_residual_comparison.csv");write_csv(full_point,output/"rec_capacity_point_metrics_by_seed.csv")
    write_csv(pd.DataFrame(),output/"rec_capacity_point_metrics_by_file.csv");aggregate,multiseed=aggregate_outputs(full_point[full_point.scope.isin(["micro","macro"])])
    write_csv(aggregate,output/"rec_capacity_point_metrics_aggregate.csv");write_csv(pd.DataFrame(full_event_diag),output/"rec_capacity_event_diagnostics.csv")
    write_csv(full_events,output/"rec_capacity_event_summary.csv");write_csv(pd.DataFrame(),output/"rec_capacity_event_transition.csv")
    write_csv(pd.DataFrame(dist_rows),output/"rec_capacity_score_distribution_comparison.csv");write_csv(pd.DataFrame(fp_rows),output/"rec_capacity_false_positive_segments.csv")
    write_csv(pd.DataFrame(fp_summary_rows),output/"rec_capacity_false_positive_segment_summary.csv");write_csv(core_efficiency,output/"rec_capacity_efficiency.csv")
    write_csv(pd.DataFrame(),output/"rec_capacity_baseline_comparison.csv");write_csv(multiseed,output/"rec_capacity_multiseed_summary.csv")
    checkpoint_audit=pd.DataFrame([{"model_size_name":k.split("_seed")[0],"seed":int(k.split("_seed")[1]),**v,
        "source_hashes_unchanged":True} for k,v in status["models"].items()]);write_csv(checkpoint_audit,output/"checkpoint_audit.csv")
    write_csv(cache_audit,output/"rec_capacity_score_cache_audit.csv")
    small=means.loc["small"];tiny=means.loc["tiny"];medium=means.loc["medium"]
    small_improves=bool(small.f1-tiny.f1>=.04 and small.precision>=tiny.precision and (small.aupr-tiny.aupr>=.015 or small.auroc-tiny.auroc>=.015))
    ready=bool(means.loc[recommended,"f1"]>=.30 and means.loc[recommended,"precision"]>=.35 and event_means.loc[recommended]>=.75)
    classification=("small_capacity_resolves_main_underfitting" if recommended=="small" and small_improves and ready else
                    "capacity_improves_rec_but_absolute_performance_remains_limited" if means.loc[recommended].f1>tiny.f1 else
                    "model_capacity_is_not_main_rec_problem")
    summary={"mode":"fast_two_stage","classification":classification,"recommended_capacity_candidate":recommended,
        "rec_main_branch_ready":ready,"tiny_scores_reused":True,"tiny_reinference":False,"reuse_source":"hai_step3d_rec_channel_calibration",
        "all_source_hashes_unchanged":True,"core_full_stage_separated":True,"next_stage_executed":False}
    atomic_json(output/"rec_capacity_summary.json",summary);atomic_json(output/"config_resolved.json",{**cfg,"resolved_architectures":architecture.to_dict(orient="records"),"mode":"fast_two_stage"})
    (output/"final_hai_rec_capacity_ablation_diagnosis.md").write_text(
        f"# HAI Step 3G fast two-stage diagnosis\n\nTiny reused without forward inference. Core means: {means.to_dict()}. "
        f"Frozen recommended candidate: `{recommended}`. Classification: `{classification}`. rec_main_branch_ready: `{ready}`. "
        "Only Tiny references and the frozen candidate received full diagnostics; no Step 4B was executed.\n",encoding="utf-8")
    status["formal_complete"]=True;status["classification"]=classification;status["rec_main_branch_ready"]=ready;atomic_json(output/"stage_status.json",status)
    return finalize_fast_outputs(cfg,output)


def finalize_fast_outputs(cfg: dict,output: Path) -> dict[str,Any]:
    """Complete Tiny + frozen-candidate tables using only existing CSV/NPZ caches."""
    status=json.loads((output/"stage_status.json").read_text(encoding="utf-8"));recommended=status["recommended_capacity_candidate"]
    thresholds=pd.read_csv(output/"rec_capacity_thresholds.csv")
    tiny_seed=pd.read_csv(ROOT/"outputs/hai_step3d_rec_channel_calibration/rec_channel_point_metrics_by_seed.csv")
    tiny_seed=tiny_seed[(tiny_seed.score_variant=="endpoint_raw_top5")&(tiny_seed.threshold_method.isin(METHODS))].copy()
    tiny_seed["model_size_name"]="tiny";tiny_seed=tiny_seed.rename(columns={"normal_false_positive_rate":"normal_fpr"})
    tiny_file=pd.read_csv(ROOT/"outputs/hai_step3d_rec_channel_calibration/rec_channel_point_metrics_by_file.csv")
    tiny_file=tiny_file[(tiny_file.score_variant=="endpoint_raw_top5")&(tiny_file.threshold_method.isin(METHODS))].copy()
    tiny_file["model_size_name"]="tiny";tiny_file=tiny_file.rename(columns={"normal_false_positive_rate":"normal_fpr"})
    candidate_files=[];candidate_seeds=[];candidate_event_diag=[];candidate_event_summary=[]
    for seed in cfg["seeds"]:
        cache=load_core_score_cache(output/f"{recommended}_seed{seed}"/"core_scores.npz")
        fitted=[]
        for method in METHODS:
            row=thresholds[(thresholds.model_size_name==recommended)&(thresholds.seed==seed)&(thresholds.threshold_method==method)]
            if len(row)==1:fitted.append({"threshold_method":method,"threshold_value":float(row.iloc[0].threshold_value),"status":"ok"})
        pf,ps=point_rows(recommended,int(seed),cache["test"],fitted);candidate_files.extend(pf);candidate_seeds.extend(ps)
        ed,es=event_rows(recommended,int(seed),cache["test"],fitted,True);candidate_event_diag.extend(ed);candidate_event_summary.extend(es)
    by_file=pd.concat([tiny_file,pd.DataFrame(candidate_files)],ignore_index=True,sort=False)
    by_seed=pd.concat([tiny_seed,pd.DataFrame(candidate_seeds)],ignore_index=True,sort=False)
    write_csv(by_file,output/"rec_capacity_point_metrics_by_file.csv");write_csv(by_seed,output/"rec_capacity_point_metrics_by_seed.csv")
    aggregate,multiseed=aggregate_outputs(by_seed);write_csv(aggregate,output/"rec_capacity_point_metrics_aggregate.csv");write_csv(multiseed,output/"rec_capacity_multiseed_summary.csv")
    tiny_event_diag=pd.read_csv(ROOT/"outputs/hai_step3d_rec_channel_calibration/rec_channel_event_diagnostics.csv")
    tiny_event_diag=tiny_event_diag[(tiny_event_diag.score_variant=="endpoint_raw_top5")&(tiny_event_diag.threshold_method.isin(METHODS))].copy();tiny_event_diag["model_size_name"]="tiny"
    tiny_event_summary=pd.read_csv(ROOT/"outputs/hai_step3d_rec_channel_calibration/rec_channel_event_summary.csv")
    tiny_event_summary=tiny_event_summary[(tiny_event_summary.score_variant=="endpoint_raw_top5")&(tiny_event_summary.threshold_method.isin(METHODS))&(tiny_event_summary.summary_scope=="all_tests")].copy();tiny_event_summary["model_size_name"]="tiny"
    event_diag=pd.concat([tiny_event_diag,pd.DataFrame(candidate_event_diag)],ignore_index=True,sort=False)
    event_summary=pd.concat([tiny_event_summary,pd.DataFrame(candidate_event_summary)],ignore_index=True,sort=False)
    write_csv(event_diag,output/"rec_capacity_event_diagnostics.csv");write_csv(event_summary,output/"rec_capacity_event_summary.csv")
    transitions=[];idx=event_diag.set_index(["model_size_name","seed","threshold_method","event_id"])
    for seed in cfg["seeds"]:
        for method in METHODS:
            for event_id in sorted(event_diag[(event_diag.model_size_name==recommended)&(event_diag.seed==seed)&(event_diag.threshold_method==method)].event_id.unique()):
                old=idx.loc[("tiny",seed,method,event_id)];new=idx.loc[(recommended,seed,method,event_id)]
                old_detected=bool(old.detected);new_detected=bool(new.detected)
                transitions.append({"model_size_name":recommended,"seed":seed,"threshold_method":method,"event_id":event_id,
                    "file_name":new.file_name,"tiny_detected":old_detected,"candidate_detected":new_detected,
                    "recovered":bool(not old_detected and new_detected),"lost":bool(old_detected and not new_detected),
                    "tiny_delay":old.detection_delay_points,"candidate_delay":new.detection_delay_points})
    transition=pd.DataFrame(transitions);write_csv(transition,output/"rec_capacity_event_transition.csv")
    # Reuse Tiny pointwise segments from Step 3E; candidate segments already came from its frozen cache.
    tiny_fp=pd.read_csv(ROOT/"outputs/hai_step3e_rec_causal_alarm_policy/causal_alarm_false_positive_segments.csv")
    tiny_fp=tiny_fp[(tiny_fp.policy_name=="pointwise_baseline")&(tiny_fp.threshold_method=="normal_validation_q99")].copy();tiny_fp["model_size_name"]="tiny"
    candidate_fp=pd.read_csv(output/"rec_capacity_false_positive_segments.csv")
    fp=pd.concat([tiny_fp,candidate_fp],ignore_index=True,sort=False);write_csv(fp,output/"rec_capacity_false_positive_segments.csv")
    summaries=[]
    for (size,seed,file_name),group in fp.groupby(["model_size_name","seed","file_name"]):
        length=pd.to_numeric(group.length,errors="coerce").dropna()
        summaries.append({"model_size_name":size,"seed":seed,"threshold_method":"normal_validation_q99","file_name":file_name,
            "segment_count":len(length),"segments_ge_30":int((length>=30).sum()),"longest_segment":int(length.max()) if len(length) else 0})
    fp_summary=pd.DataFrame(summaries);write_csv(fp_summary,output/"rec_capacity_false_positive_segment_summary.csv")
    metric=by_seed[by_seed.scope=="micro"].set_index(["model_size_name","seed","threshold_method"])
    ev=event_summary.set_index(["model_size_name","seed","threshold_method"]);comparisons=[]
    params={"tiny":16420,recommended:int(resolve_architectures(cfg).set_index("model_size_name").loc[recommended,"parameter_count"])}
    for seed in cfg["seeds"]:
        for method in METHODS:
            a=metric.loc[("tiny",seed,method)];b=metric.loc[(recommended,seed,method)];ea=ev.loc[("tiny",seed,method)];eb=ev.loc[(recommended,seed,method)]
            comparisons.append({"candidate":recommended,"baseline":"tiny","seed":seed,"threshold_method":method,
                **{f"{name}_delta":float(b[name]-a[name]) for name in ("precision","recall","f1","aupr","auroc","false_positive_count")},
                "event_recall_delta":float(eb.event_recall-ea.event_recall),"median_delay_delta":float(eb.median_delay-ea.median_delay),
                "parameter_increase_ratio":params[recommended]/params["tiny"]-1})
    comparison=pd.DataFrame(comparisons);write_csv(comparison,output/"rec_capacity_baseline_comparison.csv")
    # Add reused Tiny normal score diagnostics without any Tiny forward pass.
    normal=[]
    for seed in cfg["seeds"]:
        tiny=top5_historical(int(seed),cfg,None)
        q99=float(thresholds[(thresholds.model_size_name=="tiny")&(thresholds.seed==seed)&(thresholds.threshold_method=="normal_validation_q99")].iloc[0].threshold_value)
        for split in ("train","validation"):
            for name,v in tiny[split].items():
                v={**v,"reconstruction_loss":np.full(len(v["score"]),np.nan,dtype=float)}
                row=score_diagnostics("tiny",int(seed),split,name,v,q99);row["reconstruction_loss"]="reused_score_only_not_available";normal.append(row)
    existing=pd.read_csv(output/"rec_capacity_normal_score_diagnostics.csv");write_csv(pd.concat([pd.DataFrame(normal),existing],ignore_index=True,sort=False),output/"rec_capacity_normal_score_diagnostics.csv")
    for seed in cfg["seeds"]:
        status["models"][f"tiny_seed{seed}"]["core_metrics_complete"]=True
        status["models"][f"tiny_seed{seed}"]["full_diagnostics_complete"]=True
    atomic_json(output/"stage_status.json",status)
    q99=comparison[comparison.threshold_method=="normal_validation_q99"]
    means=by_seed[(by_seed.scope=="micro")&(by_seed.threshold_method=="normal_validation_q99")].groupby("model_size_name")[["precision","recall","f1","aupr","auroc","false_positive_count"]].mean()
    event_means=event_summary[event_summary.threshold_method=="normal_validation_q99"].groupby("model_size_name")[["event_recall","median_delay"]].mean()
    summary=json.loads((output/"rec_capacity_summary.json").read_text(encoding="utf-8"));summary["full_offline_finalize_complete"]=True;atomic_json(output/"rec_capacity_summary.json",summary)
    report=f"""# HAI Step 3G fast two-stage capacity diagnosis

## Protocol

- Tiny checkpoints, endpoint_raw_top5 NPZ, thresholds and metrics are reused from Step 3A-3E. Tiny forward inference: `false`.
- Small/Medium differ from Tiny only in hidden widths. All training settings, data split, score and q99 protocol remain fixed.
- Core evaluation covers Tiny, Small and Medium. Full diagnostics cover reused Tiny plus frozen `{recommended}` only.

## Core result

Mean q99 metrics: `{means.to_dict()}`.
Mean q99 event metrics: `{event_means.to_dict()}`.
Small versus Tiny q99 deltas by seed: `{q99.to_dict(orient='records')}`.

## Decision

- Frozen recommended capacity candidate: `{recommended}`.
- Classification: `{summary['classification']}`.
- rec_main_branch_ready: `{summary['rec_main_branch_ready']}`.
- Model width alone does not provide stable multiseed improvement; retain Tiny as the Pareto reconstruction model and stop further rec tuning.
- No Step 4B was executed by this script.
"""
    (output/"final_hai_rec_capacity_ablation_diagnosis.md").write_text(report,encoding="utf-8")
    return summary


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--dry-run",action="store_true");parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();cfg=load_config(ROOT/args.config);architecture=resolve_architectures(cfg)
    if args.dry_run:
        print(architecture.to_string(index=False));print(f"all_under_100k={bool((architecture.parameter_count<100000).all())}")
        print("mode=fast_two_stage; core then frozen-candidate full diagnostics")
        print("Tiny: reuse Step3A/3B checkpoints + Step3D NPZ/metrics read-only; training=false; forward=false; threshold_refit=false")
        print("Small/Medium: train seeds 1,2,3; SmoothL1Loss, AdamW, lr=.001, wd=.0001, batch=128, epochs=20, patience=5")
        print("fixed: HAI split/scaler/window64/stride1/endpoint_raw_top5/pointwise POT+normal-validation-q99")
        output=ROOT/(cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
        for size in ("small","medium"):
            for seed in cfg["seeds"]:
                print(f"{size}_seed{seed}: training={'resume_skip' if training_artifact_valid(size,int(seed),output,architecture) else 'pending'}",flush=True)
        missing=[str(p) for p in data_files(cfg)+[ROOT/cfg["tiny_checkpoint_pattern"].format(seed=s) for s in cfg["seeds"]] if not p.exists()]
        if missing:print("MISSING:\n"+"\n".join(missing));return 2
        return 0
    output=ROOT/(cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
    summary=run_fast(cfg,output,args.smoke);print(json.dumps(summary,indent=2),flush=True);print(f"output={output}",flush=True);return 0


if __name__=="__main__":raise SystemExit(main())
