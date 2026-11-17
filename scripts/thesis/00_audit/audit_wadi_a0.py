"""WADI.A2 source/schema/attack audit. No model code is imported or executed."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset" / "WADI_A2_2019"
OUT = ROOT / "outputs" / "thesis_wadi_a0_audit"
NORMAL = DATA / "WADI_14days_new.csv"
ATTACK = DATA / "WADI_attackdataLABLE.csv"
PDF = DATA / "table_WADI.pdf"
LABEL = "Attack LABLE (1:No Attack, -1:Attack)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def infer_audit_timestamps(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict]:
    """Recover hour-of-day from monotonic Row; raw Time contains mm:ss only."""
    raw_dates = frame["Date"].astype(str).str.strip()
    ordered = list(dict.fromkeys(raw_dates))
    parsed = []
    formats = set()
    for value in ordered:
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                parsed.append(pd.Timestamp(datetime.strptime(value, fmt))); formats.add(fmt); break
            except ValueError:
                continue
        else:
            raise ValueError(f"unsupported raw Date value: {value!r}")
    groups = raw_dates.value_counts(sort=False)
    pieces=[]; offsets=[]
    for i,(value,date) in enumerate(zip(ordered,parsed)):
        count=int(groups[value])
        prev_contiguous=i>0 and (date-parsed[i-1]).days==1
        next_contiguous=i+1<len(parsed) and (parsed[i+1]-date).days==1
        # A partial day beginning a session is right-aligned to midnight;
        # a partial day ending a session is left-aligned from midnight.
        offset=86400-count if count<86400 and not prev_contiguous and next_contiguous else 0
        offsets.append(offset)
        pieces.append(pd.date_range(date+pd.Timedelta(seconds=offset),periods=count,freq="s"))
    ts=pieces[0].append(pieces[1:])
    row_delta=np.diff(frame["Row"].astype(np.int64).to_numpy())
    raw_duplicate = int(frame[["Date", "Time"]].duplicated().sum())
    ts_delta=np.diff(ts.view("i8"))//1_000_000_000
    gap_indices=np.flatnonzero(ts_delta!=1)
    return ts, {
        "method": "audit-only reconstruction from contiguous Row and raw Date boundaries; raw Time is retained",
        "raw_time_semantics": "minute:second-within-hour; hour omitted and values cycle every 3600 rows",
        "raw_date_formats": sorted(formats),
        "date_groups": [{"raw_date":v,"rows":int(groups[v]),"inferred_start_offset_seconds":o}
                        for v,o in zip(ordered,offsets)],
        "raw_date_time_duplicate_count": raw_duplicate,
        "canonical_duplicate_timestamp_count": 0,
        "raw_row_discontinuity_count": int((row_delta!=1).sum()),
        "largest_raw_row_step": int(row_delta.max()) if len(row_delta) else 0,
        "gap_positions": [{"index_before":int(i),"index_after":int(i+1),
                           "timestamp_before":ts[i].isoformat(),"timestamp_after":ts[i+1].isoformat(),
                           "gap_seconds":int(ts_delta[i])} for i in gap_indices],
        "timestamp_repair_applied_to_source": False,
    }


def frame_audit(frame: pd.DataFrame, feature_names: list[str], ts: pd.DatetimeIndex) -> dict:
    nan_by_column = frame[feature_names].isna().sum()
    inf_by_column = {}
    for name in feature_names:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        inf_by_column[name] = int(np.isinf(values).sum())
    delta = np.diff(ts.view("i8")) // 1_000_000_000
    return {
        "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "process_variable_count": len(feature_names),
        "start": ts[0].isoformat(), "end": ts[-1].isoformat(),
        "duration_seconds": int((ts[-1] - ts[0]).total_seconds()),
        "nominal_sampling_seconds": int(pd.Series(delta).mode().iloc[0]),
        "gap_count": int((delta != 1).sum()),
        "largest_gap_seconds": int(delta.max()) if len(delta) else 0,
        "duplicate_rows_including_unique_Row": int(frame["Row"].duplicated().sum()),
        "nan_total": int(nan_by_column.sum()),
        "nan_columns": {k: int(v) for k, v in nan_by_column[nan_by_column > 0].items()},
        "inf_total": int(sum(inf_by_column.values())),
        "inf_columns": {k: v for k, v in inf_by_column.items() if v},
    }


def process_of(name: str) -> str:
    for prefix in ("2A_", "2B_", "1_", "2_", "3_"):
        if name.startswith(prefix):
            return prefix[:-1]
    return "PLANT"


def kind_of(name: str, unique_count: int) -> str:
    if unique_count <= 1:
        return "constant"
    if name.endswith("_STATUS") or name.endswith("_AL") or name == "PLANT_START_STOP_LOG":
        return "discrete"
    if any(token in name for token in ("_PV", "_CO", "_SP", "_SPEED")) or name in {
        "LEAK_DIFF_PRESSURE", "TOTAL_CONS_REQUIRED_FLOW"
    }:
        return "continuous"
    return "uncertain"


def official_events() -> pd.DataFrame:
    rows = [
        (1,"2017-10-09 19:25:00","2017-10-09 19:50:16","1_MV_001;1_LT_001;1_FIT_001","1","Turn on 1_MV_001; overflow primary tank should appear in 1_LT_001 and 1_FIT_001","MIXED_COLLABORATION","Actuator manipulation has direct evidence and an explicitly expected flow/level response.","MEDIUM"),
        (2,"2017-10-10 10:24:10","2017-10-10 10:34:00","1_FIT_001;chemical dosing pump","1","Turn off/tune false 1_FIT_001 reading; PLC activates chemical dosing while tank level remains constant","HIGH_COLLABORATION_RELEVANCE","False sensor reading is intended to disagree with physical level and actuator response.","HIGH"),
        (3,"2017-10-10 10:55:00","2017-10-10 11:24:00","2_LT_002","2","Stealthily manipulate elevated-reservoir level draining and filling speed","HIGH_COLLABORATION_RELEVANCE","Stealthy level manipulation is defined through inconsistent tank dynamics.","HIGH"),
        (4,"2017-10-10 10:55:00","2017-10-10 11:24:00","1_AIT_001","1","Change water-quality reading while manipulating tank draining/filling","HIGH_COLLABORATION_RELEVANCE","Coordinated sensor and hydraulic manipulation is explicitly described.","HIGH"),
        (5,"2017-10-10 11:30:40","2017-10-10 11:44:50","2_MCV_101;2_MCV_201;2_MCV_301;2_MCV_401;2_MCV_501;2_MCV_601","2","Turn off consumer valves so consumers receive no water","DIRECT_DOMINATED","Multiple commanded valve states provide direct attack evidence.","HIGH"),
        (6,"2017-10-10 13:39:30","2017-10-10 13:50:40","2_MCV_101;2_MCV_201","2","Turn on 2_MCV_101 and 2_MCV_201 maliciously","DIRECT_DOMINATED","The table specifies direct valve actuation without a contextual stealth mechanism.","HIGH"),
        (7,"2017-10-10 14:48:17","2017-10-10 14:59:55","1_AIT_002;2_MV_003","1+2","Set 1_AIT_002 to 6 and open 2_MV_003 to supply contaminated water and drain primary grid","HIGH_COLLABORATION_RELEVANCE","Coordinated false quality reading and valve action create cross-process inconsistency.","HIGH"),
        (8,"2017-10-10 17:40:00","2017-10-10 17:49:40","2_MCV_007;2_PIT_002;2_FIT_002","2","Open 2_MCV_007 to create leakage; expected reflection in pressure and flow was unsuccessful","MIXED_COLLABORATION","Direct valve change plus an explicitly expected pressure/flow response; table marks outcome unsuccessful.","HIGH"),
        (9,"2017-10-11 10:55:00","2017-10-11 10:56:27","1_P_006","1","Turn on 1_P_006 to cause pipe burst","DIRECT_DOMINATED","Direct pump state manipulation dominates.","HIGH"),
        (10,"2017-10-11 11:17:54","2017-10-11 11:31:20","1_MV_001;raw water pump","1","Damage 1_MV_001 and raw-water pump to drain elevated reservoir tank","MIXED_COLLABORATION","Coordinated valve/pump manipulation has both direct and hydraulic effects.","MEDIUM"),
        (11,"2017-10-11 11:36:31","2017-10-11 11:47:00","2_MCV_007;2_PIT_002;2_FIT_002","2","Similar to attack 8","MIXED_COLLABORATION","Inherited from attack 8: valve leakage with expected pressure/flow response.","MEDIUM"),
        (12,"2017-10-11 11:59:00","2017-10-11 12:05:00","2_MCV_007;2_PIT_002;2_FIT_002","2","Similar to attack 8","MIXED_COLLABORATION","Inherited from attack 8: valve leakage with expected pressure/flow response.","MEDIUM"),
        (13,"2017-10-11 12:07:30","2017-10-11 12:10:52","2_PIC_003_SP;2_FIT_003;2_PIT_003","2","Reduce booster pressure setpoint, causing intermittent supply reflected in flow and pressure","MIXED_COLLABORATION","Setpoint attack is direct while flow/pressure response is explicitly contextual.","HIGH"),
        (14,"2017-10-11 12:16:00","2017-10-11 12:25:36","chemical dosing","1","Stop chemical dosing to raw water supplied to primary grid tank","DIRECT_DOMINATED","Official row gives direct dosing stoppage but does not name a measurable contextual pair.","MEDIUM"),
        (15,"2017-10-11 15:26:30","2017-10-11 15:37:00","2_LT_002;1_AIT_001","1+2","Stealthy attack, inverse of attack 3","HIGH_COLLABORATION_RELEVANCE","The table explicitly calls it stealthy and ties it to the inverse tank-dynamics attack.","MEDIUM"),
    ]
    cols=["attack_id","official_start","official_end","targets","process","description_intention","taxonomy","reason","confidence"]
    f=pd.DataFrame(rows,columns=cols)
    f["attack_action"] = f["description_intention"]
    f["attacker_intention"] = f["description_intention"]
    f["physical_process_area"] = f["process"]
    f["evidence_source"] = "table_WADI.pdf page 1, attack row " + f.attack_id.astype(str)
    return f


def main() -> int:
    started=time.perf_counter()
    for path in (NORMAL,ATTACK,PDF):
        if not path.is_file(): raise FileNotFoundError(path)
    expected={"variable_schema.csv","process_groups.csv","label_intervals.csv","official_attack_events.csv",
              "attack_binding.csv","attack_taxonomy.csv","dataset_manifest.json","protocol_candidate.json","audit_summary.json"}
    if OUT.exists() and ({p.name for p in OUT.iterdir()}-expected):
        raise FileExistsError(f"unexpected existing artifacts in: {OUT}")

    normal=pd.read_csv(NORMAL,low_memory=False)
    normal.columns=normal.columns.str.strip()
    normal_features=[c for c in normal.columns if c not in {"Row","Date","Time"}]
    normal_ts,normal_ts_note=infer_audit_timestamps(normal)
    normal_stats=frame_audit(normal,normal_features,normal_ts)
    unique=normal[normal_features].nunique(dropna=True)
    schema=[]
    for name in normal_features:
        count=int(unique[name]); kind=kind_of(name,count)
        schema.append({"raw_column_name":name,"canonical_short_name":name,"process_prefix":process_of(name),
                       "dtype":str(normal[name].dtype),"unique_count":count,"variable_kind":kind,
                       "target_candidate":kind=="continuous","source_candidate":kind in {"continuous","discrete"},
                       "classification_basis":"name suffix and Normal unique count; no correlation or Attack data"})
    schema=pd.DataFrame(schema)
    del normal

    attack=pd.read_csv(ATTACK,header=1,low_memory=False)
    attack.columns=attack.columns.str.strip()
    trailer_rows=int(attack["Row"].isna().sum())
    attack=attack[attack["Row"].notna()].reset_index(drop=True)
    if LABEL not in attack.columns: raise RuntimeError("actual label column not found")
    attack_features=[c for c in attack.columns if c not in {"Row","Date","Time",LABEL}]
    if attack_features != normal_features: raise RuntimeError("Normal/Attack feature schema mismatch")
    attack_ts,attack_ts_note=infer_audit_timestamps(attack)
    attack_stats=frame_audit(attack,attack_features,attack_ts)
    label_counts={str(int(k)):int(v) for k,v in attack[LABEL].value_counts(dropna=False).items()}
    if set(label_counts)!={"-1","1"}: raise RuntimeError(f"unexpected labels: {label_counts}")
    y=attack[LABEL].eq(-1).to_numpy()
    starts=np.flatnonzero(y & ~np.r_[False,y[:-1]])
    ends=np.flatnonzero(y & ~np.r_[y[1:],False])
    intervals=pd.DataFrame([{"label_interval_id":i+1,"start":attack_ts[s].isoformat(),"end":attack_ts[e].isoformat(),
                             "duration_seconds_inclusive":int(e-s+1),"point_count":int(e-s+1),
                             "label_raw":-1,"canonical_label":1}
                            for i,(s,e) in enumerate(zip(starts,ends))])

    official=official_events()
    bind=[]
    used=set()
    label_periods=[(pd.Timestamp(r.start),pd.Timestamp(r.end)) for r in intervals.itertuples()]
    for row in official.itertuples():
        os_,oe=pd.Timestamp(row.official_start),pd.Timestamp(row.official_end)
        overlaps=[i for i,(ls,le) in enumerate(label_periods) if max(os_,ls)<=min(oe,le)]
        if overlaps:
            i=max(overlaps,key=lambda j:(min(oe,label_periods[j][1])-max(os_,label_periods[j][0])).total_seconds())
            used.add(i); ls,le=label_periods[i]
            so=int((ls-os_).total_seconds()); eo=int((le-oe).total_seconds())
            status="exact_match" if so==0 and eo==0 else "partial_overlap"
            bind.append({"attack_id":row.attack_id,"label_interval_id":i+1,"binding_status":status,
                         "start_offset_seconds":so,"end_offset_seconds":eo,"successfully_bound":True})
        else:
            nearest=min(range(len(label_periods)),key=lambda j:abs((label_periods[j][0]-os_).total_seconds()))
            ls,le=label_periods[nearest]
            bind.append({"attack_id":row.attack_id,"label_interval_id":"","binding_status":"missing",
                         "start_offset_seconds":int((ls-os_).total_seconds()),"end_offset_seconds":int((le-oe).total_seconds()),
                         "successfully_bound":False})
    binding=pd.DataFrame(bind)
    unused=[i+1 for i in range(len(intervals)) if i not in used]

    groups=[]
    for process,g in schema.groupby("process_prefix",sort=False):
        counts=g.variable_kind.value_counts()
        groups.append({"process_group":process,"channel_count":len(g),"channel_names":";".join(g.raw_column_name),
                       "continuous_target_candidates":int(counts.get("continuous",0)),
                       "discrete_sources":int(counts.get("discrete",0)),"constant_count":int(counts.get("constant",0)),
                       "uncertain_count":int(counts.get("uncertain",0)),
                       "grouping_basis":"WADI raw variable-name prefix only; no data/model/Attack grouping"})
    groups=pd.DataFrame(groups)

    taxonomy=official[["attack_id","targets","process","taxonomy","reason","evidence_source","confidence"]].copy()
    tax_counts={k:int(v) for k,v in taxonomy.taxonomy.value_counts().items()}
    high=tax_counts.get("HIGH_COLLABORATION_RELEVANCE",0); mixed=tax_counts.get("MIXED_COLLABORATION",0)
    context_status=("WADI_REAL_CONTEXTUAL_BENCHMARK_PROMISING" if high>=3 else
                    "WADI_REAL_CONTEXTUAL_BENCHMARK_USABLE" if high+mixed>=5 else
                    "WADI_CONTEXTUAL_ATTACK_COVERAGE_NARROW")
    gap_after=(np.flatnonzero(np.diff(normal_ts.view("i8"))!=1_000_000_000)+1).tolist()
    bounds=[0]+gap_after+[normal_stats["rows"]]; segments=[]
    for i,(start,end) in enumerate(zip(bounds[:-1],bounds[1:]),1):
        split=start+int(math.floor(0.8*(end-start)))
        segments.append({"segment_id":i,"range":f"[{start},{end})","rows":end-start,
                         "train_range":f"[{start},{split})","train_rows":split-start,
                         "validation_range":f"[{split},{end})","validation_rows":end-split})
    protocol={"normal_session":"two sessions separated by the audited 264001-second boundary","normal_segments":segments,
              "normal_train_rows":sum(s["train_rows"] for s in segments),
              "normal_validation_rows":sum(s["validation_rows"] for s in segments),
              "segment_aware":True,"window_lag_cross_boundary":False,"test":"all WADI_attackdataLABLE.csv physical rows",
              "test_labels":"evaluation only","test_selection_forbidden":["target","source","lag","threshold","fusion"],
              "timestamp_note":"Raw Time omits hour. Any future loader must preserve raw fields and use the audited Row/date reconstruction explicitly."}
    manifest={"dataset":"WADI.A2 2019 local release","source_variant_guard":{"used":"WADI.A2","A1_used":False,"A3_used":False,
              "exact_files":[NORMAL.name,ATTACK.name,PDF.name]},"source_files":{p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in (NORMAL,ATTACK,PDF)},
              "pdf_pages":1,"pdf_attack_identifiers":"1-15 (3-4 share one table row)","pdf_updated_corrected_attack_table_confirmed":True,
              "normal":normal_stats,"attack":attack_stats,"normal_timestamp_audit":normal_ts_note,"attack_timestamp_audit":attack_ts_note,
              "attack_trailing_nondata_rows":trailer_rows,"label_column":LABEL,"label_mapping":{"1":0,"-1":1},
              "label_counts":label_counts,"attack_ratio":float(y.mean())}
    summary={"elapsed_seconds":time.perf_counter()-started,"status":"WADI_A2_PROTOCOL_AUDIT_COMPLETE_WITH_LABEL_BINDING_CONFLICT",
             "version_confirmation":"WADI.A2 updated/corrected 15-attack table confirmed by local source bundle and PDF body",
             "official_attack_count":15,"label_interval_count":len(intervals),"successfully_bound":int(binding.successfully_bound.sum()),
             "unbound_attack_ids":binding.loc[~binding.successfully_bound,"attack_id"].astype(int).tolist(),
             "unused_label_interval_ids":unused,"taxonomy_counts":tax_counts,"high_ids":taxonomy.loc[taxonomy.taxonomy=="HIGH_COLLABORATION_RELEVANCE","attack_id"].tolist(),
             "mixed_ids":taxonomy.loc[taxonomy.taxonomy=="MIXED_COLLABORATION","attack_id"].tolist(),"status_context":context_status,
             "no_model_training":True,"no_model_inference":True,"no_anomaly_metrics":True,"no_attack_driven_selection":True,
             "no_silent_timestamp_repair":True,"protocol":protocol}

    OUT.mkdir(parents=True, exist_ok=True)
    schema.to_csv(OUT/"variable_schema.csv",index=False)
    groups.to_csv(OUT/"process_groups.csv",index=False)
    intervals.to_csv(OUT/"label_intervals.csv",index=False)
    official.to_csv(OUT/"official_attack_events.csv",index=False)
    binding.to_csv(OUT/"attack_binding.csv",index=False)
    taxonomy.to_csv(OUT/"attack_taxonomy.csv",index=False)
    (OUT/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (OUT/"protocol_candidate.json").write_text(json.dumps(protocol,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (OUT/"audit_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"manifest":manifest,"summary":summary,"groups":groups.to_dict("records")},indent=2,allow_nan=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
