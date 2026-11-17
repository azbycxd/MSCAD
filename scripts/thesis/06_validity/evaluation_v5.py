"""RETROSPECTIVE_VALIDITY_AUDIT: v5 shared-prediction residual controls."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import real_helper
TAG='RETROSPECTIVE_VALIDITY_AUDIT'


def evaluate(prepared,replay,normal_freeze_sha,output_dir,stats_state):
    if not replay.get('passed') or replay.get('shared_check',{}).get('status')!='PASS':
        raise RuntimeError('STOP_B: formal/shared prediction gate not passed')
    out=Path(output_dir)
    if hashlib.sha256((out/'normal_stats.csv').read_bytes()).hexdigest()!=normal_freeze_sha:
        raise RuntimeError('NORMAL_STATISTICS_HASH_MISMATCH')
    import pyarrow  # Required for the explicit per-target parquet artifact.
    dataset=prepared['dataset'];main=list(prepared['targets']);skill=list(prepared['skill_targets']);all_targets=prepared['all_targets']
    if stats_state['target_order']!=main or stats_state['skill_target_order']!=skill:raise RuntimeError('FROZEN_TARGET_ORDER_MISMATCH')
    pred=replay['prediction'];bind=replay['binding'].copy();bind['audit_label']=TAG;bind['valid_mask']=bind['valid_endpoint'].astype(bool)
    if not bind.valid_mask.all():raise RuntimeError('INVALID_FORMAL_ENDPOINT')
    labels=bind.raw_label.to_numpy(np.int8)
    # Independently rebuild the primary aggregate from the SAME saved predictions,
    # before computing any S1/S2 metric. The reference is the frozen formal reference.
    shared=np.zeros(len(bind),dtype=np.float64)
    for target in main:
        j=all_targets.index(target);median,scale=prepared['references'][target]
        np.maximum(shared,np.maximum(0.,(np.abs(pred['ps'][:,j]-pred['pc'][:,j])-median)/scale),out=shared)
    maxdiff=float(np.max(np.abs(shared-replay['S3'])))
    if not np.isfinite(shared).all() or maxdiff>1e-9:raise RuntimeError('STOP_B: shared-source aggregate mismatch at evaluation entry')
    refs=stats_state['references'];details=stats_state['details'];thresholds=stats_state['thresholds']
    relevant=[t for t in all_targets if t in set(main)|set(skill)]
    scores={s:np.zeros(len(bind),dtype=np.float64) for s in ('S1','S2','S3')}
    skill_scores={s:np.zeros(len(bind),dtype=np.float64) for s in ('S2','S3')}
    wide={c:bind[c].to_numpy() for c in bind};per_target=[]
    for target in relevant:
        j=all_targets.index(target)
        raw={'S1':np.abs(pred['z'][:,j]-pred['ps'][:,j]),'S2':np.abs(pred['z'][:,j]-pred['pc'][:,j]),'S3':np.abs(pred['ps'][:,j]-pred['pc'][:,j])}
        row={'audit_label':TAG,'dataset':dataset,'target':target,'main_frozen_target':target in main,'skill_only_target':target in skill,
             'kappa':stats_state['kappa'][target],'normal_stats_sha256':normal_freeze_sha,
             'auroc_semantics':'per-target prescribed Normal median/MAD calibration plus nonnegative truncation'}
        for score,values in raw.items():
            med,scale=refs[score][target]
            if score=='S3' and target in main and (med,scale)!=prepared['references'][target]:raise RuntimeError('S3_REFERENCE_CHANGED')
            normalized=np.maximum(0.,(values-med)/scale)
            if not np.isfinite(normalized).all():raise RuntimeError('NONFINITE_SCORE')
            wide[f'{target}__{score}_raw']=values
            row[f'{score}_auroc']=float(roc_auc_score(labels,normalized))
            if target in main:np.maximum(scores[score],normalized,out=scores[score])
            if target in skill and score in skill_scores:np.maximum(skill_scores[score],normalized,out=skill_scores[score])
        row['S3_minus_S2_auroc']=row['S3_auroc']-row['S2_auroc'];per_target.append(row)
    if not np.array_equal(scores['S3'],shared):raise RuntimeError('SHARED_AGGREGATE_INTERNAL_MISMATCH')
    metrics=[];events_by_score={};outputs=[];sensitivity=[]
    def parquet(frame,name):
        path=out/name
        if path.exists():raise RuntimeError('Refuse replacing '+str(path))
        frame.to_parquet(path,index=False,engine='pyarrow');outputs.append(str(path))
    for score,values in scores.items():
        threshold=thresholds[score]
        measured=replay['metric_fn'](labels,values,threshold)
        events=real_helper.event_rates(prepared,bind,values,threshold);events_by_score[score]=events
        row={'audit_label':TAG,'dataset':dataset,'score':score,**measured,'threshold':threshold,'target_count':len(main),'endpoint_count':len(bind),
             'event_hit_count':int((events.hit_rate>=.1).sum()),'event_count':len(events),'normal_stats_sha256':normal_freeze_sha,'prediction_source':'SAME_SAVED_SHARED_Z_SELF_CROSS'}
        if dataset=='hai':row.update(high19_hit_count=int((events.loc[events.high,'hit_rate']>=.1).sum()),high19_median_hit=float(events.loc[events.high,'hit_rate'].median()))
        metrics.append(row)
        frame=bind.copy();frame['score_id']=score;frame['score']=values;frame['prediction']=values>threshold;frame['normal_stats_sha256']=normal_freeze_sha
        parquet(frame,f'scores_{dataset}_{score}.parquet')
    parquet(pd.DataFrame(wide,copy=False),f'per_target_scores_{dataset}.parquet')
    per_frame=pd.DataFrame(per_target);per_frame.to_csv(out/f'per_target_{dataset}.csv',index=False)
    merged=events_by_score['S3'].drop(columns='hit_rate').copy();merged['audit_label']=TAG;merged['dataset']=dataset
    for score,events in events_by_score.items():
        if not events.event_id.equals(events_by_score['S3'].event_id):raise RuntimeError('EVENT_ORDER_MISMATCH')
        merged[f'{score}_hit_rate']=events.hit_rate.to_numpy();merged[f'{score}_hit_ge_0_10']=events.hit_rate.to_numpy()>=.1
    merged['S3_minus_S2_hit_rate']=merged.S3_hit_rate-merged.S2_hit_rate
    merged.to_csv(out/f'events_{dataset}.csv',index=False)
    for score,values in skill_scores.items():
        if not skill:continue
        threshold=stats_state['skill_thresholds'][score]
        measured=replay['metric_fn'](labels,values,threshold)
        sensitivity.append({'audit_label':TAG,'dataset':dataset,'score':score,'target_set':'SKILL_ONLY_SUPPLEMENTARY','target_count':len(skill),
                            'auroc':float(measured['auroc']),'aupr':float(measured['aupr']),'normal_stats_sha256':normal_freeze_sha,'included_in_decision':False})
        frame=bind.copy();frame['score_id']=score;frame['score']=values;frame['target_set']='SKILL_ONLY_SUPPLEMENTARY';frame['normal_stats_sha256']=normal_freeze_sha
        parquet(frame,f'scores_{dataset}_skill_only_{score}.parquet')
    main_frame=per_frame[per_frame.main_frozen_target]
    positive=main_frame[main_frame.kappa>0];nonpositive=main_frame[main_frame.kappa<=0]
    ppos=float((positive.S3_minus_S2_auroc>0).mean()) if len(positive) else None
    pnon=float((nonpositive.S3_minus_S2_auroc>0).mean()) if len(nonpositive) else None
    hk={'status':'TESTABLE' if len(positive) and len(nonpositive) else 'NOT_TESTABLE','positive_count':len(positive),'nonpositive_count':len(nonpositive),
        'positive_group_S3_gt_S2_fraction':ppos,'nonpositive_group_S3_gt_S2_fraction':pnon,'trend_present':ppos>pnon if ppos is not None and pnon is not None else None,
        'interpretation':'descriptive association only; no causality or guaranteed ranking performance'}
    event_diff={'S3_greater_than_S2':int((merged.S3_minus_S2_hit_rate>0).sum()),'S3_less_than_S2':int((merged.S3_minus_S2_hit_rate<0).sum()),'equal':int((merged.S3_minus_S2_hit_rate==0).sum())}
    target_diff={'S3_greater_than_S2':int((main_frame.S3_minus_S2_auroc>0).sum()),'S3_less_than_S2':int((main_frame.S3_minus_S2_auroc<0).sum()),'equal':int((main_frame.S3_minus_S2_auroc==0).sum())}
    return {'audit_label':TAG,'dataset':dataset,'metrics':metrics,'metrics_rows':metrics,'per_target':per_target,'hk':hk,'H_kappa':hk,
            'event_differences':event_diff,'target_differences':target_diff,'sensitivity':sensitivity,'outputs':outputs,'format_fallbacks':[],
            'thresholds':thresholds,'shared_recheck_max_abs_diff':maxdiff,'shared_prediction_source_used_for_all_scores':True}