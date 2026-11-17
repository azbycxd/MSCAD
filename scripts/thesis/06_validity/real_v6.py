"""RETROSPECTIVE_VALIDITY_AUDIT: v6 deterministic and shared-prediction gates.

No fitting here. Both complete Normal-only prepares must precede this call.
Only formal scoring functions are invoked; E2 archives are diagnostic references.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import real_helper as base

TAG='RETROSPECTIVE_VALIDITY_AUDIT'
TOL=1e-9

def _json(path,value):
    def cast(x):
        if isinstance(x,np.generic):return x.item()
        return str(x)
    Path(path).write_text(json.dumps({'audit_label':TAG,**value},indent=2,ensure_ascii=False,allow_nan=False,default=cast)+'\n',encoding='utf-8')

def difference(a,b,tol=TOL):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:
        return {'status':'FAIL','actual_shape':list(a.shape),'expected_shape':list(b.shape),'max_abs_diff':None,'difference_count':None,'first_difference':None}
    numeric=np.issubdtype(a.dtype,np.number) and np.issubdtype(b.dtype,np.number)
    d=np.abs(a-b) if numeric else (a.astype(str)!=b.astype(str))
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all()) if numeric else True
    mask=d>tol;positions=np.argwhere(mask)
    return {'status':'PASS' if finite and not mask.any() else 'FAIL','max_abs_diff':float(d.max()) if d.size else 0.,'difference_count':int(mask.sum()),'first_difference':positions[0].tolist() if positions.size else None,'tolerance':tol,'finite':finite}

def deterministic_check(first,second):
    a=np.concatenate(first['validation_s3']);b=np.concatenate(second['validation_s3'])
    result=difference(a,b,0)
    result['bitwise_identical']=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    result['status']='PASS' if result['status']=='PASS' and result['bitwise_identical'] else 'FAIL'
    result['sha256_first']=hashlib.sha256(a.tobytes()).hexdigest()
    result['sha256_second']=hashlib.sha256(b.tobytes()).hexdigest()
    return result

def _official(state,values):
    r,w=state['runner'],state['helper']
    if state['dataset']=='swat':
        fitted={t:{'self_model':v[0],'cross_model':v[1]} for t,v in state['models'].items()}
        return r.score_segment(w,values,state['records'],fitted,state['references'])
    return r.score_dual(w,values,state['records'],state['models'],state['references'])

def _segments(state):
    r=state['runner'];cursor=0
    if state['dataset']=='swat':
        from data.swat2015_official import load_attack_cache
        attack,labels,timestamps,_=load_attack_cache(state['cache'],state['config'])
        manifest=json.loads((r.T25/'segment_manifest.json').read_text())
        source=r.sha256(state['cache']/'attack_v0_features.npy')
        state['inputs'] += [state['cache']/name for name in ('attack_v0_features.npy','attack_v0_labels.npy','attack_v0_timestamps.npy')]+[r.T25/'segment_manifest.json',r.T25/'canonical_attack_events.csv']
        for index,segment in enumerate(manifest['test_segments']):
            end=cursor+int(segment['row_count'])
            yield str(index),np.asarray(attack[cursor:end],dtype=np.float64),np.asarray(labels[cursor:end]),np.asarray(timestamps[cursor:end]),cursor,source
            cursor=end
        if cursor!=len(attack) or len(manifest['test_segments'])!=2:
            raise RuntimeError('STOP_A: SWaT two-segment coverage changed')
    else:
        for name in r.TEST_FILES:
            path=r.DATA/name;frame=pd.read_csv(path,compression='gzip',low_memory=False)
            if list(frame.columns[1:-4])!=state['names'] or tuple(frame.columns[-4:])!=r.LABELS:
                raise RuntimeError('STOP_A: HAI schema changed '+name)
            raw=frame[state['names']].to_numpy(np.float64)
            labels=frame.attack.to_numpy(np.int8)
            timestamps=pd.to_datetime(frame.iloc[:,0]).to_numpy(dtype='datetime64[s]').astype(np.int64)
            state['inputs'].append(path)
            yield name,raw,labels,timestamps,cursor,r.sha256(path)
            cursor+=len(frame)
        state['inputs'] += [r.A0/'attack_events_50.csv',r.OLD/'high_collaboration_events_frozen.csv']

def replay_test(first_state,second_state,output_dir,log=print):
    """Returns only S3 metrics; S1/S2 evaluator must use returned prediction object."""
    out=Path(output_dir)
    if out.name!='p1_residual_controls_v6' or not out.is_dir():raise RuntimeError('V6_OUTPUT_ISOLATION_REQUIRED')
    first,second=first_state,second_state
    dataset=first['dataset'];r=first['runner']
    normal_determinism=deterministic_check(first,second)
    if not first['gate_pass'] or not second['gate_pass'] or normal_determinism['status']!='PASS':
        raise RuntimeError('STOP_B: Normal target/threshold/determinism gate failed before Test')
    if first['targets']!=second['targets'] or first['all_targets']!=second['all_targets']:
        raise RuntimeError('STOP_B: two Normal replays have different target order')
    indices=[first['all_targets'].index(t) for t in first['targets']]
    refs=first['references']
    med=np.asarray([refs[t][0] for t in first['targets']])
    scale=np.asarray([refs[t][1] for t in first['targets']])
    bindings=[];predictions=[];formal_parts=[];second_parts=[];raw_parts=[];shared_parts=[];shared_reports=[]
    for name,raw_values,labels,timestamps,start,source in _segments(first):
        # Preserve the formal float64-before-transform Test path for both runs.
        values1=first['scaler'].transform(raw_values).astype(np.float64,copy=False)
        values2=second['scaler'].transform(raw_values).astype(np.float64,copy=False)
        formal,_,formal_raw=_official(first,values1)
        repeated,_,_=_official(second,values2)
        pred=base.predictions(first,values1)
        shared_raw=np.abs(pred['ps'][:,indices]-pred['pc'][:,indices])
        shared_cal=np.maximum(0.,(shared_raw-med)/scale)
        # Official raw matrix returned by score_segment/score_dual, same frozen references.
        formal_cal=np.column_stack([np.maximum(0.,(formal_raw[:,j]-refs[t][0])/refs[t][1]) for j,t in enumerate(first['targets'])])
        shared=shared_cal.max(axis=1)
        local=np.arange(63,len(values1),dtype=np.int64)
        bind=pd.DataFrame({'audit_label':TAG,'dataset':dataset,'file_id':name,'segment_id':name,'local_index':local,'global_index':start+local,'label':labels[63:],'raw_label':labels[63:],'timestamp':timestamps[63:],'valid_endpoint':True,'valid_mask':True,'source_hash':source})
        report={'segment':name,'raw_disagreement':difference(shared_raw[33:],formal_raw[33:]),'calibrated_disagreement':difference(shared_cal[33:],formal_cal[33:]),'aggregate_disagreement':difference(shared[33:],formal[33:])}
        report['status']='PASS' if all(report[k]['status']=='PASS' for k in ('raw_disagreement','calibrated_disagreement','aggregate_disagreement')) else 'FAIL'
        report['per_target']=[{'target':t,'raw_max_abs_diff':float(np.max(np.abs(shared_raw[33:,j]-formal_raw[33:,j]))),'calibrated_max_abs_diff':float(np.max(np.abs(shared_cal[33:,j]-formal_cal[33:,j])))} for j,t in enumerate(first['targets'])]
        shared_reports.append(report)
        bindings.append(bind);predictions.append({key:val[33:] for key,val in pred.items()})
        formal_parts.append(formal[33:]);second_parts.append(repeated[33:]);raw_parts.append(formal_raw[33:]);shared_parts.append(shared[33:])
        log(f'{dataset}: v6 two S3 replays and shared-source check {name}, n={len(local)}, shared={report["status"]}')
    binding=pd.concat(bindings,ignore_index=True)
    prediction={key:np.concatenate([p[key] for p in predictions]) for key in predictions[0]}
    s3=np.concatenate(formal_parts);s3_repeat=np.concatenate(second_parts);s3_shared=np.concatenate(shared_parts)
    formal_raw=np.concatenate(raw_parts)
    shared_raw=np.abs(prediction['ps'][:,indices]-prediction['pc'][:,indices])
    shared_cal=np.maximum(0.,(shared_raw-med)/scale)
    formal_cal=np.maximum(0.,(formal_raw-med)/scale)
    test_determinism=difference(s3,s3_repeat,0)
    test_determinism['bitwise_identical']=s3.dtype==s3_repeat.dtype and s3.shape==s3_repeat.shape and s3.tobytes()==s3_repeat.tobytes()
    test_determinism['status']='PASS' if test_determinism['status']=='PASS' and test_determinism['bitwise_identical'] else 'FAIL'
    test_determinism['sha256_first']=hashlib.sha256(s3.tobytes()).hexdigest();test_determinism['sha256_second']=hashlib.sha256(s3_repeat.tobytes()).hexdigest()
    shared_check={'status':'PASS' if all(x['status']=='PASS' for x in shared_reports) else 'FAIL','tolerance':TOL,'normal_references':'unchanged official frozen Normal references','target_order':first['targets'],'segments':shared_reports,'raw_disagreement':difference(shared_raw,formal_raw),'calibrated_disagreement':difference(shared_cal,formal_cal),'aggregate_disagreement':difference(s3_shared,s3)}
    # Persist precisely the arrays consumed later by real_evaluation. No later forward.
    pred_path=out/f'shared_predictions_{dataset}.npz'
    if pred_path.exists():raise RuntimeError('Refusing prediction overwrite')
    bound={key:binding[key].to_numpy() for key in binding}
    for key in ('dataset','file_id','segment_id','source_hash','audit_label'):bound[key]=bound[key].astype(str)
    np.savez(pred_path,z=prediction['z'],pred_self=prediction['ps'],pred_cross=prediction['pc'],target_order=np.asarray(first['all_targets'],dtype=str),primary_target_order=np.asarray(first['targets'],dtype=str),S3_formal=s3,S3_replay2=s3_repeat,S3_shared=s3_shared,S3_formal_raw= formal_raw,**bound)
    _json(out/f'shared_source_{dataset}.json',{**shared_check,'prediction_archive':str(pred_path),'prediction_archive_sha256':r.sha256(pred_path),'saved_all_candidate_targets':len(first['all_targets']),'saved_primary_targets':len(first['targets'])})
    metric_fn=r.metrics if dataset=='swat' else first['h'].metrics
    metrics=metric_fn(binding.label.to_numpy(),s3,first['threshold'])
    frozen=pd.read_csv(r.OUT/'global_metrics.csv').set_index('method').loc['DUAL']
    metric_check={k:{'actual':float(metrics[k]),'expected':float(frozen[k]),'pass':round(float(metrics[k]),6)==round(float(frozen[k]),6)} for k in ('auroc','aupr','precision','recall','f1','fpr')}
    events=base.event_rates(first,binding,s3,first['threshold'])
    hits=int((events.hit_rate>=.10).sum());high_median=float(events.loc[events.high,'hit_rate'].median()) if dataset=='hai' else None
    event_check={'hit_count':hits,'expected_hit_count':2 if dataset=='swat' else 44,'event_count':len(events),'expected_event_count':36 if dataset=='swat' else 50,'high_median_hit':high_median,'expected_high_median_hit':.841085 if dataset=='hai' else None}
    event_check['status']='PASS' if hits==event_check['expected_hit_count'] and len(events)==event_check['expected_event_count'] and (dataset!='hai' or round(high_median,6)==.841085) else 'FAIL'
    archive=base.ROOT/'outputs/thesis_evaluation/literature_compatible/raw_scores'/dataset/'ours_dual/test_raw_scores.npz'
    first['inputs'].append(archive)
    endpoint_checks={};diagnostic={'status':'NOT_AVAILABLE'}
    if archive.exists():
        with np.load(archive,allow_pickle=False) as saved:
            for key in ('file_id','segment_id','local_index','global_index','label','valid_endpoint'):
                endpoint_checks[key]=difference(binding[key],saved[key],0)
            if 'timestamp' in saved:endpoint_checks['timestamp']=difference(binding.timestamp,saved['timestamp'],0)
            old=saved['score'];absolute=np.abs(s3-old);relative=absolute/np.maximum(np.abs(old),1.)
            diagnostic={'status':'DIAGNOSTIC_ONLY_NOT_A_GATE','archive':str(archive),'archive_dtype':str(old.dtype),'reason':'E2-R1D reconstructed through saved efficiency coefficients and Innovation3B path, not same formal runner path','absolute_difference':{'max':float(absolute.max()),'median':float(np.median(absolute)),'q99':float(np.quantile(absolute,.99))},'relative_difference':{'max':float(relative.max()),'median':float(np.median(relative)),'q99':float(np.quantile(relative,.99))},'relative_denominator':'max(abs(archived_score),1)'}
            top=np.argsort(-absolute,kind='stable')[:10];top_rows=binding.iloc[top].copy()
            top_rows['archive_position']=top;top_rows['formal_S3']=s3[top];top_rows['reconstructed_S3']=old[top];top_rows['absolute_difference']=absolute[top];top_rows['relative_difference']=relative[top]
            top_rows.to_csv(out/f'archive_difference_top10_{dataset}.csv',index=False)
            diagnostic['top10']=top_rows.to_dict('records')
    else:
        endpoint_checks['archive']={'status':'FAIL','reason':'No reference endpoint metadata available'}
    _json(out/f'archive_difference_{dataset}.json',diagnostic)
    passed=(all(x['status']=='PASS' for x in endpoint_checks.values()) and all(x['pass'] for x in metric_check.values()) and event_check['status']=='PASS' and normal_determinism['status']=='PASS' and test_determinism['status']=='PASS' and shared_check['status']=='PASS')
    gates={'status':'PASS' if passed else 'FAIL','target_order':first['target_check'],'threshold':first['threshold_check'],'formal_archive':{'status':'NOT_AVAILABLE','reason':'No same-path formal runner pointwise archive; E2-R1D reconstruction is diagnostic only','tolerance_if_present':TOL},'endpoints':endpoint_checks,'determinism':{'normal_validation':normal_determinism,'test':test_determinism},'shared_prediction_source':shared_check,'metrics':metric_check,'events':event_check,'reconstructed_archive_diagnostic':diagnostic}
    _json(out/f'replay_{dataset}.json',gates)
    return {'binding':binding,'prediction':prediction,'predictions':prediction,'score':s3,'S3':s3,'metrics':metrics,'events':events,'gates':gates,'check':gates,'passed':passed,'metric_fn':metric_fn,'shared_check':shared_check,'determinism':gates['determinism'],'prediction_archive':str(pred_path)}
