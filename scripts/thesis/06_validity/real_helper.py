"""Normal-only exact formal Ridge replay. No official main() is invoked."""
from __future__ import annotations
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r'D:\workspace\python\mscad-thesis')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COLLAB = ROOT / 'scripts/thesis/02_collaboration'
TAG = 'RETROSPECTIVE_VALIDITY_AUDIT'

def load(name, filename):
    spec = importlib.util.spec_from_file_location('validity_p1_' + name, COLLAB / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def modules():
    return dict(swat=load('swat', 'run_swat2015_final_unified.py'),
                hai=load('hai', 'run_hai21_final_unified.py'),
                h=load('h', 'run_hai21_b1_real.py'),
                w=load('w', 'run_wadi_b1_external.py'))

def predictions(prepared, values, records=None, first=30):
    """Per-target normal/test forward; all features come from official design()."""
    records = records if records is not None else prepared['all_records']
    times = np.arange(first, len(values), dtype=np.int64)
    zs, selfs, crosses = [], [], []
    w = prepared['helper']
    for row in records:
        ti = row['target_index']
        sm, cm = prepared['models'][row['target']]
        zs.append(values[times, ti])
        selfs.append(sm.predict(w.design(values, times, [(ti, lag) for lag in prepared['runner'].SELF_LAGS]))[:, 0])
        crosses.append(cm.predict(w.design(values, times, [(si, lag) for si in row['source_indices'] for lag in prepared['runner'].CROSS_LAGS]))[:, 0])
    return {'z': np.column_stack(zs), 'ps': np.column_stack(selfs), 'pc': np.column_stack(crosses), 'local_index': times}

def prepare(dataset, log=print):
    m = modules()
    r, w, h = m[dataset], m['w'], m['h']
    inputs = []
    if dataset == 'swat':
        a2 = load('a2', 'run_innovation2a2_joint_lagged.py')
        b2 = load('b2', 'run_innovation2b_controlled.py')
        from data.swat2015_official import load_normal_cache, resolve_repo_path, validate_config_contract
        from data.transforms import StandardScaler
        from utils.config_io import load_config
        config = load_config(str(b2.CONFIG)); validate_config_contract(config)
        cache = resolve_repo_path(ROOT, config['paths']['cache_dir'])
        normal, cache_manifest = load_normal_cache(cache, config)
        if normal.shape != (r.NORMAL_END, 51):
            raise RuntimeError('STOP_A: SWaT Normal shape mismatch')
        names = [str(x) for x in cache_manifest['column_order']]
        schema_path = ROOT / 'outputs/thesis_t2_swat2015_audit/schema_profile.json'
        schema = json.loads(schema_path.read_text(encoding='utf-8'))
        contract, _ = a2.build_contract(a2.normal_profiles(schema), names)
        scaler = StandardScaler().fit(np.asarray(normal[:r.TRAIN_END]))
        z = scaler.transform(np.asarray(normal)).astype(np.float64)
        fit, select, train, validation = [z[:r.DISCOVERY_END]], [z[r.DISCOVERY_END:r.TRAIN_END]], [z[:r.TRAIN_END]], [z[r.TRAIN_END:r.NORMAL_END]]
        midpoint = len(validation[0]) // 2
        first, second = [validation[0][:midpoint]], [validation[0][midpoint:]]
        groups = []
        for stage in ('P1','P2','P3','P4','P5','P6'):
            usable = list(contract[stage]['nonconstant_channels'])
            targets = list(contract[stage]['continuous_targets']) if len(usable) >= 2 else []
            groups.append((stage, usable, targets))
        extra = {'config':config, 'cache':cache, 'cache_manifest':cache_manifest}
        inputs = [Path(b2.CONFIG), cache/'cache_manifest.json', cache/'normal_v1_features.npy', schema_path]
    elif dataset == 'hai':
        schema = pd.read_csv(r.A0/'variable_schema.csv')
        group_table = pd.read_csv(r.A0/'process_groups.csv')
        names = schema.column_name.astype(str).tolist()
        if len(names) != 79 or set(r.LABELS) & set(names):
            raise RuntimeError('STOP_A: HAI feature schema mismatch')
        byname = schema.set_index('column_name')
        scaler = r.StandardScaler(); boundaries = {}
        for name in r.TRAIN_FILES:
            frame = pd.read_csv(r.DATA/name, compression='gzip', usecols=names, low_memory=False)
            boundary = int(math.floor(.8*len(frame))); boundaries[name] = boundary
            scaler.partial_fit(frame.iloc[:boundary].to_numpy(np.float64))
        train, validation = [], []
        for name in r.TRAIN_FILES:
            frame = pd.read_csv(r.DATA/name, compression='gzip', usecols=names, low_memory=False)
            boundary = boundaries[name]
            train.append(scaler.transform(frame.iloc[:boundary].to_numpy(np.float64)).astype(np.float64, copy=False))
            validation.append(scaler.transform(frame.iloc[boundary:].to_numpy(np.float64)).astype(np.float64, copy=False))
        fit, select = [], []
        for values in train:
            boundary = int(math.floor(.8*len(values)))
            fit.append(values[:boundary]); select.append(values[boundary:])
        first, second = h.split_halves(validation)
        groups = []
        for group in group_table.itertuples(index=False):
            usable = [name for name in str(group.channel_names).split(';') if name in names and str(byname.loc[name,'nonconstant']).lower() == 'true']
            targets = [name for name in usable if byname.loc[name,'variable_kind'] == 'continuous' and len(usable)>=2]
            groups.append((str(group.process),usable,targets))
        extra = {'train_boundaries':boundaries}
        inputs = [r.DATA/name for name in r.TRAIN_FILES]+[r.A0/'variable_schema.csv',r.A0/'process_groups.csv']
    else:
        raise ValueError(dataset)

    index = {name:i for i,name in enumerate(names)}
    candidates, fitted, full_disagreements, all_records = [], {}, {}, []
    def views(segs, tids, specs, cm, sms, bases):
        return r.evaluate_views(w, segs[0] if dataset=='swat' else segs, tids,specs,cm,sms,bases)
    for group, usable, target_names in groups:
        if not target_names:
            continue
        tids = [index[name] for name in target_names]
        specs = [(index[name],lag) for name in usable for lag in r.CROSS_LAGS]
        cd = w.leave_target_out_models(fit,specs,tids)
        cf = w.leave_target_out_models(train,specs,tids)
        sd = [w.fit_ridge(fit,[(ti,lag) for lag in r.SELF_LAGS],[ti]) for ti in tids]
        sf = [w.fit_ridge(train,[(ti,lag) for lag in r.SELF_LAGS],[ti]) for ti in tids]
        bd = [float(np.median(np.concatenate([v[:,ti] for v in fit]))) for ti in tids]
        bf = [float(np.median(np.concatenate([v[:,ti] for v in train]))) for ti in tids]
        discovery = views(select,tids,specs,cd,sd,bd)
        full = views(validation,tids,specs,cf,sf,bf)
        vfirst = views(first,tids,specs,cf,sf,bf)
        vsecond = views(second,tids,specs,cf,sf,bf)
        for j,target in enumerate(target_names):
            ti = tids[j]
            ds,fs,f1,f2 = discovery[j],full[j],vfirst[j],vsecond[j]
            m1,m2 = float(np.median(f1['disagreement'])),float(np.median(f2['disagreement']))
            q1,q2 = float(np.quantile(f1['disagreement'],.95)),float(np.quantile(f2['disagreement'],.95))
            mr = max(m1/max(m2,1e-12),m2/max(m1,1e-12)); qr=max(q1/max(q2,1e-12),q2/max(q1,1e-12))
            skills = {}
            for label,row in [('discovery',ds),('validation_full',fs),('validation_first',f1),('validation_second',f2)]:
                for view in ('self','cross'):
                    skills[f'{label}_{view}_skill']=r.skill(row[f'mae_{view}'],row['mae_base'])
            skill_only = all(v>0 for v in skills.values())
            reliable = skill_only and mr<=2 and qr<=3
            sources = [index[name] for name in usable if name!=target]
            keep=np.asarray([channel!=ti for channel,_ in specs])
            fitted[target]=(sf[j],w.LinearModel(cf.coef[keep,j:j+1],np.asarray([cf.intercept[j]])))
            full_disagreements[target]=fs['disagreement']
            all_records.append({'target':target,'target_index':ti,'source_indices':sources,'group':group})
            candidates.append({'audit_tag':TAG,'dataset':dataset,'target':target,'group':group,**skills,'median_ratio':mr,'q95_ratio':qr,'skill_only':skill_only,'reliable':reliable})
        log(f'{dataset}: normal-only exact fit {group}, candidates={len(target_names)}')
    frozen_path=r.OUT/'dual_targets_unified_frozen.csv'
    frozen_table=pd.read_csv(frozen_path)
    frozen_targets=frozen_table.target.astype(str).tolist()
    replay_targets=[row['target'] for row in candidates if row['reliable']]
    target_check={'status':'PASS' if replay_targets==frozen_targets else 'FAIL','expected':frozen_targets,'replayed':replay_targets,'order_equal':replay_targets==frozen_targets}
    records=[next(row for row in all_records if row['target']==target) for target in frozen_targets]
    refs={target:(r.robust if dataset=='swat' else h.robust)(full_disagreements[target]) for target in frozen_targets}
    state={'dataset':dataset,'names':names,'targets':frozen_targets,'records':records,'all_records':all_records,
           'models':fitted,'validation':validation,'references':refs,'helper':w,'runner':r,'h':h,'scaler':scaler,
           'candidates':pd.DataFrame(candidates),'skill_targets':[row['target'] for row in candidates if row['skill_only']],
           'all_targets':[row['target'] for row in all_records], 'inputs':inputs+[frozen_path,r.OUT/'thresholds.json',r.OUT/'global_metrics.csv',r.OUT/'manifest.json'],
           'target_check':target_check,'full_disagreements':full_disagreements, **extra}
    if dataset=='swat':
        formal_fitted={t:{'self_model':v[0],'cross_model':v[1]} for t,v in fitted.items()}
        vals=[r.score_segment(w,v,records,formal_fitted,refs)[0] for v in validation]
    else:
        vals,_,_=h.dual_scores(validation,records,fitted,refs)
    threshold=float(np.quantile(np.concatenate(vals),.99))
    threshold_ref=float(json.loads((r.OUT/'thresholds.json').read_text())['dual'])
    threshold_diff=abs(threshold-threshold_ref)
    state.update(threshold=threshold,validation_s3=vals,threshold_check={'status':'PASS' if threshold_diff<=1e-9 else 'FAIL','replayed':threshold,'frozen':threshold_ref,'max_abs_diff':threshold_diff,'tolerance':1e-9})
    state['gate_pass']=target_check['status']=='PASS' and state['threshold_check']['status']=='PASS'
    # Predictions for all pre-specified candidates permit only the declared skill sensitivity.
    state['val_predictions']=[predictions(state,v) for v in validation]
    state['normal_predictions']=state['val_predictions']
    log(f'{dataset}: Normal replay targets={target_check["status"]}, threshold={state["threshold_check"]["status"]}, all candidates={len(all_records)}')
    return state

def event_rates(prepared, binding, scores, threshold):
    r=prepared['runner']; dataset=prepared['dataset']
    rows=[]
    if dataset=='swat':
        events=pd.read_csv(r.T25/'canonical_attack_events.csv')
        events=events[events.evaluable_physical_attack_interval.astype(str).str.lower()=='true']
        pcc=load('pcc','run_swat2015_pcc1.py')
        for e in events.itertuples(index=False):
            start=pcc.timestamp_seconds(str(e.canonical_start)); end=pcc.timestamp_seconds(str(e.canonical_end))
            mask=(binding['timestamp']>=start)&(binding['timestamp']<=end)
            rows.append({'event_id':str(e.event_id),'high':False,'hit_rate':float((scores[mask]>threshold).mean()),'point_count':int(mask.sum())})
    else:
        events=pd.read_csv(r.A0/'attack_events_50.csv')
        highs=set(pd.read_csv(r.OLD/'high_collaboration_events_frozen.csv').event_id.astype(str))
        for e in events.itertuples(index=False):
            mask=(binding['file_id']==str(e.file))&(binding['local_index']>=int(e.start_index))&(binding['local_index']<=int(e.end_index_inclusive))
            rows.append({'event_id':str(e.event_id),'file':str(e.file),'high':str(e.event_id) in highs,'hit_rate':float((scores[mask]>threshold).mean()),'point_count':int(mask.sum())})
    result=pd.DataFrame(rows)
    result['HIGH']=result['high']
    return result

def _difference(actual,expected,tolerance=0):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape:
        return {'status':'FAIL','actual_shape':list(a.shape),'expected_shape':list(b.shape),'max_abs_diff':None,'difference_count':None,'first_difference':None}
    diff=np.abs(a-b) if np.issubdtype(a.dtype,np.number) and np.issubdtype(b.dtype,np.number) else (a.astype(str)!=b.astype(str))
    mask=diff>tolerance
    idx=np.flatnonzero(mask)
    return {'status':'PASS' if not mask.any() else 'FAIL','max_abs_diff':float(np.max(diff)) if diff.size else 0.,'difference_count':int(mask.sum()),'first_difference':int(idx[0]) if idx.size else None}

def replay_test(prepared,log=print):
    if not prepared['gate_pass']:
        raise RuntimeError('STOP_B: Normal replay target/threshold gate failed before Attack access')
    r,w,dataset=prepared['runner'],prepared['helper'],prepared['dataset']
    segments=[]
    if dataset=='swat':
        from data.swat2015_official import load_attack_cache
        attack,labels,ts,_=load_attack_cache(prepared['cache'],prepared['config'])
        z=prepared['scaler'].transform(np.asarray(attack,dtype=np.float64))
        manifest=json.loads((r.T25/'segment_manifest.json').read_text())
        cursor=0
        source=r.sha256(prepared['cache']/'attack_v0_features.npy')
        for i,item in enumerate(manifest['test_segments']):
            end=cursor+int(item['row_count'])
            segments.append((str(i),z[cursor:end],np.asarray(labels[cursor:end]),np.asarray(ts[cursor:end]),cursor,source))
            cursor=end
        if cursor!=len(attack) or len(segments)!=2:
            raise RuntimeError('STOP_A: SWaT frozen two-segment coverage changed')
        prepared['inputs'] += [prepared['cache']/f for f in ('attack_v0_features.npy','attack_v0_labels.npy','attack_v0_timestamps.npy')]+[r.T25/'segment_manifest.json',r.T25/'canonical_attack_events.csv']
    else:
        cursor=0
        for name in r.TEST_FILES:
            file=r.DATA/name; frame=pd.read_csv(file,compression='gzip',low_memory=False)
            if list(frame.columns[1:-4])!=prepared['names'] or tuple(frame.columns[-4:])!=r.LABELS:
                raise RuntimeError(f'STOP_A: Test schema changed {name}')
            z=prepared['scaler'].transform(frame[prepared['names']].to_numpy(np.float64)).astype(np.float64,copy=False)
            timestamps=pd.to_datetime(frame.iloc[:,0]).to_numpy(dtype='datetime64[s]').astype(np.int64)
            segments.append((name,z,frame.attack.to_numpy(np.int8),timestamps,cursor,r.sha256(file))); cursor+=len(frame)
            prepared['inputs'].append(file)
        prepared['inputs'] += [r.A0/'attack_events_50.csv',r.OLD/'high_collaboration_events_frozen.csv']
    bind_parts=[]; pred_parts=[]; official_scores=[]
    for name,values,labels,ts,start,source in segments:
        if dataset=='swat':
            fitted={t:{'self_model':v[0],'cross_model':v[1]} for t,v in prepared['models'].items()}
            score,_,raw=r.score_segment(w,values,prepared['records'],fitted,prepared['references'])
        else:
            score,_,raw=r.score_dual(w,values,prepared['records'],prepared['models'],prepared['references'])
        official_scores.append(score[33:])
        # Start at 30 to exactly preserve formal full-matrix BLAS call shape, slice afterwards.
        pred=predictions(prepared,values)
        pred_parts.append({key:value[33:] for key,value in pred.items()})
        local=np.arange(63,len(values),dtype=np.int64)
        bind_parts.append({'dataset':np.repeat(dataset,len(local)),'file_id':np.repeat(name,len(local)),
                           'segment_id':np.repeat(name,len(local)),'local_index':local,'global_index':start+local,
                           'label':labels[63:],'raw_label':labels[63:],'timestamp':ts[63:],'valid_endpoint':np.ones(len(local),dtype=bool),
                           'source_hash':np.repeat(source,len(local))})
        log(f'{dataset}: S3 replay scored {name}, endpoints={len(local)}')
    binding={key:np.concatenate([part[key] for part in bind_parts]) for key in bind_parts[0]}
    pred={key:np.concatenate([part[key] for part in pred_parts]) for key in pred_parts[0]}
    s3=np.concatenate(official_scores)
    metric_fn=r.metrics if dataset=='swat' else prepared['h'].metrics
    result=metric_fn(binding['label'],s3,prepared['threshold'])
    frozen=pd.read_csv(r.OUT/'global_metrics.csv').set_index('method').loc['DUAL']
    metric_check={key:{'actual':float(result[key]),'expected':float(frozen[key]),'pass':round(float(result[key]),6)==round(float(frozen[key]),6)} for key in ('auroc','aupr','precision','recall','f1','fpr')}
    ev=event_rates(prepared,binding,s3,prepared['threshold'])
    hit=int((ev.hit_rate>=.10).sum())
    high_med=float(ev.loc[ev.high,'hit_rate'].median()) if dataset=='hai' else None
    events_check={'hit_count':hit,'expected_hit_count':2 if dataset=='swat' else 44,'event_count':len(ev),'expected_event_count':36 if dataset=='swat' else 50,'high_median_hit':high_med,'expected_high_median_hit':.841085 if dataset=='hai' else None}
    events_check['status']='PASS' if hit==events_check['expected_hit_count'] and len(ev)==events_check['expected_event_count'] and (dataset!='hai' or round(high_med,6)==.841085) else 'FAIL'
    archive=ROOT/'outputs/thesis_evaluation/literature_compatible/raw_scores'/dataset/'ours_dual/test_raw_scores.npz'
    if archive.exists():
        prepared['inputs'].append(archive)
        with np.load(archive,allow_pickle=False) as a:
            dtype=str(a['score'].dtype); tol=1e-5 if a['score'].dtype==np.float32 else 1e-9
            score_check={**_difference(s3,a['score'],tol),'dtype':dtype,'tolerance':tol,'archive':str(archive)}
            endpoint_check={key:_difference(binding[key],a[key]) for key in ('file_id','local_index','global_index','label','valid_endpoint')}
            if 'timestamp' in a:
                endpoint_check['timestamp']=_difference(binding['timestamp'],a['timestamp'])
    else:
        score_check={'status':'NOT_AVAILABLE','archive':str(archive)}
        endpoint_check={'status':'NOT_AVAILABLE','reason':'Endpoint archive absent; formal frozen timeline requires separate audit'}
    passed=all(item['pass'] for item in metric_check.values()) and events_check['status']=='PASS' and score_check['status'] in ('PASS','NOT_AVAILABLE') and all(item.get('status')=='PASS' for item in endpoint_check.values() if isinstance(item,dict))
    check={'status':'PASS' if passed else 'FAIL','target_order':prepared['target_check'],'threshold':prepared['threshold_check'],
           'scores':score_check,'endpoints':endpoint_check,'metrics':metric_check,'events':events_check}
    return {'binding':pd.DataFrame(binding),'prediction':pred,'predictions':pred,
            'score':s3,'S3':s3,'metrics':result,'events':ev,'gates':check,'check':check,
            'passed':passed,'metric_fn':metric_fn}
