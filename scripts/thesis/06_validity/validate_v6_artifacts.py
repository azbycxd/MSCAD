"""RETROSPECTIVE_VALIDITY_AUDIT: read-only completed-artifact validation."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/thesis_validity/p1_residual_controls_v6'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OUT/'_deps'))
import numpy as np
import pandas as pd
TAG='RETROSPECTIVE_VALIDITY_AUDIT'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
checks=[]
def check(name,value,detail=None):
    checks.append({'check':name,'passed':bool(value),'detail':detail})
manifest=read('manifest.json');decision=read('decision.json');replay=read('replay_check.json')
if manifest['status']!='VALIDITY_P1_COMPLETED':
    raise SystemExit('Skip full artifact validation for incomplete audit; see replay_check/REPORT.')
check('manifest_decision_completed',decision['status']==manifest['status'])
frozen=manifest['normal_stats_sha256']
check('normal_stats_hash',sha(OUT/'normal_stats.csv')==frozen)
check('skill_list_hash',sha(OUT/'skill_only_targets.csv')==manifest['skill_only_targets_sha256'])
check('normal_freeze_precedes_all_new_metrics',all(manifest['normal_stats_frozen_utc']<manifest['steps'][k] for k in ('CONTROLLED_S1_S2_S3','NEW_CONTROLS_SWAT','NEW_CONTROLS_HAI')))
check('all_fits_before_attack',manifest['all_Normal_fits_complete_utc']<manifest['steps']['FORMAL_TEST_REPLAY_SWAT'])
check('v4_formal_history_preserved',read('preservation_check.json')['status']=='PASS')
metrics=pd.read_csv(OUT/'metrics_main.csv')
check('six_main_rows',len(metrics)==6)
check('v5_normal_statistics_exact',read('normal_statistics_v5_comparison.json')['status']=='PASS')
check('metrics_finite',np.isfinite(metrics[['auroc','aupr','precision','recall','f1','fpr','threshold']].to_numpy()).all())
normal=pd.read_csv(OUT/'normal_stats.csv')
for dataset,n,n_targets in [('swat',449793,15),('hai',401690,54)]:
    gates=replay[dataset]
    check(dataset+'_all_gates_pass',gates['status']=='PASS')
    check(dataset+'_formal_archive_unavailable',gates['formal_archive']['status']=='NOT_AVAILABLE')
    for scope in ('normal_validation','test'):
        det=gates['determinism'][scope]
        check(dataset+'_'+scope+'_bitwise',det['bitwise_identical'] and det['max_abs_diff']==0)
    check(dataset+'_shared_source',gates['shared_prediction_source']['status']=='PASS')
    with np.load(OUT/f'shared_predictions_{dataset}.npz',allow_pickle=False) as a:
        check(dataset+'_saved_primary_targets',len(a['primary_target_order'])==n_targets)
        check(dataset+'_saved_endpoint_count',len(a['S3_shared'])==n)
        aggregate=a['S3_shared'];raw_labels=a['raw_label'];local=a['local_index'];fileids=a['file_id']
        check(dataset+'_saved_shared_vs_formal',np.max(np.abs(aggregate-a['S3_formal']))<=1e-9)
    for score in ('S1','S2','S3'):
        f=pd.read_parquet(OUT/f'scores_{dataset}_{score}.parquet')
        check(dataset+'_'+score+'_bound_length',len(f)==n)
        check(dataset+'_'+score+'_finite',np.isfinite(f.score.to_numpy()).all())
        check(dataset+'_'+score+'_binding',np.array_equal(f.raw_label,raw_labels) and np.array_equal(f.local_index,local) and np.array_equal(f.file_id.astype(str),fileids))
        check(dataset+'_'+score+'_frozen_hash',f.normal_stats_sha256.eq(frozen).all())
        if score=='S3':check(dataset+'_evaluated_S3_is_saved_shared',np.array_equal(f.score.to_numpy(),aggregate))
    nf=normal[(normal.dataset==dataset)&normal.main_frozen_target.astype(str).str.lower().eq('true')]
    check(dataset+'_primary_normal_rows',len(nf)==n_targets*3)
    check(dataset+'_normal_scales',nf.sigma.ge(1e-6).all())
    for score in ('S1','S2','S3'):
        part=nf[nf.score==score]
        check(dataset+'_'+score+'_dominance_sum',abs(part.normal_max_dominance_fraction.sum()-1)<1e-12)
    import pyarrow.parquet as pq
    schema=pq.read_schema(OUT/f'per_target_scores_{dataset}.parquet')
    check(dataset+'_per_target_binding_fields',all(x in schema.names for x in ('dataset','file_id','local_index','raw_label','valid_mask','source_hash')))
    check(dataset+'_normal_reference_unchanged',sha(OUT/f'normal_reference_{dataset}.csv')==manifest['normal_reference_hashes'][dataset])
control=pd.read_csv(OUT/'controlled_swat.csv')
check('controlled_18_rows',len(control)==18)
s1=control[control.score=='S1']
check('controlled_S1_pairwise_equality',s1.s1_pair_max_abs_difference.eq(0).all() and (s1.auroc.sub(.5).abs()<=1e-12).all())
check('controlled_2000_pairs',control.sample_pairs.eq(2000).all())
for dataset,value in read('math_checks.json').items():
    if dataset=='audit_label':continue
    check(dataset+'_math_Normal_only',not value['test_data_used'] and not value['new_fitting'] and not value['model_scaler_targets_references_changed'])
result={'audit_label':TAG,'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','passed':sum(x['passed'] for x in checks),'total':len(checks),'checks':checks,'no_model_execution':True,'no_new_metric_computation':True}
path=OUT/'artifact_validation.json'
if path.exists():raise RuntimeError('Refuse validation overwrite')
path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
