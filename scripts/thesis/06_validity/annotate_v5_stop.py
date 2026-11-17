"""RETROSPECTIVE_VALIDITY_AUDIT: annotate stopped v5 evidence; no model/metric execution."""
import json,csv,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/thesis_validity/p1_residual_controls_v5'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import verify_v5_inputs
TAG='RETROSPECTIVE_VALIDITY_AUDIT'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps({'audit_label':TAG,**obj},ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
manifest=read('manifest.json');replay=read('replay_check.json')
assert manifest['status']=='VALIDITY_P1_INCOMPLETE'
assert 'AUROC FIT601' in manifest['exception_traceback']
assert 'NEW_CONTROLS_SWAT' not in manifest['steps']
diagnostic={
 'status':'STOPPED_AUDIT_HELPER_ASSERTION','target':'FIT601','score':'S1',
 'exception':'S1_CONSISTENCY_FLAG: AUROC FIT601',
 'paired_raw_max_abs_difference':0.0,
 'paired_equality_evidence':'Control flow reached line207; preceding equal_delta != 0 gate at line196 did not raise.',
 'actual_auroc':None,'actual_auroc_status':'NOT_PERSISTED_BY_EXCEPTION',
 'expected_mathematical_auroc':0.5,
 'failed_expression':'metrics[\"auroc\"] != 0.5',
 'source':str(ROOT/'scripts/thesis/06_validity/controlled_helper.py'),'source_sha256':sha(ROOT/'scripts/thesis/06_validity/controlled_helper.py'),
 'interpretation':'Audit helper exact-floating-equality assertion stopped execution; not evidence that S1 paired samples differed. Magnitude/cause of AUROC deviation cannot be established from saved values.',
 'previous_five_targets':'S1/S2/S3 metrics computed in local function memory but not returned or persisted; no numbers reconstructed.',
 'real_S1_S2_evaluation':'NOT_RUN','controlled_retry':False,'assertion_changed':False,'scoring_changed':False}
write('controlled_stop_diagnostic.json',diagnostic)
controls=[]
for case in replay['controlled']['cases']:
    t=case['target'];o=case['observed']
    for score in ('S1','S2','S3'):
        row={'audit_label':TAG,'target':t,'score':score,'status':'COMPUTED_NOT_PERSISTED','auroc':None,'aupr':None,'recall_q99':None,'fpr_q99':None,'threshold':None}
        if score=='S3':row.update(status='S3_FORMAL_REPLAY_PASS',auroc=o['dual_auroc'],aupr=o['dual_aupr'],recall_q99=o['corruption_recall'],fpr_q99=o['clean_fpr'],threshold=o['calibration_q99_threshold'])
        elif t=='FIT601':row['status']='ASSERTION_ABORT_ACTUAL_AUROC_NOT_PERSISTED' if score=='S1' else 'NOT_RUN_AFTER_ASSERTION'
        controls.append(row)
pd.DataFrame(controls).to_csv(OUT/'controlled_swat.csv',index=False)
decision=read('decision.json')
decision.update(stop_category='AUDIT_HELPER_ASSERTION_NOT_SHARED_SOURCE_FAILURE',main_replay_gates={'swat':'PASS','hai':'PASS'},normal_statistics_frozen=True,math_checks_completed=True,controlled_failure=diagnostic,
 skipped=['Real S1/S2 metrics and per-target/event comparison','Completed controlled S1/S2 table','Skill-only Test sensitivity','Historical CoMuT Test controls','H-kappa AUROC association','R-UP/R-DOWN/R-EQ label'])
write('decision.json',decision)
checks=[]
def check(name,ok):checks.append({'check':name,'passed':bool(ok)})
check('normal_statistics_hash',sha(OUT/'normal_stats.csv')==manifest['normal_stats_sha256'])
check('skill_list_hash',sha(OUT/'skill_only_targets.csv')==manifest['skill_only_targets_sha256'])
check('normal_frozen_before_partial_controls',manifest['normal_stats_frozen_utc']<manifest['steps']['CONTROLLED_S1_S2_S3'])
check('no_real_controls_after_exception','NEW_CONTROLS_SWAT' not in manifest['steps'] and 'NEW_CONTROLS_HAI' not in manifest['steps'])
for d,n in [('swat',449793),('hai',401690)]:
    r=replay[d];check(d+'_formal_gate',r['status']=='PASS')
    check(d+'_shared_raw_cal_aggregate_zero',all(r['shared_prediction_source'][k]['max_abs_diff']==0 for k in ('raw_disagreement','calibrated_disagreement','aggregate_disagreement')))
    check(d+'_two_runs_bitwise',all(r['determinism'][k]['bitwise_identical'] and r['determinism'][k]['max_abs_diff']==0 for k in ('normal_validation','test')))
    shared=read('shared_source_'+d+'.json')
    check(d+'_saved_prediction_hash',sha(OUT/f'shared_predictions_{d}.npz')==shared['prediction_archive_sha256'])
    with np.load(OUT/f'shared_predictions_{d}.npz',allow_pickle=False) as a:
        check(d+'_saved_endpoint_count',len(a['S3_shared'])==n)
        check(d+'_saved_shared_formal_equal',np.array_equal(a['S3_shared'],a['S3_formal']))
        check(d+'_saved_binding_fields',all(k in a for k in ('dataset','file_id','local_index','raw_label','valid_mask','source_hash','z','pred_self','pred_cross')))
    check(d+'_normal_reference_hash',sha(OUT/f'normal_reference_{d}.csv')==manifest['normal_reference_hashes'][d])
preservation=verify_v5_inputs.verify_preservation(read('input_integrity.json'))
write('preservation_check_after_stop.json',preservation)
check('v4_formal_history_unchanged',preservation['status']=='PASS')
write('artifact_validation.json',{'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','scope':'COMPLETED_STEPS_OF_INCOMPLETE_AUDIT_ONLY','passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'no_model_execution':True,'no_new_metrics':True})
normal=pd.read_csv(OUT/'normal_stats.csv')
main=normal[normal.main_frozen_target.astype(str).str.lower().eq('true')]
maths=read('math_checks.json')
lines=['## 实际停止原因与未完成范围',
 '本轮 SWaT/HAI 主实验复现门禁及共同预测源门禁均已通过，raw/calibrated/aggregate差值均为0。随后受控FIT601的S1触发辅助脚本的严格浮点相等断言：metrics[auroc] != 0.5。',
 '到达该断言前，逐对raw S1差值检查已经通过（最大差0）。因此这不是已证实的配对样本不一致。异常没有携带实际AUROC数值，局部结果也未返回；无法仅据保存记录确认偏差幅度或数值原因，不猜测数值。',
 '该严格断言来自复用的受控审计辅助函数。本轮没有修改断言、容差、模型或校准后继续，也没有补跑。该异常属于审计执行未完成，不能解释为方法失败。',
 '前5个目标的控制指标在函数局部内存中计算过，但未返回/持久化；本报告不重构或填补这些数值。controlled_swat.csv保留6个目标通过的正式S3复现数值，并明确登记S1/S2的未持久化/未执行状态。',
 '真实攻击S1/S2、逐目标/逐事件比较、技能敏感性、CoMuT Test、H-κ关联与描述标签均未完成。metrics_main.csv的S3是复现结果，不能作为三评分完整对照表。',
 'Normal统计和skill列表已在部分控制指标计算前冻结，math_checks.json已经完成。normal_stats.csv不是占位文件。']
lines+=['## Normal-only尺度诊断摘录','|dataset|score|sigma=1e-6目标数|目标数|max主要来源|主导比例|主导目标sigma|','|---|---|---|---|---|---|---|']
for (d,s),g in main.groupby(['dataset','score'],sort=False):
    top=g.loc[g.normal_max_dominance_fraction.idxmax()]
    floor=int(g.scale_at_floor.astype(str).str.lower().eq('true').sum())
    lines.append(f'|{d}|{s}|{floor}|{len(g)}|{top.target}|{top.normal_max_dominance_fraction:.6f}|{top.sigma:.12g}|')
lines+=['## 已完成数学检查摘录','|dataset|target|identity最大误差|当前目标S3变化|全局S3变化|','|---|---|---|---|---|']
for d,m in maths.items():
    if d=='audit_label':continue
    lines.append(f"|{d}|{m['target']}|{m['identity_all_normal_endpoints']['max_absolute_difference']:.12g}|{m['own_current']['S3']['delta']:.12g}|{m['global_S3_at_t0']['delta']:.12g}|")
lines+=['已完成工件检查：'+str(sum(c['passed'] for c in checks))+'/'+str(len(checks))+'。这些检查不改变INCOMPLETE状态。']
p=OUT/'REPORT.md';text=p.read_text(encoding='utf-8')
text=text.replace('S1配对相等且AUROC=0.5只确认构造一致。','S1理论上配对相等且AUROC=0.5，只用于构造一致性检查；本轮实际受控执行中断，详见末尾停止说明。')
p.write_text(text+'\n\n'+'\n\n'.join(lines)+'\n',encoding='utf-8')
manifest['post_stop_annotation_utc']=datetime.now(timezone.utc).isoformat()
manifest['stop_category']=decision['stop_category']
manifest['final_delivery_files']=[{'path':str(p),'sha256':sha(p)} for p in [OUT/n for n in ('REPORT.md','replay_check.json','normal_stats.csv','metrics_main.csv','controlled_swat.csv','decision.json','math_checks.json')]]
write('manifest.json',manifest)
print(json.dumps({'status':'VALIDITY_P1_INCOMPLETE','completed_step_checks_passed':sum(c['passed'] for c in checks),'total':len(checks),'new_model_or_metric_execution':False}))
