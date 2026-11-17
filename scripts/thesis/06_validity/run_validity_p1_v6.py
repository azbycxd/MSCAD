"""RETROSPECTIVE_VALIDITY_AUDIT: isolated v6 orchestration; no official main calls."""
from __future__ import annotations
import os, sys, json, hashlib, traceback, platform, gc
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/thesis_validity/p1_residual_controls_v6'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OUT/'_deps'))
import numpy as np
import pandas as pd
TAG='RETROSPECTIVE_VALIDITY_AUDIT'
START='2026-11-17T10:28:26+00:00'
DEADLINE=datetime.fromisoformat('2026-11-17T10:58:26+00:00')
manifest={'audit_label':TAG,'task':'Thesis-VALIDITY-P1 v6 fresh replay, shared source and v5 Normal equality','started_at_utc':START,'deadline_utc':DEADLINE.isoformat(),'steps':{},'inputs':[],'no_neural_training':True,'no_test_tuning':True,'no_method_change':True,'no_primary_overwrite':True,'audit_process_commits':False,'automatic_small_artifact_commit_authorized':True,'no_push':True}
states={}; repetitions={}; replays={}; actual={}; stats={}; analysis={}; failures=[]; completed=[]
initial=None; ctl=None; cm=None; normal_sha=None
status='VALIDITY_P1_INCOMPLETE'; active=False
class GateStop(RuntimeError):pass
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(name,value):
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        return str(x)
    if isinstance(value,dict):value={'audit_label':TAG,**value}
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf-8')
def frame(name,value):
    df=value.copy() if isinstance(value,pd.DataFrame) else pd.DataFrame(value)
    if 'audit_label' not in df:df.insert(0,'audit_label',TAG)
    df.to_csv(OUT/name,index=False)
    return df
def log(message):
    if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('30-minute hard deadline')
    line=now()+' '+message
    print(line,flush=True)
    with (OUT/'run_log.txt').open('a',encoding='utf-8') as f:f.write(line+'\n')
def step(name):
    log(name);manifest['steps'][name]=now();dump('manifest.json',manifest)
def gated_sequence(items,invoke):
    for name in items:
        value=invoke(name)
        if not value['passed']:raise GateStop('STOP_B: '+name+' formal/shared/deterministic replay gate failed')
def mock_gate_check():
    visited=[]
    try:
        gated_sequence(('swat','hai'),lambda name:(visited.append(name) or {'passed':False}))
    except GateStop:pass
    assert visited==['swat']
    return {'status':'PASS','test':'Failed first gate prevents second dataset invocation','visited_mock_datasets':visited,'real_data_read':False}
def record_inputs():
    paths=[]
    for s in states.values():paths+=s.get('inputs',[])
    if ctl:paths+=ctl.get('input_paths',[])
    if cm:paths+=cm.get('input_paths',[])
    for name in ('real_helper.py','controlled_helper.py','comut_helper.py','audit_stats.py','real_v6.py','stats_v5.py','evaluation_v5.py','provenance_v6.py','normal_v6_gate.py','controlled_v6.py','run_validity_p1_v6.py'):
        paths.append(Path(__file__).parent/name)
    seen=set()
    for p in paths:
        p=Path(p)
        if str(p) not in seen and p.is_file():
            seen.add(str(p));manifest['inputs'].append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})

def run():
    global active,initial,ctl,cm,normal_sha,status
    if (OUT/'manifest.json').exists():raise RuntimeError('Existing v6 run: refusing overwrite')
    if not OUT.is_dir() or not (OUT/'protocol_v6.md').is_file():raise RuntimeError('V6 bootstrap missing')
    active=True;step('BEGIN_V6')
    import scipy,sklearn,pyarrow
    manifest['environment']={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pyarrow':pyarrow.__version__,'pyarrow_scope':str(OUT/'_deps'),'platform':platform.platform()}
    import provenance_v6 as integrity
    initial=integrity.snapshot(ROOT);dump('input_integrity.json',initial)
    if initial['status']!='PASS':raise GateStop('STOP_A: input/source provenance gate failed')
    dump('orchestration_guard_test.json',mock_gate_check())
    import real_helper as formal,real_v6 as real_v5,controlled_v6 as controlled_helper,comut_helper
    for dataset in ('swat','hai'):
        step('NORMAL_REPLAY_1_'+dataset.upper());states[dataset]=formal.prepare(dataset,log)
        if not states[dataset]['gate_pass']:
            replays[dataset]={'status':'FAIL','target_order':states[dataset]['target_check'],'threshold':states[dataset]['threshold_check']}
            raise GateStop('STOP_B: '+dataset+' Normal first replay gate')
        step('NORMAL_REPLAY_2_'+dataset.upper());repetitions[dataset]=formal.prepare(dataset,log)
        det=real_v5.deterministic_check(states[dataset],repetitions[dataset])
        replays[dataset]={'status':'NORMAL_PASS_TEST_PENDING','target_order':states[dataset]['target_check'],'threshold':states[dataset]['threshold_check'],'second_target_order':repetitions[dataset]['target_check'],'second_threshold':repetitions[dataset]['threshold_check'],'normal_determinism':det}
        dump('replay_check.json',replays)
        if not repetitions[dataset]['gate_pass'] or det['status']!='PASS':raise GateStop('STOP_B: '+dataset+' Normal second replay/determinism gate')
        completed.append(dataset+' two Normal replays and exact threshold/target gate')
    step('CONTROLLED_NORMAL_PREPARE');ctl=controlled_helper.prepare(ROOT);dump('donor_manifest_audit.json',ctl['donor_audit'])
    step('COMUT_NORMAL_PREPARE');cm=comut_helper.prepare(ROOT)
    manifest['all_Normal_fits_complete_utc']=now()
    step('CONTROLLED_FORMAL_S3_GATE');replays['controlled']=controlled_helper.replay_s3(ctl)
    dump('replay_check.json',replays)
    if not replays['controlled']['passed']:raise GateStop('STOP_B: controlled S3 case gate failed')
    completed.append('controlled six-case S3 reproduction')
    def invoke(dataset):
        step('FORMAL_TEST_REPLAY_'+dataset.upper())
        value=real_v5.replay_test(states[dataset],repetitions[dataset],OUT,log)
        actual[dataset]=value;replays[dataset]=value['gates'];dump('replay_check.json',replays)
        if value['passed']:completed.append(dataset+' all S3 gates and shared-source consistency')
        return value
    gated_sequence(('swat','hai'),invoke)
    # All new Test metrics remain prohibited until the Normal-only hash below.
    step('NORMAL_STATS_AND_MATH')
    import stats_v5
    normal_rows=[];skill_rows=[];maths={}
    for dataset,state in states.items():
        stats[dataset]=stats_v5.normal_statistics(state)
        normal_rows+=stats[dataset]['rows']
        frame('normal_reference_'+dataset+'.csv',stats[dataset]['reference_rows'])
        for order,target in enumerate(state['skill_targets']):
            skill_rows.append({'dataset':dataset,'order':order,'target':target,'scope':'SUPPLEMENTARY_SKILL_ONLY','primary_frozen':target in state['targets']})
        maths[dataset]=stats_v5.math_checks(state)
    dump('math_checks.json',maths)
    frame('skill_only_targets.csv',skill_rows)
    skill_sha=sha(OUT/'skill_only_targets.csv')
    (OUT/'skill_only_targets.sha256').write_text(TAG+'\n'+skill_sha+'  skill_only_targets.csv\n',encoding='utf-8')
    manifest['skill_only_targets_frozen_utc']=now();manifest['skill_only_targets_sha256']=skill_sha
    normal_rows+=ctl['normal_stats']+cm.get('normal_stats',[])
    for row in normal_rows:row['normal_statistics_completed_utc']=now()
    frame('normal_stats.csv',normal_rows);normal_sha=sha(OUT/'normal_stats.csv')
    import normal_v6_gate
    equality=normal_v6_gate.compare_to_v5(OUT/'normal_stats.csv',ROOT/'outputs/thesis_validity/p1_residual_controls_v5/normal_stats.csv')
    dump('normal_statistics_v5_comparison.json',equality)
    replays['normal_statistics_v5_comparison']=equality;dump('replay_check.json',replays)
    if equality['status']!='PASS':raise GateStop('STOP: freshly computed Normal statistics differ from v5 after allowed timestamp exclusion')
    manifest['normal_statistics_v5_comparison_passed_utc']=now()
    completed.append('Fresh Normal statistics equal v5 excluding only run timestamp, tolerance zero')
    (OUT/'normal_stats.sha256').write_text(TAG+'\n'+normal_sha+'  normal_stats.csv\n',encoding='utf-8')
    manifest['normal_stats_frozen_utc']=now();manifest['normal_stats_sha256']=normal_sha
    manifest['normal_reference_hashes']={d:sha(OUT/('normal_reference_'+d+'.csv')) for d in states}
    step('NORMAL_HASH_FROZEN')
    completed.append('Normal statistics/hash, skill-only list/hash and mathematical operation B')
    step('CONTROLLED_S1_S2_S3')
    controls,consistency=controlled_helper.evaluate_controls(ctl,normal_sha,OUT)
    frame('controlled_swat.csv',controls);dump('controlled_consistency.json',consistency)
    completed.append('controlled all18 target-score rows and paired consistency')
    import evaluation_v5
    all_metrics=[];sensitivity=[]
    for dataset,state in states.items():
        step('NEW_CONTROLS_'+dataset.upper())
        analysis[dataset]=evaluation_v5.evaluate(state,actual[dataset],normal_sha,OUT,stats[dataset])
        all_metrics+=analysis[dataset]['metrics'];sensitivity+=analysis[dataset]['sensitivity']
        dump('analysis_'+dataset+'.json',analysis[dataset])
        frame('metrics_main.csv',all_metrics);frame('sensitivity_skill_only.csv',sensitivity)
        completed.append(dataset+' three score metrics, target/event tables and bound Parquet scores')
    step('CONDITIONAL_COMUT')
    c=comut_helper.replay_and_evaluate(cm,normal_sha)
    dump('comut_replay.json',{k:v for k,v in c.items() if k not in ('rows','score_frames','per_target_frames')})
    if c['status']=='COMPLETED':
        frame('comut.csv',c['rows'])
        pd.concat(c['score_frames'],ignore_index=True).to_parquet(OUT/'comut_scores.parquet',index=False)
        pd.concat(c['per_target_frames'],ignore_index=True).to_parquet(OUT/'comut_per_target_scores.parquet',index=False)
        completed.append('Historical CoMuT13 frozen cases and score controls')
    else:
        failures.append('Conditional CoMuT skipped: '+c['status']+' '+c.get('reason',''))
    rows=pd.DataFrame(all_metrics);ct=pd.DataFrame(controls)
    hai=rows[rows.dataset=='hai'].set_index('score')
    dh=float(hai.loc['S3','aupr']-hai.loc['S2','aupr'])
    dc=float(ct[ct.score=='S3'].auroc.median()-ct[ct.score=='S2'].auroc.median())
    label='R-UP' if dh>=.02 and dc>=.02 else 'R-DOWN' if dh<=-.02 and dc<=-.02 else 'R-EQ'
    dump('decision.json',{'status':'VALIDITY_P1_COMPLETED','descriptive_label':label,'delta_HAI':dh,'delta_CTRL':dc,'H_kappa':{d:analysis[d]['H_kappa'] for d in analysis},'event_differences':{d:analysis[d]['event_differences'] for d in analysis},'target_differences':{d:analysis[d]['target_differences'] for d in analysis},'interpretation':'Pre-specified descriptive label only; no innovation/novelty conclusion'})
    status='VALIDITY_P1_COMPLETED';step('ANALYSIS_COMPLETE')

def finish():
    global status
    if not active:return
    import provenance_v6 as integrity
    record_inputs()
    if initial:
        preserve=integrity.verify_preservation(initial);dump('preservation_check.json',preserve)
        if preserve['status']!='PASS':status='VALIDITY_P1_INCOMPLETE';failures.append('Protected history integrity check FAIL')
    dump('replay_check.json',replays)
    if not (OUT/'normal_stats.csv').exists():frame('normal_stats.csv',[{'status':'NOT_COMPLETED_NOT_FROZEN','reason':'Stopped before permitted new Test metrics'}])
    if not (OUT/'math_checks.json').exists():dump('math_checks.json',{'status':'NOT_RUN','reason':'Replay gate/exception/deadline stopped preceding steps'})
    if not (OUT/'metrics_main.csv').exists():
        rows=[]
        for dataset in ('swat','hai'):
            for score in ('S1','S2','S3'):
                row={'dataset':dataset,'score':score,'status':'NOT_EVALUATED'}
                if score=='S3' and dataset in actual:row.update(actual[dataset]['metrics'],status='FORMAL_REPLAY_ONLY')
                rows.append(row)
        frame('metrics_main.csv',rows)
    if not (OUT/'controlled_swat.csv').exists():
        rows=[]
        if replays.get('controlled',{}).get('cases'):
            for item in replays['controlled']['cases']:rows.append({'target':item['target'],'score':'S3','status':'REPLAY_ONLY',**item['observed']})
        frame('controlled_swat.csv',rows or [{'status':'NOT_EVALUATED'}])
    if not (OUT/'decision.json').exists():dump('decision.json',{'status':status,'descriptive_label':'NOT_AVAILABLE','delta_HAI':None,'delta_CTRL':None,'H_kappa':{'swat':'NOT_TESTABLE','hai':'NOT_TESTABLE'},'stop_reasons':failures})
    manifest.update(status=status,finished_at_utc=now(),completed_steps=completed,failed_or_skipped=failures)
    dump('manifest.json',manifest)
    decision=json.loads((OUT/'decision.json').read_text(encoding='utf-8'))
    lines=['# Thesis-VALIDITY-P1 v6',TAG,
       '最终状态：'+status+'。本轮是回顾性研究阶段中预先固定的分析方案，不是预注册确认性实验，也不是未见Test的独立验证。',
       '## 来源与复现门禁',
       'v6为唯一执行协议。正式runner只读；Normal-only Ridge按正式函数重新拟合两次。所有拟合完成后才读Attack特征。主实验使用冻结15/54目标且顺序不变。E2-R1D是效率系数与Innovation3B路径重建，不是正式runner逐点存档，因此正式逐点档案项NOT_AVAILABLE；不放宽容差、不切换路径。',
       '两次重放的Validation/Test S3要求字节完全相同。共同预测源检查固定使用本次保存的z/pred_self/pred_cross和正式Normal参考，原始、校准和聚合分歧均以1e-9检查。失败即退出下一数据集Test调用。',
       '详细门禁及重建差异见replay_check.json、shared_source_*.json、archive_difference_*.json和top10 CSV。',
       '## 已完成步骤',*['- '+s for s in completed],
       '## 正常统计与冻结',
       '完整Normal Validation逐目标残差方差/协方差采用ddof=0；每个S1/S2/S3各自median、MAD、q95/q99、sigma下限及max主导性见normal_stats.csv。仅技能列表先于正常统计哈希冻结；新的S1/S2 Test指标必须在冻结后计算。q99/median在median=0时空值，不引入epsilon。',
       '主导比例采用冻结顺序argmax的确定性归属，另报包括并列的比例；只诊断，不调整floor、目标或校准。',
       '新生成normal_stats.csv按v6与v5核验；仅排除normal_statistics_completed_utc运行时间戳列，所有其他列及行顺序逐值精确比较，容差0。具体比较列、哈希与差异见normal_statistics_v5_comparison.json。',
       '目标集合由正式双视角门控选出。8 个正技能条件对 Self 与 Cross 是对称的；但分歧稳定性比值筛选只针对 S3，可能使集合偏向 S3。因此本实验的结论仅限于"相同预测器与相同目标条件下"的比较，不代表 S2 在为其单独优化的目标集上的最佳表现。',
       '## 受控与数学检查',
       '受控部分沿用6目标、每目标2000 frozen donor配对、原calibration区间，逐目标评价，无跨目标聚合。S1门禁是逐对最大差0，AUROC用repr原样保存，并要求与0.5差值不超过1e-12；此为实现一致性检查。S2/S3带符号变化都等于cross(clean)-cross(donor)，区别来自正常基线位置和宽度。',
       '数学操作B只修改Normal序列副本的一个当前值并重建滞后特征；恒等式、目标当前/未来lag和其他目标/全局响应分别记录于math_checks.json。没有改变模型、参考或门控。',
       '## 结果与描述标签',json.dumps(decision,ensure_ascii=False,indent=2),
       'H-κ为描述性关联，不表示因果，不保证AUROC/AUPR；任一分组空时NOT_TESTABLE。描述标签不决定创新成立，不升级S1/S2为正式方法。',
       '## 历史披露',
       'v4因把E2-R1D重建分数当作正式存档触发STOP_B。v4编排在SWaT失败后仍进入HAI，读取了5个Test并完成test1评分后被终止；该执行偏差继续保留。v6显式门禁防止下一数据集调用，旧代码和停止产物均保留。v5的FIT601逐对相等检查通过，但AUROC严格浮点等于0.5断言失败且实际数值未保存；v6按本次先行修订的数值判据全量重跑，不使用旧中间预测。完整性见preservation_check.json。',
       'docs/thesis/v3r_selective_cross_only.md记录：S2类V3R selective cross-only残差采用目标排除、Cross lags [1,5,10,20,30]不含lag0、Top5 predictor；23目标、46受控cell的median AUROC为0.523134，状态SELECTIVE_CROSS_ONLY_NOT_SUPPORTED。双预测分歧是在这之后设计的。',
       '上述历史结果与本轮 S2 的设置不同（predictor 集合、是否含 lag 0、受控扰动方式都不同），二者不可直接比较；本轮 S2 与 S3 使用同一组预测器，目的正是把"评分公式的作用"和"预测器设置变化的作用"分开。',
       '## 未通过、异常或跳过',*['- '+s for s in failures],
       'NO TEST TUNING / NO PRIMARY OVERWRITE / NO METHOD CHANGE / NO NEURAL TRAINING / NO PUSH。审计进程不操作Git；结束后按既有授权只自动本地提交小型报告和脚本。没有修改论文章节或创新点表述。']
    (OUT/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('FINAL_STATUS '+status,flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:
        if not active:raise
        failures.append(type(e).__name__+': '+str(e));manifest['exception_traceback']=traceback.format_exc()
        status='VALIDITY_P1_INCOMPLETE'
        print(traceback.format_exc(),flush=True)
    finally:finish()
