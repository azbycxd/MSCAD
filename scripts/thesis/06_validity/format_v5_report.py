"""RETROSPECTIVE_VALIDITY_AUDIT: format existing completed audit facts only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/thesis_validity/p1_residual_controls_v5'
sys.dont_write_bytecode=True
import pandas as pd
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def md(frame):
    def fmt(value):
        if pd.isna(value):return 'N/A'
        if isinstance(value,float):return f'{value:.6f}'
        return str(value).replace('|','/')
    return '\n'.join(['|'+'|'.join(frame.columns)+'|','|'+'|'.join(['---']*len(frame.columns))+'|']+['|'+'|'.join(fmt(x) for x in row)+'|' for row in frame.itertuples(index=False,name=None)])
m=read('manifest.json')
if m['status']!='VALIDITY_P1_COMPLETED':raise SystemExit('No completed-results formatting on stopped run')
p=OUT/'REPORT.md';text=p.read_text(encoding='utf-8')
marker='## 已冻结结果数值摘录'
if marker in text:raise RuntimeError('Refuse duplicated result formatting')
metrics=pd.read_csv(OUT/'metrics_main.csv')
controls=pd.read_csv(OUT/'controlled_swat.csv')
normal=pd.read_csv(OUT/'normal_stats.csv')
maths=read('math_checks.json')
normal=normal[normal.main_frozen_target.astype(str).str.lower().eq('true')]
dominance=[]
for (dataset,score),rows in normal.groupby(['dataset','score'],sort=False):
    winner=rows.loc[rows.normal_max_dominance_fraction.idxmax()]
    dominance.append({'dataset':dataset,'score':score,'q99':winner.aggregate_q99_threshold,'floor_targets':int(rows.scale_at_floor.astype(str).str.lower().eq('true').sum()),'target_count':len(rows),'largest_dominance_target':winner.target,'dominance_fraction':winner.normal_max_dominance_fraction,'dominant_sigma':winner.sigma})
math_rows=[]
for dataset,value in maths.items():
    if dataset=='audit_label':continue
    math_rows.append({'dataset':dataset,'target':value['target'],'perturbation':value['perturbation_5MAD'],'identity_max_diff':value['identity_all_normal_endpoints']['max_absolute_difference'],'own_current_S3_delta':value['own_current']['S3']['delta'],'global_S3_delta':value['global_S3_at_t0']['delta']})
segments=[marker,
 '仅汇总已经完成、正常统计哈希冻结后的结果；没有新增评分方案或结果驱动选择。',
 md(metrics[['dataset','score','auroc','aupr','precision','recall','f1','fpr','threshold','event_hit_count']]),
 '受控6个目标的中位AUROC/AUPR：',md(controls.groupby('score',sort=False)[['auroc','aupr']].median().reset_index()),
 'Normal-only尺度下限与max主导性（不改变校准）：',md(pd.DataFrame(dominance)),
 '数学性质：当前目标和全局聚合分开记录。微小恒等式误差按浮点实测保留；若正常MAD为0，不添加人为扰动下限。',md(pd.DataFrame(math_rows))]
validation=read('artifact_validation.json')
segments+=['产物校验：'+validation['status']+f"（{validation['passed']}/{validation['total']}）。"]
p.write_text(text+'\n\n'+'\n\n'.join(segments)+'\n',encoding='utf-8')
print('REPORT_NUMERIC_SUMMARY_ADDED_FROM_EXISTING_ARTIFACTS')
