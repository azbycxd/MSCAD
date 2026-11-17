"""RETROSPECTIVE_VALIDITY_AUDIT: v6 controlled S1 numerical consistency.
Only v6 changes: paired equality gate, repr(AUROC), fixed 1e-12 sanity bound.
Formal fits, predictors, references, scores and donor samples remain unchanged.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from controlled_helper import prepare, replay_s3, TAG, TARGETS

S1_AUROC_TOLERANCE=1e-12

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def evaluate_controls(state,normal_freeze_sha,output_dir):
    """Fresh replay state only; persist every completed score before any failure."""
    out=Path(output_dir).resolve()
    if out.name!='p1_residual_controls_v6' or not out.is_dir():
        raise RuntimeError('V6_OUTPUT_ISOLATION_REQUIRED')
    if not state.get('replay') or not state['replay']['passed']:
        raise RuntimeError('STOP_B controlled S3 reproduction did not pass')
    if not isinstance(normal_freeze_sha,str) or len(normal_freeze_sha)!=64:
        raise RuntimeError('NORMAL_STATISTICS_NOT_HASH_FROZEN')
    if _sha(out/'normal_stats.csv')!=normal_freeze_sha:
        raise RuntimeError('NORMAL_STATISTICS_HASH_MISMATCH')
    if tuple(state['cases'])!=TARGETS:
        raise RuntimeError('CONTROLLED_TARGET_ORDER_MISMATCH')
    progress=out/'controlled_consistency_progress.json'
    partial=out/'controlled_swat_partial.csv'
    if progress.exists() or partial.exists():
        raise RuntimeError('Refuse replacing prior controlled v6 attempt')
    b2,d0=state['d1'].b2,state['d1'].d0
    rows,checks=[],[]

    def persist(status,failure=None):
        evidence={'audit_label':TAG,'status':status,'s1_pairwise_tolerance':0.0,
                  's1_auroc_absolute_tolerance':S1_AUROC_TOLERANCE,
                  'normal_stats_sha256':normal_freeze_sha,'checks':checks,
                  'completed_metric_rows':len(rows),'failure':failure}
        progress.write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        if rows:pd.DataFrame(rows).to_csv(partial,index=False)
        if failure:
            (out/'controlled_failure.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    for target,case in state['cases'].items():
        y,p_self,p_cross,p_donor=case['predictions']
        raw={'S1':(np.abs(y-p_self),np.abs(y-p_self)),
             'S2':(np.abs(y-p_cross),np.abs(y-p_donor)),
             'S3':(np.abs(p_self-p_cross),np.abs(p_self-p_donor))}
        pair_delta=float(np.max(np.abs(raw['S1'][0]-raw['S1'][1])))
        signed_s2=(y-p_donor)-(y-p_cross)
        signed_s3=(p_self-p_donor)-(p_self-p_cross)
        signed_delta=float(np.max(np.abs(signed_s2-signed_s3)))
        check={'target':target,'s1_pair_max_abs_difference':pair_delta if np.isfinite(pair_delta) else None,
               's1_pair_max_abs_difference_repr':repr(pair_delta),
               's1_pairwise_gate':'PASS' if pair_delta==0.0 else 'FAIL',
               's1_raw_clean_sha256':hashlib.sha256(raw['S1'][0].tobytes()).hexdigest(),
               's1_raw_corrupt_sha256':hashlib.sha256(raw['S1'][1].tobytes()).hexdigest(),
               's2_s3_signed_change_max_abs_difference':signed_delta if np.isfinite(signed_delta) else None}
        checks.append(check)
        if pair_delta!=0.0:
            failure=f'S1_CONSISTENCY_FLAG: paired difference {target}: {repr(pair_delta)}'
            persist('FAIL',failure)
            raise RuntimeError(failure)
        for score,(clean_raw,corrupt_raw) in raw.items():
            ref=case['references'][score]
            clean=d0.calibrated(clean_raw,ref['median'],ref['scale'])
            corrupt=d0.calibrated(corrupt_raw,ref['median'],ref['scale'])
            metrics=b2.balanced_metrics(clean,corrupt)
            if score=='S1':
                auc=float(metrics['auroc']);delta=abs(auc-.5)
                check.update(s1_auroc=auc if np.isfinite(auc) else None,
                             s1_auroc_repr=repr(auc),
                             s1_auroc_minus_half_repr=repr(auc-.5),
                             s1_auroc_abs_difference=delta if np.isfinite(delta) else None,
                             s1_auroc_numerical_gate='PASS' if delta<=S1_AUROC_TOLERANCE else 'FAIL')
                if not np.isfinite(auc) or delta>S1_AUROC_TOLERANCE:
                    failure=f'S1_CONSISTENCY_FLAG: AUROC {target}: actual={repr(auc)}, abs_difference={repr(delta)}'
                    persist('FAIL',failure)
                    raise RuntimeError(failure)
            rows.append({'audit_label':TAG,'target':target,'stage':case['stage'],'score':score,
                         'auroc':metrics['auroc'],'aupr':metrics['aupr'],
                         'auroc_repr':repr(float(metrics['auroc'])),
                         'recall_q99':float(np.mean(corrupt>ref['threshold'])),
                         'fpr_q99':float(np.mean(clean>ref['threshold'])),'threshold':ref['threshold'],
                         'cross_sources':';'.join(case['source_names']),'sample_pairs':len(clean),
                         'normal_median':ref['median'],'normal_mad':ref['mad'],
                         'normal_scaled_mad':ref['scaled_mad'],'normal_scale':ref['scale'],
                         'normal_q95':ref['q95'],'normal_q99':ref['q99'],
                         's1_pair_max_abs_difference':pair_delta,
                         's1_auroc_repr':check.get('s1_auroc_repr'),
                         's1_auroc_abs_difference':check.get('s1_auroc_abs_difference'),
                         's1_pairwise_gate':check['s1_pairwise_gate'],
                         's1_auroc_numerical_gate':check.get('s1_auroc_numerical_gate'),
                         'normal_stats_sha256':normal_freeze_sha})
            persist('RUNNING')
    persist('PASS')
    return rows,{'audit_label':TAG,'s1_consistency':'PASS','s1_pairwise_tolerance':0.0,
                 's1_auroc_absolute_tolerance':S1_AUROC_TOLERANCE,'checks':checks,
                 'normal_stats_sha256':normal_freeze_sha}
