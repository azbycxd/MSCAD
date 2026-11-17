"""RETROSPECTIVE_VALIDITY_AUDIT: v5 Normal-only references and dominance."""
from __future__ import annotations
import numpy as np
import pandas as pd
from audit_stats import math_checks
TAG='RETROSPECTIVE_VALIDITY_AUDIT'


def normal_statistics_v5(prepared):
    main=list(prepared['targets']); skill=list(prepared['skill_targets'])
    relevant=[t for t in prepared['all_targets'] if t in set(main)|set(skill)]
    refs={s:{} for s in ('S1','S2','S3')}; details={s:{} for s in refs}
    normal_cal={s:{} for s in refs}; rows=[]; kappas={}
    for target in relevant:
        j=prepared['all_targets'].index(target)
        z=np.concatenate([x['z'][:,j] for x in prepared['val_predictions']])
        ps=np.concatenate([x['ps'][:,j] for x in prepared['val_predictions']])
        pc=np.concatenate([x['pc'][:,j] for x in prepared['val_predictions']])
        rs,rc=z-ps,z-pc
        vs,vc=float(np.var(rs)),float(np.var(rc))
        cov=float(np.mean((rs-rs.mean())*(rc-rc.mean())))
        kappas[target]=cov-.5*vs
        raw={'S1':np.abs(rs),'S2':np.abs(rc),'S3':np.asarray(prepared['full_disagreements'][target])}
        for score,values in raw.items():
            if not np.isfinite(values).all():raise ValueError('Nonfinite Normal score')
            median=float(np.median(values));mad=float(np.median(np.abs(values-median)))
            sigma=max(1.4826*mad,1e-6)
            if score=='S3' and target in main:
                formal=tuple(float(v) for v in prepared['references'][target])
                if formal!=(median,sigma):raise ValueError('S3 frozen reference discrepancy')
                median,sigma=formal
            q95,q99=map(float,np.quantile(values,[.95,.99]))
            # References retain formal grouped arithmetic; dominance follows scalar forward.
            dominance_raw=np.abs(ps-pc) if score=='S3' else values
            normalized=np.maximum(0.,(dominance_raw-median)/sigma)
            refs[score][target]=(median,sigma)
            detail={'median':median,'mad':mad,'scaled_mad':1.4826*mad,'scale':sigma,'sigma':sigma,'scale_at_floor':sigma==1e-6,
                    'q95':q95,'q99':q99,'q99_over_median':q99/median if median else None,
                    'individual_robust_z_q99':float(np.quantile(normalized,.99))}
            details[score][target]=detail;normal_cal[score][target]=normalized
            rows.append({'audit_label':TAG,'dataset':prepared['dataset'],'target':target,'score':score,
                         'main_frozen_target':target in main,'skill_only_target':target in skill,'scope':'FULL_LEGAL_NORMAL_VALIDATION',
                         'n':len(values),'variance_ddof':0,'var_self':vs,'var_cross':vc,'cov_cross_self':cov,'kappa':kappas[target],
                         's3_reference_source':'FORMAL_GROUP_FORWARD' if score=='S3' else 'SHARED_PREDICTIONS_NORMAL_ONLY',**detail})
    thresholds={};skill_thresholds={};norm_scores={}
    for score in refs:
        matrix=np.column_stack([normal_cal[score][t] for t in main])
        maxima=matrix.max(axis=1);winner=np.argmax(matrix,axis=1)
        for j,target in enumerate(main):
            row=next(r for r in rows if r['target']==target and r['score']==score)
            row['normal_max_dominance_fraction']=float(np.mean(winner==j))
            row['normal_max_tied_fraction']=float(np.mean(matrix[:,j]==maxima))
            row['dominance_tie_policy']='argmax picks first frozen target; inclusive tied fraction reported separately'
        norm_scores[score]=maxima;thresholds[score]=float(np.quantile(maxima,.99))
        if skill:
            skillmax=np.column_stack([normal_cal[score][t] for t in skill]).max(axis=1)
            skill_thresholds[score]=float(np.quantile(skillmax,.99))
    # Frozen S3 threshold uses the formal scalar scorer path, never a newly calibrated replacement.
    s3_delta=float(np.max(np.abs(norm_scores['S3']-np.concatenate(prepared['validation_s3']))))
    norm_scores['S3']=np.concatenate(prepared['validation_s3'])
    thresholds['S3']=float(prepared['threshold'])
    reference_rows=[]
    for row in rows:
        row['aggregate_q99_threshold']=thresholds[row['score']] if row['main_frozen_target'] else None
        row['skill_only_aggregate_q99_threshold']=skill_thresholds.get(row['score']) if row['skill_only_target'] else None
        reference_rows.append(dict(row))
    return {'audit_label':TAG,'rows':rows,'table':pd.DataFrame(rows),'reference_rows':reference_rows,'references':refs,'details':details,
            'thresholds':thresholds,'skill_thresholds':skill_thresholds,'normal_scores':norm_scores,'normal_calibrated':normal_cal,
            'target_order':main,'skill_target_order':skill,'kappa':kappas,
            's3_formal_aggregate_arithmetic_max_abs_delta':s3_delta,
            'dominance_scope':'PRIMARY_FROZEN_TARGETS_ONLY','dominance_tie_policy':'first frozen target argmax; inclusive tie fraction also reported'}

normal_statistics=normal_statistics_v5