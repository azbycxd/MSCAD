"""RETROSPECTIVE_VALIDITY_AUDIT: v6 input snapshot plus v4/v5 preservation."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import verify_v5_inputs as old

TAG='RETROSPECTIVE_VALIDITY_AUDIT'
EXCLUDED_DIRS={'_deps','_tmp'}
V6_ATTACHMENT=Path(r'C:\Users\jiusi\.codex\attachments\ccf277d8-83ee-4d0c-8e1b-29f254f64f73\已粘贴的文本.txt')

def _v5_files(folder):
    return [p for p in sorted(folder.rglob('*')) if p.is_file()
            and not (set(p.relative_to(folder).parts)&EXCLUDED_DIRS)]

def snapshot(root):
    root=Path(root).resolve()
    value=old.snapshot(root)
    value['protocol_v5_historical']=value['protocol']
    value['protocol']=old._record(V6_ATTACHMENT)
    value['protocol_version']='v6'
    v5=root/'outputs/thesis_validity/p1_residual_controls_v5'
    code=root/'scripts/thesis/06_validity'
    output_records=[old._record(p) for p in _v5_files(v5)]
    code_records=[old._record(p) for p in sorted(code.iterdir()) if p.is_file() and 'v6' not in p.name.lower()]
    required=[v5/n for n in ('normal_stats.csv','normal_stats.sha256','replay_check.json','decision.json','math_checks.json')]
    missing=[str(p) for p in required if not p.is_file()]
    if not value['protocol']['exists']:missing.append(str(V6_ATTACHMENT))
    value['v5_preservation']={'root':str(v5),'output_files':output_records,
                              'pre_v6_code_files':code_records,'excluded_directory_names':sorted(EXCLUDED_DIRS)}
    value['v6_added_missing_inputs']=missing
    value['status']='PASS' if value['status']=='PASS' and not missing else 'FAIL'
    value['fresh_execution_requirement']='No v5 intermediate model/prediction/statistic reuse; v5 statistics read solely for the mandated equality gate.'
    value['captured_at_utc']=datetime.now(timezone.utc).isoformat()
    return value

def verify_preservation(initial):
    value=old.verify_preservation(initial)
    rows=[]
    records=initial['v5_preservation']['output_files']+initial['v5_preservation']['pre_v6_code_files']
    for record in records:
        path=Path(record['path'])
        current=old.sha(path) if path.is_file() else None
        rows.append({'path':str(path),'before_sha256':record['sha256'],'after_sha256':current,
                     'unchanged':current==record['sha256']})
    folder=Path(initial['v5_preservation']['root'])
    before={r['path'] for r in initial['v5_preservation']['output_files']}
    after={str(p) for p in _v5_files(folder)}
    value['v5_checks']=rows
    value['v5_added_files']=sorted(after-before)
    value['v5_removed_files']=sorted(before-after)
    value['status']='PASS' if value['status']=='PASS' and all(r['unchanged'] for r in rows) and before==after else 'FAIL'
    value['v5_excluded_directory_names']=sorted(EXCLUDED_DIRS)
    return value
