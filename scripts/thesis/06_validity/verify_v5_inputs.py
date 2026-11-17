"""RETROSPECTIVE_VALIDITY_AUDIT: read-only input and v4 preservation checks.

No model imports, fitting, Test parsing, metrics, or file writes occur here.
The caller owns writing this JSON-safe snapshot into the new v5 output only.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TAG = 'RETROSPECTIVE_VALIDITY_AUDIT'
OLD_CODE = ('run_validity_p1.py','real_helper.py','controlled_helper.py',
            'comut_helper.py','audit_stats.py','real_evaluation.py',
            'finalize_stopped_audit.py','provenance_report.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _git(root, *args):
    result = subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode:
        raise RuntimeError('Git read failed: '+result.stderr)
    return result.stdout.strip()


def _record(path):
    path = Path(path)
    return {'path':str(path),'exists':path.is_file(),
            'sha256':sha(path) if path.is_file() else None,
            'bytes':path.stat().st_size if path.is_file() else None}


def snapshot(root):
    root = Path(root).resolve()
    old_output = root/'outputs/thesis_validity/p1_residual_controls_v3'
    old_path = old_output/'provenance_report.json'
    if not old_path.is_file():
        raise FileNotFoundError('STOP_A missing v4 provenance report')
    old = json.loads(old_path.read_text(encoding='utf-8-sig'))
    publication = root.parent/'mscad-github'
    source_git = {'path':str(root),'branch':_git(root,'branch','--show-current'),
                  'head':_git(root,'rev-parse','HEAD'),'remote':_git(root,'remote','-v'),
                  'tracked_diff_stat':_git(root,'diff','--stat'),
                  'status_short':_git(root,'status','--short')}
    publication_git = {'path':str(publication),'branch':_git(publication,'branch','--show-current'),
                       'head':_git(publication,'rev-parse','HEAD'),'remote':_git(publication,'remote','-v'),
                       'status_short':_git(publication,'status','--short')}
    source_checks, artifacts, normals = [], [], []
    for section, destination in [('sources',source_checks),('artifacts',artifacts),('normal_data',normals)]:
        for item in old[section]:
            path = Path(item['path'])
            path = path if path.is_absolute() else root/path
            row = _record(path)
            row['expected_v4_sha256'] = item['sha256']
            row['matches_v4'] = row['sha256'] == item['sha256']
            if section == 'sources':
                pubpath = publication/Path(item['path'])
                row['publication_sha256'] = sha(pubpath) if pubpath.is_file() else None
                row['matches_publication'] = row['publication_sha256'] == row['sha256']
            destination.append(row)
    # Additional required frozen metadata and code imported by the formal chain.
    extra_names = [
        'outputs/thesis_t2_swat2015_audit/schema_profile.json',
        'outputs/thesis_t25_swat2015_protocol/canonical_attack_events.csv',
        'outputs/thesis_t25_swat2015_protocol/segment_manifest.json',
        'outputs/thesis_hai21_a0_audit/variable_schema.csv',
        'outputs/thesis_hai21_a0_audit/process_groups.csv',
        'outputs/thesis_hai21_a0_audit/attack_events_50.csv',
        'outputs/thesis_hai21_b1_real/high_collaboration_events_frozen.csv',
        'outputs/thesis_innovation2a2_joint_lagged/stable_targets.csv',
        'outputs/thesis_innovation2d1_swat_controlled/normal_sanity.csv',
        'scripts/thesis/02_collaboration/run_innovation2c0b_comut_structure.py',
        'outputs/thesis_innovation2d0_dual_view_comut/case_metrics.csv',
        'outputs/thesis_innovation2d0_dual_view_comut/manifest.json',
    ]
    extra = [_record(root/name) for name in extra_names]
    # Before all Normal fits, raw Attack inputs are checked for existence only.
    attack = [{'path':str(root/item['path']),'exists':(root/item['path']).is_file(),
               'semantic_content_read':False,'hash_deferred_until_fits_complete':True}
              for item in old['attack_input_existence_only']]
    v4_files = [_record(p) for p in sorted(old_output.rglob('*')) if p.is_file()]
    v4_code = [_record(root/'scripts/thesis/06_validity'/name) for name in OLD_CODE]
    all_checks = source_checks+artifacts+normals
    missing = [row['path'] for row in all_checks+extra+attack+v4_code if not row['exists']]
    mismatches = [row['path'] for row in all_checks if not row['matches_v4']]
    mismatches += [row['path'] for row in source_checks if not row['matches_publication']]
    archives = []
    for original in old['archives']:
        archives.append({**original,'formal_runner_archive_status':'NOT_AVAILABLE',
            'v5_use':'ENDPOINT_BINDING_AND_RECONSTRUCTED_SCORE_DIAGNOSTIC_ONLY',
            'formal_pointwise_gate':False,
            'reason':'E2-R1D reconstruction used saved efficiency coefficients and Innovation3B path; not the formal final-unified runner output'})
    attachment = Path(r'C:\Users\jiusi\Downloads\Thesis-VALIDITY-P1_Codex任务说明_v5.md')
    return {'audit_label':TAG,'status':'PASS' if not missing and not mismatches else 'FAIL',
        'captured_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_git':source_git,'publication_git':publication_git,
        'protocol':_record(attachment),'sources':source_checks,'artifacts':artifacts,
        'normal_data':normals,'additional_inputs':extra,'attack_input_existence_only':attack,
        'archives':archives,'v4_preservation':{'output_files':v4_files,'code_files':v4_code},
        'missing':missing,'mismatches':mismatches,
        'source_history_note':old['source_history_note'],
        'manifest_provenance_limit':old['manifest_provenance_limit'],
        'shared_source_gate_tolerance':1e-9,'determinism_tolerance':0.0,
        'neural_training':False,'test_metrics_computed':False}


def verify_preservation(initial):
    rows = []
    protected = (initial['v4_preservation']['output_files']+
                 initial['v4_preservation']['code_files']+initial['sources']+
                 initial['artifacts']+initial['normal_data']+initial['additional_inputs'])
    for old in protected:
        p = Path(old['path'])
        current = sha(p) if p.is_file() else None
        rows.append({'path':str(p),'before_sha256':old['sha256'],'after_sha256':current,
                     'unchanged':current==old['sha256']})
    before_names = {p['path'] for p in initial['v4_preservation']['output_files']}
    if before_names:
        root = Path(next(iter(before_names))).parent
        # The original output is flat today; derive its known v4 root explicitly.
        while root.name != 'p1_residual_controls_v3' and root != root.parent:
            root = root.parent
        after_names = {str(p) for p in root.rglob('*') if p.is_file()}
    else:
        after_names=set()
    return {'audit_label':TAG,'status':'PASS' if all(r['unchanged'] for r in rows) and before_names==after_names else 'FAIL',
            'checked_at_utc':datetime.now(timezone.utc).isoformat(),'checks':rows,
            'v4_added_files':sorted(after_names-before_names),'v4_removed_files':sorted(before_names-after_names)}
