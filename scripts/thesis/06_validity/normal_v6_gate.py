"""RETROSPECTIVE_VALIDITY_AUDIT: exact v6-v5 Normal statistic binding gate.

This module reads only the two statistical CSVs and the frozen v5 SHA file.
It never changes/reuses values, fits a model, or reads Test observations.
"""
from __future__ import annotations
import csv
import hashlib
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

TAG = 'RETROSPECTIVE_VALIDITY_AUDIT'
IGNORABLE_COLUMNS = ('normal_statistics_completed_utc',)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def _read(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)

def _decimal(value):
    if value is None or value == '':
        return None
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None

def compare_to_v5(v6_csv, v5_csv):
    """Require frozen v5 SHA authenticity and exact values excluding one timestamp.

    Numeric tokens are compared as exact Decimals, avoiding float-parser rounding
    that could collapse distinct CSV numbers. Non-numeric tokens compare exactly.
    Row order and every header column/order remain binding; no sorting is done.
    """
    v6_csv, v5_csv = Path(v6_csv), Path(v5_csv)
    hash_file = v5_csv.with_suffix('.sha256')
    missing = [str(p) for p in (v6_csv, v5_csv, hash_file) if not p.is_file()]
    if missing:
        return {'audit_label':TAG,'status':'FAIL','reason':'Missing Normal binding input','missing':missing}
    declared = re.findall(r'(?im)^([0-9a-f]{64})\s+normal_stats\.csv\s*$', hash_file.read_text(encoding='utf-8-sig'))
    old_sha, new_sha = sha(v5_csv), sha(v6_csv)
    result = {'audit_label':TAG,'status':'FAIL','v5_path':str(v5_csv),'v6_path':str(v6_csv),
              'v5_sha256':old_sha,'v6_sha256':new_sha,'v5_declared_sha256':declared[0].lower() if len(declared)==1 else None,
              'v5_hash_file':str(hash_file),'v5_hash_file_sha256':sha(hash_file),'tolerance':0,
              'comparison_arithmetic':'exact Decimal for numeric CSV tokens; exact strings otherwise',
              'values_copied_from_v5':False}
    if len(declared)!=1 or declared[0].lower()!=old_sha:
        result['reason']='V5_NORMAL_HASH_BINDING_FAILED'
        return result
    c5,r5=_read(v5_csv); c6,r6=_read(v6_csv)
    ignored=[c for c in IGNORABLE_COLUMNS if c in c5 and c in c6]
    compared=[c for c in c5 if c not in ignored]
    result.update(v5_hash_verified=True,full_sha256_identical=old_sha==new_sha,
                  ignored_columns=ignored,compared_columns=compared,
                  v5_columns=c5,v6_columns=c6,v5_row_count=len(r5),v6_row_count=len(r6),
                  header_and_order_equal=c5==c6,row_count_equal=len(r5)==len(r6))
    if c5!=c6 or len(r5)!=len(r6):
        result['reason']='NORMAL_SCHEMA_OR_ROW_COUNT_MISMATCH'
        return result
    mismatch_count=0;examples=[];numeric_max=Decimal(0);numeric_columns=set();numeric_count=0;string_count=0
    timestamp_differences={c:0 for c in ignored}
    for i,(old,new) in enumerate(zip(r5,r6)):
        for c in ignored:
            timestamp_differences[c] += old[c]!=new[c]
        for c in compared:
            a,b=old[c],new[c]
            da,db=_decimal(a),_decimal(b)
            if da is not None and db is not None:
                numeric_columns.add(c);numeric_count+=1
                difference=abs(da-db);numeric_max=max(numeric_max,difference)
                equal=difference==0
                detail={'absolute_numeric_difference':str(difference)}
            else:
                string_count+=1;equal=a==b
                detail={'comparison':'exact string/empty-field equality'}
            if not equal:
                mismatch_count+=1
                if len(examples)<20:
                    examples.append({'row_index':i,'csv_line':i+2,'column':c,'v5':a,'v6':b,**detail})
    passed=mismatch_count==0 and (old_sha==new_sha or bool(ignored))
    result.update(status='PASS' if passed else 'FAIL',
                  mode='SHA256_IDENTICAL' if old_sha==new_sha else 'EXACT_VALUES_EXCLUDING_ALLOWLISTED_TIMESTAMP',
                  excluded_timestamp_difference_counts=timestamp_differences,
                  numeric_compared_columns=[c for c in compared if c in numeric_columns],
                  numeric_cell_count=numeric_count,non_numeric_cell_count=string_count,
                  difference_count=mismatch_count,
                  numeric_max_abs_difference_decimal=str(numeric_max),
                  numeric_max_abs_difference=float(numeric_max),
                  first_difference=examples[0] if examples else None,
                  first_differences=examples,
                  reason='Exact Normal values unchanged; only runtime timestamp excluded' if passed and old_sha!=new_sha else
                         'Identical complete CSV SHA256' if passed else 'NORMAL_VALUES_MISMATCH')
    return result
