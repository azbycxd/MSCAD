"""Normal-only statistics and operation-B checks; no I/O or fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import real_helper

TAG = real_helper.TAG


def _robust(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, max(1.4826 * mad, 1e-6)


def _frozen_predictions(prepared):
    ids = [prepared['all_targets'].index(t) for t in prepared['targets']]
    return {key: np.concatenate([p[key][:, ids] for p in prepared['normal_predictions']])
            for key in ('z', 'ps', 'pc')}


def normal_statistics(prepared):
    """Return tables, exact references and arrays on full legal Normal Validation.

    S3 references and aggregate threshold retain the formal multi-output replay
    arithmetic. The per-target-forward discrepancy is explicitly reported.
    No Test values or labels are accepted by this function.
    """
    pred = _frozen_predictions(prepared)
    rs, rc = pred['z'] - pred['ps'], pred['z'] - pred['pc']
    forward_s3 = np.abs(pred['ps'] - pred['pc'])
    raw = {'S1': np.abs(rs), 'S2': np.abs(rc),
           'S3': np.column_stack([prepared['full_disagreements'][t] for t in prepared['targets']])}
    for values in list(raw.values()) + [rs, rc, forward_s3]:
        if not np.isfinite(values).all():
            raise ValueError('Non-finite Normal-only prediction or score')
    references = {score: {} for score in raw}
    rows, reference_rows = [], []
    for j, target in enumerate(prepared['targets']):
        rsc, rcc = rs[:, j] - rs[:, j].mean(), rc[:, j] - rc[:, j].mean()
        vs, vc = float(np.mean(rsc * rsc)), float(np.mean(rcc * rcc))
        cov = float(np.mean(rsc * rcc))
        row = {'audit_tag': TAG, 'dataset': prepared['dataset'], 'target': target,
               'endpoint_count': len(rs), 'variance_ddof': 0,
               'var_r_self': vs, 'var_r_cross': vc, 'cov_r_cross_r_self': cov,
               'kappa': cov - 0.5 * vs,
               'var_residual_difference': float(np.var(rc[:, j] - rs[:, j])),
               's3_formal_vs_per_target_forward_max_abs_diff': float(np.max(np.abs(raw['S3'][:, j] - forward_s3[:, j])))}
        for score, values in raw.items():
            median, mad, scale = _robust(values[:, j])
            q95, q99 = (float(v) for v in np.quantile(values[:, j], [.95, .99]))
            if score == 'S3':
                reference_median, reference_scale = (float(v) for v in prepared['references'][target])
                if median != reference_median or scale != reference_scale:
                    raise ValueError('Formal S3 reference differs from formal full-disagreement array')
            references[score][target] = (median, scale)
            row.update({f'{score}_median': median, f'{score}_MAD': mad,
                        f'{score}_1_4826_MAD': 1.4826 * mad, f'{score}_scale': scale,
                        f'{score}_q95': q95, f'{score}_q99': q99,
                        f'{score}_q99_over_median': q99 / median if median != 0 else None})
            reference_rows.append({'audit_tag': TAG, 'dataset': prepared['dataset'],
                                   'score': score, 'target': target, 'median': median,
                                   'MAD': mad, 'scaled_MAD': 1.4826 * mad, 'scale': scale,
                                   'reference_split': 'full_legal_Normal_Validation'})
        rows.append(row)
    normal_scores, thresholds = {}, {}
    for score, values in raw.items():
        med = np.array([references[score][t][0] for t in prepared['targets']])
        scale = np.array([references[score][t][1] for t in prepared['targets']])
        normal_scores[score] = np.maximum(0., (values - med) / scale).max(axis=1)
        thresholds[score] = float(np.quantile(normal_scores[score], .99))
    # The exact formal runner's per-target forward is the S3 aggregate used for
    # frozen operating points. Report, never conceal, its floating point delta.
    formal_s3 = np.concatenate(prepared['validation_s3'])
    s3_arithmetic_delta = float(np.max(np.abs(normal_scores['S3'] - formal_s3)))
    normal_scores['S3'] = formal_s3
    thresholds['S3'] = float(prepared['threshold'])
    for row in reference_rows:
        row['aggregate_q99_threshold'] = thresholds[row['score']]
        row['global_q99'] = thresholds[row['score']]
    return {'table': pd.DataFrame(rows), 'rows': rows, 'references': references,
            'reference_rows': reference_rows,
            'reference_table': pd.DataFrame(reference_rows), 'thresholds': thresholds,
            'normal_scores': normal_scores, 'raw_scores': raw,
            'target_order': list(prepared['targets']), 'prediction': pred,
            's3_formal_aggregate_arithmetic_max_abs_delta': s3_arithmetic_delta,
            'reference_scope': 'full_legal_Normal_Validation', 'audit_tag': TAG}


def math_checks(prepared):
    """Operation B on one Normal segment; every affected design is rebuilt."""
    pred = _frozen_predictions(prepared)
    rs, rc = pred['z'] - pred['ps'], pred['z'] - pred['pc']
    identity = np.abs(np.abs(pred['ps'] - pred['pc']) - np.abs(rc - rs))
    records = prepared['records']
    # Selection is deterministic and depends only on frozen Normal metadata.
    selected = next((r for r in records if any(r['target_index'] in k['source_indices']
                    for k in records if k['target'] != r['target'])), records[0])
    source_targets = [r for r in records if selected['target_index'] in r['source_indices']
                      and r['target'] != selected['target']]
    segment_id = next(i for i, v in enumerate(prepared['validation']) if len(v) >= 91)
    clean = prepared['validation'][segment_id]
    t0 = max(30, min(len(clean) // 2, len(clean) - 31))
    j = selected['target_index']
    channel = np.concatenate([v[:, j] for v in prepared['validation']])
    channel_median = float(np.median(channel))
    channel_mad = float(np.median(np.abs(channel - channel_median)))
    delta = 5. * channel_mad
    corrupt = clean.copy()
    corrupt[t0, j] += delta
    times = np.arange(t0, t0 + 31, dtype=np.int64)
    w, runner = prepared['helper'], prepared['runner']

    def forward(values):
        zs, ps, pc = [], [], []
        for row in records:
            ti = row['target_index']
            sm, cm = prepared['models'][row['target']]
            zs.append(values[times, ti])
            ps.append(sm.predict(w.design(values, times, [(ti, lag) for lag in runner.SELF_LAGS]))[:, 0])
            pc.append(cm.predict(w.design(values, times, [(si, lag) for si in row['source_indices'] for lag in runner.CROSS_LAGS]))[:, 0])
        z, s, c = (np.column_stack(v) for v in (zs, ps, pc))
        return {'S1': np.abs(z-s), 'S2': np.abs(z-c), 'S3': np.abs(s-c)}

    before, after = forward(clean), forward(corrupt)
    col = prepared['targets'].index(selected['target'])

    def change(score, offset, index):
        a, b = float(before[score][offset, index]), float(after[score][offset, index])
        return {'before': a, 'after': b, 'delta': b-a, 'absolute_delta': abs(b-a)}

    med = np.array([prepared['references'][t][0] for t in prepared['targets']])
    scale = np.array([prepared['references'][t][1] for t in prepared['targets']])
    global_before = float(np.max(np.maximum(0., (before['S3'][0] - med) / scale)))
    global_after = float(np.max(np.maximum(0., (after['S3'][0] - med) / scale)))
    return {'audit_tag': TAG, 'dataset': prepared['dataset'], 'split': 'Normal_Validation_only',
            'operation': 'B: single standardized raw-channel value modified; original design rebuilt',
            'target': selected['target'], 'segment_index': segment_id, 't0_local_index': t0,
            't0_minus30_same_segment': t0-30 >= 0, 't0_plus30_same_segment': t0+30 < len(clean),
            'channel_mad_scope': 'complete_Normal_Validation_standardized_raw_channel',
            'channel_median': channel_median, 'channel_MAD': channel_mad,
            'perturbation_5MAD': delta, 'effective_perturbation': delta != 0,
            'changed_raw_value_count': int(np.count_nonzero(corrupt != clean)),
            'identity_all_normal_endpoints': {'endpoint_count': len(identity), 'target_count': identity.shape[1],
                'max_absolute_difference': float(identity.max()),
                'per_target_max_absolute_difference': {t: float(identity[:, k].max()) for k, t in enumerate(prepared['targets'])}},
            'own_current': {score: change(score, 0, col) for score in ('S1', 'S2', 'S3')},
            'own_lagged_S3': {str(lag): change('S3', lag, col) for lag in (1, 5, 10, 20, 30)},
            'other_reliable_cross_source_targets_at_t0': {r['target']: change('S3', 0, prepared['targets'].index(r['target'])) for r in source_targets},
            'cross_source_selection_available': bool(source_targets),
            'global_S3_at_t0': {'before': global_before, 'after': global_after,
                                'delta': global_after-global_before, 'absolute_delta': abs(global_after-global_before)},
            'model_scaler_targets_references_changed': False, 'new_fitting': False,
            'test_data_used': False, 'interpretation': 'MATHEMATICAL_DIAGNOSTIC_ONLY'}
