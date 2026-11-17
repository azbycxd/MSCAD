from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.hai_structure_v2a1r_cache_backfill_and_grouped_aggregation as stage
from utils.dynamic_collaborative_corruptions import (
    replace_channels,
    source_block_permutation,
    source_temporal_shift,
    target_endpoint_spike,
    target_short_level_shift,
)


ROOT = Path(__file__).resolve().parents[1]


def source():
    return inspect.getsource(stage).lower()


def test_01_hai_test_not_loaded():
    assert "test1.csv" not in source() and "test2.csv" not in source()


def test_02_attack_label_not_used():
    loader = inspect.getsource(stage.load_validation_only).lower()
    assert "attack" not in loader and "label" not in loader


def test_03_ae_and_rec_not_loaded():
    assert "torch.load" not in source() and "rec_score" not in source()


def test_04_ridge_not_refit():
    assert "fit_ridge" not in source() and ".fit(" not in source()


def test_05_relation_pool_is_frozen():
    pool = pd.read_csv(stage.V2A / "selected_dynamic_relation_pool.csv")
    assert len(pool) == 12


def test_06_dependency_reference_is_read_only():
    assert "dynamic_dependency_reference_registry.csv" in source()
    assert "robust_reference" not in source()


def test_07_original_parameters_are_fixed():
    cfg = stage.load_frozen_config()
    assert cfg["seed"] == 1 and cfg["rolling_dependency_window"] == 16


def test_08_source_window_fold_is_reused():
    frame = stage.source_windows()
    assert frame.source_window_id.is_unique and set(frame.fold) == {"A", "B"}


def test_09_r1_r2_preserve_source_multiset():
    x = np.arange(96, dtype=float).reshape(48, 2)
    assert np.array_equal(np.sort(x[:, 0]), np.sort(source_temporal_shift(x, 0)[:, 0]))
    assert np.array_equal(np.sort(x[:, 0]), np.sort(source_block_permutation(x, 0)[:, 0]))


def test_10_replacement_must_be_same_sequence_and_nonoverlap():
    manifest = pd.read_csv(stage.V2A / "controlled_dynamic_corruption_manifest.csv")
    assert manifest.same_sequence.all() and manifest.nonoverlap.all()


def test_11_p1_p2_rules_match_v2a():
    x = np.zeros((48, 3))
    p1 = target_endpoint_spike(x, 1, 1.0)
    p2 = target_short_level_shift(x, 1, 1.0)
    assert np.flatnonzero(p1[:, 1]).tolist() == [47]
    assert np.flatnonzero(p2[:, 1]).tolist() == list(range(40, 48))


def test_12_c1_c2_use_frozen_replace_channels():
    x, y = np.zeros((48, 3)), np.ones((48, 3))
    assert np.all(replace_channels(x, y, [0, 1])[:, :2] == 1)
    assert np.all(replace_channels(x, y, [0, 1])[:, 2] == 0)


def test_13_cache_schema_fields_are_declared():
    required = {
        "family", "source_window_id", "sequence_id", "fold", "role", "target_edge_id",
        "relation_edge_ids", "edge_scores", "valid_mask", "target_channel", "source_channel",
        "replacement_window_id", "endpoint_row", "timestamp",
    }
    text = inspect.getsource(stage.save_score_caches)
    assert all(name in text for name in required)


def test_14_exact_reproduction_has_one_e_minus_six_gate():
    assert stage.TOLERANCE == 1e-6


def test_15_reproduction_failure_blocks_grouped_evaluation():
    run_source = inspect.getsource(stage.run)
    assert "if not exact_passed" in run_source
    assert run_source.index("if not exact_passed") < run_source.index("eval_grouped")


def test_16_incoming_edges_are_maxed_within_target():
    scores = np.asarray([[1.0, 3.0, 2.0]])
    grouped, target_ids, _ = stage.grouped_scores(scores, np.asarray([0, 0, 1]))
    assert grouped[0] == 2.5


def test_17_grouped_top2_targets_are_distinct():
    scores = np.asarray([[9.0, 8.0, 7.0, 6.0]])
    _, target_ids, _ = stage.grouped_scores(scores, np.asarray([0, 0, 1, 2]))
    assert len(set(target_ids[0].tolist())) == 2


def test_18_v2a_hashes_are_read_only():
    assert "write_text" not in inspect.getsource(stage.artifact_audit).split("pd.dataframe", 1)[0].lower()


def test_19_aggregation_outputs_are_finite():
    scores = np.arange(48, dtype=float).reshape(2, 2, 12)
    maximum, top2, _ = stage.edge_aggregates(scores)
    grouped, _, _ = stage.grouped_scores(scores, np.repeat(np.arange(6), 2))
    assert np.isfinite(maximum).all() and np.isfinite(top2).all() and np.isfinite(grouped).all()


def test_20_resume_checks_dependency_hash():
    text = inspect.getsource(stage.run)
    assert "final_report_complete" in text and "dependency_hash" in text


def test_21_scale_recovery_reads_normal_train_prefix_only():
    text = inspect.getsource(stage.load_frozen_normal_train_for_scale).lower()
    assert "nrows=int(row.split_index)" in text
    assert "skiprows" not in text and "validation" not in text


def test_22_scale_formula_matches_original_v2a_expression():
    values = np.asarray([0.0, 1.0, 2.0, 20.0], dtype=np.float32)
    median, mad, scaled_mad, std, fallback, effective = stage.original_channel_robust_scale(values)
    expected = max(1.4826 * np.median(np.abs(values - np.median(values))), np.std(values), 1e-6)
    assert effective == float(expected)


def test_23_mad_multiplier_is_exact():
    values = np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
    _, mad, scaled_mad, _, _, _ = stage.original_channel_robust_scale(values)
    expected_mad = np.median(np.abs(values - np.median(values)))
    assert scaled_mad == float(1.4826 * expected_mad)


def test_24_scale_fallback_order_is_exact():
    constant = np.ones(8, dtype=np.float32)
    *_, fallback, effective = stage.original_channel_robust_scale(constant)
    assert fallback == "epsilon" and effective == 1e-6


def test_25_fix_only_assigns_p1_p2_family_slices():
    text = inspect.getsource(stage.patch_p1_p2_cache)
    assert "scores[3] = p1" in text and "scores[4] = p2" in text
    assert "protected_families = np.asarray([0, 1, 2, 5, 6])" in text


def test_26_fix_preserves_protected_family_hash():
    text = inspect.getsource(stage.patch_p1_p2_cache)
    assert "protected_hash_before" in text and "protected_hash_after" in text
    assert "protected_unchanged" in text


def test_27_patch_uses_temporary_npz_and_atomic_replace():
    text = inspect.getsource(stage.atomic_savez)
    assert ".tmp.npz" in text and "os.replace" in text


def test_28_grouped_requires_all_66_checks():
    text = inspect.getsource(stage.run_scale_fix)
    gate = "exact_passed and reproduction[\"checks_passed\"] == 66 and reproduction[\"checks_total\"] == 66"
    assert gate in text
    assert text.index("if not (exact_passed") < text.index("eval_grouped")


def test_29_scale_cache_is_reused():
    text = inspect.getsource(stage.recover_channel_scales)
    assert "registry_path.exists() and cache_path.exists()" in text


def test_30_report_resume_skips_recomputation():
    text = inspect.getsource(stage.run_scale_fix)
    assert text.index("final_report_complete") < text.index("recover_channel_scales")
