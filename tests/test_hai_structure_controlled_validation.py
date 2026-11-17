from __future__ import annotations

import importlib.util
import inspect
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hai_step4b_validate_independent_structure_score.py"
SPEC = importlib.util.spec_from_file_location("hai_step4b", SCRIPT)
step4b = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(step4b)

from utils.controlled_relation_corruptions import (  # noqa: E402
    BLOCK_ORDER, block_permute_both, block_permute_one, circular_shift_both,
    circular_shift_one, matched_replacement_indices, replace_one, same_multiset,
)
from utils.independent_structure_score import (  # noqa: E402
    aggregate_distances, fisher_transform, fit_robust_reference, relation_distances, window_pair_fisher,
)


def test_frozen_step4a_pools_are_read_directly():
    source = inspect.getsource(step4b.load_frozen)
    assert "step4a_strict_pool" in source and "step4a_diversity_pool" in source
    assert "pd.read_csv" in source


def test_full_2278_pair_recompute_is_forbidden():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "step 4a all-pair (2278) diagnosis" in source
    assert "combinations(" not in source


def test_no_ae_checkpoint_or_rec_import():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "import torch" not in source
    assert "torch.load" not in source and "load_state_dict" not in source
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("models" in ast.unparse(node) for node in imports)


def test_train_and_validation_scopes_are_separate():
    source = inspect.getsource(step4b.load_selected_data)
    assert "values[:split_at]" in source and "values[split_at:]" in source
    assert "effective_scale" in source


def test_test_files_cannot_be_loaded():
    assert step4b.TRAIN_FILES == tuple(f"train{i}.csv" for i in range(1, 7))
    source = inspect.getsource(step4b.load_selected_data).lower()
    assert "test1" not in source and "test2" not in source


def test_pearson_fisher_definition():
    x = np.arange(32, dtype=float)
    windows = np.stack([np.column_stack([x, x]), np.column_stack([x, -x])])
    z = window_pair_fisher(windows, np.array([[0, 1]]))[:, 0]
    assert np.allclose(z, fisher_transform(np.array([1.0, -1.0])))
    assert np.isfinite(z).all()


def test_robust_fallback_order_mad_iqr_std_epsilon():
    values = np.column_stack([
        np.arange(12, dtype=float),
        np.r_[np.zeros(9), 1, 2, 3],
        np.r_[np.zeros(11), 1],
        np.ones(12),
    ])
    ref = fit_robust_reference(values, np.repeat(np.arange(6), 2), epsilon=1e-6)
    assert ref.fallback.tolist() == ["mad", "iqr", "std", "epsilon"]


def test_strict_pool_has_expected_pairs():
    cfg = step4b.load_config(ROOT / "configs" / "hai_structure_controlled_validation.yaml")
    strict, diversity, *_ = step4b.load_frozen(cfg)
    assert strict[["feature_i", "feature_j"]].values.tolist() == [
        ["P2_VIBTR01", "P2_VIBTR03"], ["P2_VIBTR02", "P2_VIBTR04"]]
    assert len(diversity) == 10


def test_nearest_component_is_identity_free_minimum():
    train = np.array([[0.0], [0.1], [10.0], [10.1], [20], [20.1], [30], [30.1], [40], [40.1], [50], [50.1]])
    ref = fit_robust_reference(train, np.repeat(np.arange(6), 2), epsilon=.01)
    score = relation_distances(np.array([[10.05]]), ref, np.array([True]))
    expected = np.min(np.abs(10.05 - ref.sequence_median[:, 0]) / ref.sequence_scale[:, 0])
    assert np.isclose(score[0, 0], expected)


def test_candidate_space_is_exactly_sixteen():
    cfg = step4b.load_config(ROOT / "configs" / "hai_structure_controlled_validation.yaml")
    assert len(cfg["reference_strategies"]) * len(cfg["windows"]) * len(cfg["aggregations"]) == 16
    assert cfg["windows"] == [64, 128]
    assert cfg["aggregations"] == ["top3_mean", "max"]


@pytest.mark.parametrize("window,shift", [(64, 16), (128, 32)])
def test_r1_preserves_target_multiset(window, shift):
    x = np.arange(window * 3).reshape(window, 3)
    y = circular_shift_one(x, 1, shift)
    assert same_multiset(x[:, 1], y[:, 1]) and np.array_equal(x[:, 0], y[:, 0])


def test_r2_preserves_target_multiset_and_block_shape():
    x = np.arange(64 * 3).reshape(64, 3)
    y = block_permute_one(x, 1)
    assert BLOCK_ORDER == (2, 0, 3, 1) and same_multiset(x[:, 1], y[:, 1])


def test_r3_replacement_is_single_channel():
    x, source = np.zeros((64, 3)), np.ones((64, 3))
    y = replace_one(x, source, 1)
    assert np.all(y[:, 1] == 1) and np.all(y[:, [0, 2]] == 0)


def test_r3_matching_is_same_sequence_nonself_nonoverlap():
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(20, 64, 2))
    starts = np.arange(20) * 64
    matched = matched_replacement_indices(windows, 0, starts, 64, quantile_bins=2)
    valid = np.flatnonzero(matched >= 0)
    assert len(valid) > 0
    assert np.all(valid != matched[valid])
    assert np.all(np.abs(starts[valid] - starts[matched[valid]]) >= 64)


def test_c1_uses_same_shift_for_both_channels():
    x = np.arange(64 * 3).reshape(64, 3)
    y = circular_shift_both(x, (0, 1), 16)
    assert np.array_equal(y[:, 0], np.roll(x[:, 0], 16))
    assert np.array_equal(y[:, 1], np.roll(x[:, 1], 16))


def test_c2_uses_same_permutation_for_both_channels():
    x = np.arange(64 * 3).reshape(64, 3)
    y = block_permute_both(x, (0, 1))
    assert same_multiset(x[:, 0], y[:, 0]) and same_multiset(x[:, 1], y[:, 1])
    assert np.array_equal(np.argsort(y[:, 0]), np.argsort(y[:, 1]))


def test_marginal_preservation_is_exact_for_r1_r2_c1_c2():
    x = np.arange(64 * 3, dtype=float).reshape(64, 3)
    variants = [circular_shift_one(x, 0, 16), block_permute_one(x, 0),
                circular_shift_both(x, (0, 1), 16), block_permute_both(x, (0, 1))]
    assert all(np.allclose([x[:, 0].mean(), x[:, 0].std(), x[:, 0].min(), x[:, 0].max()],
                           [y[:, 0].mean(), y[:, 0].std(), y[:, 0].min(), y[:, 0].max()]) for y in variants)


def test_threshold_comes_only_from_clean_scores():
    source = inspect.getsource(step4b.evaluate)
    assert "np.quantile(clean_score" in source
    threshold_line = next(line for line in source.splitlines() if "threshold =" in line)
    assert "features" not in threshold_line and "success" not in threshold_line


def test_selection_does_not_accept_test_metrics():
    signature = inspect.signature(step4b.evaluate)
    assert "test" not in str(signature).lower()
    assert "test" not in inspect.getsource(step4b.score_values).lower()


def test_clean_cache_is_written_once_and_resumed():
    source = inspect.getsource(step4b.build_clean_cache)
    assert "cache_valid(cache, metadata)" in source
    assert source.count("np.savez_compressed") == 1


def test_completed_stage_resume_short_circuits():
    source = inspect.getsource(step4b.main)
    assert "final_report_complete" in source
    assert "REUSE completed Step 4B stage" in source
    assert "COMPUTE_NEW: none" in source


def test_corruption_cache_is_written_once():
    source = inspect.getsource(step4b.main)
    assert "if corruption_path.exists()" in source
    assert source.count("np.savez_compressed(corruption_path") == 1


def test_cache_metadata_covers_minimum_recompute_contract():
    source = inspect.getsource(step4b.dependency_metadata)
    for token in ("data_file_hashes", "split_sha256", "scaler_sha256", "relation_pool_sha256",
                  "window", "train_stride", "validation_stride", "pearson_fisher", "pair_order", "metadata_complete"):
        assert token in source


def test_corruption_cache_is_dependency_bound():
    source = inspect.getsource(step4b.corruption_dependency_hash)
    assert "clean_relation_features_w" in source and "pair_order" in source
    assert "replacement_quantile_bins" in source


def test_readonly_audit_detects_training_leakage_contract():
    source = inspect.getsource(step4b.readonly_audit)
    assert "invalid_hai_rec_training_leakage" in source
    assert "validation_loss(model, val_loader" in source
    assert "sha256(checkpoint)" in source


def test_readonly_audit_recognizes_legacy_loader_separately():
    source = inspect.getsource(step4b.readonly_audit)
    assert "legacy_nonblocking_issue" in source
    assert "legacy_train_uses_validation_loader" in source


def test_dry_run_has_three_required_lists():
    source = inspect.getsource(step4b.print_plan)
    assert all(label in source for label in ("REUSE:", "COMPUTE_NEW:", "FORBIDDEN_RECOMPUTE:"))


def test_forbidden_compute_list_covers_historical_work():
    joined = " ".join(step4b.FORBIDDEN).lower()
    assert "step 2" in joined and "step 3" in joined and "2278" in joined and "hai test" in joined


def test_aggregation_definitions():
    values = np.array([[1.0, 5.0, 3.0, 2.0]])
    assert aggregate_distances(values, "max")[0] == 5
    assert np.isclose(aggregate_distances(values, "top3_mean")[0], (5 + 3 + 2) / 3)


def test_no_formal_training_or_forward_in_test_module():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    calls = [ast.unparse(node.func).lower() for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(name.endswith("torch.load") or name.endswith(".forward") for name in calls)
    assert not any("subprocess" in name or "build_hai_dataloaders" in name for name in calls)


def test_output_and_smoke_directories_are_isolated():
    cfg = step4b.load_config(ROOT / "configs" / "hai_structure_controlled_validation.yaml")
    assert cfg["output_dir"] != cfg["smoke_output_dir"]


def test_historical_steps_are_only_read_dependencies():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "step4a_output" in source and "step3g_output" in source
    assert "to_csv(paths[" not in source and "write_text(paths[" not in source


def test_formal_frozen_artifact_hash_audit_is_clean_when_present():
    audit_path = ROOT / "outputs" / "hai_step4b_structure_controlled_validation" / "artifact_reuse_audit.csv"
    if not audit_path.exists():
        pytest.skip("formal output not generated")
    audit = pd.read_csv(audit_path)
    frozen = audit[audit.artifact_id.str.startswith(("step1_", "step2_", "step3g_", "step4a_"))]
    assert len(frozen) >= 10 and frozen.hash_match.all() and not frozen.recomputed.any()


def test_score_outputs_are_finite_on_small_tensor():
    rng = np.random.default_rng(4)
    windows = rng.normal(size=(30, 64, 4))
    z = window_pair_fisher(windows, np.array([[0, 1], [2, 3]]))
    ref = fit_robust_reference(z[:24], np.repeat(np.arange(6), 4))
    score = aggregate_distances(relation_distances(z[24:], ref), "top3_mean")
    assert np.isfinite(score).all()
