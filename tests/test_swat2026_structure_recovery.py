from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.swat2026_structure import stable_pairs
from data.swat2026_structure_recovery import (
    LEGACY_EXPANDED3,
    REPRESENTATIONS,
    batch_representation_correlations,
    reproducibility_diagnostics,
    select_primary,
    select_reproducible_sets,
    train_relation_split,
    validation_usage_split,
)
from scripts.swat2026 import recover_structure_relations as recovery


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "swat2026" / "structure_recovery_seed1.yaml"


def cfg() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def timestamps() -> pd.DatetimeIndex:
    return pd.date_range("2026-03-11 09:00:00", periods=28_860, freq="s")


def pair_frame() -> pd.DataFrame:
    return stable_pairs(["A.Pv", "B.Pv", "C.Pv"], set()).head(1)


def test_discovery_and_confirmation_raw_timestamps_do_not_overlap():
    manifest, split = train_relation_split(timestamps())
    assert not set(split["discovery"]) & set(split["confirmation"])
    assert manifest["raw_timestamp_overlap"] is False


def test_boundary_guard_has_exactly_127_points():
    manifest, split = train_relation_split(timestamps())
    assert len(split["guard"]) == 127
    assert manifest["boundary_guard"]["raw_points"] == 127
    assert split["confirmation_ends"][0] - 127 >= split["confirmation"][0]


def test_reproducibility_screening_has_no_validation_or_test_argument():
    parameters = set(inspect.signature(reproducibility_diagnostics).parameters)
    assert parameters == {"pairs", "discovery", "confirmation", "representation", "cfg"}


def test_level_pearson_is_correct():
    x = np.arange(128.0)
    windows = np.stack([np.column_stack([x, x])])
    result = batch_representation_correlations(windows, pair_frame(), "level_pearson")
    assert np.allclose(result, 1.0)


def test_delta_pearson_is_correct():
    x = np.arange(128.0) ** 2
    windows = np.stack([np.column_stack([x, 3 * x + 2])])
    result = batch_representation_correlations(windows, pair_frame(), "delta_pearson")
    assert np.allclose(result, 1.0)


def test_delta_spearman_is_correct():
    x = np.arange(128.0) ** 2
    windows = np.stack([np.column_stack([x, np.exp(x / 1000)])])
    result = batch_representation_correlations(windows, pair_frame(), "delta_spearman")
    assert np.allclose(result, 1.0)


def test_ait401_is_excluded():
    pairs = stable_pairs(["A.Pv", "AIT401.Pv", "B.Pv"], {"AIT401.Pv"})
    assert not pairs.pair_name.str.contains("AIT401").any()


def test_discovery_and_confirmation_both_must_pass():
    pairs = pair_frame()
    discovery = np.full((100, 1), .8)
    confirmation = np.full((100, 1), .1)
    result = reproducibility_diagnostics(pairs, discovery, confirmation, "level_pearson", cfg()["reproducibility"])
    assert not bool(result.iloc[0].reproducible_pass)


def test_robust_median_gap_condition_is_enforced():
    pairs = pair_frame()
    discovery = np.linspace(.75, .85, 100)[:, None]
    confirmation = np.linspace(.30, .40, 100)[:, None]
    result = reproducibility_diagnostics(pairs, discovery, confirmation, "level_pearson", cfg()["reproducibility"])
    assert result.iloc[0].discovery_confirmation_median_gap > .15
    assert not bool(result.iloc[0].reproducible_pass)


def test_fewer_than_three_relations_is_narrow_only():
    rows = []
    for index in range(2):
        rows.append({"pair_name": f"P{index}", "channel_i": f"A{index}", "channel_j": f"B{index}",
                     "channel_i_index": index, "channel_j_index": index + 2, "reproducible_pass": True,
                     "confirmation_valid_window_ratio": 1.0, "discovery_median_abs_corr": .8,
                     "confirmation_median_abs_corr": .8, "discovery_confirmation_median_gap": .01,
                     "discovery_corr_iqr": .1, "confirmation_corr_iqr": .1})
    assert select_reproducible_sets(pd.DataFrame(rows))["status"] == "narrow_relation_diagnostic"


def test_legacy_expanded_is_diagnostic_only():
    assert len(LEGACY_EXPANDED3) == 3
    source = inspect.getsource(recovery.run)
    assert '"automatic_promotion": False' in source


def test_validation_threshold_and_controlled_raw_points_do_not_overlap():
    manifest, split = validation_usage_split(timestamps())
    assert not set(split["threshold_raw"]) & set(split["controlled_raw"])
    assert manifest["raw_timestamp_overlap"] is False
    assert len(split["guard"]) == 127


def test_threshold_metrics_uses_separate_normal_threshold_values():
    signature = inspect.signature(recovery.fixed_threshold_metrics)
    assert list(signature.parameters) == ["normal", "anomaly", "threshold_values", "quantile"]
    result = recovery.fixed_threshold_metrics(np.array([0.1, .2]), np.array([.8, .9]), np.array([.1, .15]), .99)
    assert result["threshold"] == pytest.approx(np.quantile([.1, .15], .99))


def test_injection_implementation_preserves_marginal_values():
    source = inspect.getsource(recovery.controlled_validation)
    assert "np.roll(original, shift)" in source
    assert "value_multiset_preserved" in source
    assert "test_used\": False" in source


def test_primary_selection_has_no_test_argument():
    assert set(inspect.signature(select_primary).parameters) == {"controlled", "representation_counts", "minimum_auroc"}
    assert select_primary(pd.DataFrame(), {}, .8)["primary_frozen"] is False


def test_primary_tie_prefers_simpler_representation():
    rows = pd.DataFrame([
        {"representation": "delta_spearman", "score_name": "representation_max", "auroc": .90,
         "marginal_statistics_preserved": True, "reference_valid": True},
        {"representation": "level_pearson", "score_name": "representation_max", "auroc": .895,
         "marginal_statistics_preserved": True, "reference_valid": True},
    ])
    selected = select_primary(rows, {name: 3 for name in REPRESENTATIONS}, .8)
    assert selected["representation"] == "level_pearson"


def test_test_labels_are_read_only_inside_post_freeze_function():
    run_source = inspect.getsource(recovery.run)
    assert "frozen_primary_structure_manifest.json" in run_source
    assert "formal_test_evaluation(" in run_source
    test_source = inspect.getsource(recovery.formal_test_evaluation)
    assert "point_labels_11mar.csv" in test_source


def test_no_point_adjustment():
    assert cfg()["evaluation"]["point_adjustment"] is False
    assert '"point_adjustment": False' in inspect.getsource(recovery.formal_test_evaluation)


def test_no_fusion_is_generated():
    assert cfg()["evaluation"]["fusion"] is False
    assert "fuse_scores(" not in inspect.getsource(recovery)


def test_m1_is_never_retrained():
    source = inspect.getsource(recovery)
    assert "optimizer" not in source.lower()
    assert ".backward(" not in source
    assert "train_epoch(" not in source


def test_checkpoint_hash_is_fixed():
    assert cfg()["data"]["checkpoint_sha256"] == "0f145cc0385bdbb6d8b638dde3c087cf757380c7f845dfdc45bb90d6fc4a5728"
    assert "validate_m1_artifacts" in inspect.getsource(recovery.run)


def test_source_csv_hash_is_fixed():
    assert cfg()["data"]["source_sha256"] == "d1dd05034b393a6b201f39e86ac3b933000551d26680c5d91c5ff8db0559cf72"
    assert "Official source CSV hash changed" in inspect.getsource(recovery.run)


def test_no_hai_dependency_or_modification():
    source = inspect.getsource(recovery)
    assert "data.hai" not in source
    assert "hai_" not in source.lower()


def test_outputs_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "outputs/" in ignore and "*.pt" in ignore and "dataset/" in ignore
