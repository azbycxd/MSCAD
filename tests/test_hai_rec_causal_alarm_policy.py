from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import hai_step3e_evaluate_rec_causal_alarm_policy as step3e
from utils.causal_alarm_policy import (
    ALL_POLICIES, apply_policy, binary_segments, causal_vote_alarm, consecutive_alarm, hysteresis_alarm,
    pointwise_alarm,
)


ROOT = Path(__file__).resolve().parents[1]


def test_directly_reads_three_step3d_score_archives():
    cfg = step3e.load_config(ROOT / "configs/hai_rec_causal_alarm_policy.yaml")
    assert all((ROOT / cfg["score_npz_pattern"].format(seed=seed)).exists() for seed in cfg["seeds"])


def test_step3e_does_not_import_or_load_a_model():
    source = inspect.getsource(step3e)
    assert "from models" not in source and "torch.load" not in source and "make_model" not in source


def test_step3e_does_not_run_inference_or_training():
    source = inspect.getsource(step3e)
    assert "model(" not in source and "optimizer" not in source and "backward(" not in source


def test_pointwise_alarm_is_exact():
    np.testing.assert_array_equal(pointwise_alarm(np.array([0., 1., 1.1]), 1.), [0, 0, 1])


def test_consecutive_2_is_causal_without_backfill():
    np.testing.assert_array_equal(consecutive_alarm(np.array([1, 1, 0, 1, 1, 1]), 2), [0, 1, 0, 0, 1, 1])


def test_consecutive_3_is_causal_without_backfill():
    np.testing.assert_array_equal(consecutive_alarm(np.array([1, 1, 0, 1, 1, 1]), 3), [0, 0, 0, 0, 0, 1])


def test_vote_2_of_3_uses_available_past_and_current_only():
    np.testing.assert_array_equal(causal_vote_alarm(np.array([1, 1, 0, 1, 0]), 2, 3), [0, 1, 1, 1, 0])


def test_vote_3_of_5_requires_three_available_points():
    np.testing.assert_array_equal(causal_vote_alarm(np.array([1, 1, 1, 0, 0, 1]), 3, 5), [0, 0, 1, 1, 1, 1])


def test_hysteresis_state_machine_exit_on_third_low_point():
    scores = np.array([0., 2.1, .5, .5, .5, 2.1, 1.1, 0., 0., 0.])
    np.testing.assert_array_equal(hysteresis_alarm(scores, 2., 1.), [0, 1, 1, 1, 0, 1, 1, 1, 1, 0])


def test_hysteresis_rejects_invalid_threshold_order():
    with pytest.raises(ValueError):
        hysteresis_alarm(np.array([1.]), 1., 2.)


def test_all_rules_are_registered_and_causal():
    cfg = step3e.load_config(ROOT / "configs/hai_rec_causal_alarm_policy.yaml")
    registry = step3e.policy_registry(cfg)
    assert set(registry.policy_name) == set(ALL_POLICIES)
    assert registry.causal.all() and not registry.uses_future.any() and not registry.backfills_history.any()


def test_primary_policy_is_pre_registered():
    cfg = step3e.load_config(ROOT / "configs/hai_rec_causal_alarm_policy.yaml")
    assert cfg["primary_policy"] == "vote_2_of_3"
    assert step3e.policy_registry(cfg).query("primary_candidate").policy_name.unique().tolist() == ["vote_2_of_3"]


def test_separate_calls_do_not_carry_csv_state():
    first = apply_policy(np.array([2., 2.]), "consecutive_2", threshold=1.)
    second = apply_policy(np.array([2.]), "consecutive_2", threshold=1.)
    assert first[-1] == 1 and second[0] == 0


def test_hysteresis_thresholds_are_fixed_step3d_validation_methods():
    cfg = step3e.load_config(ROOT / "configs/hai_rec_causal_alarm_policy.yaml")
    assert cfg["hysteresis_enter_method"] == "normal_validation_q995"
    assert cfg["hysteresis_stay_method"] == "normal_validation_q99"


def test_threshold_lookup_is_score_and_seed_specific():
    cfg = step3e.load_config(ROOT / "configs/hai_rec_causal_alarm_policy.yaml")
    frame = pd.read_csv(ROOT / cfg["threshold_file"])
    value = step3e.threshold_lookup(frame, 1, "normal_validation_q99")
    expected = frame[(frame.seed == 1) & (frame.score_variant == "endpoint_raw_top5") &
                     (frame.threshold_method == "normal_validation_q99")].threshold_value.iloc[0]
    assert value == expected


def test_threshold_and_policy_functions_do_not_accept_test_labels():
    assert "label" not in inspect.signature(apply_policy).parameters
    assert "label" not in inspect.signature(step3e.threshold_lookup).parameters


def test_full_seed_contains_fixed_58_events_without_cross_file_merge():
    cfg = step3e.load_config(ROOT / "configs/hai_rec_causal_alarm_policy.yaml")
    scores = step3e.load_seed_scores(ROOT / cfg["score_npz_pattern"].format(seed=1))
    events = step3e.extract_events(scores["test"])
    assert len(events) == 58
    assert all(event["file_name"].startswith("test") for event in events)


def test_duration_groups_are_exact():
    assert step3e.duration_group(29) == "short"
    assert step3e.duration_group(30) == "medium"
    assert step3e.duration_group(299) == "medium"
    assert step3e.duration_group(300) == "long"


def test_binary_segment_merge_is_exact():
    starts, ends = binary_segments(np.array([0, 1, 1, 0, 1, 0]))
    np.testing.assert_array_equal(starts, [1, 4]); np.testing.assert_array_equal(ends, [2, 4])


def test_event_transition_logic_records_recovered_and_lost():
    source = inspect.getsource(step3e.event_transitions)
    assert "not base.detected and new.detected" in source
    assert "base.detected and not new.detected" in source


def test_smoke_and_formal_outputs_are_isolated():
    cfg = step3e.load_config(ROOT / "configs/hai_rec_causal_alarm_policy.yaml")
    assert cfg["smoke_output_dir"] != cfg["output_dir"]


def test_step3d_sources_are_never_opened_for_writing():
    source = inspect.getsource(step3e)
    assert "np.save" not in source
    assert "to_csv(source_paths" not in source


def test_all_policy_outputs_are_binary():
    scores = np.array([0., 2., 0., 2., 2., 0.])
    for policy in ALL_POLICIES:
        alarm = (apply_policy(scores, policy, enter_threshold=1.5, stay_threshold=.5)
                 if "hysteresis" in policy else apply_policy(scores, policy, threshold=1.))
        assert np.isin(alarm, [0, 1]).all()


def test_required_output_registry_is_complete():
    assert len(step3e.OUTPUT_FILES) == 17
