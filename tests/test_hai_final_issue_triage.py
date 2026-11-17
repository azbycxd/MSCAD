from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import scripts.hai_final_issue_triage as triage


def source():
    return Path(triage.__file__).read_text(encoding="utf-8")


def issues():
    return pd.read_csv(triage.INPUT / "unresolved_issues_master.csv")


def test_01_reads_exactly_seven_issues():
    assert len(issues()) == 7


def test_02_issue_ids_unique():
    assert issues().issue_id.is_unique


def test_03_decision_map_complete():
    assert set(issues().issue_id) == set(triage.DECISIONS)


def test_04_every_issue_has_evidence():
    assert issues().evidence.astype(str).str.len().gt(0).all()


def test_05_no_raw_data_loading():
    text = source().lower()
    assert "data_root" not in text and "dataset\\dataset" not in text


def test_06_no_model_or_score_loading():
    text = source().lower()
    assert "np.load" not in text and "torch.load" not in text and "joblib.load" not in text


def test_07_no_training_or_inference():
    text = source().lower()
    assert ".fit(" not in text and ".forward(" not in text and "optimizer" not in text


def test_08_no_fusion_execution():
    assert "run_fusion" not in source().lower()


def test_09_no_external_download():
    text = source().lower()
    assert "requests" not in text and "http://" not in text and "https://" not in text


def test_10_unique_severity_each_issue():
    assert all(triage.DECISIONS[key]["severity"] in {"P0", "P1", "P2", "P3"} for key in triage.DECISIONS)


def test_11_unique_action_each_issue():
    allowed = {"must_resolve_before_thesis", "should_improve_if_time_allows", "retain_as_explicit_limitation", "already_resolved_or_superseded", "invalid_or_duplicate_issue"}
    assert all(triage.DECISIONS[key]["action_class"] in allowed for key in triage.DECISIONS)


def test_12_no_p0_is_ignored():
    for decision in triage.DECISIONS.values():
        if decision["severity"] == "P0":
            assert decision["action_class"] == "must_resolve_before_thesis"


def test_13_two_p1_limitations():
    p1 = [value for value in triage.DECISIONS.values() if value["severity"] == "P1"]
    assert len(p1) == 2 and all(value["action_class"] == "retain_as_explicit_limitation" for value in p1)


def test_14_resolved_issues_have_resolution_evidence():
    resolved = [value for value in triage.DECISIONS.values() if value["action_class"] == "already_resolved_or_superseded"]
    assert len(resolved) == 5 and all(value["why"] for value in resolved)


def test_15_input_audit_uses_inventory_and_provenance():
    text = inspect.getsource(triage.input_audit)
    assert "inventory_paths" in text and "provenance_stages" in text


def test_16_impact_dimensions_complete():
    assert len(triage.IMPACT_FIELDS) == 10


def test_17_recomputation_has_stop_rule():
    frame = triage.recomputation_matrix(issues())
    assert frame.failure_stop_rule.astype(str).str.len().gt(0).all()


def test_18_recomputation_does_not_rerun_frozen_stages():
    frame = triage.recomputation_matrix(issues())
    assert frame.stages_to_rerun.eq("none").all()


def test_19_claim_revision_removes_real_v2_claim():
    frame = triage.claim_revision({})
    assert bool(frame.loc[frame.claim_id.eq("C3"), "remove"].iloc[0])


def test_20_claim_revision_removes_fusion_claim():
    frame = triage.claim_revision({})
    assert bool(frame.loc[frame.claim_id.eq("C4"), "remove"].iloc[0])


def test_21_controlled_claim_is_weakened():
    frame = triage.claim_revision({})
    assert bool(frame.loc[frame.claim_id.eq("C2"), "weaken"].iloc[0])


def test_22_stop_list_contains_a1m_direction():
    frame = triage.experiments_to_stop()
    assert frame.existing_evidence.str.contains("V2-A1M").any()


def test_23_stop_list_contains_a1c_direction():
    frame = triage.experiments_to_stop()
    assert frame.existing_evidence.str.contains("V2-A1C").any()


def test_24_stop_list_rejects_sequence_thresholds():
    assert triage.experiments_to_stop().experiment_direction.str.contains("sequence-specific").any()


def test_25_stop_list_rejects_legacy_swat():
    assert triage.experiments_to_stop().experiment_direction.str.contains("SWaT").any()


def test_26_required_inputs_are_5a_only():
    assert all((triage.INPUT / name).exists() for name in triage.REQUIRED_INPUTS)


def test_27_fingerprint_reads_only_required_inputs():
    text = inspect.getsource(triage.fingerprint)
    assert "REQUIRED_INPUTS" in text


def test_28_resume_precedes_output_generation():
    text = inspect.getsource(triage.run)
    assert text.index("REUSE completed issue triage") < text.index("input_audit(")


def test_29_markdown_rows_match_frame():
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    text = triage.markdown_table(frame, "x")
    assert len([line for line in text.splitlines() if line.startswith("| ")]) - 2 == len(frame)


def test_30_output_directory_is_new():
    assert triage.OUT != triage.INPUT and triage.OUT.parent == triage.INPUT.parent


def test_31_no_must_resolve_in_current_decisions():
    assert not any(value["action_class"] == "must_resolve_before_thesis" for value in triage.DECISIONS.values())


def test_32_priority_order_p1_before_p2():
    decisions = triage.decision_frame(issues())
    frame = triage.priority_frame(decisions)
    assert frame.severity.tolist()[:2] == ["P1", "P1"]


def test_33_no_nan_in_generated_decision_frame():
    assert not triage.decision_frame(issues()).isna().any().any()


def test_34_external_decision_is_protocol_only_in_source():
    text = inspect.getsource(triage.run)
    assert '"execution_status": "protocol_only_not_executed"' in text


def test_35_stronger_plan_has_at_most_one_stage():
    text = inspect.getsource(triage.run)
    assert "最多一个新增实验阶段" in text


def test_36_formal_output_count_and_required_files():
    assert len(list(triage.OUT.iterdir())) == 19
    assert (triage.OUT / "final_hai_issue_triage_and_improvement_decision.md").exists()


def test_37_formal_input_audit_all_passes():
    frame = pd.read_csv(triage.OUT / "unresolved_issue_input_audit.csv")
    assert len(frame) == 7 and frame.input_complete.all()


def test_38_formal_severity_counts():
    counts = pd.read_csv(triage.OUT / "issue_severity_classification.csv").severity.value_counts().to_dict()
    assert counts == {"P2": 4, "P1": 2, "P3": 1}


def test_39_formal_action_counts():
    counts = pd.read_csv(triage.OUT / "final_unresolved_issue_decision.csv").action_class.value_counts().to_dict()
    assert counts == {"already_resolved_or_superseded": 5, "retain_as_explicit_limitation": 2}


def test_40_formal_readiness_is_level1():
    payload = yaml.safe_load((triage.OUT / "thesis_readiness_assessment.yaml").read_text(encoding="utf-8"))
    assert payload["level"] == "Level 1" and payload["new_experiment_required"] is False


def test_41_external_confirmation_is_not_executed():
    payload = yaml.safe_load((triage.OUT / "external_confirmation_decision.yaml").read_text(encoding="utf-8"))
    assert payload["decision"] == "strongly_recommended"
    assert payload["execution_status"] == "protocol_only_not_executed"


def test_42_formal_outputs_have_no_nan():
    for path in triage.OUT.glob("*.csv"):
        assert not pd.read_csv(path).isna().any().any(), path.name


def test_43_formal_markdown_rows_match_csv():
    stems = ["final_unresolved_issue_decision", "claim_revision_matrix", "improvement_priority_matrix", "experiments_not_to_continue"]
    for stem in stems:
        frame = pd.read_csv(triage.OUT / f"{stem}.csv")
        lines = [line for line in (triage.OUT / f"{stem}.md").read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
        assert len(lines) - 2 == len(frame)


def test_44_historical_hash_status_unchanged():
    payload = json.loads((triage.OUT / "stage_status.json").read_text(encoding="utf-8"))
    assert payload["historical_hashes_unchanged"] is True


def test_45_all_stage_status_flags_complete():
    payload = json.loads((triage.OUT / "stage_status.json").read_text(encoding="utf-8"))
    assert all(value is True for key, value in payload.items() if key != "input_fingerprint")
