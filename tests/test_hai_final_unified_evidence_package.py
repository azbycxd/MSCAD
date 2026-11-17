from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest
import yaml

import scripts.hai_final_unified_evidence_package as final


def source_text():
    return Path(final.__file__).read_text(encoding="utf-8")


def test_01_no_raw_hai_csv_loading():
    source = source_text()
    assert "data_root" not in source and "dataset\\dataset\\hai" not in source


def test_02_no_npz_content_loading():
    assert "np.load" not in source_text()


def test_03_no_model_loading():
    source = source_text().lower()
    assert "torch.load" not in source and "joblib.load" not in source


def test_04_no_forward_execution():
    assert ".forward(" not in source_text()


def test_05_no_score_recomputation():
    source = source_text()
    assert "edge_scores(" not in source and "dependency_loss(" not in source


def test_06_no_threshold_fit():
    source = source_text().lower()
    assert "quantile(" not in source and "thresholds_from" not in source


def test_07_fusion_not_executed():
    manifest = final.freeze_manifest()
    assert manifest["final_fusion"]["status"] == "not_executed"


def test_08_swat_invalid_for_claims():
    frame = final.validity_registry()
    assert frame.loc[frame.evidence_id.eq("SWAT_LEGACY"), "validity_status"].iloc[0] == "invalid_for_claims"


def test_09_controlled_not_real():
    frame = final.validity_registry()
    assert frame.loc[frame.evidence_id.eq("V2_CONTROLLED"), "validity_status"].iloc[0] == "valid_controlled_mechanism_evidence"


def test_10_tiny_parameter_count():
    assert final.freeze_manifest()["reconstruction_backbone"]["parameter_count"] == 16420


def test_11_tiny_status_frozen():
    assert final.freeze_manifest()["reconstruction_backbone"]["status"] == "real_evaluation_baseline_frozen"


def test_12_v1_status_consistent():
    assert final.freeze_manifest()["structure_v1"]["status"] == "frozen_failed_real_transfer_baseline"


def test_13_v2_static_status_consistent():
    assert final.freeze_manifest()["structure_v2_static"]["status"] == "frozen_controlled_validation_reference"


def test_14_a1m_negative_ablation():
    assert final.freeze_manifest()["structure_v2_a1m"]["status"] == "frozen_negative_ablation"


def test_15_a1c_negative_ablation():
    assert final.freeze_manifest()["structure_v2_a1c"]["status"] == "frozen_negative_ablation"


def test_16_claim_boundary_rejects_v2_real_superiority():
    claims = final.claim_boundary()
    assert any("real HAI" in item for item in claims["NOT_SUPPORTED"])


def test_17_claim_boundary_rejects_sota():
    assert any("SOTA" in item for item in final.claim_boundary()["NOT_SUPPORTED"])


def test_18_claim_boundary_rejects_fusion_claim():
    assert any("fusion" in item.lower() for item in final.claim_boundary()["NOT_SUPPORTED"])


def test_19_unavailable_marker_exact():
    assert final.UNAVAILABLE == "unavailable_without_recomputation"


def test_20_markdown_table_row_count():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    markdown = final.markdown_table(frame, "x")
    assert len([line for line in markdown.splitlines() if line.startswith("| ")]) - 2 == len(frame)


def test_21_metric_provenance_has_required_fields():
    provenance = final.Provenance({"x.csv": "abc"})
    provenance.add("m", "s", "v", "test", "scope", "auroc", 0.5, "x.csv", "col")
    required = {"metric_id", "stage_id", "method_variant", "dataset", "split", "evaluation_scope", "metric_name", "metric_value", "unit", "aggregation", "seed", "threshold_protocol", "source_artifact", "source_column_or_json_path", "source_hash", "validity_status", "limitation"}
    assert required <= set(provenance.frame().columns)


def test_22_resume_check_precedes_inventory_scan():
    source = inspect.getsource(final.run)
    assert source.index("REUSE completed unified evidence package") < source.index("scan_inventory()")


def test_23_resume_does_not_parse_npz():
    source = inspect.getsource(final.input_fingerprint)
    assert "open(" not in source and "np.load" not in source


def test_24_only_text_artifacts_are_summary_eligible():
    assert ".npz" in final.RAW_EXTENSIONS and ".csv" in final.TEXT_EXTENSIONS


def test_25_decision_matrix_has_required_methods():
    assert set(final.decision_matrix().method) == {"Tiny rec", "Structure V1", "Structure V2 static", "A1M", "A1C", "Fusion"}


def test_26_fusion_performance_unavailable():
    row = final.decision_matrix().query("method == 'Fusion'").iloc[0]
    assert row.performance == final.UNAVAILABLE


def test_27_method_freeze_yaml_serializable():
    text = yaml.safe_dump(final.freeze_manifest(), allow_unicode=True)
    assert yaml.safe_load(text)["structure_v2_a1c"]["status"] == "frozen_negative_ablation"


def test_28_safe_value_rejects_nonfinite():
    assert final.safe_value(float("nan")) == final.UNAVAILABLE


def test_29_required_stage_map_present():
    required = {"hai_step1_dataset_audit", "hai_step2_dataloader_validation", "hai_step3g_rec_capacity_ablation", "hai_step4cd_structure_real_and_complementarity", "hai_structure_v2a_lagged_dependency", "hai_structure_v2a1s_full_clean_stability", "hai_structure_v2a1m_mode_conditioned_reference", "hai_structure_v2a1c_causal_adaptive_reference"}
    assert required <= set(final.STAGES)


def test_30_output_does_not_overwrite_historical_directory():
    assert final.OUT.name == "hai_final_unified_evidence_package"
    assert final.OUT.parent == final.OUTPUTS


def test_31_no_external_baseline_search():
    source = source_text().lower()
    assert "requests" not in source and "http://" not in source and "https://" not in source


def test_32_no_training_api():
    source = source_text()
    assert ".fit(" not in source and "optimizer" not in source.lower()


def test_33_primary_tiny_threshold_protocol():
    assert final.freeze_manifest()["reconstruction_backbone"]["primary_threshold"] == "normal_validation_q99"


def test_34_v2_aggregation_frozen():
    assert "target-group max then top2 mean" in final.freeze_manifest()["structure_v2_static"]["method"]


def test_35_claim_categories_exact():
    assert set(final.claim_boundary()) == {"SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"}


def test_36_formal_outputs_exist():
    required = ["artifact_inventory.csv", "metric_provenance_registry.csv", "evidence_validity_registry.csv", "final_method_freeze_manifest.yaml", "final_claim_boundary.yaml", "unresolved_issues_master.csv", "final_method_decision_matrix.csv", "final_hai_unified_evidence_package.md"]
    assert all((final.OUT / name).exists() for name in required)


def test_37_provenance_ids_unique_and_sources_hashed():
    frame = pd.read_csv(final.OUT / "metric_provenance_registry.csv")
    assert frame.metric_id.is_unique
    assert not frame.source_hash.eq("source_not_in_inventory").any()


def test_38_tiny_parameter_count_matches_cost_table():
    table = pd.read_csv(final.OUT / "table_07_lightweight_cost.csv")
    value = int(table.loc[table.component.eq("tiny"), "parameter_or_state_count"].iloc[0])
    assert value == final.freeze_manifest()["reconstruction_backbone"]["parameter_count"]


def test_39_a1m_classification_matches_source():
    summary = final.read_json("outputs/hai_structure_v2a1m_mode_conditioned_reference/structure_v2a1m_summary.json")
    table = pd.read_csv(final.OUT / "table_05_structure_v2_evolution.csv")
    assert table.loc[table.variant.eq("A1M_mode_conditioned"), "classification"].iloc[0] == summary["classification"]


def test_40_a1c_classification_matches_source():
    summary = final.read_json("outputs/hai_structure_v2a1c_causal_adaptive_reference/structure_v2a1c_summary.json")
    table = pd.read_csv(final.OUT / "table_05_structure_v2_evolution.csv")
    assert table.loc[table.variant.eq("A1C_causal_adaptive"), "classification"].iloc[0] == summary["classification"]


def test_41_v1_real_auroc_matches_source():
    summary = final.read_json("outputs/hai_step4cd_structure_real_and_complementarity/structure_real_evaluation_summary.json")
    table = pd.read_csv(final.OUT / "table_04_structure_v1_controlled_and_real.csv")
    assert float(table.loc[table.method.eq("Structure_V1_real_HAI"), "auroc"].iloc[0]) == pytest.approx(summary["structure_point"]["auroc"])


def test_42_unresolved_issue_ids_are_deduplicated():
    frame = pd.read_csv(final.OUT / "unresolved_issues_master.csv")
    assert not frame.duplicated(["issue_id", "research_question"]).any()


def test_43_all_formal_csv_numeric_values_finite():
    for path in final.OUT.glob("*.csv"):
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include="number")
        assert not numeric.isna().any().any(), path.name


def test_44_markdown_csv_rows_match():
    paths = list(final.OUT.glob("table_*.csv")) + [final.OUT / "final_method_decision_matrix.csv"]
    for path in paths:
        markdown = path.with_suffix(".md").read_text(encoding="utf-8")
        rows = len([line for line in markdown.splitlines() if line.startswith("| ")]) - 2
        assert rows == len(pd.read_csv(path)), path.name


def test_45_stage_status_complete():
    import json
    status = json.loads((final.OUT / "stage_status.json").read_text(encoding="utf-8"))
    assert all(value is True for key, value in status.items() if key != "input_fingerprint")
