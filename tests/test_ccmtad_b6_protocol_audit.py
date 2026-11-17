import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_baselines" / "ccmtad_2026"
REPO = ROOT / "external" / "baselines" / "CCM-TAD"


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_official_source_chain():
    audit = load_json("repo_audit.json")
    assert audit["repository"]["head"] == "3a8bc8263b648a707cd0658bf4f06517fb9e52c5"
    assert audit["repository"]["mirror_head_match"] is True
    assert all(v == "FOUND" for v in audit["required_chain"].values())
    assert (REPO / "run_anomaly.py").is_file()
    assert (REPO / "scripts" / "config" / "config.xlsx").is_file()


def test_official_clone_was_clean():
    assert load_json("repo_audit.json")["repository"]["clean"] is True


def test_official_swat_configuration_is_exactly_recorded():
    cfg = load_json("method_manifest.json")["official_swat_configuration"]
    assert (cfg["seq_len"], cfg["pred_len"], cfg["channels"]) == (24, 1, 51)
    assert (cfg["clusters_nheads"], cfg["d_model"], cfg["num_mix"]) == (6, 128, 1)
    assert cfg["learning_rate"] == 0.00498096466936417
    assert (cfg["batch_size"], cfg["epochs"], cfg["seed"]) == (512, 30, 42)
    assert cfg["effective_early_stopping"] is False


def test_raw_score_is_traced():
    manifest = load_json("method_manifest.json")
    core = manifest["core"]
    assert core["raw_point_score"] == "mean-channel squared reconstruction error"
    assert "point_based_anomaly_evaluation.py" in manifest["source_locations"]["point_score_and_threshold"]


def test_cluster_implementation_is_traced():
    manifest = load_json("method_manifest.json")
    assert "correlation profiles" in manifest["core"]["channel_dependency"]
    assert "feature_clustering.py" in manifest["source_locations"]["cluster_algorithm"]


def test_pf1_and_sf1_flows_are_traced():
    manifest = load_json("method_manifest.json")
    assert "alpha" in manifest["core"]["sequential_score"]
    assert "Test labels" in manifest["official_swat_configuration"]["point_target"]
    assert "Test labels" in manifest["official_swat_configuration"]["sequential_target"]
    assert "sequence_based_anomaly_evaluation.py" in manifest["source_locations"]["sequential_score"]


def test_checkpoint_leakage_is_not_sanitized():
    protocol = load_json("protocol.json")
    assert "Test PF1 or SF1" in protocol["official_protocol"]["model_selection"]


def test_threshold_and_alpha_leakage_are_not_sanitized():
    protocol = load_json("protocol.json")
    assert "Test-label" in protocol["official_protocol"]["sequential_score"]
    assert protocol["strict_thesis_candidate"]["sequential_alpha"] == "UNRESOLVED_REQUIRES_REVIEW"


def test_leakage_forces_review_status():
    protocol = load_json("protocol.json")
    assert protocol["status"] == "CCM_TAD_2026_PROTOCOL_REQUIRES_REVIEW"
    assert protocol["strict_thesis_candidate"]["hai_fallback_frozen"] is False


def test_centered_future_looking_preprocessing_is_disclosed():
    protocol = load_json("protocol.json")
    assert protocol["strict_thesis_candidate"]["moving_average"].startswith("UNRESOLVED")


def test_no_training_or_test_read_occurred():
    audit = load_json("repo_audit.json")["audit_boundaries"]
    protocol = load_json("protocol.json")
    assert audit["model_training"] is False
    assert audit["test_data_read"] is False
    assert audit["hyperparameter_search"] is False
    assert protocol["model_training"] is False
    assert protocol["test_data_read"] is False
