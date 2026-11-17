import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "thesis_baselines" / "baseline_manifest.json"
DOC = ROOT / "docs" / "thesis" / "baseline_b2_usad_normalization_policy.md"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_usad_minmax_is_train_only_and_unclipped():
    policy = load_manifest()["normalization_policy"]
    usad = policy["methods"]["USAD"]
    assert usad["scaler"] == "TRAIN_ONLY_MINMAX_0_1"
    assert usad["fit_scope"] == "Normal Train only"
    assert usad["validation_rows_seen_during_fit"] == 0
    assert usad["test_rows_seen_during_fit"] == 0
    assert usad["clip"] is False


def test_official_usad_core_and_score_remain_frozen():
    usad = load_manifest()["baselines"]["USAD"]
    assert "shared encoder" in usad["architecture"]
    assert "two decoders" in usad["architecture"]
    assert usad["decoder_output"] == "sigmoid"
    assert usad["score_weights"] == {"alpha": 0.5, "beta": 0.5}
    assert "0.5*(x-w1)^2 + 0.5*(x-w3)^2" in usad["score"]


def test_other_scalers_and_evaluation_protocol_are_unchanged():
    manifest = load_manifest()
    methods = manifest["normalization_policy"]["methods"]
    assert methods["LSTM-AE"]["scaler"] == "TRAIN_ONLY_STANDARD"
    assert methods["Ours"]["scaler"] == "TRAIN_ONLY_STANDARD"
    assert methods["GDN"]["scaler"] == "TRAIN_ONLY_STANDARD"
    assert manifest["common"]["window"] == 64
    assert manifest["common"]["threshold"]["source"] == "Normal Validation endpoint scores only"
    assert manifest["common"]["point_adjustment"] is False
    assert manifest["common"]["pot"] is False
    assert manifest["common"]["test_tuning"] is False


def test_range_audit_is_normal_train_only():
    audit = load_manifest()["normalization_policy"]["normal_train_range_audit"]
    assert audit["validation_data_read"] is False
    assert audit["test_data_read"] is False
    assert audit["SWaT"]["train_rows"] == 396000
    assert audit["HAI21"]["train_rows"] == 737280
    for dataset in ("SWaT", "HAI21"):
        assert audit[dataset]["standard_min"] < 0 < audit[dataset]["standard_max"]
        assert abs(audit[dataset]["minmax_min"]) <= 1e-12
        assert abs(audit[dataset]["minmax_max"] - 1.0) <= 1e-12


def test_policy_phase_did_not_train_or_read_test():
    manifest = load_manifest()
    text = DOC.read_text(encoding="utf-8")
    assert manifest["status"] == "USAD_NORMALIZATION_POLICY_FROZEN"
    assert manifest["model_training"] is False
    assert manifest["commit"] is False and manifest["push"] is False
    assert "No Validation or Test rows were opened" in text
