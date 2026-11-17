import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "outputs" / "thesis_baselines" / "madtcn_2026"
REPO = ROOT / "external" / "baselines" / "MAD-TCN"


def load_json(name: str):
    return json.loads((AUDIT_DIR / name).read_text(encoding="utf-8"))


def test_official_repo_head_and_clean_status():
    audit = load_json("repo_audit.json")
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--short"], text=True
    ).strip()
    assert head == audit["repository"]["head"]
    assert status == ""
    assert audit["repository"]["clean"] is True


def test_reproducibility_matrix_records_required_gaps():
    with (AUDIT_DIR / "reproducibility_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = {row["component"]: row for row in csv.DictReader(handle)}
    assert rows["model_definition"]["status"] == "FOUND"
    for component in (
        "dataset_loader",
        "preprocessing",
        "training_script",
        "validation_protocol",
        "test_evaluation",
        "configuration_hyperparameters",
        "training_objective_loss",
        "anomaly_score",
    ):
        assert rows[component]["status"] == "MISSING"


def test_parameter_claim_is_not_relabelled_as_source_measurement():
    method = load_json("method_manifest.json")
    assert method["paper"]["paper_parameter_claim"] == "approximately 0.026M"
    assert method["paper"]["paper_parameter_claim_status"] == "REFERENCE_ONLY"
    assert method["source_derived_parameter_count"] is None


def test_score_normalization_and_leakage_are_not_invented():
    protocol = load_json("protocol.json")
    assert protocol["anomaly_score"]["formula"] is None
    assert protocol["anomaly_score"]["status"] == "MISSING_DO_NOT_INVENT"
    assert protocol["normalization"]["official_transform"] is None
    assert set(protocol["leakage_audit"].values()) >= {"UNKNOWN"}


def test_hai_fallback_is_not_frozen_before_swat_protocol():
    protocol = load_json("protocol.json")
    assert protocol["official_swat_configuration"]["status"] == "MISSING"
    assert protocol["datasets"]["HAI21"]["fallback_frozen"] is False
    assert protocol["unified_thesis_adaptation"]["frozen"] is False


def test_runtime_constructor_dependency_is_recorded():
    source = (REPO / "models" / "MAD_TCN.py").read_text(encoding="utf-8")
    audit = load_json("repo_audit.json")
    assert "args.alpha_value" in source
    assert audit["source_runtime_issue"]["present"] is True


def test_no_training_and_no_test_data_read():
    protocol = load_json("protocol.json")
    controls = protocol["controls"]
    assert controls["model_training"] is False
    assert controls["test_data_read"] is False
    assert controls["hyperparameter_search"] is False
    assert controls["score_invented"] is False
    assert controls["commit"] is False
    assert controls["push"] is False
    assert protocol["final_status"] == "MAD_TCN_2026_OFFICIAL_CODE_INCOMPLETE"
