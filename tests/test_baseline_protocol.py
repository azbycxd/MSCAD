import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_baselines"


def load_manifest():
    return json.loads((OUT / "baseline_manifest.json").read_text(encoding="utf-8"))


def test_dataset_protocols_match_finals():
    manifest = load_manifest()
    swat = manifest["datasets"]["SWaT"]
    hai = manifest["datasets"]["HAI21"]
    assert swat["normal_train"] == [0, 396000]
    assert swat["normal_validation"] == [396000, 495000]
    assert swat["features"] == 51 and swat["events"] == 36
    assert len(hai["train_files"]) == 3 and len(hai["test_files"]) == 5
    assert hai["features"] == 79 and hai["events"] == 50


def test_common_protocol_is_frozen():
    manifest = load_manifest()
    common = manifest["common"]
    assert common["window"] == 64
    assert common["train_stride"] == common["validation_stride"] == common["test_stride"] == 1
    assert common["scaler_fit"] == "Normal Train only"
    assert common["threshold"]["source"] == "Normal Validation endpoint scores only"
    assert common["threshold"]["quantile"] == 0.99
    assert common["point_adjustment"] is False
    assert common["test_tuning"] is False
    assert manifest["seed"] == 1


def test_baseline_cores_and_endpoint_alignment():
    manifest = load_manifest()
    baselines = manifest["baselines"]
    assert set(baselines) == {"LSTM-AE", "USAD", "GDN"}
    assert "two decoders" in baselines["USAD"]["architecture"]
    assert "top-k graph" in baselines["GDN"]["architecture"]
    assert "endpoint" in baselines["LSTM-AE"]["score"]
    assert "local endpoint >=64" in manifest["common_endpoint_mask"]


def test_event_and_efficiency_freezes():
    manifest = load_manifest()
    efficiency = json.loads((OUT / "efficiency_protocol.json").read_text(encoding="utf-8"))
    assert manifest["datasets"]["SWaT"]["events"] == 36
    assert manifest["datasets"]["HAI21"]["events"] == 50
    assert efficiency["latency"]["batch_size"] == 1
    assert efficiency["latency"]["timed_legal_endpoints"] == 1000
    assert efficiency["latency"]["statistics_ms"] == ["mean", "median", "p95"]


def test_table_template_and_no_training():
    with (OUT / "baseline_table_template.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert set(row["Dataset"] for row in rows) == {"SWaT", "HAI21"}
    assert set(row["Method"] for row in rows) == {
        "LSTM-AE", "USAD", "GDN", "Ours-REC", "Ours-DUAL", "Ours-Tail"
    }
    manifest = load_manifest()
    assert manifest["model_training"] is False
    assert manifest["commit"] is False and manifest["push"] is False
