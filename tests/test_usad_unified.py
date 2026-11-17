import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "thesis" / "01_baseline" / "run_usad_unified.py"
spec = importlib.util.spec_from_file_location("usad_b2", SCRIPT)
usad = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = usad
spec.loader.exec_module(usad)


def test_frozen_architecture_dimensions_and_sigmoid():
    for channels in (51, 79):
        model = usad.USAD(channels)
        w = 64 * channels
        assert model.encoder.linear1.in_features == w
        assert model.encoder.linear1.out_features == w // 2
        assert model.encoder.linear2.out_features == w // 4
        assert model.encoder.linear3.out_features == 6400
        assert model.decoder1.linear1.out_features == w // 4
        assert model.decoder1.linear2.out_features == w // 2
        assert model.decoder1.linear3.out_features == w
        assert isinstance(model.decoder1.sigmoid, torch.nn.Sigmoid)
        assert isinstance(model.decoder2.sigmoid, torch.nn.Sigmoid)


def test_loss_formula_and_score_are_frozen():
    source = inspect.getsource(usad)
    assert "fraction = 1.0 / epoch" in source
    assert "loss1 = fraction" in source and "loss2 = fraction" in source
    assert "0.5 * (x_endpoint - s1) ** 2 + 0.5 * (x_endpoint - s3) ** 2" in source
    assert usad.BATCH == 256 and usad.WINDOW == 64 and usad.Z_SIZE == 6400


def test_protocol_and_prohibitions():
    manifest = json.loads((ROOT / "outputs" / "thesis_baselines" / "baseline_manifest.json").read_text(encoding="utf-8"))
    policy = manifest["normalization_policy"]["methods"]["USAD"]
    assert policy["scaler"] == "TRAIN_ONLY_MINMAX_0_1" and policy["clip"] is False
    assert manifest["common"]["point_adjustment"] is False
    assert manifest["common"]["pot"] is False
    assert manifest["common"]["test_tuning"] is False
    assert manifest["common"]["early_stopping"] == {"source": "Normal Validation loss only", "patience": 10, "min_delta": 0.0, "restore_best": True}


def test_preflight_loader_names_are_train_only():
    source = inspect.getsource(usad.main)
    before_output = source.split("OUT.mkdir", 1)[0]
    assert "load_attack_cache" not in before_output
    assert "test" not in before_output.lower()


def test_completed_outputs_follow_frozen_protocol():
    out = ROOT / "outputs" / "thesis_baselines" / "usad"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    swat = json.loads((out / "swat_metrics.json").read_text(encoding="utf-8"))
    hai = json.loads((out / "hai_metrics.json").read_text(encoding="utf-8"))
    efficiency = json.loads((out / "efficiency.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "USAD_BASELINE_COMPLETED"
    assert manifest["resource_preflight"]["status"] == "RESOURCE_PREFLIGHT_PASS"
    assert manifest["normalization"] == {"scaler": "TRAIN_ONLY_MINMAX_0_1", "clip": False}
    assert manifest["validation_monitor"] == "official val_loss1"
    assert manifest["no_test_tuning"] is True and manifest["point_adjustment"] is False and manifest["pot"] is False
    assert swat["endpoint_count"] == 449791 and swat["event_count"] == 36
    assert hai["endpoint_count"] == 401685 and hai["events"]["ALL"]["event_count"] == 50
    assert efficiency["datasets"]["SWaT"]["best_epoch"] == 1
    assert efficiency["datasets"]["HAI21"]["best_epoch"] == 1
    assert (out / "latest_training_state.pt").is_file()
