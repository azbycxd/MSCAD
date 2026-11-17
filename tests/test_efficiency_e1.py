import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_efficiency_final"
SCRIPT = ROOT / "scripts" / "thesis" / "05_efficiency" / "run_efficiency_e1.py"
EXPECTED = {
    "SWaT": "af9f9aac9cb8b369015a94d4b88a4729d817c219d305eb3276050c5d33ad31bf",
    "HAI21": "a78e820567eaa15dd5a0cd2a9f6491d04102c0a50732a4de7aa918f1b20be5bb",
}


def load():
    summary = json.loads((OUT / "efficiency_summary.json").read_text())
    params = pd.read_csv(OUT / "ours_parameter_breakdown.csv")
    sizes = pd.read_csv(OUT / "ours_model_sizes.csv")
    latency = pd.read_csv(OUT / "ours_latency.csv")
    return summary, params, sizes, latency


def test_01_frozen_target_hashes():
    summary, *_ = load()
    assert summary["target_hashes"] == EXPECTED


def test_02_rec_params_are_actual_expected_counts():
    _, _, sizes, _ = load()
    got = sizes[sizes.method == "Ours-REC"].set_index("dataset").trainable_params.to_dict()
    assert got == {"SWaT": 11730, "HAI21": 15482}


def test_03_ridge_coefficients_are_complete():
    for dataset, count in (("swat", 15), ("hai21", 54)):
        with np.load(OUT / f"{dataset}_dual_coefficients.npz", allow_pickle=False) as artifact:
            keys = set(artifact.files)
        assert sum(key.endswith("_self_coef") for key in keys) == count
        assert sum(key.endswith("_self_intercept") for key in keys) == count
        assert sum(key.endswith("_cross_coef") for key in keys) == count
        assert sum(key.endswith("_cross_intercept") for key in keys) == count


def test_04_intercepts_are_counted():
    _, params, *_ = load()
    targets = params[params.row_type == "TARGET"]
    assert (targets.self_params == 6).all()
    assert (targets.cross_params == targets.cross_feature_dimension + 1).all()


def test_05_tail_has_zero_trainable_parameters():
    _, _, sizes, _ = load()
    tail = sizes[sizes.method == "Ours-TAIL-CALIBRATION"]
    assert (tail.trainable_params == 0).all()


def test_06_full_parameters_equal_rec_plus_dual():
    _, _, sizes, _ = load()
    pivot = sizes.pivot(index="dataset", columns="method", values="trainable_params")
    assert (pivot["Ours-FULL"] == pivot["Ours-REC"] + pivot["Ours-DUAL"]).all()


def test_07_score_equivalence_gate():
    *_, latency = load()
    assert latency.score_equivalence_max_abs_diff.max() <= 1e-10


def test_08_warmup_is_100():
    *_, latency = load()
    assert (latency.warmup == 100).all()


def test_09_timed_endpoints_are_1000():
    *_, latency = load()
    assert (latency.timed_endpoints == 1000).all()


def test_10_batch_size_is_one():
    *_, latency = load()
    assert (latency.batch_size == 1).all()


def test_11_baseline_values_are_read_only():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "baseline_table_template.csv" in source
    assert ".to_csv(" not in source


def test_12_no_training_api():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert ".train()" not in source


def test_13_no_pruning_or_quantization_api():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "torch.quantization" not in source
    assert "torch.nn.utils.prune" not in source


def test_14_latency_is_finite():
    *_, latency = load()
    assert np.isfinite(latency[["mean_latency_ms", "median_latency_ms", "p95_latency_ms"]]).all().all()


def test_15_artifact_sizes_are_real():
    _, _, sizes, _ = load()
    paths = {
        ("SWaT", "Ours-REC"): ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline" / "best.pt",
        ("SWaT", "Ours-DUAL"): OUT / "swat_dual_coefficients.npz",
        ("SWaT", "Ours-TAIL-CALIBRATION"): OUT / "swat_tail_calibration.npz",
        ("HAI21", "Ours-REC"): ROOT / "outputs" / "thesis_hai21_b1_real" / "rec_best.pt",
        ("HAI21", "Ours-DUAL"): OUT / "hai21_dual_coefficients.npz",
        ("HAI21", "Ours-TAIL-CALIBRATION"): OUT / "hai21_tail_calibration.npz",
    }
    indexed = sizes.set_index(["dataset", "method"])
    for key, path in paths.items():
        assert path.stat().st_size == int(indexed.loc[key, "model_size_bytes"])
    for dataset in ("SWaT", "HAI21"):
        components = sum(int(indexed.loc[(dataset, method), "model_size_bytes"]) for method in ("Ours-REC", "Ours-DUAL", "Ours-TAIL-CALIBRATION"))
        assert components == int(indexed.loc[(dataset, "Ours-FULL"), "model_size_bytes"])
