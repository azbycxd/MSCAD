from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hai_step4b_a_repair_relation_aggregation.py"
SPEC = importlib.util.spec_from_file_location("step4ba", SCRIPT)
step4ba = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(step4ba)


def test_no_raw_hai_or_test_loading():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "data_root" not in source and "hai-22.04" not in source
    assert "test1.csv" not in source and "test2.csv" not in source


def test_no_ae_checkpoint_or_rec_loading():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = "\n".join(ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))).lower()
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "torch" not in imports and "models" not in imports
    assert "torch.load" not in source and "load_state_dict" not in source


def test_relation_features_are_loaded_not_recomputed():
    source = inspect.getsource(step4ba.main)
    assert "clean_relation_features_w128.npz" not in source  # resolved through frozen artifact registry
    assert "validation_z" in source and "features_w128" in source
    assert "corrcoef" not in SCRIPT.read_text(encoding="utf-8")


def test_reference_is_not_refit():
    source = inspect.getsource(step4ba.main)
    assert "global_median" in source and "global_scale" in source
    assert "fit_robust" not in source and "median(clean" not in source


def test_crossfit_source_window_is_never_in_own_calibration():
    sequence = np.repeat(np.arange(6), 10)
    starts = np.tile(np.arange(10), 6)
    frame, masks = step4ba.build_crossfit_manifest(sequence, starts)
    assert (frame.source_window_overlap_count == 0).all()
    assert not set(masks["A_calibration"]) & set(masks["A_evaluation"])
    assert not set(masks["B_calibration"]) & set(masks["B_evaluation"])


def test_crossfit_halves_are_chronological_complements():
    sequence = np.repeat(np.arange(6), 12)
    starts = np.tile(np.arange(12), 6)
    _, masks = step4ba.build_crossfit_manifest(sequence, starts)
    assert set(masks["A_calibration"]) == set(masks["B_evaluation"])
    assert set(masks["B_calibration"]) == set(masks["A_evaluation"])


def test_each_relation_gets_independent_ecdf():
    calibration = np.column_stack([np.arange(10), np.arange(10) * 100])
    values = np.array([[5.0, 500.0], [9.0, 0.0]])
    actual = step4ba.relation_tail(calibration, values)
    assert np.isclose(actual[0, 0], actual[0, 1])
    assert not np.isclose(actual[1, 0], actual[1, 1])


def test_tail_probability_is_positive_finite_and_tie_deterministic():
    calibration = np.array([1.0, 1.0, 2.0, 3.0])
    values = np.array([1.0, 1.0, 100.0])
    result = step4ba.empirical_tail_surprisal(calibration, values)
    assert np.isfinite(result).all() and (result >= 0).all()
    assert result[0] == result[1]
    assert np.isclose(result[-1], -np.log10(1 / 5))


def test_candidate_set_is_strictly_three():
    assert step4ba.CANDIDATES == ("original_distance_max", "relation_tail_max", "relation_tail_top2_mean")


def test_q99_threshold_uses_calibration_clean_only():
    source = inspect.getsource(step4ba.run_crossfit)
    line = next(line for line in source.splitlines() if "threshold = float(np.quantile" in line)
    assert "aggregate_cal" in line and "positive" not in line and "altered" not in line


def test_global_controls_are_exact_clean_aliases():
    source = inspect.getsource(step4ba.main)
    assert "distance_alias_source" in source and "absolute_fpr_gap\": 0.0" in source
    assert "max_score_difference\": 0.0" in source


def test_unrelated_edge_identity_has_chance_auroc():
    clean = np.arange(20, dtype=float)
    labels = np.r_[np.zeros(20), np.ones(20)]
    assert np.isclose(step4ba.roc_auc_score(labels, np.r_[clean, clean]), .5)


def test_historical_hashes_checked_before_and_after():
    source = inspect.getsource(step4ba.main)
    assert "before4b" in source and "after4b" in source and "before4br" in source and "after4br" in source
    assert "historical artifact changed" in source


def test_resume_short_circuits_completed_output():
    source = inspect.getsource(step4ba.main)
    assert "REUSE completed Step 4B-A" in source
    assert "aggregation_repair_summary.json" in source


def test_formal_outputs_are_finite_when_present():
    if not step4ba.OUTPUT.exists():
        pytest.skip("formal Step 4B-A not run")
    for path in step4ba.OUTPUT.glob("*.csv"):
        numeric = pd.read_csv(path).select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all(), path.name
