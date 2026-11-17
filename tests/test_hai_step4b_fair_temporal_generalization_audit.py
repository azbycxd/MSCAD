from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hai_step4b_b_audit_fair_temporal_generalization.py"
SPEC = importlib.util.spec_from_file_location("step4bb", SCRIPT)
step4bb = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(step4bb)


def test_directly_reuses_crossfit_manifest():
    source = inspect.getsource(step4bb.inspect_manifest)
    assert "crossfit_manifest.csv" in source and "pd.read_csv" in source


def test_existing_manifest_reports_zero_summary_overlap():
    frame, audit = step4bb.inspect_manifest()
    assert audit["source_window_overlap_count"] == 0
    assert len(frame) == 12


def test_missing_per_window_fold_map_blocks_fair_comparison():
    _, audit = step4bb.inspect_manifest()
    assert audit["per_window_assignment_available"] is False


def test_both_scores_require_the_same_source_fold_fields():
    required = set(step4bb.REQUIRED_SCORE_FIELDS)
    assert {"source_window_id", "sequence_id", "fold"}.issubset(required)
    assert any(name.startswith("original_distance_max") for name in required)
    assert any(name.startswith("relation_tail_top2_mean") for name in required)


def test_original_max_is_not_recomputed():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "robust_distances" not in source and "np.max(" not in source
    assert "validation_z" not in source and "features_w128" not in source


def test_tail_top2_is_not_refit_or_recomputed():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "empirical_tail" not in source and "searchsorted" not in source
    assert "global_median" not in source and "global_scale" not in source


def test_no_new_q99_is_fit_without_scores():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "quantile" not in source and "percentile" not in source


def test_in_sample_rows_are_diagnostic_only():
    source = inspect.getsource(step4bb.main)
    assert '"diagnostic_only" if protocol.endswith("in_sample")' in source


def test_no_raw_test_or_model_loading():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = "\n".join(ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))).lower()
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "torch" not in imports and "models" not in imports
    assert "data_root" not in source and "test1.csv" not in source
    assert "torch.load" not in source and "load_state_dict" not in source


def test_three_historical_trees_are_hash_guarded():
    source = inspect.getsource(step4bb.main).lower()
    assert all(token in source for token in ("before", "after", "step4b_r", "step4b_a", "historical artifact changed"))


def test_formal_outputs_have_no_nonfinite_numbers_when_present():
    if not step4bb.OUTPUT.exists():
        pytest.skip("blocked formal audit has not run")
    for path in step4bb.OUTPUT.glob("*.csv"):
        numeric = pd.read_csv(path).select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all(), path.name
