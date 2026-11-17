from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hai_step4b_r_audit_preserving_controls.py"
SPEC = importlib.util.spec_from_file_location("step4br", SCRIPT)
step4br = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(step4br)

from utils.controlled_relation_corruptions import block_permute_both, circular_shift_both  # noqa: E402
from utils.independent_structure_score import window_pair_fisher  # noqa: E402


@pytest.mark.parametrize("operation", ["shift", "permute"])
def test_common_permutation_preserves_pearson(operation):
    rng = np.random.default_rng(12)
    window = rng.normal(size=(128, 4))
    changed = circular_shift_both(window, (0, 1), 32) if operation == "shift" else block_permute_both(window, (0, 1))
    before = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
    after = np.corrcoef(changed[:, 0], changed[:, 1])[0, 1]
    assert np.isclose(before, after, atol=1e-14)


def test_common_permutation_preserves_fisher_z():
    rng = np.random.default_rng(8)
    window = rng.normal(size=(1, 64, 3))
    changed = np.asarray([circular_shift_both(window[0], (0, 1), 16)])
    pair = np.array([[0, 1]])
    assert np.allclose(window_pair_fisher(window, pair), window_pair_fisher(changed, pair), atol=1e-12)


def test_all_pool_edges_are_invariant_under_global_permutation():
    rng = np.random.default_rng(19)
    windows = rng.normal(size=(7, 64, 15))
    pairs = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13],
                        [0, 14], [2, 10], [5, 12]])
    changed = np.asarray([np.roll(window, 16, axis=0) for window in windows])
    assert np.allclose(window_pair_fisher(windows, pairs), window_pair_fisher(changed, pairs), atol=1e-12)


def test_target_collateral_unrelated_classification():
    pool = pd.DataFrame({"pair_id": [1, 2, 3, 4], "feature_i": ["A", "A", "C", "E"],
                         "feature_j": ["B", "C", "D", "F"]})
    assert step4br.edge_classes(pool, 0).tolist() == ["target_edge", "collateral_edges", "unrelated_edges", "unrelated_edges"]


def test_target_pair_threshold_is_clean_validation_q99():
    source = inspect.getsource(step4br.audit_breaking)
    assert "np.quantile(clean_score, .99)" in source
    assert "positive" not in next(line for line in source.splitlines() if "threshold = global_threshold" in line)


def test_no_hai_test_or_model_loading_code():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = "\n".join(ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))).lower()
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "torch" not in imports and "models" not in imports
    assert "torch.load" not in source and "load_state_dict" not in source
    assert "test1.csv" not in source and "test2.csv" not in source


def test_r1_r2_r3_are_only_read_from_frozen_cache():
    source = inspect.getsource(step4br.audit_breaking)
    assert "controlled[f\"features_w{window}\"]" in source
    assert "circular_shift" not in source and "block_permute" not in source and "replace_one" not in source


def test_reference_is_loaded_not_refit():
    source = inspect.getsource(step4br.references)
    assert "structure_reference_fallback_audit.csv" in source
    assert "median(" not in source and "quantile(" not in source and "fit" not in source


def test_source_hashes_are_checked_before_and_after():
    source = inspect.getsource(step4br.main)
    assert "before = source_hashes()" in source and "after = source_hashes()" in source
    assert "step4b_hashes_unchanged" in source


def test_global_control_uses_frozen_cache_without_raw_csv():
    source = inspect.getsource(step4br.audit_global_controls)
    assert "sampled_z.copy()" in source
    assert "pd.read_csv" not in source
    assert "raw_csv_loaded" in source


def test_formal_outputs_are_finite_when_present():
    output = ROOT / "outputs" / "hai_step4b_preserving_control_audit"
    if not output.exists():
        pytest.skip("formal audit has not run")
    for path in output.glob("*.csv"):
        numeric = pd.read_csv(path).select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all(), path.name
