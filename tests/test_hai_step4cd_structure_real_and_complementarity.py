from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"hai_step4cd_evaluate_structure_and_complementarity.py"
SPEC=importlib.util.spec_from_file_location("step4cd",SCRIPT); step4cd=importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None; SPEC.loader.exec_module(step4cd)


def test_config_is_frozen_before_test_loading():
    source=inspect.getsource(step4cd.main)
    assert source.index("write_frozen_config()") < source.index("generate_score_cache(")


def test_frozen_relation_pool_window_and_aggregation():
    cfg=step4cd.frozen_config()
    assert cfg["relation_pool"]["pair_count"]==10 and cfg["relation_pool"]["channel_count"]==15
    assert cfg["window"]==128 and cfg["aggregation"]=="original_distance_max"


def test_reference_and_threshold_are_directly_reused():
    source=inspect.getsource(step4cd.load_registry)
    assert "structure_reference_fallback_audit.csv" in source and "structure_controlled_thresholds.csv" in source
    assert "quantile" not in source and "fit" not in source


def test_score_function_has_no_label_argument():
    assert list(inspect.signature(step4cd.score_numeric_test).parameters)==["values","pairs","median","scale"]


def test_tiny_score_is_loaded_from_npz():
    source=inspect.getsource(step4cd.main)
    assert "rec_channel_scores_seed1.npz" in SCRIPT.read_text(encoding="utf-8")
    assert "endpoint_raw_top5" in source


def test_no_ae_checkpoint_or_training_import():
    tree=ast.parse(SCRIPT.read_text(encoding="utf-8")); imports=" ".join(ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))).lower()
    source=SCRIPT.read_text(encoding="utf-8").lower()
    assert "torch" not in imports and "models" not in imports and "torch.load" not in source


def test_score_cache_is_dependency_bound_and_resumed():
    source=inspect.getsource(step4cd.generate_score_cache)
    assert "dependency_hash" in source and "return cache, True" in source
    assert source.count("np.savez_compressed") == 1


def test_timestamp_label_and_endpoint_alignment_is_strict():
    source=inspect.getsource(step4cd.main)
    assert "np.intersect1d" in source and "NPZ alignment error" in source
    assert "timestamp" in source and "label" in source


def test_or_and_logic():
    rec=np.array([0,1,0,1],bool); structure=np.array([0,0,1,1],bool)
    assert np.array_equal(rec|structure,[0,1,1,1]); assert np.array_equal(rec&structure,[0,0,0,1])


def test_no_point_adjustment_backfill_or_hysteresis():
    source=SCRIPT.read_text(encoding="utf-8").lower()
    assert "point_adjust" not in source.replace('"point_adjustment":false','')
    assert "backfill" not in source and "hysteresis" not in source


def test_event_detection_is_pointwise_inside_frozen_intervals():
    source=inspect.getsource(step4cd.event_table)
    assert "rows>=e.start_row" in source and "rows<=e.end_row" in source
    assert "hit" in source and "delay" in source


def test_historical_hash_guard_exists():
    source=inspect.getsource(step4cd.main)
    assert "before" in source and "after" in source and "historical_hashes_unchanged" in source


def test_rolling_fisher_shape_and_invariance():
    rng=np.random.default_rng(1); x=rng.normal(size=(200,3)); pairs=np.array([[0,1],[1,2]])
    z=step4cd.rolling_fisher(x,pairs,128)
    assert z.shape==(73,2) and np.isfinite(z).all()


def test_point_metrics_are_unadjusted():
    labels=np.array([0,0,1,1]); score=np.array([0.1,0.9,0.2,0.8]); alarm=score>.5
    result=step4cd.point_metrics(labels,score,alarm)
    assert result["tp"]==1 and result["fp"]==1 and result["fn"]==1


def test_formal_outputs_finite_when_present():
    if not step4cd.OUTPUT.exists(): pytest.skip("formal Step 4C+D not run")
    for path in step4cd.OUTPUT.glob("*.csv"):
        numeric=pd.read_csv(path).select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all(),path.name
