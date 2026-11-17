from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
A0 = ROOT / "outputs" / "thesis_midterm_s1" / "a0_rec_baseline"
S4 = ROOT / "outputs" / "thesis_midterm_s4_oracle" / "oracle_f1_diagnostic.json"


def result() -> dict:
    return json.loads(S4.read_text(encoding="utf-8"))


def test_01_score_is_the_exact_stored_a0_score():
    with np.load(A0 / "test_scores.npz", allow_pickle=False) as payload:
        assert len(payload["endpoint_mae"]) == result()["test_endpoint_count"]
        assert np.isfinite(payload["endpoint_mae"]).all()


def test_02_labels_are_the_exact_stored_a0_labels():
    with np.load(A0 / "test_scores.npz", allow_pickle=False) as payload:
        assert len(payload["label"]) == result()["test_endpoint_count"]
        assert set(np.unique(payload["label"]).tolist()) == {0, 1}


def test_03_auroc_is_invariant():
    a0 = json.loads((A0 / "metrics.json").read_text(encoding="utf-8"))["primary_endpoint_mae"]
    assert result()["auroc"] == pytest.approx(a0["auroc"], abs=1e-6)


def test_04_aupr_is_invariant():
    a0 = json.loads((A0 / "metrics.json").read_text(encoding="utf-8"))["primary_endpoint_mae"]
    assert result()["aupr"] == pytest.approx(a0["aupr"], abs=1e-6)


def test_05_oracle_f1_is_not_below_formal_q99_f1():
    assert result()["oracle_f1"] >= result()["formal_q99_f1"]


def test_06_oracle_threshold_is_explicitly_selected_from_test_labels():
    assert result()["threshold_selected_from"] == "Test point labels; diagnostic only"
    assert result()["formal_threshold_change"] is False


def test_07_output_is_explicitly_oracle_only_and_nonformal():
    assert result()["oracle_only"] is True
    assert result()["formal_result"] is False
    assert "INVALID_FOR_FORMAL_BENCHMARK" in result()["scientific_status"]
