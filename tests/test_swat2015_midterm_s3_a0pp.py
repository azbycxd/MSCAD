from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import genpareto

from data.swat2015_official import OfficialSWaTError


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts" / "thesis" / "01_baseline" / "evaluate_swat2015_midterm_a0pp_pot.py"
    spec = importlib.util.spec_from_file_location("evaluate_swat2015_midterm_a0pp_pot", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def scores() -> np.ndarray:
    return np.linspace(0.001, 1.0, 10000, dtype=np.float64)


def test_01_pot_accepts_only_validation_normal_scores():
    m = module()
    m.fit_static_pot(scores(), source_split="normal_validation")
    with pytest.raises(OfficialSWaTError):
        m.fit_static_pot(scores(), source_split="attack_test")


def test_02_u_is_fixed_validation_q95():
    m = module()
    fit = m.fit_static_pot(scores(), source_split="normal_validation")
    assert fit["u_quantile"] == 0.95
    assert fit["u"] == pytest.approx(np.quantile(scores(), 0.95))


def test_03_alpha_is_fixed_to_one_percent():
    m = module()
    assert m.ALPHA == 0.01


def test_04_exceedance_definition_is_strictly_greater_than_u():
    m = module()
    fit = m.fit_static_pot(scores(), source_split="normal_validation")
    assert fit["exceedance_count"] == int(np.sum(scores() > fit["u"]))


def test_05_gpd_location_is_fixed_to_zero():
    m = module()
    assert m.fit_static_pot(scores(), source_split="normal_validation")["gpd_loc_fixed"] == 0.0


def test_06_threshold_formula_matches_frozen_gpd_ppf():
    m = module()
    fit = m.fit_static_pot(scores(), source_split="normal_validation")
    expected = fit["u"] + genpareto.ppf(fit["q_cond"], fit["shape"], loc=0, scale=fit["scale"])
    assert fit["threshold_pot"] == pytest.approx(expected)


def test_07_test_prediction_has_no_label_dependency():
    m = module()
    score = np.asarray([0.1, 0.3])
    assert m.predict_static(score, 0.2).tolist() == [0, 1]


def test_08_a0pp_requires_frozen_endpoint_identity():
    m = module()
    timestamp = np.asarray([1, 2, 3], dtype=np.int64)
    labels = np.asarray([0, 1, 0], dtype=np.int8)
    m.assert_frozen_endpoints(timestamp, labels, 3)
    with pytest.raises(OfficialSWaTError):
        m.assert_frozen_endpoints(timestamp, labels, 4)


def test_09_auroc_must_be_invariant():
    m = module()
    m.assert_ranking_invariant({"auroc": 0.8, "aupr": 0.7}, {"auroc": 0.8, "aupr": 0.7})
    with pytest.raises(OfficialSWaTError):
        m.assert_ranking_invariant({"auroc": 0.81, "aupr": 0.7}, {"auroc": 0.8, "aupr": 0.7})


def test_10_aupr_must_be_invariant():
    m = module()
    with pytest.raises(OfficialSWaTError):
        m.assert_ranking_invariant({"auroc": 0.8, "aupr": 0.71}, {"auroc": 0.8, "aupr": 0.7})
