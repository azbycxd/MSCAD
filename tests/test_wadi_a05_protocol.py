"""Contract tests for the frozen WADI-A0.5 protocol."""
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_wadi_a05_protocol"
P = json.loads((OUT / "protocol_freeze.json").read_text(encoding="utf-8"))
M = pd.read_csv(OUT / "missingness_by_column.csv")
B = pd.read_csv(OUT / "event_binding_final.csv")
F = pd.read_csv(OUT / "feature_schema_frozen.csv")
R = pd.read_csv(OUT / "attack13_raw_timestamp_audit.csv")


def test_01_a2_only():
    assert P["dataset_variant"] == "WADI.A2 only"
    assert P["exact_sources"] == ["WADI_14days_new.csv", "WADI_attackdataLABLE.csv", "table_WADI.pdf"]


def test_02_exclusion_is_train_only():
    assert P["exclusion_rule"].startswith("exclude iff")
    assert P["attack_test_used_for_exclusion"] is False
    assert set(M.loc[M.missingness_class == "ALL_NAN_NORMAL_TRAIN", "column"]) == set(P["excluded_features"])


def test_03_no_test_driven_exclusion():
    assert P["no_attack_driven_feature_exclusion"] is True
    assert (F.decision_data == "Normal Train only").all()


def test_04_no_imputation():
    assert P["imputation"] == "NONE"


def test_05_nan_breaks_windows_and_lags():
    assert P["window_or_lag_crosses_invalid_point"] is False
    assert "hard session boundary" in P["invalid_point_policy"]


def test_06_raw_timestamp_ranges_checked():
    assert set(R.audit_range) == {"canonical_00h", "official_12h"}
    assert P["attack13"]["csv_raw_ranges"]["canonical_00h"]["rows_checked"] == 481
    assert P["attack13"]["csv_raw_ranges"]["official_12h"]["rows_checked"] == 481


def test_07_pdf_id13_verified():
    a = P["attack13"]
    assert a["pdf_visual_and_text_verified"] is True
    assert a["official_pdf_start"] == "2017-10-11T12:07:30"
    assert a["official_pdf_end"] == "2017-10-11T12:10:52"
    assert {"2_FIT_003", "2_PIT_003"}.issubset(a["official_targets"])


def test_08_no_event_specific_12h_patch():
    t = P["timestamp_parser"]
    assert t["twelve_hour_clock_used"] is False
    assert t["twelve_to_zero_conversion_used"] is False
    assert t["event_specific_12h_patch"] is False


def test_09_attack13_conflict_preserved():
    row = B.loc[B.attack_id == 13].iloc[0]
    assert row.final_binding_status == "SOURCE_CONFLICT_UNRESOLVED"
    assert not bool(row.include_in_primary_event_evaluation)
    assert P["attack13_status"] == "ATTACK13_SOURCE_CONFLICT"
    assert P["attack13"]["csv_attack_points_during_exact_official_id13_interval"] == 0


def test_10_high_bindings_confirmed():
    high = B.loc[B.attack_id.isin([2, 3, 4, 7, 15])]
    assert len(high) == 5 and high.final_binding_status.eq("BOUND").all()
    assert P["event_protocol"]["all_high_relevance_bound"] is True


def test_11_feature_count_consistent():
    assert len(F) == 127
    assert int(F.included.sum()) == P["frozen_feature_count"]
    assert int((~F.included).sum()) == P["excluded_feature_count"]


def test_12_valid_ratio_gate():
    ratios = [P["splits"][x]["valid_point_ratio"] for x in ("train", "validation", "test")]
    assert (P["missingness_status"] == "MISSINGNESS_PROTOCOL_READY") == (min(ratios) >= 0.99)


def test_13_no_model():
    assert P["no_model_training"] is True and P["no_model_inference"] is True


def test_14_official_primary_has_14_bound():
    assert P["event_protocol"]["bound_primary_event_count"] == 14
    assert P["event_protocol"]["unresolved_ids"] == [13]
