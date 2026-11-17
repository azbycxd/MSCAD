"""Contract checks for the WADI.A2 A0 audit artifacts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"thesis_wadi_a0_audit"
DATA=ROOT/"dataset"/"WADI_A2_2019"
M=json.loads((OUT/"dataset_manifest.json").read_text(encoding="utf-8"))
S=json.loads((OUT/"audit_summary.json").read_text(encoding="utf-8"))
P=json.loads((OUT/"protocol_candidate.json").read_text(encoding="utf-8"))

def test_01_normal_filename():
    assert (DATA/"WADI_14days_new.csv").is_file()
    assert M["source_variant_guard"]=={"used":"WADI.A2","A1_used":False,"A3_used":False,
                                       "exact_files":["WADI_14days_new.csv","WADI_attackdataLABLE.csv","table_WADI.pdf"]}
def test_02_attack_filename(): assert (DATA/"WADI_attackdataLABLE.csv").is_file()
def test_03_pdf_parsed(): assert (DATA/"table_WADI.pdf").is_file() and M["pdf_pages"]==1
def test_04_pdf_is_a2_table(): assert M["pdf_updated_corrected_attack_table_confirmed"] is True
def test_05_label_mapping_is_observed(): assert M["label_mapping"]=={"1":0,"-1":1} and set(M["label_counts"])=={"1","-1"}
def test_06_label_intervals_automatic(): assert len(pd.read_csv(OUT/"label_intervals.csv"))==14
def test_07_official_count(): assert set(pd.read_csv(OUT/"official_attack_events.csv").attack_id)==set(range(1,16))
def test_08_binding_complete_audit():
    b=pd.read_csv(OUT/"attack_binding.csv")
    assert len(b)==15 and int(b.successfully_bound.sum())==14 and S["unbound_attack_ids"]==[13]
def test_09_variables_complete(): assert len(pd.read_csv(OUT/"variable_schema.csv"))==127
def test_10_label_not_feature(): assert M["label_column"] not in set(pd.read_csv(OUT/"variable_schema.csv").raw_column_name)
def test_11_grouping_not_data_driven():
    g=pd.read_csv(OUT/"process_groups.csv")
    assert g.channel_count.sum()==127 and g.grouping_basis.str.contains("no data/model/Attack").all()
def test_12_taxonomy_pre_model(): assert len(pd.read_csv(OUT/"attack_taxonomy.csv"))==15 and S["no_model_inference"]
def test_13_no_model_training(): assert S["no_model_training"] is True
def test_14_no_anomaly_metrics(): assert S["no_anomaly_metrics"] is True
def test_15_no_silent_timestamp_repair():
    assert S["no_silent_timestamp_repair"] is True
    assert M["normal_timestamp_audit"]["timestamp_repair_applied_to_source"] is False
    assert P["segment_aware"] is True and P["window_lag_cross_boundary"] is False
    gaps=M["normal_timestamp_audit"]["gap_positions"]
    assert gaps==[{"index_before":335998,"index_after":335999,
                   "timestamp_before":"2017-09-29T15:19:58","timestamp_after":"2017-10-02T16:39:59",
                   "gap_seconds":264001}]
