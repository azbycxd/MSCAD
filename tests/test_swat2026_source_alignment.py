from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scripts.swat2026.inventory_and_align_sources import (
    align_attack_to_session,
    build_session_timeline,
    classify_session,
    determine_readiness,
    profile_ot_csv,
)


def write_csv(path: Path, start: str, periods: int = 5, label: bool = False, alarm: str = "Inactive") -> str:
    frame = pd.DataFrame({
        "t_stamp": pd.date_range(start, periods=periods, freq="s").strftime("%d-%b-%Y %H:%M:%S"),
        "LIT101.Pv": range(periods), "MV101.Status": [1] * periods, "LS201.Alarm": [alarm] * periods,
    })
    if label:
        frame["Label"] = ["Normal"] * periods
    frame.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_csv_time_range_and_source_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "session.csv"
    before = write_csv(path, "2026-03-11 09:00:00")
    profile = profile_ot_csv(path, "S1")
    assert profile.row["start_time"] == "2026-03-11T09:00:00"
    assert profile.row["end_time"] == "2026-03-11T09:00:04"
    assert profile.row["duration_seconds"] == 4
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_gap_overlap_and_no_cross_file_windows(tmp_path: Path) -> None:
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    write_csv(first, "2026-03-11 09:00:00", 5)
    write_csv(second, "2026-03-11 09:00:10", 5)
    timeline = build_session_timeline([profile_ot_csv(first, "A"), profile_ot_csv(second, "B")])
    assert timeline.iloc[0].gap_to_next_seconds == 6
    assert not timeline.iloc[0].overlaps_next
    assert not timeline.iloc[0].cross_file_window_allowed
    write_csv(second, "2026-03-11 09:00:03", 5)
    overlap = build_session_timeline([profile_ot_csv(first, "A"), profile_ot_csv(second, "B")])
    assert overlap.iloc[0].overlaps_next
    assert overlap.iloc[0].duplicate_timestamp_count_between_files == 2


def test_no_label_or_filename_normal_is_not_official_evidence() -> None:
    assert not classify_session("normal_session")["confirmed_normal_session"]
    assert classify_session("normal_session")["confidence"] == "unresolved"


def test_full_and_partial_attack_alignment() -> None:
    session = {"session_id": "S", "start_time": "2026-03-11T09:00:00", "end_time": "2026-03-11T10:00:00"}
    full = align_attack_to_session({"attack_id": "A1", "attack_start": "2026-03-11T09:10:00", "attack_end": "2026-03-11T09:20:00"}, session)
    partial = align_attack_to_session({"attack_id": "A2", "attack_start": "2026-03-11T08:50:00", "attack_end": "2026-03-11T09:10:00"}, session)
    assert full["fully_contained"] and full["candidate_for_labeling"]
    assert partial["partially_contained"] and not partial["fully_contained"]


def test_cross_day_attack_and_unknown_timezone_preserved() -> None:
    session = {"session_id": "S", "start_time": "2026-03-11T23:00:00", "end_time": "2026-03-12T01:00:00"}
    attack = {"attack_id": "A", "attack_start": "2026-03-11T23:50:00", "attack_end": "2026-03-12T00:10:00", "timezone": "unknown"}
    aligned = align_attack_to_session(attack, session)
    assert aligned["fully_contained"]
    assert attack["timezone"] == "unknown"


def test_readiness_single_unlabelled_and_complete_sources() -> None:
    unresolved = [classify_session("S")]
    assert determine_readiness(1, unresolved, 0, False).startswith("E.")
    ready = [classify_session("N", official_normal=True, evidence="official"), classify_session("A", official_attack=True, evidence="official")]
    assert determine_readiness(2, ready, 1, True).startswith("H.")


def test_alarm_values_do_not_infer_attack(tmp_path: Path) -> None:
    path = tmp_path / "alarm.csv"
    write_csv(path, "2026-03-11 09:00:00", alarm="Active")
    profile = profile_ot_csv(path, "S")
    semantic = classify_session(profile.row["session_id"])
    assert not semantic["confirmed_attack_session"]


def test_source_contains_no_model_checkpoint_label_or_split_writes() -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "swat2026" / "inventory_and_align_sources.py"
    text = source.read_text(encoding="utf-8").lower()
    assert "import torch" not in text and "load_state_dict" not in text and "torch.load" not in text
    assert "data.hai" not in text and "data.swat2026" not in text
    assert "def generate_labels" not in text and "labels.csv" not in text and "to_parquet" not in text
