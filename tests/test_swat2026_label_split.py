from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.swat2026.build_labels_and_split import (
    build_point_labels,
    build_split,
    audit_model_features,
    load_approved_schedule,
    validate_label_counts,
)
from scripts.swat2026.extract_and_review_info_sheet import (
    KNOWN_SCHEDULE_IMAGE_SHA256,
    make_schedule,
)


def full_timestamps() -> pd.Series:
    return pd.Series(pd.date_range("2026-03-11 09:00:00", "2026-03-11 17:00:59", freq="s"))


def write_schedule(path: Path) -> None:
    rows = make_schedule({"sha256": KNOWN_SCHEDULE_IMAGE_SHA256, "extracted_path": "media/image1.png"}, Path("official.docx"))
    pd.DataFrame(rows).to_csv(path, index=False)


def approved_schedule(tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "schedule.csv"
    write_schedule(path)
    approved, validation = load_approved_schedule(path)
    assert validation["all_rows_human_approved"]
    return approved


def test_loads_eleven_human_approved_rows_without_mutating_candidate(tmp_path: Path) -> None:
    path = tmp_path / "schedule.csv"
    write_schedule(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    approved, validation = load_approved_schedule(path)
    assert len(approved) == 11 and validation["issues"] == []
    assert approved["human_review_status"].eq("approved").all()
    assert approved["description_source"].eq("copied_from_attack_name").all()
    assert approved["timezone"].eq("local_clock_unspecified").all()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_left_open_labels_shared_boundary_and_expected_counts(tmp_path: Path) -> None:
    labels = build_point_labels(full_timestamps(), approved_schedule(tmp_path))
    counts = validate_label_counts(labels)
    assert counts["test_phase_points"] == 14400
    assert counts["attack_point_count"] == 3120
    assert counts["test_normal_point_count"] == 11280
    assert counts["attack_prevalence"] == 3120 / 14400
    boundary = labels[labels.timestamp.eq("2026-03-11T15:02:00")].iloc[0]
    assert boundary.point_label == 1 and boundary.attack_id == "7"
    assert labels[labels.phase.eq("attack_experiment_phase")].point_label.nunique() == 2


def test_outside_sixty_seconds_not_evaluation_eligible(tmp_path: Path) -> None:
    labels = build_point_labels(full_timestamps(), approved_schedule(tmp_path))
    outside = labels[labels.phase.eq("outside_declared_phase")]
    assert len(outside) == 60 and outside.point_label.eq(0).all()
    assert not outside.evaluation_eligible.any()
    assert labels.timezone_policy.eq("local_clock_unspecified").all()


def quality_fixture(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    timestamps = pd.Series(pd.date_range("2026-03-11 09:00:00", periods=6, freq="s"))
    labels = pd.DataFrame({
        "timestamp": timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S"), "phase": ["normal_phase"] * 6,
        "point_label": [0] * 6, "evaluation_eligible": [False] * 6,
    })
    features = [f"F{i:02d}.Pv" for i in range(31)]
    data = {"t_stamp": timestamps.dt.strftime("%d-%b-%Y %H:%M:%S")}
    data.update({name: ["1", "2", "3", "4", "5", "6"] for name in features})
    data.update({f"Other{i:02d}.Status": ["0"] * 6 for i in range(55)})
    frame = pd.DataFrame(data)
    frame.loc[0, features[0]] = "Bad Input"
    frame.loc[1, features[1]] = ""
    frame.loc[2, features[2]] = "not-number"
    frame.loc[3, features[3]] = "inf"
    frame.loc[4, "Other00.Status"] = "Bad Input"
    return frame, labels, features


def test_31_feature_invalid_counts_are_separate_from_all_87_columns(tmp_path: Path) -> None:
    frame, labels, features = quality_fixture(tmp_path)
    quality, cells, invalid_rows, validity, details = audit_model_features(frame, labels, features)
    assert frame.shape[1] == 87 and details["feature_count"] == 31
    assert details["all_87_columns_bad_input_count"] == 2
    assert details["model_31_features_bad_input_count"] == 1
    assert details["model_invalid_cell_count"] == 4
    assert len(cells) == 4 and len(invalid_rows) == 4 and len(validity) == len(frame)
    assert validity.invalid_feature_count.sum() == 4
    aggregate = quality[quality.feature.eq("__ALL_31_MODEL_FEATURES__")].set_index("region")
    assert aggregate.loc["full_session", "invalid_cell_count"] == 4


def test_invalid_mask_preserves_timeline_and_raw_values(tmp_path: Path) -> None:
    frame, labels, features = quality_fixture(tmp_path)
    before = frame.copy(deep=True)
    _, _, _, validity, _ = audit_model_features(frame, labels, features)
    assert len(validity) == len(frame)
    assert frame.equals(before)
    assert validity.loc[0, "all_model_features_valid"] == False  # noqa: E712


def test_chronological_split_ranges_context_and_128_windows(tmp_path: Path) -> None:
    timestamps = full_timestamps()
    labels = build_point_labels(timestamps, approved_schedule(tmp_path))
    validity = labels[["timestamp", "phase", "point_label", "evaluation_eligible"]].copy()
    validity["all_model_features_valid"] = True
    manifest, ranges, windows, leakage = build_split(timestamps, validity, 128, 0.8)
    by_role = ranges.set_index("split_role")
    assert manifest["train_raw_point_count"] == 11418
    assert manifest["validation_raw_point_count"] == 2855
    assert manifest["test_context_point_count"] == 127
    assert by_role.loc["test_context", "start_time_inclusive"] == "2026-03-11T12:57:53"
    assert by_role.loc["test_evaluation", "start_time_inclusive"] == "2026-03-11T13:00:00"
    assert windows.set_index("split_role").loc["train", "candidate_window_count"] == 11291
    assert windows.set_index("split_role").loc["validation", "candidate_window_count"] == 2728
    assert windows.set_index("split_role").loc["test_evaluation", "candidate_window_count"] == 14400
    assert leakage["status"] == "A. leakage_free"


def test_split_excludes_february_and_has_no_raw_overlap(tmp_path: Path) -> None:
    timestamps = full_timestamps()
    labels = build_point_labels(timestamps, approved_schedule(tmp_path))
    validity = labels[["timestamp", "phase", "point_label", "evaluation_eligible"]].copy()
    validity["all_model_features_valid"] = True
    _, ranges, _, leakage = build_split(timestamps, validity, 128, 0.8)
    assert leakage["raw_timestamp_overlap_pairs"] == []
    assert leakage["excluded_unresolved_sessions"] == ["19-Feb-2026", "20-Feb-2026"]
    assert leakage["windows_cross_csv"] is False
    assert ranges.source_file.nunique() == 1
    assert leakage["scaler_fitted"] is False and leakage["model_trained"] is False


def test_source_contains_no_training_scaler_checkpoint_hai_or_formal_arrays() -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "swat2026" / "build_labels_and_split.py"
    text = source.read_text(encoding="utf-8").lower()
    assert "torch" not in text and "load_state_dict" not in text
    assert ".fit(" not in text and "standardscaler" not in text and "minmaxscaler" not in text
    assert "data/hai.py" not in text and "np.save" not in text and "to_parquet" not in text
    assert "19-feb-2026_" not in text and "20-feb-2026_" not in text
