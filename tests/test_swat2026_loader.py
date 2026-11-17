from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.swat2026 import build_swat2026_windows, load_swat2026_csv


def _write_fixture(path: Path, alarm_values: tuple[str, ...] = ("Inactive", "Active", "Inactive", "Active")) -> None:
    pd.DataFrame(
        {
            "t_stamp": pd.date_range("2026-03-11 09:00:00", periods=4, freq="s").strftime("%d-%b-%Y %H:%M:%S"),
            "P1_STATE": [0, 1, 1, 0],
            "LIT101.Pv": [1.5, 2.5, 3.5, 4.5],
            "FIT101.Pv": [0.0, 0.1, 0.2, 0.3],
            "MV101.Status": [0, 1, 1, 0],
            "LS201.Alarm": list(alarm_values),
            "Label": ["Normal", "Normal", "Attack", "Normal"],
        }
    ).to_csv(path, index=False)


def test_process_values_excludes_timestamp_label_status_and_alarm(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_fixture(path)
    loaded = load_swat2026_csv(path)
    assert loaded.feature_names == ["LIT101.Pv", "FIT101.Pv"]
    assert loaded.features.shape == (4, 2)
    assert loaded.labels.tolist() == [0, 0, 1, 0]
    assert loaded.metadata["timestamp_column"] == "t_stamp"
    assert loaded.metadata["scaler_fitted"] is False
    assert loaded.metadata["checkpoint_loaded"] is False
    assert np.isfinite(loaded.features).all()


def test_status_switch_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_fixture(path)
    loaded = load_swat2026_csv(path, include_status_columns=True)
    assert loaded.feature_names == ["LIT101.Pv", "FIT101.Pv", "P1_STATE", "MV101.Status"]


def test_active_inactive_is_explicitly_mapped(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_fixture(path)
    loaded = load_swat2026_csv(path, include_alarm_columns=True)
    index = loaded.feature_names.index("LS201.Alarm")
    assert loaded.features[:, index].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert loaded.categorical_mappings["LS201.Alarm"] == {"Active": 1.0, "Inactive": 0.0}


def test_unknown_category_does_not_silently_become_zero(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_fixture(path, ("Inactive", "Bad Input", "Active", "Inactive"))
    with pytest.raises(ValueError, match="Unknown categorical"):
        load_swat2026_csv(path, include_alarm_columns=True)


def test_missing_numeric_does_not_silently_become_zero(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_fixture(path)
    frame = pd.read_csv(path)
    frame.loc[1, "LIT101.Pv"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing numeric"):
        load_swat2026_csv(path)


def test_explicit_drop_rows_policy_is_audited(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_fixture(path)
    frame = pd.read_csv(path)
    frame["LIT101.Pv"] = frame["LIT101.Pv"].astype(object)
    frame.loc[1, "LIT101.Pv"] = "Bad Input"
    frame.to_csv(path, index=False)
    with pytest.warns(RuntimeWarning, match="drop_rows"):
        loaded = load_swat2026_csv(path, missing_numeric_policy="drop_rows")
    assert loaded.features.shape == (3, 2)
    assert loaded.metadata["dropped_row_count"] == 1
    assert loaded.metadata["dropped_row_indices"] == [1]


def test_windows_align_to_endpoint_and_input_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_fixture(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_swat2026_csv(path)
    windows = build_swat2026_windows(loaded.features, loaded.labels, loaded.timestamps, window_size=3, stride=1)
    assert windows.features.shape == (2, 3, 2)
    assert windows.labels.tolist() == [1, 0]
    assert windows.timestamps.tolist() == loaded.timestamps[[2, 3]].tolist()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_label_values_never_select_features(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_fixture(first)
    _write_fixture(second)
    frame = pd.read_csv(second)
    frame["Label"] = ["Attack"] * len(frame)
    frame.to_csv(second, index=False)
    assert load_swat2026_csv(first).feature_names == load_swat2026_csv(second).feature_names
