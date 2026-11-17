from __future__ import annotations

from pathlib import Path

import pytest

from utils.project_paths import (
    CHECKPOINT_ROOT_ENV,
    DATA_ROOT_ENV,
    DataRootNotFound,
    get_repo_root,
    resolve_checkpoint_path,
    resolve_data_path,
    resolve_output_path,
    resolve_repo_path,
)


def test_repo_root_and_config_are_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = get_repo_root()
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("configs/swat2026/rec_baseline_seed1.yaml") == root / "configs/swat2026/rec_baseline_seed1.yaml"


def test_relative_outputs_stay_in_active_worktree() -> None:
    root = get_repo_root()
    output = resolve_output_path("outputs/swat2026_path_test")
    assert output == root / "outputs/swat2026_path_test"
    assert root in output.parents
    assert output != root.parent / "mscad" / "outputs/swat2026_path_test"


def test_data_root_environment_and_explicit_absolute_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data-root"
    relative = Path("swat/a12_2026/ot/session.csv")
    expected = root / relative
    expected.parent.mkdir(parents=True)
    expected.write_text("fixture", encoding="utf-8")
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    assert resolve_data_path(relative) == expected

    absolute = tmp_path / "explicit.csv"
    absolute.write_text("fixture", encoding="utf-8")
    assert resolve_data_path(absolute) == absolute


def test_missing_data_root_reports_request_environment_and_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, r"D:\missing-mscad-data")
    with pytest.raises(DataRootNotFound) as caught:
        resolve_data_path("swat/classic/physical/unknown.csv")
    message = str(caught.value)
    assert "Data root not found" in message
    assert "requested path=swat/classic/physical/unknown.csv" in message
    assert DATA_ROOT_ENV in message and "resolved path=" in message


def test_windows_paths_and_checkpoints_are_compatible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    absolute = tmp_path / "best.pt"
    assert resolve_checkpoint_path(absolute) == absolute
    assert resolve_checkpoint_path("outputs/m1/checkpoints/best.pt") == get_repo_root() / "outputs/m1/checkpoints/best.pt"
    monkeypatch.setenv(CHECKPOINT_ROOT_ENV, str(tmp_path / "checkpoint-root"))
    assert resolve_checkpoint_path("m1/best.pt") == tmp_path / "checkpoint-root" / "m1/best.pt"


def test_current_a12_compatibility_path_is_read_only() -> None:
    path = resolve_data_path("Swat_new/11-Mar-2026_0900_1700.csv")
    assert path.is_file()
    assert path.name == "11-Mar-2026_0900_1700.csv"
