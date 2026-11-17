from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import data.factory as factory
import data.hai as hai
from data.transforms import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
HAI_CONFIG = ROOT / "configs" / "hai_rec_only_seed1.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_import_isolated_from_hai_state_and_factory() -> None:
    before = {
        "features": hai.load_hai_csv.__defaults__,
        "labels": hai.LABEL_CANDIDATES,
        "factory": factory.build_dataloaders,
        "config": _sha(HAI_CONFIG),
    }
    swat2026 = importlib.import_module("data.swat2026")
    assert swat2026.__name__ == "data.swat2026"
    assert hai.load_hai_csv.__defaults__ == before["features"]
    assert hai.LABEL_CANDIDATES == before["labels"]
    assert factory.build_dataloaders is before["factory"]
    assert _sha(HAI_CONFIG) == before["config"]
    assert "swat2026" not in Path(factory.__file__).read_text(encoding="utf-8").lower()


def test_swat_has_no_hai_scaler_checkpoint_or_output_state() -> None:
    swat2026 = importlib.import_module("data.swat2026")
    assert not hasattr(swat2026, "scaler")
    assert not hasattr(swat2026, "checkpoint")
    assert StandardScaler.__module__ == "data.transforms"
    assert "outputs/swat2026" != "outputs/hai_step3a_rec_baseline_seed1"


def test_main_worktree_baseline_files_are_not_written() -> None:
    main_root = ROOT.parent / "mscad"
    assert main_root.exists()
    assert not (main_root / "data" / "swat2026.py").exists()
    assert not (main_root / "configs" / "swat2026").exists()
