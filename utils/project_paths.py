"""Central path resolution for repository resources and local run artifacts."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "MSCAD_DATA_ROOT"
CHECKPOINT_ROOT_ENV = "MSCAD_CHECKPOINT_ROOT"


class DataRootNotFound(FileNotFoundError):
    """Raised when an external dataset path cannot be resolved."""


def get_repo_root() -> Path:
    """Return the worktree root without depending on the process CWD."""
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve an absolute path unchanged or a relative path under this worktree."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else get_repo_root() / path


def _legacy_data_root() -> Path:
    """The sole compatibility location for the existing local A12 dataset."""
    return get_repo_root().parent / "mscad" / "dataset" / "dataset"


def resolve_data_path(value: str | Path, *, data_root: str | Path | None = None) -> Path:
    """Resolve data by explicit path, configured root, environment, then compatibility root."""
    requested = Path(value).expanduser()
    if requested.is_absolute():
        resolved = requested
        root_display = "<explicit absolute path>"
    else:
        configured_root = data_root if data_root is not None else os.environ.get(DATA_ROOT_ENV)
        root = Path(configured_root).expanduser() if configured_root else _legacy_data_root()
        resolved = root / requested
        root_display = str(root)
    if not resolved.exists():
        raise DataRootNotFound(
            "Data root not found; "
            f"requested path={value!s}; {DATA_ROOT_ENV}={os.environ.get(DATA_ROOT_ENV)!r}; "
            f"resolved path={resolved}; data root={root_display}"
        )
    return resolved


def resolve_output_path(value: str | Path) -> Path:
    """Resolve relative outputs under the active worktree."""
    return resolve_repo_path(value)


def resolve_checkpoint_path(value: str | Path, *, checkpoint_root: str | Path | None = None) -> Path:
    """Resolve checkpoints as absolute, configured-root, environment-root, or repo-relative paths."""
    requested = Path(value).expanduser()
    if requested.is_absolute():
        return requested
    root = checkpoint_root if checkpoint_root is not None else os.environ.get(CHECKPOINT_ROOT_ENV)
    return Path(root).expanduser() / requested if root else resolve_repo_path(requested)
