from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy
import sklearn
import torch
import yaml

from data.transforms import StandardScaler
from losses.recon import ReconstructionLoss
from models.backbone_ae import BackboneAE
from utils.config_io import load_config
from utils.project_paths import get_repo_root, resolve_output_path
from utils.seed import set_seed


def test_required_environment_imports_have_versions() -> None:
    modules = (np, pd, scipy, sklearn, torch, yaml, pytest)
    assert all(getattr(module, "__version__", None) for module in modules)


def test_core_project_modules_import() -> None:
    assert StandardScaler.__module__ == "data.transforms"
    assert ReconstructionLoss.__module__ == "losses.recon"
    assert callable(load_config)
    assert callable(set_seed)


def test_repo_and_output_roots_are_local_to_thesis_worktree() -> None:
    root = get_repo_root()
    assert root == Path(__file__).resolve().parents[1]
    output = resolve_output_path("outputs/thesis_environment_smoke")
    assert output == root / "outputs" / "thesis_environment_smoke"
    assert root in output.parents


def test_basic_torch_tensor_computation() -> None:
    values = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(values.square(), torch.tensor([1.0, 4.0, 9.0]))
    assert isinstance(torch.cuda.is_available(), bool)
    if torch.cuda.is_available():
        assert torch.cuda.get_device_name(0)


def test_reconstruction_backbone_preserves_btc_shape() -> None:
    torch.manual_seed(7)
    model = BackboneAE(in_channels=5, hidden_channels=8, latent_channels=4, dropout=0.0).eval()
    batch = torch.randn(2, 16, 5)
    with torch.no_grad():
        reconstruction = model(batch)
    assert reconstruction.shape == batch.shape
    assert torch.isfinite(reconstruction).all()
