from __future__ import annotations

import torch
from torch import nn

from utils.cov import structure_distance_batch


class StructureLoss(nn.Module):
    def __init__(self, mode: str = "corr") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, pair_index: torch.Tensor | None = None) -> torch.Tensor:
        return structure_distance_batch(x, x_hat, mode=self.mode, pair_index=pair_index).mean()
