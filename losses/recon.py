from __future__ import annotations

import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.crit = nn.SmoothL1Loss(reduction="mean")

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.crit(x_hat, x)
