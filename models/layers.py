from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=pad, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.GELU(),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBNAct1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDSBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1, kernel_size: int = 3) -> None:
        super().__init__()
        self.main = DepthwiseSeparableConv1d(channels, channels, kernel_size=kernel_size, dropout=dropout)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main(x)
        return self.norm(x + y)
