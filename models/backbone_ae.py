from __future__ import annotations

import torch
from torch import nn

from models.layers import DepthwiseSeparableConv1d, ConvBNAct1d, ResidualDSBlock


class BackboneAE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, latent_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, hidden_channels, kernel_size=3, dropout=dropout),
            DepthwiseSeparableConv1d(hidden_channels, latent_channels, kernel_size=3, dropout=dropout),
        )
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv1d(latent_channels, hidden_channels, kernel_size=3, dropout=dropout),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> conv over time with channel=D
        x_c = x.transpose(1, 2)  # [B, D, L]
        z = self.encoder(x_c)
        x_hat_c = self.decoder(z)
        return x_hat_c.transpose(1, 2)  # [B, L, D]


class BackboneAEPlus(nn.Module):
    """
    A slightly stronger but still lightweight backbone:
    1 stem Conv + 2 residual depthwise blocks + latent bottleneck + symmetric decoder.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 96, latent_channels: int = 48, dropout: float = 0.1):
        super().__init__()
        self.stem = ConvBNAct1d(in_channels, hidden_channels, kernel_size=3, dropout=dropout)
        self.enc_res1 = ResidualDSBlock(hidden_channels, dropout=dropout, kernel_size=3)
        self.enc_res2 = ResidualDSBlock(hidden_channels, dropout=dropout, kernel_size=3)
        self.to_latent = DepthwiseSeparableConv1d(hidden_channels, latent_channels, kernel_size=3, dropout=dropout)
        self.from_latent = DepthwiseSeparableConv1d(latent_channels, hidden_channels, kernel_size=3, dropout=dropout)
        self.dec_res = ResidualDSBlock(hidden_channels, dropout=dropout, kernel_size=3)
        self.out = nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=True)
        self.skip = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.transpose(1, 2)  # [B, D, L]
        h = self.stem(x_c)
        h = self.enc_res1(h)
        h = self.enc_res2(h)
        z = self.to_latent(h)
        h2 = self.from_latent(z)
        h2 = self.dec_res(h2)
        out = self.out(h2) + self.skip(x_c)
        return out.transpose(1, 2)  # [B, L, D]
