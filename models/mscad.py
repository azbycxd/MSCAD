from __future__ import annotations

import torch
from torch import nn

from losses.recon import ReconstructionLoss
from losses.structure import StructureLoss
from models.backbone_ae import BackboneAE, BackboneAEPlus
from utils.cov import structure_distance_batch


class MSCAD(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        latent_channels: int = 32,
        dropout: float = 0.1,
        cov_type: str = "corr",
        short_window: int = 16,
        backbone_variant: str = "base",
    ) -> None:
        super().__init__()
        self.backbone_variant = backbone_variant
        if backbone_variant == "plus":
            self.backbone = BackboneAEPlus(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                latent_channels=latent_channels,
                dropout=dropout,
            )
        else:
            self.backbone = BackboneAE(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                latent_channels=latent_channels,
                dropout=dropout,
            )
        self.rec_loss_fn = ReconstructionLoss()
        self.struct_loss_fn = StructureLoss(mode=cov_type)
        self.short_window = short_window
        self.cov_type = cov_type
        self.register_buffer("structure_pair_index", torch.empty((0, 2), dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def set_structure_pairs(self, pair_index: list[list[int]] | torch.Tensor | None) -> None:
        if pair_index is None:
            self.structure_pair_index = torch.empty((0, 2), dtype=torch.long, device=self.structure_pair_index.device)
            return
        if isinstance(pair_index, torch.Tensor):
            p = pair_index.long()
        else:
            if len(pair_index) == 0:
                self.structure_pair_index = torch.empty((0, 2), dtype=torch.long, device=self.structure_pair_index.device)
                return
            p = torch.tensor(pair_index, dtype=torch.long)
        if p.ndim != 2 or p.size(1) != 2:
            raise ValueError("pair_index must be shape [P,2].")
        self.structure_pair_index = p.to(self.structure_pair_index.device)

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        lambda_rec: float,
        lambda_s: float,
        lambda_l: float,
    ) -> dict:
        x_short = x[:, -self.short_window :, :]
        xhat_short = x_hat[:, -self.short_window :, :]

        rec = self.rec_loss_fn(x_hat, x)
        pair_idx = self.structure_pair_index if self.structure_pair_index.numel() > 0 else None
        s = self.struct_loss_fn(x_short, xhat_short, pair_index=pair_idx)
        l = self.struct_loss_fn(x, x_hat, pair_index=pair_idx)
        total = lambda_rec * rec + lambda_s * s + lambda_l * l
        return {"loss": total, "rec_loss": rec, "short_loss": s, "long_loss": l}

    @torch.no_grad()
    def score_components(self, x: torch.Tensor, x_hat: torch.Tensor) -> dict:
        rec = (x_hat - x).abs().mean(dim=(1, 2))
        pair_idx = self.structure_pair_index if self.structure_pair_index.numel() > 0 else None
        short = structure_distance_batch(
            x[:, -self.short_window :, :], x_hat[:, -self.short_window :, :], mode=self.cov_type, pair_index=pair_idx
        )
        long = structure_distance_batch(x, x_hat, mode=self.cov_type, pair_index=pair_idx)
        return {"rec": rec, "short": short, "long": long}
