from __future__ import annotations

import torch
from torch import nn


def ot_svd_denoise(x: torch.Tensor, rank_ratio: float = 0.5, shrink: float = 0.1) -> torch.Tensor:
    """
    Lightweight OT-SVD-style denoise over each sample window.
    x: [B, L, D], denoise each [L, D] matrix by low-rank SVD shrink.
    """
    # Fast low-rank denoise via batched covariance eigendecomposition.
    # Compared with batched SVD this is much more stable/efficient on SWaT scale.
    b, l, d = x.shape
    k = max(1, min(d, int(round(d * float(rank_ratio)))))
    mu = x.mean(dim=1, keepdim=True)
    xc = x - mu
    cov = torch.matmul(xc.transpose(1, 2), xc) / max(1, l - 1)  # [B, D, D]
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    # take top-k eigenvectors
    vk = evecs[:, :, -k:]  # [B, D, k]
    x_low = torch.matmul(torch.matmul(xc, vk), vk.transpose(1, 2))
    if shrink > 0:
        # blend low-rank signal and residual as soft shrinkage.
        x_res = xc - x_low
        x_low = x_low + (1.0 - float(shrink)) * x_res
    return x_low + mu


class MTADCore(nn.Module):
    """
    MTAD-core baseline under unified protocol:
    Transformer encoder + reconstruction head + one-step prediction head.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        ff_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 512,
        decoder_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.decoder_type = str(decoder_type).lower()

        self.input_proj = nn.Linear(self.in_channels, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.dropout = nn.Dropout(float(dropout))

        self.rec_head_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.in_channels),
        )
        self.rec_head_conv = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(self.d_model, self.in_channels, kernel_size=1, bias=True),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.in_channels),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, D]
        b, l, d = x.shape
        if d != self.in_channels:
            raise ValueError(f"Input channel mismatch: got {d}, expected {self.in_channels}")
        if l > self.max_len:
            raise ValueError(f"Sequence length {l} exceeds max_len={self.max_len}")
        h = self.input_proj(x) + self.pos_emb[:, :l, :]
        h = self.dropout(h)
        z = self.encoder(h)  # [B, L, d_model]
        if self.decoder_type == "conv":
            zc = z.transpose(1, 2)  # [B, d_model, L]
            x_rec = self.rec_head_conv(zc).transpose(1, 2)
        else:
            x_rec = self.rec_head_mlp(z)  # [B, L, D]
        x_pred = self.pred_head(z[:, -1, :])  # [B, D]
        return x_rec, x_pred
