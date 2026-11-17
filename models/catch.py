from __future__ import annotations

import torch
from torch import nn


def _safe_corr(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [B, L, D]
    xc = x - x.mean(dim=1, keepdim=True)
    std = torch.sqrt(torch.mean(xc * xc, dim=1, keepdim=False) + eps)  # [B, D]
    cov = torch.einsum("bld,ble->bde", xc, xc) / max(1, x.size(1) - 1)  # [B, D, D]
    den = torch.einsum("bd,be->bde", std, std) + eps
    c = cov / den
    c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return c


class ChannelMaskGenerator(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.in_channels = int(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        mean = x.mean(dim=1)  # [B, D]
        std = x.std(dim=1, unbiased=False)  # [B, D]
        feat = torch.stack([mean, std], dim=-1)  # [B, D, 2]
        h = self.mlp(feat)  # [B, D, H]
        h = nn.functional.normalize(h, p=2, dim=-1)
        sim = torch.einsum("bdh,beh->bde", h, h) * torch.clamp(self.scale, 0.1, 10.0)
        mask = torch.sigmoid(sim)
        eye = torch.eye(self.in_channels, device=x.device).unsqueeze(0)
        mask = torch.maximum(mask, eye)
        return mask


class CFMBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, C], attn_bias: [T, T] additive mask (0 for allowed, negative for penalized)
        if attn_bias is None:
            y, _ = self.attn(x, x, x, need_weights=False)
        else:
            y, _ = self.attn(x, x, x, need_weights=False, attn_mask=attn_bias)
        x = self.norm1(x + self.dropout(y))
        z = self.ffn(x)
        x = self.norm2(x + self.dropout(z))
        return x


class CATCHModel(nn.Module):
    """
    Unified-protocol CATCH-style model:
    - frequency patching
    - channel fusion (plain/static/learned mask)
    - time-frequency reconstruction
    """

    def __init__(
        self,
        in_channels: int,
        window_size: int,
        patch_size: int = 8,
        patch_stride: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        ff_dim: int = 128,
        dropout: float = 0.1,
        variant: str = "core",
        mask_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.window_size = int(window_size)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.variant = str(variant).lower()
        self.freq_bins = self.window_size // 2 + 1
        self.num_patches = 1 + max(0, (self.freq_bins - self.patch_size) // self.patch_stride)
        self.tokens = self.in_channels * self.num_patches

        self.patch_proj = nn.Linear(2 * self.patch_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.tokens, d_model))
        self.blocks = nn.ModuleList([CFMBlock(d_model, nhead, ff_dim, dropout) for _ in range(int(num_layers))])
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * self.patch_size),
        )
        self.mask_gen = ChannelMaskGenerator(self.in_channels, hidden=int(mask_hidden))

        # token -> channel id mapping for attention masking
        channel_ids = []
        for c in range(self.in_channels):
            channel_ids.extend([c] * self.num_patches)
        self.register_buffer("token_channel_ids", torch.tensor(channel_ids, dtype=torch.long), persistent=False)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def _freq_patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, L, D]
        xf = torch.fft.rfft(x.transpose(1, 2), dim=-1)  # [B, D, F]
        real = xf.real
        imag = xf.imag
        pr = real.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)  # [B,D,P,patch]
        pi = imag.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        p = torch.cat([pr, pi], dim=-1)  # [B,D,P,2*patch]
        return p, real, imag

    def _freq_unpatchify(self, p_hat: torch.Tensor, f_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
        # p_hat: [B, D, P, 2*patch]
        b, d, npatch, _ = p_hat.shape
        patch = self.patch_size
        real_patch = p_hat[..., :patch]
        imag_patch = p_hat[..., patch:]
        real = torch.zeros((b, d, f_bins), device=p_hat.device, dtype=p_hat.dtype)
        imag = torch.zeros((b, d, f_bins), device=p_hat.device, dtype=p_hat.dtype)
        cnt = torch.zeros((b, d, f_bins), device=p_hat.device, dtype=p_hat.dtype)
        for i in range(npatch):
            s = i * self.patch_stride
            e = s + patch
            real[..., s:e] += real_patch[:, :, i, :]
            imag[..., s:e] += imag_patch[:, :, i, :]
            cnt[..., s:e] += 1.0
        cnt = torch.clamp(cnt, min=1.0)
        return real / cnt, imag / cnt

    def _build_attn_bias(self, mask_ch: torch.Tensor | None) -> torch.Tensor | None:
        # mask_ch: [D, D] in [0,1], build token-level additive bias [T, T].
        if mask_ch is None:
            return None
        ids = self.token_channel_ids
        m = mask_ch[ids[:, None], ids[None, :]]  # [T, T]
        # allowed close to 1, penalize low mask entries
        bias = -6.0 * (1.0 - torch.clamp(m, 0.0, 1.0))
        return bias

    def _channel_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        # return [D,D] or None
        if self.variant == "w":
            return None
        if self.variant == "core":
            c = _safe_corr(x).abs().mean(dim=0)
            c = c / (c.max() + 1e-6)
            eye = torch.eye(self.in_channels, device=x.device)
            return torch.maximum(c, eye)
        if self.variant == "full":
            m = self.mask_gen(x).mean(dim=0)
            return m
        # default: no mask
        return None

    def forward(self, x: torch.Tensor) -> dict:
        # x: [B,L,D]
        b, l, d = x.shape
        if d != self.in_channels:
            raise ValueError(f"Input channel mismatch: got {d}, expected {self.in_channels}")
        p, real, imag = self._freq_patchify(x)  # p [B,D,P,2*patch]
        b, d, npatch, pdim = p.shape
        tokens = p.reshape(b, d * npatch, pdim)
        h = self.patch_proj(tokens) + self.pos_emb[:, : d * npatch, :]
        mask_ch = self._channel_mask(x)
        attn_bias = self._build_attn_bias(mask_ch)
        for blk in self.blocks:
            h = blk(h, attn_bias=attn_bias)
        out_patch = self.recon_head(h).reshape(b, d, npatch, pdim)
        real_hat, imag_hat = self._freq_unpatchify(out_patch, f_bins=real.size(-1))
        xf_hat = torch.complex(real_hat, imag_hat)
        x_time_hat = torch.fft.irfft(xf_hat, n=l, dim=-1).transpose(1, 2)  # [B,L,D]
        return {
            "x_time_hat": x_time_hat,
            "real_hat": real_hat,
            "imag_hat": imag_hat,
            "real": real,
            "imag": imag,
            "mask_ch": mask_ch,
        }

    def clustering_regular_loss(self, mask_ch: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask_ch is None:
            z = torch.tensor(0.0, device=self.pos_emb.device)
            return z, z
        m = torch.clamp(mask_ch, 1e-6, 1.0)
        p = m / torch.clamp(m.sum(dim=-1, keepdim=True), min=1e-6)
        cluster = -(p * torch.log(p)).sum(dim=-1).mean()  # entropy minimization
        reg = m.abs().mean()
        return cluster, reg
