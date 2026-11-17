from __future__ import annotations

import torch


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=1, keepdim=True)


def covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    # x: [B, L, D]
    xc = _center(x)
    denom = max(1, x.size(1) - 1)
    return xc.transpose(1, 2).matmul(xc) / float(denom)  # [B, D, D]


def correlation_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    cov = covariance_matrix(x)
    std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps))  # [B, D]
    denom = std.unsqueeze(-1) * std.unsqueeze(-2) + eps
    return cov / denom


def shrinkage_covariance_matrix(x: torch.Tensor, shrinkage: float = 0.1) -> torch.Tensor:
    cov = covariance_matrix(x)
    d = cov.size(-1)
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype).unsqueeze(0).expand_as(cov)
    tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    target = (tr / float(d)) * eye
    return (1.0 - shrinkage) * cov + shrinkage * target


def build_structure(x: torch.Tensor, mode: str = "corr") -> torch.Tensor:
    m = mode.lower()
    if m == "cov":
        return covariance_matrix(x)
    if m == "corr":
        return correlation_matrix(x)
    if m == "shrink_cov":
        return shrinkage_covariance_matrix(x)
    raise ValueError(f"Unsupported cov mode: {mode}")


def structure_distance_batch(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mode: str = "corr",
    pair_index: torch.Tensor | None = None,
) -> torch.Tensor:
    sx = build_structure(x, mode=mode)
    sy = build_structure(x_hat, mode=mode)
    if pair_index is not None and pair_index.numel() > 0:
        # pair_index: [P, 2], evaluate only selected relation entries.
        i = pair_index[:, 0].long()
        j = pair_index[:, 1].long()
        dx = sx[:, i, j]
        dy = sy[:, i, j]
        diff = dx - dy
        return torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-8)
    diff = sx - sy
    return torch.sqrt(torch.sum(diff * diff, dim=(-2, -1)) + 1e-8)
