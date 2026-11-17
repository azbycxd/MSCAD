from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from data.factory import build_dataloaders
from models.mtad import MTADCore, ot_svd_denoise
from utils.config_io import load_config, save_json
from utils.seed import set_seed


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_effective_loss_weights(cfg: dict) -> tuple[float, float]:
    model_variant = str(cfg["model"].get("variant", "core")).lower()
    w_rec = float(cfg["train"].get("lambda_rec", 0.5))
    w_pred = float(cfg["train"].get("lambda_pred", 0.5))
    if model_variant == "r":
        w_pred = 0.0
    elif model_variant == "p":
        w_rec = 0.0
    s = w_rec + w_pred
    if s <= 0:
        raise ValueError("Invalid MTAD weights: lambda_rec + lambda_pred must be > 0")
    return w_rec, w_pred


def is_ot_svd_enabled(cfg: dict) -> bool:
    variant = str(cfg["model"].get("variant", "core")).lower()
    if variant == "full":
        return True
    if variant in {"w", "core", "r", "p"}:
        return False
    return bool(cfg["model"].get("ot_svd_enabled", False))


def get_decoder_type(cfg: dict) -> str:
    variant = str(cfg["model"].get("variant", "core")).lower()
    if variant in {"w", "full"}:
        return "conv"
    return str(cfg["model"].get("decoder_type", "mlp")).lower()


def get_targets(xb: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    variant = str(cfg["model"].get("variant", "core")).lower()
    x_ctx = xb[:, :-1, :]
    x_next = xb[:, -1, :]
    if variant != "full":
        return x_ctx, x_next
    with torch.no_grad():
        den = ot_svd_denoise(
            xb,
            rank_ratio=float(cfg["model"].get("ot_svd_rank_ratio", 0.5)),
            shrink=float(cfg["model"].get("ot_svd_shrink", 0.1)),
        )
    return den[:, :-1, :], den[:, -1, :]


def mtad_loss(
    x_rec: torch.Tensor,
    x_ctx: torch.Tensor,
    x_pred: torch.Tensor,
    x_next: torch.Tensor,
    w_rec: float,
    w_pred: float,
) -> dict:
    rec_loss = ((x_rec - x_ctx) ** 2).mean()
    pred_loss = ((x_pred - x_next) ** 2).mean()
    total = w_rec * rec_loss + w_pred * pred_loss
    return {"loss": total, "rec_loss": rec_loss, "pred_loss": pred_loss}


def train_one_epoch(model: MTADCore, loader, optimizer, device: torch.device, w_rec: float, w_pred: float) -> dict:
    model.train()
    running = {"loss": 0.0, "rec_loss": 0.0, "pred_loss": 0.0}
    n = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        if xb.size(1) < 2:
            continue
        x_ctx = xb[:, :-1, :]
        x_ctx_tgt, x_next_tgt = get_targets(xb, model.cfg_ref)
        x_rec, x_pred = model(x_ctx)
        out = mtad_loss(x_rec, x_ctx_tgt, x_pred, x_next_tgt, w_rec=w_rec, w_pred=w_pred)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        optimizer.step()
        bs = xb.size(0)
        n += bs
        for k in running:
            running[k] += float(out[k].detach().item()) * bs
    for k in running:
        running[k] /= max(1, n)
    return running


@torch.no_grad()
def eval_loss(model: MTADCore, loader, device: torch.device, w_rec: float, w_pred: float) -> float:
    model.eval()
    total = 0.0
    n = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        if xb.size(1) < 2:
            continue
        x_ctx = xb[:, :-1, :]
        x_ctx_tgt, x_next_tgt = get_targets(xb, model.cfg_ref)
        x_rec, x_pred = model(x_ctx)
        out = mtad_loss(x_rec, x_ctx_tgt, x_pred, x_next_tgt, w_rec=w_rec, w_pred=w_pred)
        bs = xb.size(0)
        total += float(out["loss"].detach().item()) * bs
        n += bs
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mtad_swat.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(cfg.get("device", "cuda"))

    bundle = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))

    model_cfg = cfg["model"]
    model = MTADCore(
        in_channels=len(bundle.feature_names),
        d_model=int(model_cfg.get("d_model", 64)),
        nhead=int(model_cfg.get("nhead", 4)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        ff_dim=int(model_cfg.get("ff_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_len=int(model_cfg.get("max_len", 512)),
        decoder_type=get_decoder_type(cfg),
    ).to(device)
    model.cfg_ref = cfg
    w_rec, w_pred = get_effective_loss_weights(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    best_val = float("inf")
    best_epoch = -1
    bad = 0
    patience = int(cfg["train"]["patience"])
    hist = []
    t0_all = time.perf_counter()

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(model, bundle.train_loader, optimizer, device, w_rec=w_rec, w_pred=w_pred)
        val = eval_loss(model, bundle.val_loader, device, w_rec=w_rec, w_pred=w_pred)
        dt = time.perf_counter() - t0
        row = {"epoch": epoch, **tr, "val_loss": float(val), "epoch_seconds": float(dt)}
        hist.append(row)
        print(
            f"[Epoch {epoch:03d}] loss={tr['loss']:.4f} rec={tr['rec_loss']:.4f} "
            f"pred={tr['pred_loss']:.4f} val={val:.4f} time={dt:.2f}s"
        )
        if val < best_val:
            best_val = float(val)
            best_epoch = epoch
            bad = 0
            ckpt = {
                "model_state": model.state_dict(),
                "cfg": cfg,
                "feature_names": bundle.feature_names,
                "variant": str(cfg["model"].get("variant", "core")),
                "lambda_rec_eff": w_rec,
                "lambda_pred_eff": w_pred,
            }
            torch.save(ckpt, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    total_time = time.perf_counter() - t0_all
    train_summary = {
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "total_train_seconds": float(total_time),
        "variant": str(cfg["model"].get("variant", "core")),
        "lambda_rec_eff": float(w_rec),
        "lambda_pred_eff": float(w_pred),
        "history": hist,
    }
    save_json(str(out_dir / "train_metrics.json"), train_summary)
    print(f"Training done. Best epoch: {best_epoch}, best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
