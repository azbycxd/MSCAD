from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Must be set before importing numpy/torch on some Windows Conda stacks.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import numpy as np

from data.factory import build_dataloaders
from models.mscad import MSCAD
from utils.config_io import load_config, save_json
from utils.seed import set_seed


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_corr_pairs(x_train: np.ndarray, topk: int = 60, min_abs_corr: float = 0.7) -> list[list[int]]:
    c = np.corrcoef(x_train.T)
    np.fill_diagonal(c, 0.0)
    # sanitize invalid entries from near-constant channels
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(c), axis=None), c.shape))[0]
    out = []
    seen = set()
    for i, j in idx:
        if i == j:
            continue
        if abs(c[i, j]) < min_abs_corr:
            break
        a, b = sorted((int(i), int(j)))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append([a, b])
        if len(out) >= topk:
            break
    return out


def train_one_epoch(
    model: MSCAD,
    loader,
    optimizer,
    device: torch.device,
    lambda_rec: float,
    lambda_s: float,
    lambda_l: float,
) -> dict:
    model.train()
    running = {"loss": 0.0, "rec_loss": 0.0, "short_loss": 0.0, "long_loss": 0.0}
    n = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xh = model(xb)
        out = model.loss(
            xb,
            xh,
            lambda_rec=lambda_rec,
            lambda_s=lambda_s,
            lambda_l=lambda_l,
        )
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
def eval_loss(model: MSCAD, loader, device: torch.device, lambda_rec: float, lambda_s: float, lambda_l: float) -> float:
    model.eval()
    total = 0.0
    n = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xh = model(xb)
        out = model.loss(
            xb,
            xh,
            lambda_rec=lambda_rec,
            lambda_s=lambda_s,
            lambda_l=lambda_l,
        )
        bs = xb.size(0)
        total += float(out["loss"].detach().item()) * bs
        n += bs
    return total / max(1, n)


def apply_freeze_policy(model: MSCAD, train_cfg: dict) -> dict:
    freeze_backbone = bool(train_cfg.get("freeze_backbone", False))
    unfreeze_keywords = [str(x) for x in train_cfg.get("unfreeze_keywords", [])]
    total_params = 0
    trainable_params = 0

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    if unfreeze_keywords:
        for n, p in model.named_parameters():
            if any(k in n for k in unfreeze_keywords):
                p.requires_grad = True

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    if trainable_params <= 0:
        raise RuntimeError("No trainable parameters left after applying freeze policy.")
    return {
        "freeze_backbone": freeze_backbone,
        "unfreeze_keywords": unfreeze_keywords,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swat.yaml")
    parser.add_argument("--init-ckpt", default=None, help="Optional checkpoint to initialize model weights.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(cfg.get("device", "cuda"))

    data_cfg = cfg["data"]
    bundle = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))

    model_cfg = cfg["model"]
    model = MSCAD(
        in_channels=len(bundle.feature_names),
        hidden_channels=int(model_cfg["hidden_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        dropout=float(model_cfg["dropout"]),
        cov_type=cfg["train"]["cov_type"],
        short_window=int(data_cfg["short_window"]),
        backbone_variant=str(model_cfg.get("backbone_variant", "base")),
    ).to(device)
    if args.init_ckpt is not None:
        init = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(init["model_state"], strict=False)
    pair_cfg = cfg["train"].get("structure_pairs", {})
    if bool(pair_cfg.get("enabled", False)):
        pairs = select_corr_pairs(
            bundle.x_train_norm,
            topk=int(pair_cfg.get("topk", 60)),
            min_abs_corr=float(pair_cfg.get("min_abs_corr", 0.7)),
        )
        model.set_structure_pairs(pairs)
    else:
        pairs = []
    freeze_info = apply_freeze_policy(model, cfg["train"])
    print(
        f"Freeze policy | freeze_backbone={freeze_info['freeze_backbone']} "
        f"unfreeze={freeze_info['unfreeze_keywords']} "
        f"trainable={freeze_info['trainable_params']}/{freeze_info['total_params']}"
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    best_val = float("inf")
    best_epoch = -1
    patience = int(cfg["train"]["patience"])
    bad = 0
    hist = []
    t0_all = time.perf_counter()

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        t0 = time.perf_counter()
        warmup_ep = int(cfg["train"].get("structure_warmup_epochs", 0))
        if warmup_ep > 0:
            scale = min(1.0, float(epoch) / float(warmup_ep))
        else:
            scale = 1.0
        lambda_rec = float(cfg["train"]["lambda_rec"])
        lambda_s = float(cfg["train"]["lambda_s"]) * scale
        lambda_l = float(cfg["train"]["lambda_l"]) * scale

        tr = train_one_epoch(model, bundle.train_loader, optimizer, device, lambda_rec, lambda_s, lambda_l)
        val = eval_loss(model, bundle.val_loader, device, lambda_rec, lambda_s, lambda_l)
        dt = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            **tr,
            "val_loss": val,
            "epoch_seconds": dt,
            "lambda_rec_eff": lambda_rec,
            "lambda_s_eff": lambda_s,
            "lambda_l_eff": lambda_l,
        }
        hist.append(row)
        print(
            f"[Epoch {epoch:03d}] loss={tr['loss']:.4f} rec={tr['rec_loss']:.4f} "
            f"s={tr['short_loss']:.4f} l={tr['long_loss']:.4f} val={val:.4f} "
            f"lambda_s={lambda_s:.3f} lambda_l={lambda_l:.3f} time={dt:.2f}s"
        )
        if val < best_val:
            best_val = val
            best_epoch = epoch
            bad = 0
            ckpt = {
                "model_state": model.state_dict(),
                "cfg": cfg,
                "feature_names": bundle.feature_names,
                "structure_pairs": pairs,
            }
            torch.save(ckpt, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    total_time = time.perf_counter() - t0_all
    train_summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "total_train_seconds": total_time,
        "freeze_info": freeze_info,
        "history": hist,
    }
    save_json(str(out_dir / "train_metrics.json"), train_summary)
    print(f"Training done. Best epoch: {best_epoch}, best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
