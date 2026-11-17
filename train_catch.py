from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from data.factory import build_dataloaders
from models.catch import CATCHModel
from utils.config_io import load_config, save_json
from utils.seed import set_seed


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_variant(cfg: dict) -> str:
    return str(cfg["model"].get("variant", "core")).lower()


def get_weights(cfg: dict) -> tuple[float, float, float, float]:
    v = get_variant(cfg)
    lt = float(cfg["train"].get("lambda_time", 1.0))
    lf = float(cfg["train"].get("lambda_freq", 1.0))
    lc = float(cfg["train"].get("lambda_cluster", 0.0))
    lr = float(cfg["train"].get("lambda_reg", 0.0))
    if v == "timeonly":
        lf = 0.0
    if v != "full":
        lc = 0.0
        lr = 0.0
    return lt, lf, lc, lr


def catch_loss(out: dict, x: torch.Tensor, lt: float, lf: float, lc: float, lr: float, model: CATCHModel) -> dict:
    x_time_hat = out["x_time_hat"]
    l_time = (x_time_hat - x).abs().mean()

    re = out["real"]
    im = out["imag"]
    reh = out["real_hat"]
    imh = out["imag_hat"]
    l_freq = ((reh - re) ** 2).mean() + ((imh - im) ** 2).mean()
    l_cluster, l_reg = model.clustering_regular_loss(out.get("mask_ch"))

    total = lt * l_time + lf * l_freq + lc * l_cluster + lr * l_reg
    return {
        "loss": total,
        "time_loss": l_time,
        "freq_loss": l_freq,
        "cluster_loss": l_cluster,
        "reg_loss": l_reg,
    }


def split_params_for_bilevel(model: CATCHModel) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    mask_params = []
    main_params = []
    mask_set = set(id(p) for p in model.mask_gen.parameters())
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in mask_set:
            mask_params.append(p)
        else:
            main_params.append(p)
    return main_params, mask_params


def train_one_epoch(
    model: CATCHModel,
    loader,
    opt_main,
    opt_mask,
    device: torch.device,
    lt: float,
    lf: float,
    lc: float,
    lr: float,
    variant: str,
) -> dict:
    model.train()
    running = {"loss": 0.0, "time_loss": 0.0, "freq_loss": 0.0, "cluster_loss": 0.0, "reg_loss": 0.0}
    n = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        l = catch_loss(out, xb, lt=lt, lf=lf, lc=lc, lr=lr, model=model)

        opt_main.zero_grad(set_to_none=True)
        l["loss"].backward()
        opt_main.step()

        # Bi-level-like alternating update for learned mask branch.
        if variant == "full" and opt_mask is not None:
            out2 = model(xb)
            l2 = catch_loss(out2, xb, lt=lt, lf=lf, lc=lc, lr=lr, model=model)
            opt_mask.zero_grad(set_to_none=True)
            l2["loss"].backward()
            opt_mask.step()

        bs = xb.size(0)
        n += bs
        for k in running:
            running[k] += float(l[k].detach().item()) * bs
    for k in running:
        running[k] /= max(1, n)
    return running


@torch.no_grad()
def eval_loss(model: CATCHModel, loader, device: torch.device, lt: float, lf: float, lc: float, lr: float) -> float:
    model.eval()
    total = 0.0
    n = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        l = catch_loss(out, xb, lt=lt, lf=lf, lc=lc, lr=lr, model=model)
        bs = xb.size(0)
        total += float(l["loss"].detach().item()) * bs
        n += bs
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/catch_swat.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(cfg.get("device", "cuda"))
    bundle = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    variant = get_variant(cfg)
    model = CATCHModel(
        in_channels=len(bundle.feature_names),
        window_size=int(data_cfg["window_size"]),
        patch_size=int(model_cfg.get("patch_size", 8)),
        patch_stride=int(model_cfg.get("patch_stride", 4)),
        d_model=int(model_cfg.get("d_model", 64)),
        nhead=int(model_cfg.get("nhead", 4)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        ff_dim=int(model_cfg.get("ff_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        variant=variant,
        mask_hidden=int(model_cfg.get("mask_hidden", 32)),
    ).to(device)
    lt, lf, lc, lr = get_weights(cfg)

    main_params, mask_params = split_params_for_bilevel(model)
    opt_main = torch.optim.AdamW(
        main_params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    opt_mask = None
    if variant == "full" and len(mask_params) > 0:
        opt_mask = torch.optim.AdamW(
            mask_params,
            lr=float(cfg["train"].get("mask_lr", cfg["train"]["lr"])),
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
        tr = train_one_epoch(
            model, bundle.train_loader, opt_main, opt_mask, device, lt=lt, lf=lf, lc=lc, lr=lr, variant=variant
        )
        val = eval_loss(model, bundle.val_loader, device, lt=lt, lf=lf, lc=lc, lr=lr)
        dt = time.perf_counter() - t0
        row = {"epoch": epoch, **tr, "val_loss": float(val), "epoch_seconds": float(dt)}
        hist.append(row)
        print(
            f"[Epoch {epoch:03d}] loss={tr['loss']:.4f} time={tr['time_loss']:.4f} "
            f"freq={tr['freq_loss']:.4f} c={tr['cluster_loss']:.4f} r={tr['reg_loss']:.4f} "
            f"val={val:.4f} t={dt:.2f}s"
        )
        if val < best_val:
            best_val = float(val)
            best_epoch = epoch
            bad = 0
            ckpt = {
                "model_state": model.state_dict(),
                "cfg": cfg,
                "feature_names": bundle.feature_names,
                "variant": variant,
                "lambda_time": lt,
                "lambda_freq": lf,
                "lambda_cluster": lc,
                "lambda_reg": lr,
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
        "variant": variant,
        "lambda_time": lt,
        "lambda_freq": lf,
        "lambda_cluster": lc,
        "lambda_reg": lr,
        "history": hist,
    }
    save_json(str(out_dir / "train_metrics.json"), train_summary)
    print(f"Training done. Best epoch: {best_epoch}, best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
