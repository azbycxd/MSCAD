from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn

from data.factory import build_dataloaders
from models.backbone_ae import BackboneAE
from utils.config_io import load_config, save_json
from utils.metrics import binary_metrics
from utils.pot import POTThreshold
from utils.seed import set_seed
from utils.profiler import benchmark_inference_ms, ram_usage_mb, gpu_peak_mem_mb

try:
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover
    PCA = None
    IncrementalPCA = None
    IsolationForest = None


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LSTMAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, latent_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.to_latent_h = nn.Linear(hidden_dim, latent_dim)
        self.to_latent_c = nn.Linear(hidden_dim, latent_dim)
        self.to_dec_h = nn.Linear(latent_dim, hidden_dim)
        self.to_dec_c = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        _, (h, c) = self.encoder(x)
        h0 = h[-1]
        c0 = c[-1]
        zh = self.to_latent_h(h0)
        zc = self.to_latent_c(c0)
        dh = self.to_dec_h(zh).unsqueeze(0)
        dc = self.to_dec_c(zc).unsqueeze(0)
        dec_in = torch.zeros((b, l, d), dtype=x.dtype, device=x.device)
        y, _ = self.decoder(dec_in, (dh, dc))
        return self.out(y)


def train_nn_baseline(model: nn.Module, bundle, device: torch.device, epochs: int, lr: float, patience: int) -> nn.Module:
    crit = nn.SmoothL1Loss(reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        for xb, _ in bundle.train_loader:
            xb = xb.to(device, non_blocking=True)
            xh = model(xb)
            loss = crit(xh, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = xb.size(0)
            tr_loss += float(loss.item()) * bs
            n += bs
        tr_loss /= max(1, n)

        model.eval()
        va_loss = 0.0
        vn = 0
        with torch.no_grad():
            for xb, _ in bundle.val_loader:
                xb = xb.to(device, non_blocking=True)
                xh = model(xb)
                loss = crit(xh, xb)
                bs = xb.size(0)
                va_loss += float(loss.item()) * bs
                vn += bs
        va_loss /= max(1, vn)
        print(f"[{model.__class__.__name__}] epoch={ep:03d} train={tr_loss:.4f} val={va_loss:.4f}")
        if va_loss < best:
            best = va_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def nn_scores(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    s, y = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        xh = model(xb)
        score = (xh - xb).abs().mean(dim=(1, 2))
        s.append(score.cpu().numpy())
        y.append(yb.numpy())
    return np.concatenate(s, axis=0), np.concatenate(y, axis=0)


def _iter_flat_batches(loader):
    for xb, yb in loader:
        x = xb.numpy().reshape(xb.shape[0], -1)
        y = yb.numpy()
        yield x, y


def _collect_train_sample(loader, max_samples: int) -> np.ndarray:
    chunks = []
    total = 0
    for x, _ in _iter_flat_batches(loader):
        if total >= max_samples:
            break
        need = max_samples - total
        x_use = x[:need]
        chunks.append(x_use)
        total += len(x_use)
    return np.concatenate(chunks, axis=0)


def _score_if(clf, loader) -> tuple[np.ndarray, np.ndarray]:
    scores, ys = [], []
    for x, y in _iter_flat_batches(loader):
        scores.append(-clf.score_samples(x))
        ys.append(y)
    return np.concatenate(scores, axis=0), np.concatenate(ys, axis=0)


def _score_pca_batchwise(pca, loader) -> tuple[np.ndarray, np.ndarray]:
    scores, ys = [], []
    for x, y in _iter_flat_batches(loader):
        xr = pca.inverse_transform(pca.transform(x))
        s = np.mean(np.abs(x - xr), axis=1)
        scores.append(s)
        ys.append(y)
    return np.concatenate(scores, axis=0), np.concatenate(ys, axis=0)


def eval_with_pot(train_score: np.ndarray, test_score: np.ndarray, y_test: np.ndarray) -> dict:
    pot = POTThreshold(q=0.95, target_prob=0.005)
    pot.fit(train_score)
    pred = (test_score > pot.tau).astype(int)
    m = binary_metrics(y_test, pred, test_score)
    m["y_true_pos"] = int((y_test == 1).sum())
    m["y_true_neg"] = int((y_test == 0).sum())
    m["y_pred_pos"] = int((pred == 1).sum())
    m["y_pred_neg"] = int((pred == 0).sum())
    m["threshold_tau"] = float(pot.tau)
    m["threshold_mode"] = "pot"
    return m


def _nn_profile(model: nn.Module, loader, device: torch.device) -> dict:
    first_batch = next(iter(loader))[0]
    infer_ms = float(benchmark_inference_ms(model, first_batch, warmup=10, runs=30))
    return {
        "inference_ms_per_batch": infer_ms,
        "cpu_ram_mb": ram_usage_mb(),
        "gpu_peak_mem_mb": gpu_peak_mem_mb(device),
    }


def _sk_profile_score_fn(score_fn, loader) -> tuple[np.ndarray, np.ndarray, float]:
    scores, ys = [], []
    first = True
    t0 = None
    t1 = None
    for x, y in _iter_flat_batches(loader):
        if first:
            t0 = time.perf_counter()
            first = False
        scores.append(score_fn(x))
        ys.append(y)
    t1 = time.perf_counter()
    elapsed = (t1 - t0) * 1000.0 if t0 is not None else 0.0
    n_batches = max(1, len(scores))
    return np.concatenate(scores, axis=0), np.concatenate(ys, axis=0), float(elapsed / n_batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swat_main.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--models", default="AE,LSTM-AE,IF,PCA")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cpu or cuda")
    parser.add_argument("--if-max-train-samples", type=int, default=200000)
    parser.add_argument("--pca-max-train-samples", type=int, default=300000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    out_dir = Path("outputs/baselines_main")
    out_dir.mkdir(parents=True, exist_ok=True)
    enabled = {m.strip() for m in args.models.split(",") if m.strip()}
    device_name = args.device if args.device is not None else cfg.get("device", "cuda")
    device = build_device(device_name)

    data_cfg = cfg["data"]
    bundle = build_dataloaders(cfg, pin_memory=(device.type == "cuda"))
    d = len(bundle.feature_names)

    summary = {}

    # AE baseline
    if "AE" in enabled:
        ae = BackboneAE(in_channels=d, hidden_channels=64, latent_channels=32, dropout=0.1).to(device)
        ae = train_nn_baseline(ae, bundle, device=device, epochs=args.epochs, lr=1e-3, patience=args.patience)
        ae_train_score, _ = nn_scores(ae, bundle.train_loader, device)
        ae_test_score, ae_y = nn_scores(ae, bundle.test_loader, device)
        summary["AE"] = eval_with_pot(ae_train_score, ae_test_score, ae_y)
        summary["AE"]["params"] = int(sum(p.numel() for p in ae.parameters() if p.requires_grad))
        summary["AE"].update(_nn_profile(ae, bundle.test_loader, device))
        torch.save({"model_state": ae.state_dict(), "feature_dim": d}, out_dir / "ae_best.pt")
        print(f"AE: F1={summary['AE']['f1']:.4f}")

    # LSTM-AE baseline
    if "LSTM-AE" in enabled:
        lstm = LSTMAE(in_dim=d, hidden_dim=64, latent_dim=32).to(device)
        lstm = train_nn_baseline(lstm, bundle, device=device, epochs=args.epochs, lr=1e-3, patience=args.patience)
        l_train_score, _ = nn_scores(lstm, bundle.train_loader, device)
        l_test_score, l_y = nn_scores(lstm, bundle.test_loader, device)
        summary["LSTM-AE"] = eval_with_pot(l_train_score, l_test_score, l_y)
        summary["LSTM-AE"]["params"] = int(sum(p.numel() for p in lstm.parameters() if p.requires_grad))
        summary["LSTM-AE"].update(_nn_profile(lstm, bundle.test_loader, device))
        torch.save({"model_state": lstm.state_dict(), "feature_dim": d}, out_dir / "lstm_ae_best.pt")
        print(f"LSTM-AE: F1={summary['LSTM-AE']['f1']:.4f}")

    # IF baseline
    if "IF" in enabled and IsolationForest is not None:
        xtr = _collect_train_sample(bundle.train_loader, max_samples=args.if_max_train_samples)
        clf = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=int(cfg["seed"]),
            n_jobs=1,
        )
        clf.fit(xtr)
        tr_score = -clf.score_samples(xtr)
        te_score, yte, infer_ms = _sk_profile_score_fn(lambda xx: -clf.score_samples(xx), bundle.test_loader)
        summary["IF"] = eval_with_pot(tr_score, te_score, yte)
        summary["IF"]["params"] = None
        summary["IF"]["inference_ms_per_batch"] = infer_ms
        summary["IF"]["cpu_ram_mb"] = ram_usage_mb()
        summary["IF"]["gpu_peak_mem_mb"] = None
        print(f"IF: F1={summary['IF']['f1']:.4f}")
    elif "IF" in enabled:
        summary["IF"] = {"error": "sklearn is unavailable"}

    # PCA baseline
    if "PCA" in enabled and (PCA is not None or IncrementalPCA is not None):
        x_probe, _ = next(_iter_flat_batches(bundle.train_loader))
        n_comp = max(2, min(20, x_probe.shape[1] - 1))
        if IncrementalPCA is not None:
            pca = IncrementalPCA(n_components=n_comp, batch_size=int(data_cfg["batch_size"]))
            seen = 0
            for x, _ in _iter_flat_batches(bundle.train_loader):
                if seen >= args.pca_max_train_samples:
                    break
                need = args.pca_max_train_samples - seen
                x_use = x[:need]
                pca.partial_fit(x_use)
                seen += len(x_use)
        else:
            xtr = _collect_train_sample(bundle.train_loader, max_samples=args.pca_max_train_samples)
            pca = PCA(n_components=n_comp, random_state=int(cfg["seed"]))
            pca.fit(xtr)
        tr_score, _ = _score_pca_batchwise(pca, bundle.train_loader)
        te_score, yte, infer_ms = _sk_profile_score_fn(
            lambda xx: np.mean(np.abs(xx - pca.inverse_transform(pca.transform(xx))), axis=1),
            bundle.test_loader,
        )
        summary["PCA"] = eval_with_pot(tr_score, te_score, yte)
        summary["PCA"]["params"] = None
        summary["PCA"]["inference_ms_per_batch"] = infer_ms
        summary["PCA"]["cpu_ram_mb"] = ram_usage_mb()
        summary["PCA"]["gpu_peak_mem_mb"] = None
        print(f"PCA: F1={summary['PCA']['f1']:.4f}")
    elif "PCA" in enabled:
        summary["PCA"] = {"error": "sklearn is unavailable"}

    save_json(str(out_dir / "baselines_summary.json"), summary)
    print(f"Saved: {out_dir / 'baselines_summary.json'}")


if __name__ == "__main__":
    main()
