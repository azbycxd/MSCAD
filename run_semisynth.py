from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.swat import build_swat_dataloaders
from evaluate import build_device
from experiments.correlation_anomaly_synthesis import synthesize
from models.mscad import MSCAD
from run_baselines import LSTMAE, train_nn_baseline, nn_scores, eval_with_pot
from utils.config_io import load_config, save_json
from utils.metrics import fuse_scores
from utils.seed import set_seed

try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover
    PCA = None
    IsolationForest = None


def _window_matrix(loader) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for xb, yb in loader:
        x = xb.numpy().reshape(xb.shape[0], -1)
        xs.append(x)
        ys.append(yb.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _make_window_loader(x: np.ndarray, y: np.ndarray, window: int, stride: int, batch_size: int) -> DataLoader:
    starts = list(range(0, len(x) - window + 1, stride))
    xw = np.stack([x[s : s + window] for s in starts], axis=0)
    yw = np.array([int(y[s : s + window].max()) for s in starts], dtype=np.int64)
    ds = TensorDataset(torch.from_numpy(xw).float(), torch.from_numpy(yw).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def mscad_scores(model: MSCAD, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rec, short, long, y = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        xh = model(xb)
        comp = model.score_components(xb, xh)
        rec.append(comp["rec"].detach().cpu().numpy())
        short.append(comp["short"].detach().cpu().numpy())
        long.append(comp["long"].detach().cpu().numpy())
        y.append(yb.numpy())
    return (
        np.concatenate(rec, axis=0),
        np.concatenate(short, axis=0),
        np.concatenate(long, axis=0),
        np.concatenate(y, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swat_main.yaml")
    parser.add_argument("--ckpt", default="outputs/swat_main/best.pt")
    parser.add_argument("--methods", default="MSCAD,AE,LSTM-AE,IF,PCA")
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--min-gap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    out_dir = Path("outputs/semisynth")
    out_dir.mkdir(parents=True, exist_ok=True)
    enabled = {m.strip() for m in args.methods.split(",") if m.strip()}

    data_cfg = cfg["data"]
    bundle = build_swat_dataloaders(
        train_csv=data_cfg["train_csv"],
        test_csv=data_cfg["test_csv"],
        label_col=data_cfg["label_col"],
        timestamp_col=data_cfg["timestamp_col"],
        window_size=int(data_cfg["window_size"]),
        stride_train=int(data_cfg["stride_train"]),
        stride_test=int(data_cfg["stride_test"]),
        val_ratio=float(data_cfg.get("val_ratio", 0.2)),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
    )
    device = build_device(cfg.get("device", "cuda"))

    # Build semisynthetic dataset from normal data only.
    df_normal = pd.read_csv(data_cfg["train_csv"])
    ts_name = str(data_cfg["timestamp_col"]).strip().lower()
    lb_name = str(data_cfg["label_col"]).strip().lower()
    numeric_cols = []
    for c in df_normal.columns:
        cl = str(c).strip().lower()
        if cl in {ts_name, lb_name, "syntheticanomaly"}:
            continue
        numeric_cols.append(c)
    df_numeric = df_normal.copy()
    df_numeric[numeric_cols] = df_numeric[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    synth_df, events = synthesize(
        df=df_numeric,
        numeric_cols=numeric_cols,
        window=args.window,
        count=args.count,
        min_gap=args.min_gap,
        seed=args.seed,
        ratio_strength=0.6,
        ratio_blend=0.7,
        max_lag=12,
        rotation_alpha=0.6,
        types=["ratio_mismatch", "decorrelation", "lag_shift", "subgroup_rotation"],
    )
    synth_csv = out_dir / "swat_semisynth.csv"
    meta_json = out_dir / "swat_semisynth_events.json"
    synth_df.to_csv(synth_csv, index=False)
    save_json(str(meta_json), {"events": [e.__dict__ for e in events]})

    x_synth = synth_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    y_synth = synth_df["SyntheticAnomaly"].to_numpy(np.int64)
    x_synth = bundle.scaler.transform(x_synth)
    synth_loader = _make_window_loader(
        x_synth, y_synth, window=int(data_cfg["window_size"]), stride=int(data_cfg["stride_test"]), batch_size=int(data_cfg["batch_size"])
    )

    summary = {}

    if "MSCAD" in enabled:
        model_cfg = cfg["model"]
        model = MSCAD(
            in_channels=len(bundle.feature_names),
            hidden_channels=int(model_cfg["hidden_channels"]),
            latent_channels=int(model_cfg["latent_channels"]),
            dropout=float(model_cfg["dropout"]),
            cov_type=cfg["train"]["cov_type"],
            short_window=int(data_cfg["short_window"]),
        ).to(device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)

        tr_rec, tr_short, tr_long, _ = mscad_scores(model, bundle.train_loader, device)
        te_rec, te_short, te_long, te_y = mscad_scores(model, synth_loader, device)
        rec_stat = (float(np.mean(tr_rec)), float(np.std(tr_rec) + 1e-8))
        short_stat = (float(np.mean(tr_short)), float(np.std(tr_short) + 1e-8))
        long_stat = (float(np.mean(tr_long)), float(np.std(tr_long) + 1e-8))
        tr_score = fuse_scores(
            tr_rec,
            tr_short,
            tr_long,
            rec_stat=rec_stat,
            short_stat=short_stat,
            long_stat=long_stat,
            alpha=float(cfg["eval"]["alpha"]),
            beta=float(cfg["eval"]["beta"]),
            gamma=float(cfg["eval"]["gamma"]),
        )
        te_score = fuse_scores(
            te_rec,
            te_short,
            te_long,
            rec_stat=rec_stat,
            short_stat=short_stat,
            long_stat=long_stat,
            alpha=float(cfg["eval"]["alpha"]),
            beta=float(cfg["eval"]["beta"]),
            gamma=float(cfg["eval"]["gamma"]),
        )
        summary["MSCAD"] = eval_with_pot(tr_score, te_score, te_y)
        summary["MSCAD"]["params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    d = len(bundle.feature_names)
    if "AE" in enabled:
        from models.backbone_ae import BackboneAE

        ae = BackboneAE(in_channels=d, hidden_channels=64, latent_channels=32, dropout=0.1).to(device)
        ae = train_nn_baseline(ae, bundle, device=device, epochs=20, lr=1e-3, patience=5)
        tr, _ = nn_scores(ae, bundle.train_loader, device)
        te, y = nn_scores(ae, synth_loader, device)
        summary["AE"] = eval_with_pot(tr, te, y)

    if "LSTM-AE" in enabled:
        lstm = LSTMAE(in_dim=d, hidden_dim=64, latent_dim=32).to(device)
        lstm = train_nn_baseline(lstm, bundle, device=device, epochs=20, lr=1e-3, patience=5)
        tr, _ = nn_scores(lstm, bundle.train_loader, device)
        te, y = nn_scores(lstm, synth_loader, device)
        summary["LSTM-AE"] = eval_with_pot(tr, te, y)

    if "IF" in enabled and IsolationForest is not None:
        xtr, _ = _window_matrix(bundle.train_loader)
        xte, yte = _window_matrix(synth_loader)
        clf = IsolationForest(n_estimators=200, contamination="auto", random_state=args.seed, n_jobs=-1)
        clf.fit(xtr)
        tr = -clf.score_samples(xtr)
        te = -clf.score_samples(xte)
        summary["IF"] = eval_with_pot(tr, te, yte)

    if "PCA" in enabled and PCA is not None:
        xtr, _ = _window_matrix(bundle.train_loader)
        xte, yte = _window_matrix(synth_loader)
        n_comp = min(20, xtr.shape[1] - 1)
        pca = PCA(n_components=max(2, n_comp), random_state=args.seed)
        pca.fit(xtr)
        xtr_r = pca.inverse_transform(pca.transform(xtr))
        xte_r = pca.inverse_transform(pca.transform(xte))
        tr = np.mean(np.abs(xtr - xtr_r), axis=1)
        te = np.mean(np.abs(xte - xte_r), axis=1)
        summary["PCA"] = eval_with_pot(tr, te, yte)

    save_json(str(out_dir / "semisynth_summary.json"), summary)
    print(f"Saved semisynth dataset: {synth_csv}")
    print(f"Saved semisynth metrics: {out_dir / 'semisynth_summary.json'}")


if __name__ == "__main__":
    main()
