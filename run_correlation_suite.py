from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.factory import build_dataloaders
from models.backbone_ae import BackboneAE
from models.mscad import MSCAD
from run_baselines import LSTMAE
from utils.config_io import load_config, save_json
from utils.metrics import binary_metrics, fuse_scores
from utils.pot import POTThreshold
from utils.seed import set_seed

try:
    from sklearn.decomposition import IncrementalPCA
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover
    IncrementalPCA = None
    IsolationForest = None


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _windows_from_array(x: np.ndarray, window: int, stride: int, max_n: int | None = None) -> np.ndarray:
    starts = list(range(0, len(x) - window + 1, stride))
    if max_n is not None and len(starts) > max_n:
        starts = starts[:max_n]
    return np.stack([x[s : s + window] for s in starts], axis=0)


def _corr_pairs(x_train: np.ndarray, topk: int = 50, min_abs_corr: float = 0.0) -> List[Tuple[int, int]]:
    c = np.corrcoef(x_train.T)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 0.0)
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(c), axis=None), c.shape))[0]
    pairs = []
    seen = set()
    for i, j in idx:
        if i == j:
            continue
        a, b = sorted((int(i), int(j)))
        if (a, b) in seen:
            continue
        if abs(c[a, b]) < float(min_abs_corr):
            break
        seen.add((a, b))
        pairs.append((a, b))
        if len(pairs) >= topk:
            break
    return pairs


def _pairs_from_ckpt(ckpt: dict) -> List[Tuple[int, int]]:
    pairs = ckpt.get("structure_pairs", []) or []
    out: List[Tuple[int, int]] = []
    for p in pairs:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        a, b = int(p[0]), int(p[1])
        if a == b:
            continue
        if a > b:
            a, b = b, a
        out.append((a, b))
    # unique while preserving order
    uniq = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _choose_segment(xw: np.ndarray, length: int, locality: str, rng: np.random.Generator, ref_idx: int) -> Tuple[int, int]:
    L = xw.shape[0]
    length = min(length, L - 2)
    if locality == "contextual":
        s = xw[:, ref_idx]
        thr = np.percentile(s, 75)
        cand = np.where(s >= thr)[0]
        cand = cand[cand <= L - length]
        if len(cand) > 0:
            t0 = int(rng.choice(cand))
            return t0, t0 + length
    t0 = int(rng.integers(0, L - length + 1))
    return t0, t0 + length


def inject_ratio_mismatch(xw: np.ndarray, dims: List[int], eps: float, length: int, locality: str, rng) -> np.ndarray:
    y = xw.copy()
    i, j = dims[0], dims[1]
    t0, t1 = _choose_segment(y, length, locality, rng, i)
    xi = y[t0:t1, i]
    xj = y[t0:t1, j]
    delta = eps * np.std(xj) * np.sign(xi - np.mean(xi))
    y[t0:t1, j] = xj + delta
    return y


def inject_decorrelation(xw: np.ndarray, dims: List[int], eps: float, length: int, locality: str, rng) -> np.ndarray:
    y = xw.copy()
    ref = dims[0]
    t0, t1 = _choose_segment(y, length, locality, rng, ref)
    j = dims[1]
    seg = y[t0:t1, j].copy()
    rng.shuffle(seg)
    # keep close marginal moments
    seg = (seg - seg.mean()) / (seg.std() + 1e-6) * (y[t0:t1, j].std() + 1e-6) + y[t0:t1, j].mean()
    y[t0:t1, j] = (1 - eps) * y[t0:t1, j] + eps * seg
    return y


def inject_lag_mismatch(xw: np.ndarray, dims: List[int], eps: float, length: int, locality: str, rng) -> np.ndarray:
    y = xw.copy()
    i, j = dims[0], dims[1]
    t0, t1 = _choose_segment(y, length, locality, rng, i)
    tau = int(max(1, round(eps * 8)))
    seg = y[t0:t1, j].copy()
    y[t0:t1, j] = np.roll(seg, tau)
    return y


def inject_group_rotation(xw: np.ndarray, dims: List[int], eps: float, length: int, locality: str, rng) -> np.ndarray:
    y = xw.copy()
    ref = dims[0]
    t0, t1 = _choose_segment(y, length, locality, rng, ref)
    X = y[t0:t1][:, dims]
    k = len(dims)
    A = rng.normal(size=(k, k))
    M = A - A.T
    R = np.eye(k) + eps * 0.2 * M
    y[t0:t1][:, dims] = X @ R
    return y


def inject_point_amplitude(xw: np.ndarray, dims: List[int], eps: float, length: int, locality: str, rng) -> np.ndarray:
    y = xw.copy()
    d = dims[0]
    t0, t1 = _choose_segment(y, length, locality, rng, d)
    y[t0:t1, d] += eps * np.std(y[:, d]) * 2.0
    return y


INJECTORS = {
    "ratio_mismatch": inject_ratio_mismatch,
    "decorrelation": inject_decorrelation,
    "lag_mismatch": inject_lag_mismatch,
    "group_rotation": inject_group_rotation,
    "point_amplitude": inject_point_amplitude,
}


def make_synthetic_set(
    normal_pool: np.ndarray,
    corr_pairs: List[Tuple[int, int]],
    kinds: List[str],
    n_samples: int,
    anomaly_ratio: float,
    eps: float,
    length: int,
    arity: int,
    locality: str,
    seed: int,
    extra_dims_pool: List[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_anom = int(round(n_samples * anomaly_ratio))
    n_norm = n_samples - n_anom
    replace_norm = n_norm > len(normal_pool)
    idx_norm = rng.choice(len(normal_pool), size=n_norm, replace=replace_norm)
    x_norm = normal_pool[idx_norm].copy()
    y_norm = np.zeros(n_norm, dtype=np.int64)

    x_anom = []
    y_anom = []
    for _ in range(n_anom):
        base = normal_pool[int(rng.integers(0, len(normal_pool)))].copy()
        kind = str(rng.choice(kinds))
        pair = corr_pairs[int(rng.integers(0, len(corr_pairs)))]
        dims = list(pair)
        D = base.shape[1]
        if arity > 2:
            if extra_dims_pool is None:
                extra = [d for d in range(D) if d not in dims]
            else:
                extra = [d for d in extra_dims_pool if d not in dims and 0 <= d < D]
                if len(extra) < max(0, arity - 2):
                    fallback = [d for d in range(D) if d not in dims and d not in extra]
                    extra.extend(fallback)
            rng.shuffle(extra)
            dims.extend(extra[: max(0, arity - 2)])
        inj = INJECTORS[kind]
        out = inj(base, dims=dims, eps=eps, length=length, locality=locality, rng=rng)
        x_anom.append(out)
        y_anom.append(1)
    x_anom = np.stack(x_anom, axis=0) if len(x_anom) else np.zeros((0,) + normal_pool.shape[1:], dtype=normal_pool.dtype)
    y_anom = np.array(y_anom, dtype=np.int64)

    X = np.concatenate([x_norm, x_anom], axis=0)
    y = np.concatenate([y_norm, y_anom], axis=0)
    p = rng.permutation(len(X))
    return X[p], y[p]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


@torch.no_grad()
def mscad_scores(model: MSCAD, loader: DataLoader, device: torch.device):
    model.eval()
    rec, short, long, y = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        xh = model(xb)
        comp = model.score_components(xb, xh)
        rec.append(comp["rec"].cpu().numpy())
        short.append(comp["short"].cpu().numpy())
        long.append(comp["long"].cpu().numpy())
        y.append(yb.numpy())
    return np.concatenate(rec), np.concatenate(short), np.concatenate(long), np.concatenate(y)


@torch.no_grad()
def recon_scores(model, loader: DataLoader, device: torch.device):
    model.eval()
    s, y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        xh = model(xb)
        score = (xh - xb).abs().mean(dim=(1, 2))
        s.append(score.cpu().numpy())
        y.append(yb.numpy())
    return np.concatenate(s), np.concatenate(y)


def fit_threshold(train_score: np.ndarray, q: float = 0.95, target_prob: float = 0.005) -> float:
    pot = POTThreshold(q=q, target_prob=target_prob)
    pot.fit(train_score)
    return float(pot.tau)


def eval_binary(train_score: np.ndarray, test_score: np.ndarray, y_test: np.ndarray) -> Dict:
    tau = fit_threshold(train_score)
    pred = (test_score > tau).astype(int)
    m = binary_metrics(y_test, pred, test_score)
    m["threshold_tau"] = tau
    return m


def score_gap_stats(score: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    normal = score[y == 0]
    anom = score[y == 1]
    if len(normal) == 0 or len(anom) == 0:
        return {
            "normal_mean": float("nan"),
            "anomaly_mean": float("nan"),
            "mean_gap": float("nan"),
            "relative_gap": float("nan"),
            "cohen_d": float("nan"),
        }
    nm = float(np.mean(normal))
    am = float(np.mean(anom))
    gap = am - nm
    rel = float(gap / (abs(nm) + 1e-8))
    ns = float(np.var(normal, ddof=1)) if len(normal) > 1 else 0.0
    as_ = float(np.var(anom, ddof=1)) if len(anom) > 1 else 0.0
    pooled = np.sqrt(max(1e-12, (ns + as_) / 2.0))
    d = float(gap / pooled)
    return {
        "normal_mean": nm,
        "anomaly_mean": am,
        "mean_gap": float(gap),
        "relative_gap": rel,
        "cohen_d": d,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swat_main.yaml")
    parser.add_argument("--mscad-ckpt", default="outputs/swat_main/best.pt")
    parser.add_argument("--ae-ckpt", default="outputs/baselines_main/ae_best.pt")
    parser.add_argument("--lstm-ckpt", default="outputs/baselines_main/lstm_ae_best.pt")
    parser.add_argument("--out-dir", default="outputs/correlation_suite")
    parser.add_argument("--tag", default="default", help="Tag name for this run.")
    parser.add_argument(
        "--preset",
        default="easy",
        choices=["easy", "medium", "hard", "balanced"],
        help="Synthesis difficulty preset. 'balanced' kept for backward compatibility.",
    )
    parser.add_argument(
        "--pair-source",
        default="structure",
        choices=["structure", "corr"],
        help="Where anomaly pairs are sampled from. 'structure' aligns injection to model structure_pairs.",
    )
    parser.add_argument(
        "--pairs-ckpt",
        default=None,
        help="Optional checkpoint used only to extract structure_pairs for anomaly injection.",
    )
    parser.add_argument("--pair-topk", type=int, default=80, help="Used when --pair-source=corr.")
    parser.add_argument("--pair-min-abs-corr", type=float, default=0.7, help="Used when --pair-source=corr.")
    parser.add_argument("--normal-pool", type=int, default=8000)
    parser.add_argument("--dataset-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(cfg.get("device", "cuda"))

    data_cfg = cfg["data"]
    bundle = build_dataloaders(cfg, pin_memory=False)
    D = len(bundle.feature_names)
    train_windows = _windows_from_array(
        bundle.x_train_norm, window=int(data_cfg["window_size"]), stride=int(data_cfg["stride_train"]), max_n=args.normal_pool
    )
    corr_pairs = _corr_pairs(
        bundle.x_train_norm,
        topk=int(args.pair_topk),
        min_abs_corr=float(args.pair_min_abs_corr),
    )

    # Load models
    model_cfg = cfg["model"]
    mscad = MSCAD(
        in_channels=D,
        hidden_channels=int(model_cfg["hidden_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        dropout=float(model_cfg["dropout"]),
        cov_type=cfg["train"]["cov_type"],
        short_window=int(data_cfg["short_window"]),
        backbone_variant=str(model_cfg.get("backbone_variant", "base")),
    ).to(device)
    ckpt = torch.load(args.mscad_ckpt, map_location=device, weights_only=False)
    mscad.load_state_dict(ckpt["model_state"], strict=True)
    pairs_src_ckpt = ckpt
    if args.pairs_ckpt is not None and Path(args.pairs_ckpt).exists():
        pairs_src_ckpt = torch.load(args.pairs_ckpt, map_location="cpu", weights_only=False)
    ckpt_pairs = _pairs_from_ckpt(pairs_src_ckpt)
    if len(ckpt_pairs) > 0:
        mscad.set_structure_pairs([[a, b] for a, b in ckpt_pairs])
    if args.pair_source == "structure" and len(ckpt_pairs) > 0:
        inject_pairs = ckpt_pairs
    else:
        inject_pairs = corr_pairs
    if len(inject_pairs) == 0:
        raise RuntimeError("No candidate pairs for synthetic injection. Check pair source/topk/min corr.")
    inject_dims_pool = sorted({d for a, b in inject_pairs for d in (a, b)})

    ae = BackboneAE(in_channels=D, hidden_channels=64, latent_channels=32, dropout=0.1).to(device)
    if Path(args.ae_ckpt).exists():
        ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
        ae.load_state_dict(ae_ck["model_state"], strict=True)
    lstm = LSTMAE(in_dim=D, hidden_dim=64, latent_dim=32).to(device)
    if Path(args.lstm_ckpt).exists():
        l_ck = torch.load(args.lstm_ckpt, map_location=device, weights_only=False)
        lstm.load_state_dict(l_ck["model_state"], strict=True)

    if_model = None
    pca_model = None
    if IsolationForest is not None:
        Xtr = train_windows.reshape(train_windows.shape[0], -1)
        if_model = IsolationForest(n_estimators=200, contamination="auto", random_state=args.seed, n_jobs=1)
        if_model.fit(Xtr)
    if IncrementalPCA is not None:
        pca_model = IncrementalPCA(n_components=max(2, min(20, train_windows.shape[1] * train_windows.shape[2] - 1)), batch_size=256)
        Xtr = train_windows.reshape(train_windows.shape[0], -1)
        for i in range(0, len(Xtr), 256):
            pca_model.partial_fit(Xtr[i : i + 256])

    # Train scores for thresholds
    train_loader = make_loader(train_windows, np.zeros(len(train_windows), dtype=np.int64), int(data_cfg["batch_size"]))
    m_tr_rec, m_tr_short, m_tr_long, _ = mscad_scores(mscad, train_loader, device)
    m_stats = {
        "rec": (float(np.mean(m_tr_rec)), float(np.std(m_tr_rec) + 1e-8)),
        "short": (float(np.mean(m_tr_short)), float(np.std(m_tr_short) + 1e-8)),
        "long": (float(np.mean(m_tr_long)), float(np.std(m_tr_long) + 1e-8)),
    }
    m_train_multi = fuse_scores(
        m_tr_rec,
        m_tr_short,
        m_tr_long,
        rec_stat=m_stats["rec"],
        short_stat=m_stats["short"],
        long_stat=m_stats["long"],
        alpha=float(cfg["eval"]["alpha"]),
        beta=float(cfg["eval"]["beta"]),
        gamma=float(cfg["eval"]["gamma"]),
    )
    m_train_single = fuse_scores(
        m_tr_rec,
        m_tr_short,
        m_tr_long,
        rec_stat=m_stats["rec"],
        short_stat=m_stats["short"],
        long_stat=m_stats["long"],
        alpha=0.5,
        beta=0.0,
        gamma=0.5,
    )
    ae_train, _ = recon_scores(ae, train_loader, device)
    lstm_train, _ = recon_scores(lstm, train_loader, device)
    Xtr_flat = train_windows.reshape(train_windows.shape[0], -1)
    if_train = -if_model.score_samples(Xtr_flat) if if_model is not None else None
    if pca_model is not None:
        xtr_r = pca_model.inverse_transform(pca_model.transform(Xtr_flat))
        pca_train = np.mean(np.abs(Xtr_flat - xtr_r), axis=1)
    else:
        pca_train = None

    def evaluate_dataset(X: np.ndarray, y: np.ndarray) -> Dict:
        loader = make_loader(X, y, int(data_cfg["batch_size"]))
        m_rec, m_short, m_long, yy = mscad_scores(mscad, loader, device)
        m_test_multi = fuse_scores(
            m_rec,
            m_short,
            m_long,
            rec_stat=m_stats["rec"],
            short_stat=m_stats["short"],
            long_stat=m_stats["long"],
            alpha=float(cfg["eval"]["alpha"]),
            beta=float(cfg["eval"]["beta"]),
            gamma=float(cfg["eval"]["gamma"]),
        )
        m_test_single = fuse_scores(
            m_rec,
            m_short,
            m_long,
            rec_stat=m_stats["rec"],
            short_stat=m_stats["short"],
            long_stat=m_stats["long"],
            alpha=0.5,
            beta=0.0,
            gamma=0.5,
        )
        ae_test, _ = recon_scores(ae, loader, device)
        lstm_test, _ = recon_scores(lstm, loader, device)
        out = {}
        out["MSCAD-multi"] = eval_binary(m_train_multi, m_test_multi, yy)
        out["MSCAD-multi"]["score_gap"] = score_gap_stats(m_test_multi, yy)
        out["MSCAD-single"] = eval_binary(m_train_single, m_test_single, yy)
        out["MSCAD-single"]["score_gap"] = score_gap_stats(m_test_single, yy)
        out["AE"] = eval_binary(ae_train, ae_test, yy)
        out["AE"]["score_gap"] = score_gap_stats(ae_test, yy)
        out["LSTM-AE"] = eval_binary(lstm_train, lstm_test, yy)
        out["LSTM-AE"]["score_gap"] = score_gap_stats(lstm_test, yy)
        Xf = X.reshape(X.shape[0], -1)
        if if_model is not None and if_train is not None:
            if_test = -if_model.score_samples(Xf)
            out["IF"] = eval_binary(if_train, if_test, yy)
            out["IF"]["score_gap"] = score_gap_stats(if_test, yy)
        if pca_model is not None and pca_train is not None:
            xr = pca_model.inverse_transform(pca_model.transform(Xf))
            pca_test = np.mean(np.abs(Xf - xr), axis=1)
            out["PCA"] = eval_binary(pca_train, pca_test, yy)
            out["PCA"]["score_gap"] = score_gap_stats(pca_test, yy)
        return out

    if args.preset == "balanced":
        syn1_eps, syn1_len, syn1_arity, syn1_loc = 0.3, 16, 4, "global"
        syn2_eps, syn2_len, syn2_arity, syn2_loc = 0.3, 16, 4, "global"
        syn3_levels = [("weak", 0.1), ("medium", 0.3), ("strong", 0.5)]
        curve_strength = [0.1, 0.2, 0.3, 0.5]
        curve_length = [8, 16, 32]
        curve_arity = [2, 4, 8]
        curve_eps_default = 0.3
        curve_len_default = 16
        curve_arity_default = 4
    elif args.preset == "easy":
        # Easy must be truly separable: structure-pair aligned, global, longer segment, higher arity.
        syn1_eps, syn1_len, syn1_arity, syn1_loc = 0.8, 32, 8, "global"
        syn2_eps, syn2_len, syn2_arity, syn2_loc = 0.7, 28, 8, "global"
        syn3_levels = [("weak", 0.5), ("medium", 0.7), ("strong", 0.9)]
        curve_strength = [0.4, 0.6, 0.8, 1.0]
        curve_length = [16, 24, 32]
        curve_arity = [4, 8, 12]
        curve_eps_default = 0.8
        curve_len_default = 32
        curve_arity_default = 8
    elif args.preset == "medium":
        syn1_eps, syn1_len, syn1_arity, syn1_loc = 0.55, 24, 4, "global"
        syn2_eps, syn2_len, syn2_arity, syn2_loc = 0.50, 20, 4, "global"
        syn3_levels = [("weak", 0.35), ("medium", 0.5), ("strong", 0.65)]
        curve_strength = [0.3, 0.45, 0.6, 0.75]
        curve_length = [12, 20, 28]
        curve_arity = [2, 4, 8]
        curve_eps_default = 0.55
        curve_len_default = 24
        curve_arity_default = 4
    else:  # hard
        syn1_eps, syn1_len, syn1_arity, syn1_loc = 0.35, 16, 2, "contextual"
        syn2_eps, syn2_len, syn2_arity, syn2_loc = 0.30, 12, 2, "contextual"
        syn3_levels = [("weak", 0.2), ("medium", 0.3), ("strong", 0.4)]
        curve_strength = [0.2, 0.3, 0.4, 0.5]
        curve_length = [8, 12, 16]
        curve_arity = [2, 4, 8]
        curve_eps_default = 0.35
        curve_len_default = 12
        curve_arity_default = 2

    # Dataset-Syn-1: pure correlation anomalies
    syn1_X, syn1_y = make_synthetic_set(
        normal_pool=train_windows,
        corr_pairs=inject_pairs,
        kinds=["ratio_mismatch", "decorrelation", "lag_mismatch", "group_rotation"],
        n_samples=args.dataset_size,
        anomaly_ratio=0.3,
        eps=syn1_eps,
        length=syn1_len,
        arity=syn1_arity,
        locality=syn1_loc,
        seed=args.seed,
        extra_dims_pool=inject_dims_pool,
    )
    syn1 = evaluate_dataset(syn1_X, syn1_y)

    # Dataset-Syn-2: mixed anomalies
    syn2_X, syn2_y = make_synthetic_set(
        normal_pool=train_windows,
        corr_pairs=inject_pairs,
        # 4:1 weighting keeps mixed set focused on correlation anomalies.
        kinds=[
            "ratio_mismatch",
            "decorrelation",
            "lag_mismatch",
            "group_rotation",
            "ratio_mismatch",
            "decorrelation",
            "lag_mismatch",
            "group_rotation",
            "point_amplitude",
            "point_amplitude",
        ],
        n_samples=args.dataset_size,
        anomaly_ratio=0.3,
        eps=syn2_eps,
        length=syn2_len,
        arity=syn2_arity,
        locality=syn2_loc,
        seed=args.seed + 1,
        extra_dims_pool=inject_dims_pool,
    )
    syn2 = evaluate_dataset(syn2_X, syn2_y)

    # Dataset-Syn-3: strength stratified (weak/medium/strong)
    syn3 = {}
    for tag, eps in syn3_levels:
        X, y = make_synthetic_set(
            normal_pool=train_windows,
            corr_pairs=inject_pairs,
            kinds=["ratio_mismatch", "decorrelation", "lag_mismatch", "group_rotation"],
            n_samples=args.dataset_size,
            anomaly_ratio=0.3,
            eps=eps,
            length=syn1_len,
            arity=syn1_arity,
            locality=syn1_loc,
            seed=args.seed + 10 + int(eps * 10),
            extra_dims_pool=inject_dims_pool,
        )
        syn3[tag] = evaluate_dataset(X, y)

    # Factor curves
    curves = {"strength": {}, "length": {}, "arity": {}, "locality": {}}
    for eps in curve_strength:
        X, y = make_synthetic_set(
            normal_pool=train_windows,
            corr_pairs=inject_pairs,
            kinds=["ratio_mismatch", "decorrelation", "lag_mismatch", "group_rotation"],
            n_samples=4000,
            anomaly_ratio=0.3,
            eps=eps,
            length=curve_len_default,
            arity=curve_arity_default,
            locality="global",
            seed=args.seed + int(eps * 100),
            extra_dims_pool=inject_dims_pool,
        )
        curves["strength"][str(eps)] = evaluate_dataset(X, y)
    for L in curve_length:
        X, y = make_synthetic_set(
            normal_pool=train_windows,
            corr_pairs=inject_pairs,
            kinds=["ratio_mismatch", "decorrelation", "lag_mismatch", "group_rotation"],
            n_samples=4000,
            anomaly_ratio=0.3,
            eps=curve_eps_default,
            length=L,
            arity=curve_arity_default,
            locality="global",
            seed=args.seed + L,
            extra_dims_pool=inject_dims_pool,
        )
        curves["length"][str(L)] = evaluate_dataset(X, y)
    for ar in curve_arity:
        X, y = make_synthetic_set(
            normal_pool=train_windows,
            corr_pairs=inject_pairs,
            kinds=["ratio_mismatch", "decorrelation", "lag_mismatch", "group_rotation"],
            n_samples=4000,
            anomaly_ratio=0.3,
            eps=curve_eps_default,
            length=curve_len_default,
            arity=ar,
            locality="global",
            seed=args.seed + ar,
            extra_dims_pool=inject_dims_pool,
        )
        curves["arity"][str(ar)] = evaluate_dataset(X, y)
    for loc in ["global", "contextual"]:
        X, y = make_synthetic_set(
            normal_pool=train_windows,
            corr_pairs=inject_pairs,
            kinds=["ratio_mismatch", "decorrelation", "lag_mismatch", "group_rotation"],
            n_samples=4000,
            anomaly_ratio=0.3,
            eps=curve_eps_default,
            length=curve_len_default,
            arity=curve_arity_default,
            locality=loc,
            seed=args.seed + (1 if loc == "contextual" else 0),
            extra_dims_pool=inject_dims_pool,
        )
        curves["locality"][loc] = evaluate_dataset(X, y)

    def _top_model_by(section: Dict, metric: str) -> Tuple[str, float]:
        items = [(k, float(v.get(metric, float("nan")))) for k, v in section.items() if isinstance(v, dict) and metric in v]
        items = [x for x in items if not np.isnan(x[1])]
        if not items:
            return "N/A", float("nan")
        items.sort(key=lambda x: x[1], reverse=True)
        return items[0]

    q1_model_aupr, q1_aupr = _top_model_by(syn1, "aupr")
    q1_model_auroc, q1_auroc = _top_model_by(syn1, "auroc")
    q1_gap = float(syn1["MSCAD-multi"]["score_gap"]["cohen_d"])
    q2_rank = sorted(
        [(m, float(v["aupr"]), float(v["auroc"])) for m, v in syn1.items() if isinstance(v, dict)],
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )
    q3_multi = syn1["MSCAD-multi"]
    q3_single = syn1["MSCAD-single"]
    q4_mixed = syn2["MSCAD-multi"]

    result = {
        "dataset_syn_1": syn1,
        "dataset_syn_2": syn2,
        "dataset_syn_3": syn3,
        "curves": curves,
        "protocol": {
            "train_data": "SWaT normal train split only",
            "threshold": "POT fitted on train scores only",
            "models": ["MSCAD-multi", "MSCAD-single", "AE", "LSTM-AE", "IF", "PCA"],
            "preset": args.preset,
            "pair_source": args.pair_source,
            "n_injection_pairs": len(inject_pairs),
            "pair_topk": int(args.pair_topk),
            "pair_min_abs_corr": float(args.pair_min_abs_corr),
        },
        "q1_q4_summary": {
            "Q1": {
                "dataset": "dataset_syn_1",
                "primary_metrics": {"best_aupr_model": q1_model_aupr, "best_aupr": q1_aupr, "best_auroc_model": q1_model_auroc, "best_auroc": q1_auroc},
                "mscad_multi_score_gap_cohen_d": q1_gap,
            },
            "Q2": {
                "dataset": "dataset_syn_1",
                "ranking_by_aupr_then_auroc": q2_rank,
            },
            "Q3": {
                "dataset": "dataset_syn_1",
                "mscad_multi": {
                    "aupr": float(q3_multi["aupr"]),
                    "auroc": float(q3_multi["auroc"]),
                    "cohen_d": float(q3_multi["score_gap"]["cohen_d"]),
                },
                "mscad_single": {
                    "aupr": float(q3_single["aupr"]),
                    "auroc": float(q3_single["auroc"]),
                    "cohen_d": float(q3_single["score_gap"]["cohen_d"]),
                },
            },
            "Q4": {
                "dataset": "dataset_syn_2",
                "mscad_multi": {
                    "aupr": float(q4_mixed["aupr"]),
                    "auroc": float(q4_mixed["auroc"]),
                    "cohen_d": float(q4_mixed["score_gap"]["cohen_d"]),
                },
            },
        },
    }
    save_json(str(out_dir / "correlation_suite_results.json"), result)
    save_json(str(out_dir / "q1_q4_summary.json"), result["q1_q4_summary"])
    print(f"Saved: {out_dir / 'correlation_suite_results.json'}")


if __name__ == "__main__":
    main()
