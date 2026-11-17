"""Read-only frozen A0 Validation scoring isolated from S6 NumPy audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import EXPECTED_NORMAL_ROWS, TRAIN_RAW_ROWS, WINDOW, iter_window_batches, load_normal_cache, load_scaler, make_window_starts, resolve_repo_path, validate_config_contract
from models.backbone_ae import BackboneAE
from utils.config_io import load_config


def main() -> None:
    config = load_config(str(ROOT / "configs" / "thesis" / "swat2015_midterm_seed1.yaml")); validate_config_contract(config)
    output = ROOT / "outputs" / "thesis_s6_swat_phenomenon_audit"; output.mkdir(parents=True, exist_ok=True)
    score_path = output / "a0_validation_scores.npy"; manifest_path = output / "a0_validation_score_manifest.json"
    if score_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite existing S6 A0 Validation score artifact")
    normal, _ = load_normal_cache(resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
    scaler, metadata = load_scaler(resolve_repo_path(ROOT, config["paths"]["a0_output"]) / "scaler.npz")
    if metadata["fit_row_count"] != TRAIN_RAW_ROWS or metadata["attack_rows_seen"] != 0:
        raise RuntimeError("A0 scaler is not frozen Train-only")
    checkpoint_path = resolve_repo_path(ROOT, config["paths"]["a0_output"]) / "best.pt"; before = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = config["model"]; model = BackboneAE(int(model_cfg["in_channels"]), int(model_cfg["hidden_channels"]), int(model_cfg["latent_channels"]), float(model_cfg["dropout"]))
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    if checkpoint["best_epoch"] != 13 or sum(p.numel() for p in model.parameters()) != 11730:
        raise RuntimeError("frozen A0 identity mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    starts = make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS); chunks = []
    with torch.inference_mode():
        for windows, _ in iter_window_batches(normal, starts, scaler, int(config["training"]["batch_size"]), shuffle=False, seed=0):
            x = torch.from_numpy(windows).to(device); chunks.append(torch.abs(model(x)[:, -1] - x[:, -1]).mean(dim=1).cpu().numpy())
    scores = np.concatenate(chunks).astype(np.float32)
    if len(scores) != 99000 - WINDOW + 1 or not np.isfinite(scores).all() or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != before:
        raise RuntimeError("A0 score or checkpoint integrity failure")
    np.save(score_path, scores)
    manifest_path.write_text(json.dumps({"checkpoint_sha256": before, "checkpoint_epoch": 13, "parameter_count": 11730, "source": "Normal Validation only", "attack_rows_seen": 0, "training": False, "score_count": len(scores)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"score_count": len(scores), "training": False, "device": str(device)}))


if __name__ == "__main__":
    main()
