"""Evaluate one frozen static POT threshold on the unchanged S1 A0 endpoint score."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import genpareto

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    OfficialSWaTError,
    iter_window_batches,
    load_normal_cache,
    load_scaler,
    make_window_starts,
    resolve_repo_path,
    validate_config_contract,
)
from models.backbone_ae import BackboneAE  # noqa: E402
from utils.config_io import load_config  # noqa: E402
from utils.metrics import binary_metrics  # noqa: E402


U_QUANTILE = 0.95
ALPHA = 0.01
RANKING_TOLERANCE = 1e-6


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build_model(config: dict[str, Any]) -> BackboneAE:
    model = config["model"]
    return BackboneAE(
        in_channels=int(model["in_channels"]),
        hidden_channels=int(model["hidden_channels"]),
        latent_channels=int(model["latent_channels"]),
        dropout=float(model["dropout"]),
    )


def validation_endpoint_scores(
    model: torch.nn.Module,
    normal: np.ndarray,
    starts: np.ndarray,
    scaler: Any,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for windows, _ in iter_window_batches(normal, starts, scaler, batch_size, shuffle=False, seed=0):
            x = torch.from_numpy(windows).to(device)
            chunks.append(torch.abs(model(x)[:, -1, :] - x[:, -1, :]).mean(dim=1).cpu().numpy())
    scores = np.concatenate(chunks).astype(np.float32, copy=False)
    if len(scores) != len(starts) or not np.isfinite(scores).all():
        raise OfficialSWaTError("Validation A0 endpoint scores are incomplete or non-finite")
    return scores


def fit_static_pot(scores: np.ndarray, *, source_split: str) -> dict[str, Any]:
    if source_split != "normal_validation":
        raise OfficialSWaTError("POT parameters may only use Normal Validation endpoint scores")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise OfficialSWaTError("POT requires finite one-dimensional Validation scores")
    u = float(np.quantile(values, U_QUANTILE))
    exceedances = values[values > u] - u
    if len(exceedances) < 100:
        raise OfficialSWaTError("POT exceedance count is insufficient")
    p_u = float(len(exceedances) / len(values))
    if not ALPHA < p_u:
        raise OfficialSWaTError("POT requires alpha < empirical initialization-tail probability")
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    shape = float(shape)
    loc = float(loc)
    scale = float(scale)
    if loc != 0.0 or not math.isfinite(shape) or not math.isfinite(scale) or scale <= 0:
        raise OfficialSWaTError("GPD loc/shape/scale numerical safety check failed")
    q_cond = float(1.0 - ALPHA / p_u)
    if not 0.0 < q_cond < 1.0:
        raise OfficialSWaTError("conditional exceedance quantile is outside (0,1)")
    y_star = float(genpareto.ppf(q_cond, shape, loc=0, scale=scale))
    threshold = float(u + y_star)
    if not math.isfinite(threshold) or threshold <= u:
        raise OfficialSWaTError("POT threshold is non-finite or does not exceed u")
    return {
        "source": "Normal Validation endpoint scores only",
        "u_quantile": U_QUANTILE,
        "u": u,
        "validation_count": len(values),
        "exceedance_definition": "score > u; y = score - u",
        "exceedance_count": len(exceedances),
        "p_u": p_u,
        "gpd_loc_fixed": loc,
        "shape": shape,
        "scale": scale,
        "alpha": ALPHA,
        "q_cond": q_cond,
        "y_star": y_star,
        "threshold_pot": threshold,
        "parameter_search": False,
    }


def predict_static(score: np.ndarray, threshold: float) -> np.ndarray:
    """Prediction intentionally has no label argument."""
    return (np.asarray(score) > threshold).astype(np.int8)


def evaluate(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = predict_static(score, threshold)
    result = binary_metrics(y_true, prediction, score)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    fn = int(((y_true == 1) & (prediction == 0)).sum())
    tp = int(((y_true == 1) & (prediction == 1)).sum())
    result.update({"fpr": fp / max(tn + fp, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return result


def assert_ranking_invariant(current: dict[str, Any], a0: dict[str, Any]) -> None:
    for name in ("auroc", "aupr"):
        if abs(float(current[name]) - float(a0[name])) > RANKING_TOLERANCE:
            raise OfficialSWaTError(f"implementation mismatch: {name} is not invariant")


def assert_frozen_endpoints(
    timestamp: np.ndarray,
    labels: np.ndarray,
    expected_count: int,
    segment_manifest: dict[str, Any] | None = None,
) -> None:
    if len(timestamp) != expected_count or len(labels) != expected_count:
        raise OfficialSWaTError("A0++ endpoint count differs from A0")
    if len(timestamp) != len(np.unique(timestamp)):
        raise OfficialSWaTError("A0++ endpoint timestamps are not unique")
    if segment_manifest is not None:
        offset = 0
        for segment in segment_manifest["test_segments"]:
            count = int(segment["row_count"]) - 63
            expected_start = int(np.datetime64(segment["start"], "s").astype(np.int64)) + 63
            expected_end = int(np.datetime64(segment["end_inclusive"], "s").astype(np.int64))
            part = timestamp[offset : offset + count]
            if len(part) != count or int(part[0]) != expected_start or int(part[-1]) != expected_end:
                raise OfficialSWaTError("A0++ endpoint segment boundary differs from A0")
            if len(part) > 1 and not np.all(np.diff(part) == 1):
                raise OfficialSWaTError("A0++ endpoint segment is not internally contiguous")
            offset += count
        if offset != expected_count:
            raise OfficialSWaTError("A0++ segment endpoint counts do not cover the A0 endpoint set")


def choose_status(current: dict[str, float], a0: dict[str, float]) -> str:
    if current["precision"] > a0["precision"] and current["f1"] > a0["f1"] and current["fpr"] <= a0["fpr"] - 0.10:
        return "STATIC_POT_BASELINE_CANDIDATE"
    if all(abs(current[key] - a0[key]) <= 0.02 for key in ("precision", "recall", "f1", "fpr")):
        return "STATIC_POT_NOT_SUFFICIENT"
    return "STATIC_POT_NOT_SUPPORTED"


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(str(config_path))
    validate_config_contract(config)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing S3 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    a0_output = resolve_repo_path(ROOT, config["paths"]["a0_output"])
    a0_metrics = json.loads((a0_output / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(a0_output / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["best_epoch"] != 13 or a0_metrics["parameter_count"] != 11730:
        raise OfficialSWaTError("A0++ requires the frozen epoch-13 A0 checkpoint")
    if not torch.cuda.is_available():
        raise OfficialSWaTError("A0++ requires the same available CUDA runtime")
    device = torch.device("cuda")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    normal, manifest = load_normal_cache(resolve_repo_path(ROOT, config["paths"]["cache_dir"]), config)
    scaler, scaler_metadata = load_scaler(a0_output / "scaler.npz")
    if scaler_metadata["fit_row_count"] != TRAIN_RAW_ROWS or scaler_metadata["attack_rows_seen"] != 0:
        raise OfficialSWaTError("A0++ scaler provenance differs from A0")
    validation_starts = make_window_starts(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS)
    validation_score = validation_endpoint_scores(
        model, normal, validation_starts, scaler, int(config["training"]["batch_size"]), device
    )
    pot_fit = fit_static_pot(validation_score, source_split="normal_validation")
    threshold = float(pot_fit["threshold_pot"])
    validation_exceedance_rate = float(np.mean(validation_score > threshold))

    # Test arrays are read only after POT fitting and are the exact S1 A0 arrays.
    with np.load(a0_output / "test_scores.npz", allow_pickle=False) as payload:
        timestamp = np.asarray(payload["endpoint_timestamp"], dtype=np.int64)
        y_true = np.asarray(payload["label"], dtype=np.int8)
        test_score = np.asarray(payload["endpoint_mae"], dtype=np.float32)
    segment_manifest = json.loads(
        resolve_repo_path(ROOT, config["paths"]["t25_segment_manifest"]).read_text(encoding="utf-8")
    )
    assert_frozen_endpoints(
        timestamp, y_true, int(a0_metrics["test_evaluation_point_count"]), segment_manifest
    )
    if not np.isfinite(test_score).all():
        raise OfficialSWaTError("A0 Test score contains NaN/Inf")
    current = evaluate(y_true, test_score, threshold)
    a0 = a0_metrics["primary_endpoint_mae"]
    assert_ranking_invariant(current, a0)
    delta = {key: float(current[key] - a0[key]) for key in ("precision", "recall", "f1", "fpr")}
    delta["threshold"] = threshold - float(a0_metrics["thresholds"]["endpoint_mae"])
    status = choose_status(current, a0)
    metrics = {
        "experiment_id": "S15-A0++",
        "scientific_status": "VALID_DIAGNOSTIC / PRELIMINARY",
        "status": status,
        "a0_checkpoint_epoch": 13,
        "score": "unchanged A0 raw endpoint 51-channel MAE",
        "pot_threshold": threshold,
        "validation_exceedance_rate_at_pot_threshold": validation_exceedance_rate,
        "a0_empirical_q99": {"threshold": a0_metrics["thresholds"]["endpoint_mae"], **a0},
        "a0pp_static_pot": current,
        "delta_a0pp_minus_a0": delta,
        "ranking_invariant": "PASS",
        "ranking_tolerance": RANKING_TOLERANCE,
        "test_evaluation_point_count": len(test_score),
        "cache_protocol_fingerprint": manifest["protocol_fingerprint"],
        "retraining": False,
        "score_modification": False,
        "threshold_search": False,
        "pot_parameter_search": False,
        "adaptive_threshold": False,
        "test_driven_tuning": False,
        "point_adjustment": False,
    }
    write_json(output_dir / "pot_fit.json", pot_fit)
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "validation_tail_summary.json",
        {
            "score_count": len(validation_score),
            "score_finite": True,
            "score_median": float(np.median(validation_score)),
            "score_q95": float(np.quantile(validation_score, 0.95)),
            "score_q99": float(np.quantile(validation_score, 0.99)),
            "empirical_exceedance_rate_at_pot_threshold": validation_exceedance_rate,
            "target_alpha": ALPHA,
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/thesis_midterm_s3_a0pp")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(resolve_repo_path(ROOT, args.config), resolve_repo_path(ROOT, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
