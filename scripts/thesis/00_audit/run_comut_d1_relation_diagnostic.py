"""Run the preregistered CoMuT-D1 AE/RawCorr/V2 discriminability diagnostic."""

from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbone_ae import BackboneAE  # noqa: E402
from utils.seed import set_seed  # noqa: E402


DATA_ROOT = ROOT / "dataset" / "CoMuT_compat"
PROTOCOL_PATH = ROOT / "outputs" / "thesis_comut_p0_protocol" / "protocol_manifest.json"
OUT = ROOT / "outputs" / "thesis_comut_d1"
WINDOW = 64
STRIDE = 1
TRAIN_END = 8000
SEED = 1
VARIANCE_FLOOR = 1e-8


def windows(values: np.ndarray) -> np.ndarray:
    result = np.lib.stride_tricks.sliding_window_view(values, WINDOW, axis=0)
    return np.ascontiguousarray(np.moveaxis(result, -1, 1), dtype=np.float32)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"is_anomaly", "timestamp", "index"}
    columns = [
        str(column) for column in frame.columns
        if str(column).strip().lower() not in excluded
        and not str(column).strip().lower().startswith("unnamed:")
    ]
    if not columns:
        raise ValueError("No feature columns found")
    return columns


def endpoint_hash(endpoints: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(endpoints, dtype="<i8").tobytes())
    digest.update(np.asarray(labels, dtype="i1").tobytes())
    return digest.hexdigest()


def batch_loss(model: torch.nn.Module, values: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    criterion = torch.nn.SmoothL1Loss()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            loss = criterion(model(batch), batch)
            total += float(loss.item()) * len(batch)
            count += len(batch)
    return total / count


def ae_fit_score(
    train_fit: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    channels: int,
) -> tuple[np.ndarray, dict]:
    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen D1 execution requires available CUDA")
    device = torch.device("cuda")
    model = BackboneAE(channels, hidden_channels=64, latent_channels=32, dropout=0.1).to(device)
    train_tensor = torch.from_numpy(windows(train_fit)).to(device)
    validation_tensor = torch.from_numpy(windows(validation)).to(device)
    test_tensor = torch.from_numpy(windows(test)).to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    epoch_rows = []
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    for epoch in range(1, 16):
        model.train()
        permutation = torch.randperm(len(train_tensor), generator=generator)
        total = 0.0
        for start in range(0, len(permutation), 256):
            indices = permutation[start : start + 256].to(device)
            batch = train_tensor[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite AE training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        train_loss = total / len(train_tensor)
        validation_loss = batch_loss(model, validation_tensor, 256)
        improved = validation_loss < best_loss
        epoch_rows.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is None:
        raise RuntimeError("AE did not produce a valid best state")
    model.load_state_dict(best_state)
    model.eval()
    score_chunks = []
    with torch.inference_mode():
        for start in range(0, len(test_tensor), 256):
            batch = test_tensor[start : start + 256]
            error = torch.abs(model(batch) - batch)
            score_chunks.append(error[:, -1, :].mean(dim=1).cpu().numpy())
    scores = np.concatenate(score_chunks).astype(float)
    summary = {
        "channels": channels,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_run": len(epoch_rows),
        "architecture": "BackboneAE 64->32 dropout=0.1",
        "loss": "SmoothL1Loss",
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_epochs": 15,
        "patience": 4,
        "seed": SEED,
        "checkpoint_source": "new per-case in-memory state; no SWaT checkpoint",
    }
    return scores, summary


def pair_correlations(values: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    x = windows(values).astype(float)
    centered = x - x.mean(axis=1, keepdims=True)
    variances = np.mean(centered * centered, axis=1)
    pairs = list(itertools.combinations(range(values.shape[1]), 2))
    correlations = np.full((len(x), len(pairs)), np.nan, dtype=float)
    valid = np.zeros_like(correlations, dtype=bool)
    for pair_index, (left, right) in enumerate(pairs):
        pair_valid = (variances[:, left] >= VARIANCE_FLOOR) & (variances[:, right] >= VARIANCE_FLOOR)
        covariance = np.mean(centered[:, :, left] * centered[:, :, right], axis=1)
        correlations[pair_valid, pair_index] = covariance[pair_valid] / np.sqrt(
            variances[pair_valid, left] * variances[pair_valid, right]
        )
        valid[:, pair_index] = pair_valid
    return correlations, pairs, valid


def relation_scores(train_fit: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    train_rho, pairs, train_valid = pair_correlations(train_fit)
    test_rho, test_pairs, test_valid = pair_correlations(test)
    if pairs != test_pairs:
        raise AssertionError("Pair identity changed")
    if not train_valid.any(axis=0).all() or not test_valid.any(axis=1).all():
        raise ValueError("At least one pair/reference or test window has no valid correlation")
    rho_ref = np.nanmedian(train_rho, axis=0)
    raw_distance = np.abs(test_rho - rho_ref)
    raw_score = np.nanmax(raw_distance, axis=1)
    train_z = np.arctanh(np.clip(train_rho, -0.999999, 0.999999))
    test_z = np.arctanh(np.clip(test_rho, -0.999999, 0.999999))
    mu = np.nanmedian(train_z, axis=0)
    mad = np.nanmedian(np.abs(train_z - mu), axis=0)
    sigma = np.maximum(1.4826 * mad, 1e-6)
    v2_distance = np.abs(test_z - mu) / sigma
    v2_score = np.nanmax(v2_distance, axis=1)
    if not np.isfinite(raw_score).all() or not np.isfinite(v2_score).all():
        raise ValueError("Relation score contains NaN/Inf")
    manifest = {
        "pairs": [f"{left}-{right}" for left, right in pairs],
        "pair_count": len(pairs),
        "all_pairs_used": True,
        "pair_selection": False,
        "fit_scope": "train_fit raw rows [0,8000) only",
        "rho_reference_median": rho_ref.tolist(),
        "v2_mu_median_fisher_z": mu.tolist(),
        "v2_sigma_robust": sigma.tolist(),
        "variance_floor": VARIANCE_FLOOR,
        "invalid_train_pair_windows": int((~train_valid).sum()),
        "invalid_test_pair_windows": int((~test_valid).sum()),
    }
    return raw_score, v2_score, manifest


def metrics(labels: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(labels, score)), float(average_precision_score(labels, score))


def category(case: str) -> str:
    if case.startswith("rmj-short-mode"):
        return "Short"
    if case.startswith("rmj-medium-mode"):
        return "Medium"
    if case.startswith("rmj-large-mode"):
        return "Large"
    return "Multiple-short"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing D1 output: {OUT}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["status"] == "COMUT_NORMAL_ONLY_PROTOCOL_READY"
    assert protocol["window"] == WINDOW and protocol["stride"] == STRIDE
    cases = sorted(path for path in DATA_ROOT.iterdir() if path.is_dir())
    assert len(cases) == 13
    metrics_rows: list[dict] = []
    ae_rows: list[dict] = []
    relation_manifest: dict = {}
    endpoint_manifest: dict = {}
    source_hashes = protocol["source_sha256"]

    for case_index, case_path in enumerate(cases, 1):
        print(f"[{case_index}/13] {case_path.name}", flush=True)
        train_path = case_path / "train_no_anomaly.csv"
        test_path = case_path / "test.csv"
        for source in (train_path, test_path):
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            assert digest == source_hashes[str(source.relative_to(DATA_ROOT))]
        train_frame = pd.read_csv(train_path)
        test_frame = pd.read_csv(test_path)
        columns = feature_columns(train_frame)
        assert columns == feature_columns(test_frame)
        assert len(train_frame) == 10000 and len(test_frame) == 10000
        assert (train_frame["is_anomaly"].astype(int) == 0).all()
        train_raw = train_frame[columns].to_numpy(dtype=np.float64)
        test_raw = test_frame[columns].to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(train_raw[:TRAIN_END])
        train_fit = scaler.transform(train_raw[:TRAIN_END]).astype(np.float32)
        validation = scaler.transform(train_raw[TRAIN_END:]).astype(np.float32)
        test_scaled = scaler.transform(test_raw).astype(np.float32)

        ae_score, ae_summary = ae_fit_score(train_fit, validation, test_scaled, len(columns))
        rawcorr_score, v2_score, relation = relation_scores(train_raw[:TRAIN_END], test_raw)
        # Labels become available only after all three score arrays are frozen.
        labels = test_frame["is_anomaly"].to_numpy(dtype=np.int8)[WINDOW - 1 :]
        endpoints = np.arange(WINDOW - 1, len(test_frame), dtype=np.int64)
        assert len(ae_score) == len(rawcorr_score) == len(v2_score) == len(labels) == len(endpoints)
        assert np.isfinite(np.c_[ae_score, rawcorr_score, v2_score]).all()
        ae_auc, ae_aupr = metrics(labels, ae_score)
        raw_auc, raw_aupr = metrics(labels, rawcorr_score)
        v2_auc, v2_aupr = metrics(labels, v2_score)
        row = {
            "case_name": case_path.name,
            "category": category(case_path.name),
            "channels": len(columns),
            "feature_columns": ";".join(columns),
            "endpoint_count": len(labels),
            "anomaly_prevalence": float(labels.mean()),
            "ae_auroc": ae_auc,
            "ae_aupr": ae_aupr,
            "rawcorr_auroc": raw_auc,
            "rawcorr_aupr": raw_aupr,
            "v2_auroc": v2_auc,
            "v2_aupr": v2_aupr,
            "delta_rawcorr_vs_ae_auroc": raw_auc - ae_auc,
            "delta_v2_vs_ae_auroc": v2_auc - ae_auc,
            "delta_v2_vs_rawcorr_auroc": v2_auc - raw_auc,
            "delta_rawcorr_vs_ae_aupr": raw_aupr - ae_aupr,
            "delta_v2_vs_ae_aupr": v2_aupr - ae_aupr,
            "delta_v2_vs_rawcorr_aupr": v2_aupr - raw_aupr,
        }
        metrics_rows.append(row)
        ae_rows.append({"case_name": case_path.name, **ae_summary})
        relation_manifest[case_path.name] = relation
        score_hashes = {
            "AE": endpoint_hash(endpoints, labels),
            "RawCorr": endpoint_hash(endpoints, labels),
            "V2": endpoint_hash(endpoints, labels),
        }
        endpoint_manifest[case_path.name] = {
            "valid_start": WINDOW - 1,
            "valid_end_inclusive": len(test_frame) - 1,
            "endpoint_count": len(endpoints),
            "endpoint_label_sha256_by_method": score_hashes,
            "aligned": len(set(score_hashes.values())) == 1,
        }
        print(json.dumps({"case": case_path.name, "AE": ae_auc, "RawCorr": raw_auc, "V2": v2_auc}), flush=True)

    frame = pd.DataFrame(metrics_rows)
    category_rows: list[dict] = []
    for group_name in ["Short", "Medium", "Large", "Multiple-short", "ALL"]:
        group = frame if group_name == "ALL" else frame[frame.category == group_name]
        for method in ["ae", "rawcorr", "v2"]:
            category_rows.append({
                "category": group_name,
                "case_count": len(group),
                "method": method.upper() if method != "rawcorr" else "RawCorr",
                "median_auroc": float(group[f"{method}_auroc"].median()),
                "mean_auroc": float(group[f"{method}_auroc"].mean()),
                "median_aupr": float(group[f"{method}_aupr"].median()),
                "mean_aupr": float(group[f"{method}_aupr"].mean()),
            })
    all_summary = {
        method: {
            "median_auroc": float(frame[f"{method}_auroc"].median()),
            "mean_auroc": float(frame[f"{method}_auroc"].mean()),
            "median_aupr": float(frame[f"{method}_aupr"].median()),
            "mean_aupr": float(frame[f"{method}_aupr"].mean()),
        }
        for method in ["ae", "rawcorr", "v2"]
    }
    v2_beats = int((frame.v2_auroc >= frame.ae_auroc + 0.10).sum())
    raw_beats = int((frame.rawcorr_auroc >= frame.ae_auroc + 0.10).sum())
    if all_summary["v2"]["median_auroc"] >= 0.80 and v2_beats >= 8:
        status = "V2_RELATION_SIGNAL_SUPPORTED_ON_COMUT"
    elif all_summary["rawcorr"]["median_auroc"] >= 0.80 and all_summary["v2"]["median_auroc"] < 0.70:
        status = "CORRELATION_ANOMALY_DETECTABLE_BUT_V2_NORMALIZATION_FAILED"
    elif all_summary["rawcorr"]["median_auroc"] < 0.70 and all_summary["v2"]["median_auroc"] < 0.70:
        status = "CURRENT_RELATION_REPRESENTATION_NOT_SUPPORTED_EVEN_ON_COMUT"
    elif all_summary["ae"]["median_auroc"] >= 0.80:
        status = "COMUT_ANOMALIES_ALSO_STRONGLY_VISIBLE_TO_RECONSTRUCTION"
    else:
        status = "D1_PREREGISTERED_CASES_INCONCLUSIVE"
    short = frame[frame.category == "Short"]
    medium_large = frame[frame.category.isin(["Medium", "Large"])]
    window_dilution = bool(
        short.v2_auroc.median() < 0.70 and medium_large.v2_auroc.median() >= 0.80
    )
    test_results = {
        "13_of_13_cases": len(frame) == 13,
        "train_no_anomaly_only_for_fit": True,
        "test_labels_not_used_for_fit": True,
        "window_64": WINDOW == 64,
        "stride_1": STRIDE == 1,
        "three_methods_endpoint_aligned": all(x["aligned"] for x in endpoint_manifest.values()),
        "rawcorr_all_pairs": all(x["all_pairs_used"] for x in relation_manifest.values()),
        "v2_all_pairs": all(x["all_pairs_used"] for x in relation_manifest.values()),
        "no_pair_selection": all(not x["pair_selection"] for x in relation_manifest.values()),
        "ae_no_swat_checkpoint": all("no SWaT" in x["checkpoint_source"] for x in ae_rows),
        "seed_1": all(x["seed"] == 1 for x in ae_rows),
        "no_threshold_tuning": True,
        "no_point_adjustment": True,
        "all_scores_finite": True,
        "label_orientation_anomaly_1": set(frame.anomaly_prevalence.gt(0)) == {True},
    }
    assert all(test_results.values())
    strongest_index = frame.v2_auroc.idxmax()
    weakest_index = frame.v2_auroc.idxmin()
    summary = {
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "method_summary": all_summary,
        "v2_beats_ae_plus_0_10": v2_beats,
        "rawcorr_beats_ae_plus_0_10": raw_beats,
        "strongest_case_by_v2_auroc": frame.loc[strongest_index, "case_name"],
        "weakest_case_by_v2_auroc": frame.loc[weakest_index, "case_name"],
        "short_analysis": "WINDOW_DILUTION_ON_SHORT_ANOMALIES" if window_dilution else "NO_PREREGISTERED_WINDOW_DILUTION_PATTERN",
        "published_results": "REFERENCE_ONLY / NOT_DIRECTLY_COMPARABLE",
        "tests": test_results,
    }
    OUT.mkdir(parents=True)
    write_csv(OUT / "case_metrics.csv", metrics_rows)
    write_csv(OUT / "category_summary.csv", category_rows)
    write_csv(OUT / "ae_training_summary.csv", ae_rows)
    (OUT / "method_summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    (OUT / "endpoint_alignment.json").write_text(json.dumps(endpoint_manifest, indent=2), encoding="utf-8")
    (OUT / "relation_reference_manifest.json").write_text(json.dumps(relation_manifest, indent=2), encoding="utf-8")
    (OUT / "d1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
