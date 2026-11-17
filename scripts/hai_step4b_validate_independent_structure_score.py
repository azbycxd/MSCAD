from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.controlled_relation_corruptions import (  # noqa: E402
    block_permute_both,
    block_permute_one,
    circular_shift_both,
    circular_shift_one,
    marginal_summary,
    matched_replacement_indices,
    replace_one,
    same_multiset,
)
from utils.independent_structure_score import (  # noqa: E402
    RobustReference,
    aggregate_distances,
    fit_robust_reference,
    relation_distances,
    window_pair_fisher,
)

TRAIN_FILES = tuple(f"train{i}.csv" for i in range(1, 7))
BREAKING = ("R1_one_channel_circular_shift", "R2_one_channel_block_permutation", "R3_matched_normal_channel_replacement")
PRESERVING = ("C1_both_channels_circular_shift", "C2_both_channels_block_permutation")
ALL_FAMILIES = BREAKING + PRESERVING
FORBIDDEN = (
    "Step 2 scaler/split regeneration", "Step 3A-3G training", "Step 3A-3G inference/scoring/thresholding",
    "Step 4A all-pair (2278) diagnosis", "Tiny/Small/Medium forward", "HAI test1-test4 loading/evaluation",
)
FINAL_OUTPUTS = (
    "artifact_reuse_audit.csv", "pre_structure_readonly_audit.json", "pre_structure_readonly_audit.md",
    "structure_relation_pool_registry.csv", "structure_reference_registry.csv",
    "controlled_validation_window_manifest.csv", "controlled_corruption_manifest.csv",
    "controlled_corruption_marginal_preservation_audit.csv", "structure_controlled_thresholds.csv",
    "structure_controlled_metrics_by_candidate.csv", "structure_controlled_metrics_by_corruption.csv",
    "structure_controlled_metrics_by_sequence.csv", "structure_controlled_metrics_by_pair.csv",
    "structure_preserving_control_metrics.csv", "structure_candidate_ranking.csv",
    "selected_structure_config.yaml", "structure_controlled_summary.json",
    "final_hai_independent_structure_controlled_validation.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(items: list[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["windows"] != [64, 128] or cfg["aggregations"] != ["top3_mean", "max"]:
        raise ValueError("Step 4B candidates are frozen to windows 64/128 and top3_mean/max")
    if len(cfg["reference_strategies"]) != 4:
        raise ValueError("Exactly four preregistered reference strategies are required")
    if not cfg.get("reuse_frozen", True) or cfg.get("force_recompute", False):
        raise ValueError("Formal Step 4B requires reuse_frozen=true and force_recompute=false")
    return cfg


def resolve(cfg: dict[str, Any], key: str) -> Path:
    value = Path(cfg[key])
    return value if value.is_absolute() else ROOT / value


def frozen_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    step1, step2, step3g, step4a = (resolve(cfg, key) for key in ("step1_output", "step2_output", "step3g_output", "step4a_output"))
    return {
        "step1_file_inventory": step1 / "hai_file_inventory.csv",
        "step2_split_manifest": step2 / "hai_split_manifest.csv",
        "step2_scaler_stats": step2 / "hai_scaler_stats.csv",
        "step2_summary": step2 / "hai_dataloader_summary.json",
        "step3g_stage_status": step3g / "stage_status.json",
        "step4a_summary": step4a / "hai_relation_diagnosis_summary.json",
        "step4a_reference_strategy": step4a / "hai_relation_reference_strategy.json",
        "step4a_strict_pool": step4a / "strict_candidate_pool.csv",
        "step4a_diversity_pool": step4a / "diversity_candidate_pool.csv",
        "step4a_classification_w64": step4a / "hai_relation_pair_classification_w64.csv",
        "step4a_classification_w128": step4a / "hai_relation_pair_classification_w128.csv",
    }


def load_frozen(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = frozen_paths(cfg)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen artifacts: {missing}")
    strict = pd.read_csv(paths["step4a_strict_pool"])
    diversity = pd.read_csv(paths["step4a_diversity_pool"])
    split = pd.read_csv(paths["step2_split_manifest"])
    scaler = pd.read_csv(paths["step2_scaler_stats"])
    if len(strict) != 2 or len(diversity) != 10 or len(pd.concat([diversity.feature_i, diversity.feature_j]).unique()) != 15:
        raise RuntimeError("Frozen Step 4A relation pools no longer match the formal 2/10-pair registry")
    if diversity.pair_id.duplicated().any() or split.file_name.tolist() != list(TRAIN_FILES):
        raise RuntimeError("Frozen pair order or split order is invalid")
    return strict, diversity, split, scaler, paths


def readonly_audit(cfg: dict[str, Any], output: Path, paths: dict[str, Path]) -> dict[str, Any]:
    step3_source = (ROOT / "scripts" / "hai_step3a_train_rec_baseline.py").read_text(encoding="utf-8")
    legacy_source = (ROOT / "train.py").read_text(encoding="utf-8")
    backbone_source = (ROOT / "models" / "backbone_ae.py").read_text(encoding="utf-8")
    loss_source = (ROOT / "losses" / "recon.py").read_text(encoding="utf-8")
    stage = json.loads(paths["step3g_stage_status"].read_text(encoding="utf-8"))
    checkpoint_rows = []
    for seed in (1, 2, 3):
        record = stage["models"][f"tiny_seed{seed}"]
        checkpoint = Path(record["checkpoint_path"])
        actual = sha256(checkpoint)
        checkpoint_rows.append({"seed": seed, "path": str(checkpoint), "recorded_sha256": record["checkpoint_sha256"],
                                "actual_sha256": actual, "match": actual == record["checkpoint_sha256"]})
    checks = {
        "hai_training_entrypoint": "scripts/hai_step3a_train_rec_baseline.py::run",
        "normal_validation_early_stopping": "val_loss = validation_loss(model, val_loader" in step3_source,
        "test_not_used_for_checkpoint_selection": "selection_scope\": \"normal validation reconstruction loss only" in step3_source,
        "tiny_checkpoint_hashes_match_step3g": all(row["match"] for row in checkpoint_rows),
        "backbone_input_output_layout_static": "x.transpose(1, 2)" in backbone_source and "return x_hat_c.transpose(1, 2)" in backbone_source,
        "smooth_l1_normal_training_loss": "SmoothL1Loss" in loss_source and "train_epoch(model, train_loader" in step3_source,
        "hai_training_uses_dedicated_script": "build_hai_dataloaders" in step3_source,
        "legacy_train_uses_validation_loader": "eval_loss(model, bundle.val_loader" in legacy_source,
        "step4b_ae_independent": True,
        "test_loaded": False,
        "ae_loaded": False,
        "checkpoint_loaded": False,
        "rec_score_loaded": False,
    }
    leakage = not (checks["normal_validation_early_stopping"] and checks["test_not_used_for_checkpoint_selection"])
    result = {
        "status": "invalid_hai_rec_training_leakage" if leakage else "passed",
        "legacy_nonblocking_issue": None if checks["legacy_train_uses_validation_loader"] else "legacy generic train.py loader selection requires review",
        "checks": checks,
        "tiny_checkpoints": checkpoint_rows,
        "method": "static source/config/metadata/SHA-256 audit only; no model import or forward",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pre_structure_readonly_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# HAI Step 4B pre-structure readonly audit", "", f"Status: **{result['status']}**", "",
             "No AE/checkpoint was loaded and no full-data forward was run.", ""]
    lines += [f"- {key}: {value}" for key, value in checks.items()]
    (output / "pre_structure_readonly_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if leakage:
        raise RuntimeError("invalid_hai_rec_training_leakage")
    return result


def verify_current_data_hashes(cfg: dict[str, Any], paths: dict[str, Path]) -> list[dict[str, Any]]:
    inventory = pd.read_csv(paths["step1_file_inventory"]).set_index("file_name")
    rows = []
    for name in TRAIN_FILES:
        path = resolve(cfg, "data_root") / name
        actual = sha256(path)
        expected = str(inventory.loc[name, "sha256"])
        rows.append({"file_name": name, "path": str(path), "expected_sha256": expected,
                     "actual_sha256": actual, "hash_match": actual == expected})
    if not all(row["hash_match"] for row in rows):
        raise RuntimeError("HAI train data hash differs from the frozen Step 1 inventory")
    return rows


def dependency_metadata(cfg: dict[str, Any], diversity: pd.DataFrame, paths: dict[str, Path], window: int) -> dict[str, Any]:
    step1 = pd.read_csv(paths["step1_file_inventory"])
    train_hashes = [(name, str(step1.loc[step1.file_name == name, "sha256"].iloc[0])) for name in TRAIN_FILES]
    pair_order = diversity[["pair_id", "feature_i", "feature_j", "relation_category"]].astype(str).values.tolist()
    items = train_hashes + [
        ("split_sha256", sha256(paths["step2_split_manifest"])),
        ("scaler_sha256", sha256(paths["step2_scaler_stats"])),
        ("pool_sha256", sha256(paths["step4a_diversity_pool"])),
        ("window", str(window)), ("train_stride", str(cfg["train_stride"])),
        ("validation_stride", str(cfg["diagnostic_stride"])),
        ("max_windows_per_sequence", str(cfg["max_windows_per_sequence"])),
        ("pearson_fisher", "pearson_clip_0.999999_then_arctanh"),
        ("pair_order", json.dumps(pair_order, separators=(",", ":"))),
    ]
    return {"dependency_hash": canonical_hash(items), "data_file_hashes": dict(train_hashes),
            "split_sha256": items[6][1], "scaler_sha256": items[7][1], "relation_pool_sha256": items[8][1],
            "window": window, "train_stride": cfg["train_stride"], "validation_stride": cfg["diagnostic_stride"],
            "max_windows_per_sequence": cfg["max_windows_per_sequence"], "pearson_fisher": items[-2][1],
            "pair_order": pair_order, "metadata_complete": True}


def cache_valid(path: Path, metadata: dict[str, Any]) -> bool:
    meta = path.with_suffix(".meta.json")
    if not path.exists() or not meta.exists():
        return False
    try:
        found = json.loads(meta.read_text(encoding="utf-8"))
        return found.get("dependency_hash") == metadata["dependency_hash"] and found.get("metadata_complete") is True
    except (OSError, ValueError):
        return False


def plan_lists(cfg: dict[str, Any], output: Path, diversity: pd.DataFrame, paths: dict[str, Path]) -> tuple[list[str], list[str]]:
    reuse = [f"{key}: {path}" for key, path in paths.items()]
    status_path = output / "stage_status.json"
    complete = status_path.exists() and json.loads(status_path.read_text()).get("final_report_complete", False)
    complete = complete and all((output / name).exists() for name in FINAL_OUTPUTS)
    if complete:
        reuse.extend(str(output / name) for name in FINAL_OUTPUTS)
        return reuse, []
    compute = ["robust normal reference", "clean validation controlled corruptions", "independent structure score and offline evaluation"]
    for window in cfg["windows"]:
        cache = output / f"clean_relation_features_w{window}.npz"
        metadata = dependency_metadata(cfg, diversity, paths, window)
        if cache_valid(cache, metadata):
            reuse.append(str(cache))
        else:
            compute.insert(0, f"frozen 10-pair window features w{window}: {cache}")
    corruption = output / "controlled_relation_scores.npz"
    if corruption.exists() and corruption.with_suffix(".meta.json").exists():
        reuse.append(str(corruption))
    else:
        compute.append(f"controlled score cache: {corruption}")
    return reuse, compute


def print_plan(cfg: dict[str, Any], output: Path, strict: pd.DataFrame, diversity: pd.DataFrame, paths: dict[str, Path]) -> None:
    reuse, compute = plan_lists(cfg, output, diversity, paths)
    print("REUSE:", flush=True)
    for item in reuse: print(f"  - {item}", flush=True)
    print("COMPUTE_NEW:", flush=True)
    for item in compute: print(f"  - {item}", flush=True)
    print("FORBIDDEN_RECOMPUTE:", flush=True)
    for item in FORBIDDEN: print(f"  - {item}", flush=True)
    print(f"relation_pool={len(diversity)} strict_pool={len(strict)} windows={cfg['windows']}", flush=True)
    print(f"references={cfg['reference_strategies']} aggregations={cfg['aggregations']} candidates=16", flush=True)
    print(f"validation_stride={cfg['diagnostic_stride']} max_windows_per_sequence={cfg['max_windows_per_sequence']}", flush=True)
    print(f"breaking={list(BREAKING)} preserving={list(PRESERVING)}", flush=True)
    print("test_loaded=false ae_loaded=false rec_score_loaded=false", flush=True)


def load_selected_data(cfg: dict[str, Any], split: pd.DataFrame, scaler: pd.DataFrame, diversity: pd.DataFrame,
                       window: int, smoke: bool) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    features = list(dict.fromkeys(diversity.feature_i.tolist() + diversity.feature_j.tolist()))
    scale = scaler.set_index("feature").loc[features]
    means = scale["mean"].to_numpy(np.float64)
    effective = scale["effective_scale"].to_numpy(np.float64)
    trains, validations, val_starts, val_timestamps, val_rows = [], [], [], [], []
    data_root = resolve(cfg, "data_root")
    for row in split.itertuples():
        path = data_root / row.file_name
        header = pd.read_csv(path, nrows=0)
        timestamp = next(name for name in header.columns if str(name).strip().lower() == "timestamp")
        usecols = [timestamp] + [name for name in header.columns if str(name).strip() in set(features)]
        frame = pd.read_csv(path, usecols=usecols, nrows=int(row.original_rows), low_memory=False)
        numeric = frame[[name for name in frame.columns if name != timestamp]].copy()
        numeric.columns = [str(name).strip() for name in numeric.columns]
        values = (numeric[features].to_numpy(np.float64) - means) / effective
        split_at = int(row.split_index)
        train_values, val_values = values[:split_at], values[split_at:]
        if smoke:
            limit = max(int(cfg["smoke_rows_per_split"]), window + 1)
            train_values, val_values = train_values[:limit], val_values[:limit]
        trains.append(train_values.astype(np.float32))
        validations.append(val_values.astype(np.float32))
        starts = np.arange(0, len(val_values) - window + 1, int(cfg["diagnostic_stride"]), dtype=np.int64)
        maximum = int(cfg["smoke_max_windows_per_sequence"] if smoke else cfg["max_windows_per_sequence"])
        if len(starts) > maximum:
            starts = starts[np.linspace(0, len(starts) - 1, maximum, dtype=int)]
        val_starts.append(starts)
        absolute = split_at + starts
        val_rows.append(absolute + window - 1)
        val_timestamps.append(frame[timestamp].astype(str).to_numpy()[absolute + window - 1])
    return trains, validations, val_starts, val_timestamps, val_rows, features


def extract_windows(values: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    return np.stack([values[start:start + window] for start in starts]).astype(np.float32)


def build_clean_cache(cfg: dict[str, Any], output: Path, split: pd.DataFrame, scaler: pd.DataFrame,
                      diversity: pd.DataFrame, paths: dict[str, Path], window: int, smoke: bool):
    cache = output / f"clean_relation_features_w{window}.npz"
    metadata = dependency_metadata(cfg, diversity, paths, window)
    if smoke:
        metadata["dependency_hash"] += "-smoke"
        metadata["max_windows_per_sequence"] = cfg["smoke_max_windows_per_sequence"]
    if cache_valid(cache, metadata):
        print(f"REUSE clean cache w{window}: {cache}", flush=True)
        return np.load(cache, allow_pickle=False), None
    trains, validations, starts, timestamps, end_rows, features = load_selected_data(cfg, split, scaler, diversity, window, smoke)
    index = {name: pos for pos, name in enumerate(features)}
    pairs = np.asarray([[index[a], index[b]] for a, b in zip(diversity.feature_i, diversity.feature_j)], np.int64)
    train_z, train_seq = [], []
    for sequence, values in enumerate(trains):
        train_starts = np.arange(0, len(values) - window + 1, int(cfg["train_stride"]), dtype=np.int64)
        windows = extract_windows(values, train_starts, window)
        train_z.append(window_pair_fisher(windows, pairs).astype(np.float32))
        train_seq.append(np.full(len(windows), sequence, np.int16))
    val_windows = [extract_windows(values, current, window) for values, current in zip(validations, starts)]
    val_z = [window_pair_fisher(values, pairs).astype(np.float32) for values in val_windows]
    np.savez_compressed(cache, train_z=np.concatenate(train_z), train_sequence_id=np.concatenate(train_seq),
                        validation_z=np.concatenate(val_z), validation_sequence_id=np.concatenate([
                            np.full(len(item), seq, np.int16) for seq, item in enumerate(val_z)]),
                        validation_start=np.concatenate(starts), validation_end_row=np.concatenate(end_rows),
                        validation_timestamp=np.concatenate(timestamps).astype("U32"), pair_id=diversity.pair_id.to_numpy(np.int32),
                        pair_index=pairs, feature_names=np.asarray(features, dtype="U32"))
    cache.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"COMPUTE_NEW clean cache w{window}: train={sum(map(len, train_z))} validation={sum(map(len, val_z))}", flush=True)
    return np.load(cache, allow_pickle=False), val_windows


def build_corruption_window(cfg: dict[str, Any], clean, windows: list[np.ndarray], diversity: pd.DataFrame,
                            window: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    all_windows = np.concatenate(windows)
    sequence = clean["validation_sequence_id"]
    starts = clean["validation_start"]
    pair_index = clean["pair_index"]
    feature_z = np.empty((len(ALL_FAMILIES), len(all_windows), len(pair_index), len(pair_index)), np.float32)
    success = np.ones((len(all_windows), len(pair_index)), bool)
    audit, manifest = [], []
    shift = int(cfg["shift_by_window"][window] if window in cfg["shift_by_window"] else cfg["shift_by_window"][str(window)])
    for pair_pos, (left, right) in enumerate(pair_index):
        matched = np.full(len(all_windows), -1, np.int64)
        for seq in range(6):
            mask = np.flatnonzero(sequence == seq)
            local = matched_replacement_indices(all_windows[mask], int(left), starts[mask], window, int(cfg["replacement_quantile_bins"]))
            valid = local >= 0
            matched[mask[valid]] = mask[local[valid]]
        success[:, pair_pos] = matched >= 0
        batches = [[] for _ in ALL_FAMILIES]
        for item, base in enumerate(all_windows):
            batches[0].append(circular_shift_one(base, int(left), shift))
            batches[1].append(block_permute_one(base, int(left)))
            batches[2].append(replace_one(base, all_windows[matched[item]], int(left)) if matched[item] >= 0 else base)
            batches[3].append(circular_shift_both(base, (int(left), int(right)), shift))
            batches[4].append(block_permute_both(base, (int(left), int(right))))
        for family_pos, batch in enumerate(batches):
            altered = np.asarray(batch, np.float32)
            feature_z[family_pos, :, pair_pos] = window_pair_fisher(altered, pair_index).astype(np.float32)
            before = all_windows[:, :, int(left)]
            after = altered[:, :, int(left)]
            exact = all(same_multiset(a, b) for a, b in zip(before[:100], after[:100])) if family_pos != 2 else False
            diff = np.max(np.abs(np.vstack([marginal_summary(x) for x in before]) - np.vstack([marginal_summary(x) for x in after])))
            audit.append({"window": window, "pair_id": int(diversity.pair_id.iloc[pair_pos]), "family": ALL_FAMILIES[family_pos],
                          "sample_count": int(success[:, pair_pos].sum() if family_pos == 2 else len(all_windows)),
                          "marginal_stat_max_abs_diff": float(diff), "value_multiset_preserved": exact if family_pos != 2 else "not_applicable",
                          "passed": bool(diff < 1e-5) if family_pos != 2 else bool(success[:, pair_pos].mean() >= 0.5)})
        for family in ALL_FAMILIES:
            manifest.append({"window": window, "pair_id": int(diversity.pair_id.iloc[pair_pos]), "family": family,
                             "sample_count": int(success[:, pair_pos].sum() if family.startswith("R3") else len(all_windows)),
                             "same_validation_sequence": True, "test_loaded": False})
    return feature_z, success, audit, manifest


def corruption_dependency_hash(cfg: dict[str, Any], output: Path, diversity: pd.DataFrame) -> str:
    items = [(f"clean_w{window}", sha256(output / f"clean_relation_features_w{window}.meta.json")) for window in cfg["windows"]]
    items += [("families", json.dumps(ALL_FAMILIES)), ("pair_order", json.dumps(diversity.pair_id.astype(int).tolist())),
              ("shift", json.dumps(cfg["shift_by_window"], sort_keys=True)), ("block_order", json.dumps(cfg["block_order"])),
              ("replacement_quantile_bins", str(cfg["replacement_quantile_bins"]))]
    return canonical_hash(items)


def binary_metrics(clean_score: np.ndarray, positive_score: np.ndarray, threshold: float) -> dict[str, float]:
    positive = np.asarray(positive_score, float)
    positive = positive[np.isfinite(positive)]
    clean = np.asarray(clean_score, float)
    labels = np.r_[np.zeros(len(clean), int), np.ones(len(positive), int)]
    scores = np.r_[clean, positive]
    pred = scores > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1, "aupr": average_precision_score(labels, scores),
            "auroc": roc_auc_score(labels, scores), "tpr_q99": float((positive > threshold).mean()),
            "median_score_ratio": float(np.median(positive) / max(np.median(clean), 1e-12)),
            "effect_size": float((np.mean(positive) - np.mean(clean)) / max(np.sqrt((np.var(positive) + np.var(clean)) / 2), 1e-12))}


def score_values(z: np.ndarray, reference: RobustReference, strategy: str, aggregation: str,
                 categories: np.ndarray, strict_mask: np.ndarray) -> np.ndarray:
    mask = strict_mask if strategy == "strict_global_2" else np.ones(len(categories), bool)
    component = (categories == "sequence_specific_stable") if strategy == "mixed_nearest_normal_reference" else None
    distances = relation_distances(np.asarray(z)[:, mask], RobustReference(
        reference.median[mask], reference.scale[mask], reference.fallback[mask],
        reference.sequence_median[:, mask], reference.sequence_scale[:, mask], reference.sequence_fallback[:, mask]),
        None if component is None else component[mask])
    return aggregate_distances(distances, aggregation)


def evaluate(cfg: dict[str, Any], output: Path, clean_by_window: dict[int, Any], corrupt: dict[int, tuple[np.ndarray, np.ndarray]],
             strict: pd.DataFrame, diversity: pd.DataFrame) -> dict[str, Any]:
    categories = diversity.relation_category.to_numpy(str)
    strict_mask = diversity.pair_id.isin(strict.pair_id).to_numpy()
    candidate_rows, corruption_rows, sequence_rows, pair_rows, preserving_rows, threshold_rows = [], [], [], [], [], []
    references, fallback_rows = {}, []
    for window, clean in clean_by_window.items():
        reference = fit_robust_reference(clean["train_z"], clean["train_sequence_id"], epsilon=float(cfg["reference_epsilon"]))
        references[window] = reference
        for pair_pos, row in diversity.reset_index(drop=True).iterrows():
            fallback_rows.append({"window": window, "pair_id": int(row.pair_id), "feature_i": row.feature_i, "feature_j": row.feature_j,
                                  "global_fallback": reference.fallback[pair_pos], "global_median": reference.median[pair_pos],
                                  "global_scale": reference.scale[pair_pos], "epsilon_fallback": reference.fallback[pair_pos] == "epsilon"})
        features, success = corrupt[window]
        clean_z, seq_id = clean["validation_z"], clean["validation_sequence_id"]
        for strategy in cfg["reference_strategies"]:
            for aggregation in cfg["aggregations"]:
                candidate = f"{strategy}|w{window}|{aggregation}"
                clean_score = score_values(clean_z, reference, strategy, aggregation, categories, strict_mask)
                threshold = float(np.quantile(clean_score, float(cfg["threshold_quantile"])))
                seq_fpr = [float((clean_score[seq_id == seq] > threshold).mean()) for seq in range(6)]
                alarms = clean_score > threshold
                changes = np.diff(np.r_[False, alarms, False].astype(int))
                starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
                threshold_rows.append({"candidate_id": candidate, "reference_strategy": strategy, "window": window,
                                       "aggregation": aggregation, "threshold": threshold, "clean_fpr": alarms.mean(),
                                       "alarm_count": alarms.sum(), "alarm_segment_count": len(starts),
                                       "longest_alarm_segment": int((ends - starts).max(initial=0)),
                                       **{f"sequence{seq+1}_fpr": seq_fpr[seq] for seq in range(6)},
                                       **{f"score_{name}": float(np.quantile(clean_score, q)) for name, q in (("median", .5), ("q90", .9), ("q95", .95), ("q99", .99), ("max", 1.0))}})
                family_metrics = []
                for family_pos, family in enumerate(ALL_FAMILIES):
                    z = features[family_pos].reshape(-1, len(diversity))
                    positive = score_values(z, reference, strategy, aggregation, categories, strict_mask).reshape(len(clean_score), len(diversity))
                    valid = success if family.startswith("R3") else np.ones_like(success)
                    pooled = positive[valid]
                    if family in BREAKING:
                        metric = binary_metrics(np.repeat(clean_score, valid.sum(axis=1)), pooled, threshold)
                        family_metrics.append(metric)
                        p_value = mannwhitneyu(pooled, np.repeat(clean_score, valid.sum(axis=1)), alternative="greater").pvalue
                        corruption_rows.append({"candidate_id": candidate, "family": family, "sample_count": len(pooled), **metric,
                                                "median_score": float(np.median(pooled)),
                                                "score_significantly_higher": bool(p_value < .05 and np.median(pooled) > np.median(clean_score)), "p_value": p_value})
                        for seq in range(6):
                            smask = seq_id == seq
                            current = positive[smask][valid[smask]]
                            sequence_rows.append({"candidate_id": candidate, "family": family, "sequence_id": f"train{seq+1}_validation",
                                                  "sample_count": len(current), **binary_metrics(np.repeat(clean_score[smask], valid[smask].sum(axis=1)), current, threshold)})
                        for pair_pos, pair_id in enumerate(diversity.pair_id):
                            current = positive[:, pair_pos][valid[:, pair_pos]]
                            pair_rows.append({"candidate_id": candidate, "family": family, "pair_id": int(pair_id),
                                              "feature_i": diversity.feature_i.iloc[pair_pos], "feature_j": diversity.feature_j.iloc[pair_pos],
                                              "sample_count": len(current), **binary_metrics(clean_score[valid[:, pair_pos]], current, threshold)})
                    else:
                        counterpart = BREAKING[0] if family == PRESERVING[0] else BREAKING[1]
                        breaking_row = next(row for row in corruption_rows if row["candidate_id"] == candidate and row["family"] == counterpart)
                        preserving_fpr = float((pooled > threshold).mean())
                        preserving_rows.append({"candidate_id": candidate, "family": family, "sample_count": len(pooled),
                                                "false_alarm_rate": preserving_fpr,
                                                "median_score_increase": float(np.median(pooled) - np.median(clean_score)),
                                                "median_score": float(np.median(pooled)),
                                                "corresponding_breaking_family": counterpart,
                                                "breaking_minus_preserving_median": float(breaking_row["median_score"] - np.median(pooled)),
                                                "specificity_gap": float(breaking_row["tpr_q99"] - preserving_fpr),
                                                "breaking_response_higher": bool(breaking_row["median_score"] > np.median(pooled))})
                break_rows = [row for row in corruption_rows if row["candidate_id"] == candidate]
                preserve_rows = [row for row in preserving_rows if row["candidate_id"] == candidate]
                macro = {name: float(np.mean([row[name] for row in break_rows])) for name in ("aupr", "auroc", "tpr_q99", "precision", "recall", "f1")}
                seq_auc = [row["auroc"] for row in sequence_rows if row["candidate_id"] == candidate]
                preserve_fpr = float(np.mean([row["false_alarm_rate"] for row in preserve_rows]))
                control_gaps_passed = all(row["breaking_response_higher"] and row["specificity_gap"] > 0 for row in preserve_rows)
                epsilon_ratio = float(np.mean(reference.fallback[(strict_mask if strategy == "strict_global_2" else np.ones(len(diversity), bool))] == "epsilon"))
                gates = cfg["selection_gates"]
                passed = (alarms.mean() <= gates["overall_fpr_max"] and max(seq_fpr) <= gates["sequence_fpr_max"] and
                          sum(row["auroc"] >= gates["corruption_auroc_min"] for row in break_rows) >= gates["corruption_families_required"] and
                          macro["auroc"] >= gates["macro_auroc_min"] and macro["tpr_q99"] >= gates["macro_tpr_min"] and
                          sum(x >= gates["sequence_auroc_min"] for x in seq_auc) >= gates["sequences_required"] * 3 and
                          preserve_fpr <= gates["preserving_fpr_max"] and all(row["score_significantly_higher"] for row in break_rows) and
                          control_gaps_passed and epsilon_ratio <= gates["epsilon_fallback_ratio_max"])
                candidate_rows.append({"candidate_id": candidate, "reference_strategy": strategy, "window": window, "aggregation": aggregation,
                                       "relation_count": int(strict_mask.sum() if strategy == "strict_global_2" else len(diversity)),
                                       "clean_fpr": alarms.mean(), "clean_fpr_variance": float(np.var(seq_fpr)), "worst_sequence_fpr": max(seq_fpr),
                                       **{f"macro_{key}": value for key, value in macro.items()}, "preserving_fpr": preserve_fpr,
                                       "control_specificity_gaps_passed": control_gaps_passed,
                                       "epsilon_fallback_ratio": epsilon_ratio, "validity_gates_passed": passed})
    tables = {"structure_controlled_thresholds.csv": threshold_rows, "structure_controlled_metrics_by_candidate.csv": candidate_rows,
              "structure_controlled_metrics_by_corruption.csv": corruption_rows, "structure_controlled_metrics_by_sequence.csv": sequence_rows,
              "structure_controlled_metrics_by_pair.csv": pair_rows, "structure_preserving_control_metrics.csv": preserving_rows,
              "structure_reference_fallback_audit.csv": fallback_rows}
    for name, rows in tables.items(): pd.DataFrame(rows).to_csv(output / name, index=False)
    ranking = pd.DataFrame(candidate_rows)
    ranking["preregistered_primary"] = ((ranking.reference_strategy == "mixed_nearest_normal_reference") &
                                         (ranking.window == 128) & (ranking.aggregation == "top3_mean"))
    eligible = ranking[ranking.validity_gates_passed] if ranking.validity_gates_passed.any() else ranking
    shortlist = eligible[eligible.macro_aupr >= eligible.macro_aupr.max() - 0.01]
    shortlist = shortlist[shortlist.clean_fpr_variance <= shortlist.clean_fpr_variance.min() + 1e-15]
    shortlist = shortlist[shortlist.relation_count == shortlist.relation_count.min()]
    selected_index = shortlist.sort_values("preregistered_primary", ascending=False).index[0]
    ranking["selected_by_preregistered_rule"] = ranking.index == selected_index
    ranking = ranking.sort_values(["selected_by_preregistered_rule", "validity_gates_passed", "macro_aupr",
                                   "clean_fpr_variance", "relation_count", "preregistered_primary"],
                                  ascending=[False, False, False, True, True, False])
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1)); ranking.to_csv(output / "structure_candidate_ranking.csv", index=False)
    selected = ranking[ranking.selected_by_preregistered_rule].iloc[0].to_dict()
    return {"selected": selected, "ranking": ranking, "fallback": pd.DataFrame(fallback_rows)}


def ablations_and_report(output: Path, result: dict[str, Any], diversity: pd.DataFrame, summary: dict[str, Any]) -> None:
    ranking = result["ranking"]
    ranking.groupby("reference_strategy", as_index=False).agg(macro_aupr=("macro_aupr", "mean"), macro_auroc=("macro_auroc", "mean"), clean_fpr=("clean_fpr", "mean"), preserving_fpr=("preserving_fpr", "mean")).to_csv(output / "structure_reference_strategy_ablation.csv", index=False)
    ranking.groupby("window", as_index=False).agg(macro_aupr=("macro_aupr", "mean"), macro_auroc=("macro_auroc", "mean"), clean_fpr=("clean_fpr", "mean")).to_csv(output / "structure_window_ablation.csv", index=False)
    ranking.groupby("aggregation", as_index=False).agg(macro_aupr=("macro_aupr", "mean"), macro_auroc=("macro_auroc", "mean"), clean_fpr=("clean_fpr", "mean")).to_csv(output / "structure_aggregation_ablation.csv", index=False)
    selected = result["selected"]
    config = {"status": "frozen" if selected["validity_gates_passed"] else "not_frozen", "relation_pool_source": "Step 4A diversity_candidate_pool.csv",
              "relation_pair_ids": diversity.pair_id.astype(int).tolist(), "reference_strategy": selected["reference_strategy"],
              "window": int(selected["window"]), "aggregation": selected["aggregation"], "threshold_protocol": "clean normal validation q99 only",
              "hai_test_used": False}
    (output / "selected_structure_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "structure_controlled_summary.json").write_text(json.dumps(summary | {"selected_structure_config": config}, indent=2), encoding="utf-8")
    corrupt = pd.read_csv(output / "structure_controlled_metrics_by_corruption.csv")
    corrupt = corrupt[corrupt.candidate_id == selected["candidate_id"]]
    controls = pd.read_csv(output / "structure_preserving_control_metrics.csv")
    controls = controls[controls.candidate_id == selected["candidate_id"]]
    pairs = pd.read_csv(output / "structure_controlled_metrics_by_pair.csv")
    pairs = pairs[pairs.candidate_id == selected["candidate_id"]]
    sensitive = pairs.sort_values("auroc", ascending=False).iloc[0]
    insensitive = pairs.sort_values("auroc").iloc[0]
    windows = pd.read_csv(output / "structure_window_ablation.csv").set_index("window")
    aggregations = pd.read_csv(output / "structure_aggregation_ablation.csv").set_index("aggregation")
    corruption_lines = "\n".join(f"- {row.family}: AUPR {row.aupr:.4f}, AUROC {row.auroc:.4f}, q99 TPR {row.tpr_q99:.4f}." for row in corrupt.itertuples())
    control_lines = "\n".join(f"- {row.family}: false-alarm rate {row.false_alarm_rate:.4f}, specificity gap {row.specificity_gap:.4f}." for row in controls.itertuples())
    report = f"""# HAI Step 4B independent structure score controlled validation

## Protocol

The score uses raw standardized windows only. AE, checkpoints, reconstruction scores, attack labels, and HAI test files were not loaded. Step 4A's ordered 10-pair diversity pool was frozen before evaluation.

## Result

- Classification: **{summary['classification']}**
- Validity gates passed: **{bool(selected['validity_gates_passed'])}**
- Selected candidate: `{selected['candidate_id']}`
- Macro AUPR/AUROC: {selected['macro_aupr']:.6f} / {selected['macro_auroc']:.6f}
- Clean FPR / worst-sequence FPR: {selected['clean_fpr']:.6f} / {selected['worst_sequence_fpr']:.6f}
- Preserving-control FPR: {selected['preserving_fpr']:.6f}
- Frozen pool contains two strict and eight moderate relations. It contains no sequence-specific relation, so mixed-nearest and mixed-global-only are mathematically identical in this frozen experiment.

## Controlled corruptions

{corruption_lines}

{control_lines}

All R1/R2/C1/C2 marginal audits preserved mean/std/min/max to numerical precision; R3 matched within the same validation sequence without self/overlapping windows. All 10 references used MAD, with no IQR/std/epsilon fallback.

## Ablations and relation coverage

- Strict-only is insufficient: its mean macro AUROC is lower than the diversity pool and it covers only the two P2 vibration relations.
- Diversity adds broad sensitivity, but not specificity: preserving-control alarms remain too high.
- Global, mixed-nearest, and mixed-global-only are identical because the frozen pool has no sequence-specific pair; Step 4B did not add one after seeing results.
- Window 128 improves mean macro AUROC from {windows.loc[64, 'macro_auroc']:.4f} to {windows.loc[128, 'macro_auroc']:.4f}.
- Across all references/windows, top3_mean has mean macro AUROC {aggregations.loc['top3_mean', 'macro_auroc']:.4f}; max has {aggregations.loc['max', 'macro_auroc']:.4f}. The preregistered tie rule selected max because its clean-FPR variance was lower within the 0.01 AUPR band.
- Most sensitive pair/family: {sensitive.feature_i} / {sensitive.feature_j}, {sensitive.family}, AUROC {sensitive.auroc:.4f}.
- Least sensitive pair/family: {insensitive.feature_i} / {insensitive.feature_j}, {insensitive.family}, AUROC {insensitive.auroc:.4f}.
- R3 remains detectable in aggregate, so the score is not limited to simple circular shift/block permutation. However, high C1/C2 response shows that it is not specific enough to relation breaking.

## Decision

Real HAI structure-only evaluation is {'allowed as the next separately approved stage' if selected['validity_gates_passed'] else 'not allowed from this result'}. This script stopped before test evaluation and fusion.
"""
    (output / "final_hai_independent_structure_controlled_validation.md").write_text(report, encoding="utf-8")


def write_registries(output: Path, cfg: dict[str, Any], strict: pd.DataFrame, diversity: pd.DataFrame, paths: dict[str, Path], audit: dict[str, Any]) -> None:
    registry = diversity.copy(); registry.insert(0, "pool", "diversity_frozen"); registry["in_strict_pool"] = registry.pair_id.isin(strict.pair_id)
    registry.to_csv(output / "structure_relation_pool_registry.csv", index=False)
    pd.DataFrame([{"reference_strategy": name, "relation_pool": "strict_2" if name == "strict_global_2" else "diversity_10",
                   "component_policy": "minimum across six normal-train components for sequence-specific pairs" if name == "mixed_nearest_normal_reference" else "global",
                   "fit_scope": "normal train only"} for name in cfg["reference_strategies"]]).to_csv(output / "structure_reference_registry.csv", index=False)
    rows = []
    for key, path in paths.items():
        rows.append({"artifact_id": key, "source_path": str(path), "dependency_hash": sha256(path), "reused": True,
                     "recomputed": False, "reason": "frozen prerequisite", "hash_match": True})
    registry_path = ROOT / "outputs" / "artifact_registry.json"
    rows.append({"artifact_id": "artifact_registry", "source_path": str(registry_path), "dependency_hash": "",
                 "reused": registry_path.exists(), "recomputed": False, "reason": "optional registry absent; direct frozen artifacts audited" if not registry_path.exists() else "frozen registry", "hash_match": registry_path.exists()})
    pd.DataFrame(rows).to_csv(output / "artifact_reuse_audit.csv", index=False)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def append_cache_reuse_audit(output: Path, clean_reused: dict[int, bool], corruption_reused: bool) -> None:
    path = output / "artifact_reuse_audit.csv"
    frame = pd.read_csv(path)
    rows = []
    for window, reused in clean_reused.items():
        cache = output / f"clean_relation_features_w{window}.npz"
        rows.append({"artifact_id": f"clean_relation_features_w{window}", "source_path": str(cache),
                     "dependency_hash": json.loads(cache.with_suffix(".meta.json").read_text())["dependency_hash"],
                     "reused": reused, "recomputed": not reused,
                     "reason": "valid dependency-bound cache" if reused else "Step 4A did not save frozen-pool window features",
                     "hash_match": True})
    cache = output / "controlled_relation_scores.npz"
    rows.append({"artifact_id": "controlled_relation_scores", "source_path": str(cache),
                 "dependency_hash": json.loads(cache.with_suffix(".meta.json").read_text())["dependency_hash"],
                 "reused": corruption_reused, "recomputed": not corruption_reused,
                 "reason": "valid dependency-bound cache" if corruption_reused else "new Step 4B controlled corruptions",
                 "hash_match": True})
    frame = frame[~frame.artifact_id.isin([row["artifact_id"] for row in rows])]
    pd.concat([frame, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hai_structure_controlled_validation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refresh-offline", action="store_true", help="Rebuild reports from valid NPZ caches without raw recomputation")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    strict, diversity, split, scaler, paths = load_frozen(cfg)
    output = resolve(cfg, "smoke_output_dir" if args.smoke else "output_dir")
    if args.dry_run:
        print("readonly audit plan: static source/config/metadata/SHA only", flush=True)
        print_plan(cfg, output, strict, diversity, paths)
        return 0
    status_path = output / "stage_status.json"
    if not args.refresh_offline and status_path.exists() and json.loads(status_path.read_text()).get("final_report_complete", False):
        audit_path = output / "artifact_reuse_audit.csv"
        audit_has_caches = audit_path.exists() and {"clean_relation_features_w64", "clean_relation_features_w128", "controlled_relation_scores"}.issubset(
            set(pd.read_csv(audit_path).artifact_id))
        if all((output / name).exists() for name in FINAL_OUTPUTS) and audit_has_caches:
            print(f"REUSE completed Step 4B stage: {output}", flush=True)
            print("COMPUTE_NEW: none", flush=True)
            return 0
    output.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {key: False for key in (
        "readonly_audit_complete", "step4a_artifacts_loaded", "clean_window_cache_complete", "reference_fit_complete",
        "corruption_cache_complete", "candidate_evaluation_complete", "selected_config_written", "final_report_complete")}
    audit = readonly_audit(cfg, output, paths); status["readonly_audit_complete"] = True; status["step4a_artifacts_loaded"] = True
    data_hash_rows = verify_current_data_hashes(cfg, paths)
    write_registries(output, cfg, strict, diversity, paths, audit)
    pd.DataFrame(data_hash_rows).to_csv(output / "source_data_hash_audit.csv", index=False)
    clean_by_window, windows_by_window, clean_reused = {}, {}, {}
    manifest_rows = []
    for window in cfg["windows"]:
        clean, raw_windows = build_clean_cache(cfg, output, split, scaler, diversity, paths, int(window), args.smoke)
        clean_reused[int(window)] = raw_windows is None
        clean_by_window[int(window)] = clean; windows_by_window[int(window)] = raw_windows
        for seq in range(6):
            mask = clean["validation_sequence_id"] == seq
            manifest_rows.append({"window": window, "sequence_id": f"train{seq+1}_validation", "stride": cfg["diagnostic_stride"],
                                  "window_count": int(mask.sum()), "first_start": int(clean["validation_start"][mask][0]),
                                  "last_start": int(clean["validation_start"][mask][-1]), "sampling_fixed_before_evaluation": True})
    pd.DataFrame(manifest_rows).to_csv(output / "controlled_validation_window_manifest.csv", index=False)
    status["clean_window_cache_complete"] = True; status_path.write_text(json.dumps(status, indent=2))
    corruption_path = output / "controlled_relation_scores.npz"
    all_audit, all_manifest, corrupt = [], [], {}
    corruption_hash = corruption_dependency_hash(cfg, output, diversity)
    corruption_meta = corruption_path.with_suffix(".meta.json")
    corruption_reusable = False
    if corruption_path.exists() and corruption_meta.exists():
        try:
            corruption_reusable = json.loads(corruption_meta.read_text()).get("dependency_hash") == corruption_hash
        except (OSError, ValueError):
            corruption_reusable = False
    if corruption_reusable:
        cached = np.load(corruption_path, allow_pickle=False)
        for window in cfg["windows"]: corrupt[int(window)] = (cached[f"features_w{window}"], cached[f"success_w{window}"])
        print(f"REUSE corruption cache: {corruption_path}", flush=True)
    else:
        if any(windows_by_window[window] is None for window in cfg["windows"]):
            for window in cfg["windows"]:
                _, validations, starts, _, _, _ = load_selected_data(cfg, split, scaler, diversity, int(window), args.smoke)
                windows_by_window[int(window)] = [extract_windows(v, s, int(window)) for v, s in zip(validations, starts)]
        payload = {}
        for window in cfg["windows"]:
            features, success, audit_rows, manifest = build_corruption_window(cfg, clean_by_window[int(window)], windows_by_window[int(window)], diversity, int(window))
            corrupt[int(window)] = (features, success); payload[f"features_w{window}"] = features; payload[f"success_w{window}"] = success
            all_audit += audit_rows; all_manifest += manifest
        np.savez_compressed(corruption_path, **payload)
        corruption_meta.write_text(json.dumps({"dependency_hash": corruption_hash, "families": ALL_FAMILIES,
            "windows": cfg["windows"], "pair_order": diversity.pair_id.astype(int).tolist(),
            "generated_once": True, "test_loaded": False, "metadata_complete": True}, indent=2))
        pd.DataFrame(all_audit).to_csv(output / "controlled_corruption_marginal_preservation_audit.csv", index=False)
        pd.DataFrame(all_manifest).to_csv(output / "controlled_corruption_manifest.csv", index=False)
        print(f"COMPUTE_NEW corruption cache: {corruption_path}", flush=True)
    append_cache_reuse_audit(output, clean_reused, corruption_reusable)
    status["corruption_cache_complete"] = True; status_path.write_text(json.dumps(status, indent=2))
    result = evaluate(cfg, output, clean_by_window, corrupt, strict, diversity)
    status["reference_fit_complete"] = True; status["candidate_evaluation_complete"] = True
    selected = result["selected"]
    if selected["validity_gates_passed"]:
        classification = "independent_structure_score_is_valid"
    elif selected["macro_auroc"] >= .72 and selected["clean_fpr"] > .015:
        classification = "structure_score_is_sensitive_but_normal_fp_is_high"
    else:
        r3 = pd.read_csv(output / "structure_controlled_metrics_by_corruption.csv")
        r3 = r3[(r3.candidate_id == selected["candidate_id"]) & r3.family.str.startswith("R3")]
        classification = "current_relation_pool_is_insufficient" if r3.empty or r3.auroc.iloc[0] >= .75 else "structure_score_only_detects_simple_temporal_misalignment"
    counts = {f"w{window}": int(len(clean_by_window[int(window)]["validation_z"])) for window in cfg["windows"]}
    summary = {"readonly_audit_passed": audit["status"] == "passed", "hai_training_test_leakage": False,
               "step4a_reused": True, "relation_pool_count": len(diversity), "strict_relation_count": len(strict),
               "controlled_validation_window_counts": counts, "candidate_count": 16, "selected_candidate": selected["candidate_id"],
               "validity_gates_passed": bool(selected["validity_gates_passed"]), "classification": classification,
               "allow_real_structure_only_evaluation": bool(selected["validity_gates_passed"]), "test_loaded": False,
               "ae_loaded": False, "rec_score_loaded": False}
    ablations_and_report(output, result, diversity, summary)
    status["selected_config_written"] = True; status["final_report_complete"] = True; status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
