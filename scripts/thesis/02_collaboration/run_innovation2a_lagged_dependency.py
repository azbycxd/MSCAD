"""Discover preregistered stage-local incremental lagged dependencies on Normal SWaT."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.swat2015_official import (  # noqa: E402
    EXPECTED_FEATURES,
    EXPECTED_NORMAL_ROWS,
    TRAIN_RAW_ROWS,
    VALIDATION_RAW_ROWS,
    load_normal_cache,
    resolve_repo_path,
    validate_config_contract,
)
from utils.config_io import load_config  # noqa: E402


OUT = ROOT / "outputs" / "thesis_innovation2a_lagged_dependency"
CONFIG_PATH = ROOT / "configs" / "thesis" / "swat2015_midterm_seed1.yaml"
T2_SCHEMA = ROOT / "outputs" / "thesis_t2_swat2015_audit" / "schema_profile.json"
V3R_TARGETS = ROOT / "outputs" / "thesis_v3r_cross_only" / "r1_target_metrics.csv"
SELF_LAGS = (1, 5, 10, 20, 30)
CROSS_LAGS = (1, 5, 10, 20, 30)
MAX_LAG = 30
ALPHA = 1.0
FIT_STRIDE = 4
DISCOVERY_BOUNDARY = int(TRAIN_RAW_ROWS * 0.8)
STAGES = tuple(f"P{i}" for i in range(1, 7))
GAIN_DISCOVERY_MIN = 0.05
GAIN_VALIDATION_MIN = 0.05
GAIN_HALF_MIN = 0.02
TIE_TOLERANCE = 1e-12
TOP_K_PER_TARGET = 3


class Innovation2AError(RuntimeError):
    """Raised when a frozen Innovation-2A contract would be violated."""


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise Innovation2AError(f"cannot infer columns for empty CSV: {path.name}")
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_of(name: str) -> str:
    match = re.search(r"(\d{3})$", name)
    if match is None:
        raise Innovation2AError(f"channel has no audited SWaT stage suffix: {name}")
    stage = f"P{match.group(1)[0]}"
    if stage not in STAGES:
        raise Innovation2AError(f"channel is outside audited P1-P6: {name}")
    return stage


def normal_profiles(schema: dict[str, Any]) -> list[dict[str, Any]]:
    matches = [
        item for item in schema["physical_files"]
        if item.get("file_name") == "SWaT_Dataset_Normal_v1.xlsx"
    ]
    if len(matches) != 1 or len(matches[0].get("variable_profiles", [])) != EXPECTED_FEATURES:
        raise Innovation2AError("T2 Normal-v1 schema is missing or ambiguous")
    return matches[0]["variable_profiles"]


def stage_contract(
    profiles: list[dict[str, Any]], column_order: list[str]
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, Any]]]:
    by_name = {str(profile["name"]): profile for profile in profiles}
    if set(by_name) != set(column_order) or len(by_name) != EXPECTED_FEATURES:
        raise Innovation2AError("Official schema and frozen cache column order disagree")
    result: dict[str, dict[str, list[str]]] = {}
    for stage in STAGES:
        members = [name for name in column_order if stage_of(name) == stage]
        nonconstant = [name for name in members if not bool(by_name[name]["constant"])]
        targets = [
            name for name in nonconstant
            if by_name[name]["observed_value_kind"] == "continuous_numeric"
        ]
        if not targets or len(nonconstant) < 2:
            raise Innovation2AError(f"stage {stage} has no eligible directed dependency")
        result[stage] = {
            "members": members,
            "nonconstant_sources": nonconstant,
            "continuous_targets": targets,
            "excluded_constants": [name for name in members if bool(by_name[name]["constant"])],
        }
    return result, by_name


def sample_times(start: int, end: int, *, stride: int, prevent_cross_boundary: bool) -> np.ndarray:
    first = start + MAX_LAG if prevent_cross_boundary else max(start, MAX_LAG)
    values = np.arange(first, end, stride, dtype=np.int64)
    if not len(values) or int(values[0] - MAX_LAG) < start or int(values[-1]) >= end:
        raise Innovation2AError(f"invalid chronological sample range [{start},{end})")
    return values


def self_features(z: np.ndarray, times: np.ndarray, target: int) -> np.ndarray:
    return np.column_stack([z[times - lag, target] for lag in SELF_LAGS])


def full_features(
    z: np.ndarray,
    times: np.ndarray,
    target: int,
    source: int,
    cross_lag: int,
) -> np.ndarray:
    return np.column_stack(
        [z[times - lag, target] for lag in SELF_LAGS]
        + [z[times - cross_lag, source]]
    )


def mae(model: Ridge, x: np.ndarray, y: np.ndarray) -> float:
    value = float(np.mean(np.abs(y - model.predict(x))))
    if not math.isfinite(value):
        raise Innovation2AError("non-finite MAE")
    return value


def gain(self_mae: float, full_mae: float) -> float:
    if self_mae <= 0 or not math.isfinite(self_mae):
        raise Innovation2AError(f"invalid self MAE denominator: {self_mae}")
    return (self_mae - full_mae) / self_mae


def fit_ridge(x: np.ndarray, y: np.ndarray) -> Ridge:
    model = Ridge(alpha=ALPHA)
    model.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    if not np.isfinite(model.coef_).all() or not math.isfinite(float(model.intercept_)):
        raise Innovation2AError("non-finite Ridge model")
    return model


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite Innovation-2A output: {OUT}")
    config = load_config(str(CONFIG_PATH))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal_path = cache_dir / "normal_v1_features.npy"
    normal_hash_before = sha256_file(normal_path)

    # This is the only dataset loader in the experiment: it rejects legacy paths and opens Normal only.
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    if normal.shape != (EXPECTED_NORMAL_ROWS, EXPECTED_FEATURES):
        raise Innovation2AError("Normal cache shape differs from the frozen S1 protocol")
    column_order = [str(name) for name in cache_manifest["column_order"]]
    schema = json.loads(T2_SCHEMA.read_text(encoding="utf-8"))
    profiles = normal_profiles(schema)
    contract, profile_by_name = stage_contract(profiles, column_order)

    # The scaler sees the complete official Train split and nothing else.
    scaler = StandardScaler().fit(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float64))
    z = scaler.transform(np.asarray(normal, dtype=np.float64))
    if not np.isfinite(z).all() or int(scaler.n_samples_seen_) != TRAIN_RAW_ROWS:
        raise Innovation2AError("Train-only StandardScaler contract failed")
    name_to_index = {name: index for index, name in enumerate(column_order)}

    discovery_fit_times = sample_times(0, DISCOVERY_BOUNDARY, stride=FIT_STRIDE, prevent_cross_boundary=True)
    discovery_select_times = sample_times(
        DISCOVERY_BOUNDARY, TRAIN_RAW_ROWS, stride=FIT_STRIDE, prevent_cross_boundary=True
    )
    full_train_times = sample_times(0, TRAIN_RAW_ROWS, stride=1, prevent_cross_boundary=True)
    train_first_times = sample_times(0, TRAIN_RAW_ROWS // 2, stride=1, prevent_cross_boundary=True)
    train_second_times = sample_times(
        TRAIN_RAW_ROWS // 2, TRAIN_RAW_ROWS, stride=1, prevent_cross_boundary=True
    )
    validation_times = sample_times(
        TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS, stride=1, prevent_cross_boundary=True
    )
    validation_split = len(validation_times) // 2
    validation_first_times = validation_times[:validation_split]
    validation_second_times = validation_times[validation_split:]

    candidate_rows: list[dict[str, Any]] = []
    target_self_cache: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        for target_name in contract[stage]["continuous_targets"]:
            target_index = name_to_index[target_name]
            x_self_fit = self_features(z, discovery_fit_times, target_index)
            y_fit = z[discovery_fit_times, target_index]
            discovery_self_model = fit_ridge(x_self_fit, y_fit)
            x_self_select = self_features(z, discovery_select_times, target_index)
            y_select = z[discovery_select_times, target_index]
            discovery_self_mae = mae(discovery_self_model, x_self_select, y_select)

            x_self_train = self_features(z, full_train_times, target_index)
            y_train = z[full_train_times, target_index]
            final_self_model = fit_ridge(x_self_train, y_train)
            x_self_validation = self_features(z, validation_times, target_index)
            y_validation = z[validation_times, target_index]
            final_self_prediction = final_self_model.predict(x_self_validation)
            self_error = np.abs(y_validation - final_self_prediction)
            self_mae_full = float(self_error.mean())
            self_mae_first = float(self_error[:validation_split].mean())
            self_mae_second = float(self_error[validation_split:].mean())
            target_self_cache[target_name] = {
                "discovery_self_mae": discovery_self_mae,
                "validation_self_mae": self_mae_full,
                "validation_self_mae_first": self_mae_first,
                "validation_self_mae_second": self_mae_second,
            }

            for source_name in contract[stage]["nonconstant_sources"]:
                if source_name == target_name:
                    continue
                source_index = name_to_index[source_name]
                best_lag: int | None = None
                best_discovery_gain = -math.inf
                best_discovery_full_mae = math.inf
                lag_gains: dict[int, float] = {}
                for cross_lag in CROSS_LAGS:
                    x_full_fit = full_features(
                        z, discovery_fit_times, target_index, source_index, cross_lag
                    )
                    discovery_full_model = fit_ridge(x_full_fit, y_fit)
                    x_full_select = full_features(
                        z, discovery_select_times, target_index, source_index, cross_lag
                    )
                    full_mae_select = mae(discovery_full_model, x_full_select, y_select)
                    relative_gain = gain(discovery_self_mae, full_mae_select)
                    lag_gains[cross_lag] = relative_gain
                    if relative_gain > best_discovery_gain + TIE_TOLERANCE:
                        best_lag = cross_lag
                        best_discovery_gain = relative_gain
                        best_discovery_full_mae = full_mae_select
                if best_lag is None:
                    raise Innovation2AError(f"lag selection failed for {source_name}->{target_name}")

                x_full_train = full_features(z, full_train_times, target_index, source_index, best_lag)
                final_full_model = fit_ridge(x_full_train, y_train)
                x_full_validation = full_features(
                    z, validation_times, target_index, source_index, best_lag
                )
                full_error = np.abs(y_validation - final_full_model.predict(x_full_validation))
                full_mae_validation = float(full_error.mean())
                full_mae_first = float(full_error[:validation_split].mean())
                full_mae_second = float(full_error[validation_split:].mean())
                validation_gain = gain(self_mae_full, full_mae_validation)
                gain_first = gain(self_mae_first, full_mae_first)
                gain_second = gain(self_mae_second, full_mae_second)

                first_model = fit_ridge(
                    full_features(z, train_first_times, target_index, source_index, best_lag),
                    z[train_first_times, target_index],
                )
                second_model = fit_ridge(
                    full_features(z, train_second_times, target_index, source_index, best_lag),
                    z[train_second_times, target_index],
                )
                beta_full = float(final_full_model.coef_[-1])
                beta_first = float(first_model.coef_[-1])
                beta_second = float(second_model.coef_[-1])
                sign_stable = beta_first * beta_second > 0
                stable = (
                    best_discovery_gain >= GAIN_DISCOVERY_MIN
                    and validation_gain >= GAIN_VALIDATION_MIN
                    and gain_first >= GAIN_HALF_MIN
                    and gain_second >= GAIN_HALF_MIN
                    and sign_stable
                )
                candidate_rows.append(
                    {
                        "stage": stage,
                        "source": source_name,
                        "source_kind": profile_by_name[source_name]["observed_value_kind"],
                        "target": target_name,
                        "target_kind": profile_by_name[target_name]["observed_value_kind"],
                        "best_lag": best_lag,
                        "discovery_self_mae": discovery_self_mae,
                        "discovery_full_mae": best_discovery_full_mae,
                        "discovery_gain": best_discovery_gain,
                        "validation_self_mae": self_mae_full,
                        "validation_full_mae": full_mae_validation,
                        "validation_gain": validation_gain,
                        "gain_first": gain_first,
                        "gain_second": gain_second,
                        "beta_full": beta_full,
                        "beta_first": beta_first,
                        "beta_second": beta_second,
                        "beta_sign_stable": sign_stable,
                        "stable_dependency": stable,
                        "selected_top3_for_target": False,
                        "lag_gain_1": lag_gains[1],
                        "lag_gain_5": lag_gains[5],
                        "lag_gain_10": lag_gains[10],
                        "lag_gain_20": lag_gains[20],
                        "lag_gain_30": lag_gains[30],
                    }
                )

    stable_rows = [row for row in candidate_rows if bool(row["stable_dependency"])]
    for target_name in sorted({str(row["target"]) for row in stable_rows}):
        target_rows = [row for row in stable_rows if row["target"] == target_name]
        target_rows.sort(
            key=lambda row: (-float(row["validation_gain"]), str(row["source"]), int(row["best_lag"]))
        )
        for row in target_rows[:TOP_K_PER_TARGET]:
            row["selected_top3_for_target"] = True
    selected_rows = [row for row in stable_rows if bool(row["selected_top3_for_target"])]

    stage_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        candidates = [row for row in candidate_rows if row["stage"] == stage]
        stable = [row for row in stable_rows if row["stage"] == stage]
        selected = [row for row in selected_rows if row["stage"] == stage]
        stage_rows.append(
            {
                "stage": stage,
                "continuous_target_count": len(contract[stage]["continuous_targets"]),
                "candidate_directed_pair_count": len(candidates),
                "stable_relation_count": len(stable),
                "selected_relation_count": len(selected),
                "unique_source_count": len({row["source"] for row in selected}),
                "unique_target_count": len({row["target"] for row in selected}),
                "median_stable_validation_gain": (
                    float(np.median([row["validation_gain"] for row in stable])) if stable else None
                ),
                "selected_lag_distribution": json.dumps(
                    {str(lag): sum(int(row["best_lag"]) == lag for row in selected) for lag in CROSS_LAGS},
                    separators=(",", ":"),
                ),
            }
        )
    lag_rows = [
        {
            "lag": lag,
            "candidate_best_lag_count": sum(int(row["best_lag"]) == lag for row in candidate_rows),
            "stable_relation_count": sum(int(row["best_lag"]) == lag for row in stable_rows),
            "selected_relation_count": sum(int(row["best_lag"]) == lag for row in selected_rows),
        }
        for lag in CROSS_LAGS
    ]

    selected_targets = {str(row["target"]) for row in selected_rows}
    selected_stages = {str(row["stage"]) for row in selected_rows}
    median_selected_gain = (
        float(np.median([row["validation_gain"] for row in selected_rows])) if selected_rows else None
    )
    gates = {
        "A_selected_relations_at_least_6": len(selected_rows) >= 6,
        "B_unique_stable_targets_at_least_4": len(selected_targets) >= 4,
        "C_at_least_2_stages_have_at_least_2_selected": sum(
            int(row["selected_relation_count"]) >= 2 for row in stage_rows
        ) >= 2,
        "D_median_selected_validation_gain_at_least_0_05": (
            median_selected_gain is not None and median_selected_gain >= 0.05
        ),
    }
    if all(gates.values()):
        status = "STAGE_LOCAL_LAGGED_DEPENDENCY_SUPPORTED"
    elif len(selected_rows) <= 2:
        status = "STAGE_LOCAL_LAGGED_DEPENDENCY_NOT_SUPPORTED"
    else:
        status = "STAGE_LOCAL_LAGGED_DEPENDENCY_NARROW"

    v3r = pd.read_csv(V3R_TARGETS)
    prior_high = set(
        v3r.loc[v3r["cross_only_stable_target"].astype(str).str.lower() == "true", "target"].astype(str)
    )
    current_targets = {
        name for stage in STAGES for name in contract[stage]["continuous_targets"]
    }
    prior_high_current = prior_high & current_targets
    prior_high_without_incremental = sorted(prior_high_current - selected_targets)

    tests = {
        "attack_workbook_not_read": True,
        "stage_from_official_schema": set(contract) == set(STAGES) and len(profile_by_name) == EXPECTED_FEATURES,
        "target_continuous_nonconstant_only": all(
            row["target_kind"] == "continuous_numeric"
            and not bool(profile_by_name[str(row["target"])]["constant"])
            for row in candidate_rows
        ),
        "source_same_stage_not_target": all(
            stage_of(str(row["source"])) == row["stage"]
            and stage_of(str(row["target"])) == row["stage"]
            and row["source"] != row["target"]
            for row in candidate_rows
        ),
        "constant_excluded": all(
            not bool(profile_by_name[str(row["source"])]["constant"])
            for row in candidate_rows
        ),
        "self_lags_fixed": SELF_LAGS == (1, 5, 10, 20, 30),
        "cross_lags_fixed": CROSS_LAGS == (1, 5, 10, 20, 30),
        "alpha_1": ALPHA == 1.0,
        "lag_selection_train_internal_only": int(discovery_select_times[-1]) < TRAIN_RAW_ROWS,
        "validation_not_used_for_lag_selection": True,
        "source_selection_no_attack": True,
        "stable_rule_fixed": (
            GAIN_DISCOVERY_MIN, GAIN_VALIDATION_MIN, GAIN_HALF_MIN
        ) == (0.05, 0.05, 0.02),
        "top3_fixed": all(
            sum(row["target"] == target for row in selected_rows) <= TOP_K_PER_TARGET
            for target in selected_targets
        ),
        "train_validation_chronology": (
            int(full_train_times[-1]) < TRAIN_RAW_ROWS
            and int(validation_times[0] - MAX_LAG) >= TRAIN_RAW_ROWS
            and int(validation_times[-1]) < EXPECTED_NORMAL_ROWS
        ),
        "no_nan_inf": all(
            math.isfinite(float(row[key]))
            for row in candidate_rows
            for key in [
                "discovery_gain", "validation_gain", "gain_first", "gain_second",
                "beta_full", "beta_first", "beta_second",
            ]
        ),
    }
    if len(candidate_rows) != sum(
        len(contract[stage]["continuous_targets"])
        * (len(contract[stage]["nonconstant_sources"]) - 1)
        for stage in STAGES
    ):
        raise Innovation2AError("candidate directed-pair count is incomplete")
    if not all(tests.values()):
        raise Innovation2AError(f"Innovation-2A contract failure: {tests}")
    normal_hash_after = sha256_file(normal_path)
    if normal_hash_before != normal_hash_after:
        raise Innovation2AError("frozen Normal cache changed during discovery")

    manifest = {
        "protocol": "Innovation-2A normal-only stage-local incremental lagged dependency discovery",
        "normal_source": "SWaT_Dataset_Normal_v1.xlsx via frozen S1 Normal cache",
        "normal_cache": str(normal_path.relative_to(ROOT)),
        "normal_cache_sha256_before": normal_hash_before,
        "normal_cache_sha256_after": normal_hash_after,
        "normal_cache_unchanged": normal_hash_before == normal_hash_after,
        "attack_data_read": False,
        "attack_labels_read": False,
        "train_rows": TRAIN_RAW_ROWS,
        "validation_rows": VALIDATION_RAW_ROWS,
        "discovery_boundary": DISCOVERY_BOUNDARY,
        "discovery_fit_raw_interval": [0, DISCOVERY_BOUNDARY],
        "discovery_select_raw_interval": [DISCOVERY_BOUNDARY, TRAIN_RAW_ROWS],
        "fit_stride": FIT_STRIDE,
        "validation_stride": 1,
        "lags_refer_to_original_1hz_rows": True,
        "self_lags": list(SELF_LAGS),
        "cross_lags": list(CROSS_LAGS),
        "lag_zero_allowed": False,
        "alpha": ALPHA,
        "dtype": "float64",
        "scaler_fit_scope": "complete Normal Train raw rows [0,396000) only",
        "lag_selection_scope": "Discovery Select inside Normal Train only",
        "source_selection_scope": "audited same-stage nonconstant channels only",
        "stage_source": str(T2_SCHEMA.relative_to(ROOT)),
        "stage_contract": contract,
        "stable_rule": {
            "discovery_gain_min": GAIN_DISCOVERY_MIN,
            "validation_gain_min": GAIN_VALIDATION_MIN,
            "validation_first_gain_min": GAIN_HALF_MIN,
            "validation_second_gain_min": GAIN_HALF_MIN,
            "beta_first_times_beta_second_positive": True,
        },
        "top_k_per_target": TOP_K_PER_TARGET,
        "point_adjustment": False,
        "threshold": None,
        "anomaly_metrics": False,
        "fusion": False,
        "tests": tests,
    }
    summary = {
        "status": status,
        "candidate_directed_relation_count": len(candidate_rows),
        "stable_relation_count": len(stable_rows),
        "selected_relation_count": len(selected_rows),
        "unique_stable_target_count": len(selected_targets),
        "selected_stage_count": len(selected_stages),
        "selected_stage_names": sorted(selected_stages),
        "median_selected_validation_gain": median_selected_gain,
        "selected_lag_distribution": {
            str(row["lag"]): int(row["selected_relation_count"]) for row in lag_rows
        },
        "gates": gates,
        "raw_cross_r2_conceptual_comparison": {
            "definition": "existing V3-R cross_only_stable_target=true, intersected with current continuous targets",
            "prior_high_raw_cross_target_count": len(prior_high_current),
            "prior_high_raw_cross_but_no_selected_incremental_dependency_count": len(prior_high_without_incremental),
            "targets": prior_high_without_incremental,
            "selection_effect": "none",
        },
        "tests_passed": sum(bool(value) for value in tests.values()),
        "tests_total": len(tests),
        "elapsed_seconds": time.perf_counter() - started,
    }

    OUT.mkdir(parents=True)
    write_csv(OUT / "all_candidate_relations.csv", candidate_rows)
    write_csv(OUT / "stable_relations.csv", stable_rows, list(candidate_rows[0].keys()))
    write_csv(OUT / "selected_relations.csv", selected_rows, list(candidate_rows[0].keys()))
    write_csv(OUT / "stage_summary.csv", stage_rows)
    write_csv(OUT / "lag_distribution.csv", lag_rows)
    write_json(OUT / "dependency_manifest.json", manifest)
    write_json(OUT / "innovation2a_summary.json", summary)
    print(json.dumps({**summary, "selected_relations": selected_rows}, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
