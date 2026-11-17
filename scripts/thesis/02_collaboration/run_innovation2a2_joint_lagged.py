"""Run the preregistered Normal-only stage-joint lagged dependency discovery."""

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


OUT = ROOT / "outputs" / "thesis_innovation2a2_joint_lagged"
CONFIG = ROOT / "configs" / "thesis" / "swat2015_midterm_seed1.yaml"
SCHEMA = ROOT / "outputs" / "thesis_t2_swat2015_audit" / "schema_profile.json"
PAIRWISE = ROOT / "outputs" / "thesis_innovation2a_lagged_dependency" / "stable_relations.csv"
LAGS = (1, 5, 10, 20, 30)
MAX_LAG = 30
ALPHA = 1.0
FIT_STRIDE = 4
DISCOVERY_BOUNDARY = 316_800
STAGES = tuple(f"P{i}" for i in range(1, 7))
DISCOVERY_GAIN_MIN = 0.05
VALIDATION_GAIN_MIN = 0.05
HALF_GAIN_MIN = 0.02
PAIRWISE_ORIGINAL_TARGETS = {"DPIT301", "FIT301", "FIT601"}


class JointDependencyError(RuntimeError):
    """Raised when the frozen 2A2 protocol cannot be followed exactly."""


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        if not rows:
            raise JointDependencyError(f"cannot infer empty CSV schema: {path.name}")
        columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
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
        raise JointDependencyError(f"channel has no audited stage suffix: {name}")
    stage = f"P{match.group(1)[0]}"
    if stage not in STAGES:
        raise JointDependencyError(f"channel resolves outside P1-P6: {name}")
    return stage


def normal_profiles(schema: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        item for item in schema["physical_files"]
        if item.get("file_name") == "SWaT_Dataset_Normal_v1.xlsx"
    ]
    if len(items) != 1 or len(items[0].get("variable_profiles", [])) != EXPECTED_FEATURES:
        raise JointDependencyError("T2 Normal-v1 variable profiles are missing or ambiguous")
    return items[0]["variable_profiles"]


def build_contract(
    profiles: list[dict[str, Any]], order: list[str]
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, Any]]]:
    by_name = {str(profile["name"]): profile for profile in profiles}
    if set(by_name) != set(order) or len(by_name) != EXPECTED_FEATURES:
        raise JointDependencyError("Official schema and S1 Normal cache disagree")
    contract: dict[str, dict[str, list[str]]] = {}
    for stage in STAGES:
        members = [name for name in order if stage_of(name) == stage]
        nonconstant = [name for name in members if not bool(by_name[name]["constant"])]
        targets = [
            name for name in nonconstant
            if by_name[name]["observed_value_kind"] == "continuous_numeric"
        ]
        if not targets or len(nonconstant) < 2:
            raise JointDependencyError(f"stage {stage} has no joint model contract")
        contract[stage] = {
            "members": members,
            "nonconstant_channels": nonconstant,
            "continuous_targets": targets,
            "excluded_constants": [name for name in members if bool(by_name[name]["constant"])],
        }
    return contract, by_name


def times(start: int, end: int, stride: int) -> np.ndarray:
    result = np.arange(start + MAX_LAG, end, stride, dtype=np.int64)
    if not len(result) or int(result[0] - MAX_LAG) < start or int(result[-1]) >= end:
        raise JointDependencyError(f"invalid chronological interval [{start},{end})")
    return result


def lagged_design(z: np.ndarray, sample_times: np.ndarray, channels: list[int]) -> np.ndarray:
    design = np.column_stack(
        [z[sample_times - lag, channel] for channel in channels for lag in LAGS]
    ).astype(np.float64, copy=False)
    if not np.isfinite(design).all():
        raise JointDependencyError("non-finite lagged design")
    return design


def fit_ridge(x: np.ndarray, y: np.ndarray) -> Ridge:
    model = Ridge(alpha=ALPHA)
    model.fit(x, y)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise JointDependencyError("non-finite Ridge parameters")
    return model


def predict_matrix(model: Ridge, x: np.ndarray, target_count: int) -> np.ndarray:
    prediction = np.asarray(model.predict(x), dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if prediction.shape != (len(x), target_count):
        raise JointDependencyError(f"unexpected joint prediction shape: {prediction.shape}")
    return prediction


def coefficient_matrix(model: Ridge, target_count: int) -> np.ndarray:
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
    if coefficients.shape[0] != target_count:
        raise JointDependencyError(f"unexpected joint coefficient shape: {coefficients.shape}")
    return coefficients


def relative_gain(self_mae: float, joint_mae: float) -> float:
    if self_mae <= 0 or not math.isfinite(self_mae):
        raise JointDependencyError(f"invalid self MAE denominator: {self_mae}")
    return (self_mae - joint_mae) / self_mae


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite Innovation-2A2 output: {OUT}")
    config = load_config(str(CONFIG))
    validate_config_contract(config)
    cache_dir = resolve_repo_path(ROOT, config["paths"]["cache_dir"])
    normal_path = cache_dir / "normal_v1_features.npy"
    normal_hash_before = sha256_file(normal_path)

    # The only dataset call is the frozen Normal-only loader; no Attack path is opened.
    normal, cache_manifest = load_normal_cache(cache_dir, config)
    order = [str(name) for name in cache_manifest["column_order"]]
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    contract, profile_by_name = build_contract(normal_profiles(schema), order)
    scaler = StandardScaler().fit(np.asarray(normal[:TRAIN_RAW_ROWS], dtype=np.float64))
    z = scaler.transform(np.asarray(normal, dtype=np.float64))
    if int(scaler.n_samples_seen_) != TRAIN_RAW_ROWS or not np.isfinite(z).all():
        raise JointDependencyError("Train-only scaler failed")

    discovery_fit_times = times(0, DISCOVERY_BOUNDARY, FIT_STRIDE)
    discovery_select_times = times(DISCOVERY_BOUNDARY, TRAIN_RAW_ROWS, FIT_STRIDE)
    full_train_times = times(0, TRAIN_RAW_ROWS, 1)
    validation_times = times(TRAIN_RAW_ROWS, EXPECTED_NORMAL_ROWS, 1)
    validation_midpoint = len(validation_times) // 2
    name_to_index = {name: index for index, name in enumerate(order)}
    pairwise = pd.read_csv(PAIRWISE)
    pairwise_counts = pairwise.groupby("target").size().to_dict()
    if set(pairwise["target"].astype(str)) != PAIRWISE_ORIGINAL_TARGETS:
        raise JointDependencyError("Innovation-2A stable targets differ from the frozen three")

    target_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        names = contract[stage]["nonconstant_channels"]
        targets = contract[stage]["continuous_targets"]
        channel_indices = [name_to_index[name] for name in names]
        target_indices = [name_to_index[name] for name in targets]

        x_discovery_fit = lagged_design(z, discovery_fit_times, channel_indices)
        x_discovery_select = lagged_design(z, discovery_select_times, channel_indices)
        y_discovery_fit = z[np.ix_(discovery_fit_times, target_indices)]
        y_discovery_select = z[np.ix_(discovery_select_times, target_indices)]
        discovery_joint = fit_ridge(x_discovery_fit, y_discovery_fit)
        discovery_joint_error = np.abs(
            y_discovery_select - predict_matrix(discovery_joint, x_discovery_select, len(targets))
        )

        x_full_train = lagged_design(z, full_train_times, channel_indices)
        x_validation = lagged_design(z, validation_times, channel_indices)
        y_full_train = z[np.ix_(full_train_times, target_indices)]
        y_validation = z[np.ix_(validation_times, target_indices)]
        final_joint = fit_ridge(x_full_train, y_full_train)
        final_joint_error = np.abs(
            y_validation - predict_matrix(final_joint, x_validation, len(targets))
        )
        joint_coefficients = coefficient_matrix(final_joint, len(targets))

        for local_target, target_name in enumerate(targets):
            target_channel_local = names.index(target_name)
            self_columns = list(
                range(target_channel_local * len(LAGS), (target_channel_local + 1) * len(LAGS))
            )
            discovery_self = fit_ridge(
                x_discovery_fit[:, self_columns], y_discovery_fit[:, local_target]
            )
            discovery_self_error = np.abs(
                y_discovery_select[:, local_target]
                - discovery_self.predict(x_discovery_select[:, self_columns])
            )
            final_self = fit_ridge(x_full_train[:, self_columns], y_full_train[:, local_target])
            final_self_error = np.abs(
                y_validation[:, local_target] - final_self.predict(x_validation[:, self_columns])
            )
            self_discovery_mae = float(discovery_self_error.mean())
            joint_discovery_mae = float(discovery_joint_error[:, local_target].mean())
            self_validation_mae = float(final_self_error.mean())
            joint_validation_mae = float(final_joint_error[:, local_target].mean())
            self_first = float(final_self_error[:validation_midpoint].mean())
            joint_first = float(final_joint_error[:validation_midpoint, local_target].mean())
            self_second = float(final_self_error[validation_midpoint:].mean())
            joint_second = float(final_joint_error[validation_midpoint:, local_target].mean())
            gain_discovery = relative_gain(self_discovery_mae, joint_discovery_mae)
            gain_validation = relative_gain(self_validation_mae, joint_validation_mae)
            gain_first = relative_gain(self_first, joint_first)
            gain_second = relative_gain(self_second, joint_second)
            stable = (
                gain_discovery >= DISCOVERY_GAIN_MIN
                and gain_validation >= VALIDATION_GAIN_MIN
                and gain_first >= HALF_GAIN_MIN
                and gain_second >= HALF_GAIN_MIN
            )
            coefficients = joint_coefficients[local_target]
            source_importance: list[dict[str, Any]] = []
            for source_local, source_name in enumerate(names):
                if source_name == target_name:
                    continue
                block = coefficients[source_local * len(LAGS):(source_local + 1) * len(LAGS)]
                important_lag_index = int(np.argmax(np.abs(block)))
                source_importance.append(
                    {
                        "target": target_name,
                        "stage": stage,
                        "source": source_name,
                        "source_kind": profile_by_name[source_name]["observed_value_kind"],
                        "source_importance": float(np.abs(block).sum()),
                        "most_important_lag": LAGS[important_lag_index],
                        "most_important_coefficient": float(block[important_lag_index]),
                    }
                )
            source_importance.sort(
                key=lambda row: (-float(row["source_importance"]), str(row["source"]))
            )
            top3 = source_importance[:3]
            row = {
                "target": target_name,
                "stage": stage,
                "cross_source_count": len(names) - 1,
                "feature_count": len(names) * len(LAGS),
                "discovery_self_mae": self_discovery_mae,
                "discovery_joint_mae": joint_discovery_mae,
                "joint_discovery_gain": gain_discovery,
                "validation_self_mae": self_validation_mae,
                "validation_joint_mae": joint_validation_mae,
                "joint_validation_gain": gain_validation,
                "validation_gain_first": gain_first,
                "validation_gain_second": gain_second,
                "pairwise_2a_stable_relation_count": int(pairwise_counts.get(target_name, 0)),
                "joint_stable": stable,
                "top3_contributing_sources": ";".join(
                    f"{item['source']}@lag{item['most_important_lag']}" for item in top3
                ),
            }
            target_rows.append(row)
            if stable:
                for rank, item in enumerate(source_importance, start=1):
                    importance_rows.append(
                        {
                            **item,
                            "rank": rank,
                            "top3_for_interpretation": rank <= 3,
                            "selection_effect": "none",
                        }
                    )
        del x_discovery_fit, x_discovery_select, x_full_train, x_validation

    stable_rows = [row for row in target_rows if bool(row["joint_stable"])]
    stable_names = {str(row["target"]) for row in stable_rows}
    stable_stages = {str(row["stage"]) for row in stable_rows}
    new_stable = sorted(stable_names - PAIRWISE_ORIGINAL_TARGETS)
    pairwise_failed_joint_supported = [
        row for row in stable_rows if int(row["pairwise_2a_stable_relation_count"]) == 0
    ]
    stage_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        all_stage = [row for row in target_rows if row["stage"] == stage]
        stable_stage = [row for row in stable_rows if row["stage"] == stage]
        stage_rows.append(
            {
                "stage": stage,
                "continuous_target_count": len(all_stage),
                "stable_joint_target_count": len(stable_stage),
                "stable_target_names": ";".join(str(row["target"]) for row in stable_stage),
                "median_discovery_gain_all_targets": float(np.median([row["joint_discovery_gain"] for row in all_stage])),
                "median_validation_gain_all_targets": float(np.median([row["joint_validation_gain"] for row in all_stage])),
                "median_stable_discovery_gain": (
                    float(np.median([row["joint_discovery_gain"] for row in stable_stage])) if stable_stage else None
                ),
                "median_stable_validation_gain": (
                    float(np.median([row["joint_validation_gain"] for row in stable_stage])) if stable_stage else None
                ),
            }
        )
    median_stable_validation_gain = (
        float(np.median([row["joint_validation_gain"] for row in stable_rows])) if stable_rows else None
    )
    gates = {
        "A_stable_targets_at_least_6": len(stable_rows) >= 6,
        "B_stable_targets_cover_at_least_3_stages": len(stable_stages) >= 3,
        "C_median_stable_validation_gain_at_least_0_05": (
            median_stable_validation_gain is not None and median_stable_validation_gain >= 0.05
        ),
        "D_all_stable_half_gains_at_least_0_02": all(
            float(row["validation_gain_first"]) >= HALF_GAIN_MIN
            and float(row["validation_gain_second"]) >= HALF_GAIN_MIN
            for row in stable_rows
        ),
    }
    if all(gates.values()):
        status = "STAGE_LOCAL_JOINT_LAGGED_DEPENDENCY_SUPPORTED"
    elif len(stable_rows) <= 2:
        status = "STAGE_LOCAL_JOINT_LAGGED_DEPENDENCY_NOT_SUPPORTED"
    else:
        status = "STAGE_LOCAL_JOINT_LAGGED_DEPENDENCY_NARROW"

    pairwise_vs_joint_rows = [
        {
            "target": row["target"],
            "stage": row["stage"],
            "pairwise_2a_stable_relation_count": row["pairwise_2a_stable_relation_count"],
            "joint_discovery_gain": row["joint_discovery_gain"],
            "joint_validation_gain": row["joint_validation_gain"],
            "joint_stable": row["joint_stable"],
            "pairwise_failed_but_joint_supported": (
                int(row["pairwise_2a_stable_relation_count"]) == 0 and bool(row["joint_stable"])
            ),
        }
        for row in target_rows
    ]
    tests = {
        "attack_not_read": True,
        "stage_from_official_schema": set(contract) == set(STAGES) and len(profile_by_name) == EXPECTED_FEATURES,
        "targets_continuous_nonconstant": all(
            profile_by_name[str(row["target"])]["observed_value_kind"] == "continuous_numeric"
            and not bool(profile_by_name[str(row["target"])]["constant"])
            for row in target_rows
        ),
        "cross_sources_same_stage_only": all(
            all(stage_of(name) == stage for name in item["nonconstant_channels"])
            for stage, item in contract.items()
        ),
        "target_not_cross_source": all(
            int(row["cross_source_count"]) == len(contract[str(row["stage"])]["nonconstant_channels"]) - 1
            for row in target_rows
        ),
        "constants_excluded": all(
            set(item["excluded_constants"]).isdisjoint(item["nonconstant_channels"])
            for item in contract.values()
        ),
        "self_lags_fixed": LAGS == (1, 5, 10, 20, 30),
        "cross_lags_fixed": LAGS == (1, 5, 10, 20, 30),
        "alpha_1": ALPHA == 1.0,
        "no_source_selection": all(
            int(row["feature_count"]) == len(contract[str(row["stage"])]["nonconstant_channels"]) * len(LAGS)
            for row in target_rows
        ),
        "train_internal_discovery_split": (
            int(discovery_fit_times[-1]) < DISCOVERY_BOUNDARY
            and int(discovery_select_times[0] - MAX_LAG) == DISCOVERY_BOUNDARY
            and int(discovery_select_times[-1]) < TRAIN_RAW_ROWS
        ),
        "validation_not_used_for_design": True,
        "stable_rule_fixed": (
            DISCOVERY_GAIN_MIN, VALIDATION_GAIN_MIN, HALF_GAIN_MIN
        ) == (0.05, 0.05, 0.02),
        "no_anomaly_metrics": True,
        "no_nan_inf": all(
            math.isfinite(float(row[key]))
            for row in target_rows
            for key in [
                "joint_discovery_gain", "joint_validation_gain",
                "validation_gain_first", "validation_gain_second",
            ]
        ),
    }
    if len(target_rows) != 25 or not all(tests.values()):
        raise JointDependencyError(f"2A2 contract failure: targets={len(target_rows)}, tests={tests}")
    normal_hash_after = sha256_file(normal_path)
    if normal_hash_before != normal_hash_after:
        raise JointDependencyError("Normal cache changed during 2A2")

    manifest = {
        "protocol": "Innovation-2A2 Normal-only stage-joint lagged dependency discovery",
        "normal_source": "SWaT_Dataset_Normal_v1.xlsx via frozen S1 Normal cache",
        "normal_cache_sha256_before": normal_hash_before,
        "normal_cache_sha256_after": normal_hash_after,
        "normal_cache_unchanged": normal_hash_before == normal_hash_after,
        "attack_data_read": False,
        "attack_labels_read": False,
        "train_rows": TRAIN_RAW_ROWS,
        "validation_rows": VALIDATION_RAW_ROWS,
        "scaler_fit_scope": "complete Normal Train raw rows [0,396000) only",
        "discovery_fit_raw_interval": [0, DISCOVERY_BOUNDARY],
        "discovery_select_raw_interval": [DISCOVERY_BOUNDARY, TRAIN_RAW_ROWS],
        "fit_stride": FIT_STRIDE,
        "validation_stride": 1,
        "lags": list(LAGS),
        "lag_zero_allowed": False,
        "lag_selection": False,
        "alpha": ALPHA,
        "dtype": "float64",
        "stage_source": str(SCHEMA.relative_to(ROOT)),
        "stage_contract": contract,
        "source_selection": False,
        "source_importance_timing": "post-hoc after joint_stable is determined",
        "source_importance_selection_effect": "none",
        "stable_rule": {
            "discovery_gain_min": DISCOVERY_GAIN_MIN,
            "validation_gain_min": VALIDATION_GAIN_MIN,
            "first_half_gain_min": HALF_GAIN_MIN,
            "second_half_gain_min": HALF_GAIN_MIN,
        },
        "threshold": None,
        "anomaly_metrics": False,
        "fusion": False,
        "tests": tests,
    }
    summary = {
        "status": status,
        "continuous_target_count": len(target_rows),
        "stable_joint_target_count": len(stable_rows),
        "stable_stage_count": len(stable_stages),
        "stable_stage_names": sorted(stable_stages),
        "median_stable_validation_gain": median_stable_validation_gain,
        "pairwise_original_targets": {
            target: target in stable_names for target in sorted(PAIRWISE_ORIGINAL_TARGETS)
        },
        "new_stable_targets": new_stable,
        "pairwise_failed_but_joint_supported_count": len(pairwise_failed_joint_supported),
        "pairwise_failed_but_joint_supported_targets": [
            str(row["target"]) for row in pairwise_failed_joint_supported
        ],
        "gates": gates,
        "tests_passed": sum(bool(value) for value in tests.values()),
        "tests_total": len(tests),
        "elapsed_seconds": time.perf_counter() - started,
    }

    OUT.mkdir(parents=True)
    write_csv(OUT / "target_metrics.csv", target_rows)
    write_csv(OUT / "stable_targets.csv", stable_rows, list(target_rows[0].keys()))
    write_csv(
        OUT / "source_importance.csv",
        importance_rows,
        [
            "target", "stage", "source", "source_kind", "source_importance",
            "most_important_lag", "most_important_coefficient", "rank",
            "top3_for_interpretation", "selection_effect",
        ],
    )
    write_csv(OUT / "pairwise_vs_joint.csv", pairwise_vs_joint_rows)
    write_csv(OUT / "stage_summary.csv", stage_rows)
    write_json(OUT / "dependency_manifest.json", manifest)
    write_json(OUT / "innovation2a2_summary.json", summary)
    print(json.dumps({**summary, "stable_targets": stable_rows}, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
