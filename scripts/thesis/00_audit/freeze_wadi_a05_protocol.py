"""Freeze WADI.A2 missingness and Attack 13 binding protocol. No model code."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset" / "WADI_A2_2019"
A0 = ROOT / "outputs" / "thesis_wadi_a0_audit"
OUT = ROOT / "outputs" / "thesis_wadi_a05_protocol"
DOC = ROOT / "docs" / "thesis" / "wadi_a05_missingness_binding_freeze.md"
NORMAL = DATA / "WADI_14days_new.csv"
ATTACK = DATA / "WADI_attackdataLABLE.csv"
PDF = DATA / "table_WADI.pdf"
LABEL = "Attack LABLE (1:No Attack, -1:Attack)"
TRAIN_SLICES = ((0, 268799), (335999, 694856))
VAL_SLICES = ((268799, 335999), (694856, 784571))
HIGH_IDS = {2, 3, 4, 7, 15}


def load_a0_module():
    path = Path(__file__).with_name("audit_wadi_a0.py")
    spec = importlib.util.spec_from_file_location("wadi_a0_source", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def intervals(mask: np.ndarray, split: str, segment: str, offset: int = 0) -> list[dict]:
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [
        {
            "split": split,
            "segment_id": segment,
            "start_index": int(offset + s),
            "end_index_exclusive": int(offset + e + 1),
            "point_count": int(e - s + 1),
            "boundary_policy": "hard segment break; no window or lag may cross",
        }
        for s, e in zip(starts, ends)
    ]


def split_stats(frame: pd.DataFrame, features: list[str], pieces, split: str):
    masks = []
    records = []
    affected_columns: set[str] = set()
    for number, (start, end) in enumerate(pieces, 1):
        part = frame.iloc[start:end][features]
        column_mask = part.isna().any(axis=0)
        affected_columns.update(column_mask.index[column_mask].tolist())
        mask = part.isna().any(axis=1).to_numpy()
        masks.append(mask)
        records.extend(intervals(mask, split, f"{split.lower()}_{number}", start))
    all_mask = np.concatenate(masks)
    total = int(len(all_mask))
    invalid = int(all_mask.sum())
    return {
        "points": total,
        "invalid_points": invalid,
        "valid_points": total - invalid,
        "valid_point_ratio": float((total - invalid) / total),
        "affected_column_count": len(affected_columns),
        "affected_columns": sorted(affected_columns),
        "invalid_interval_count": len(records),
    }, records


def main() -> int:
    started = time.perf_counter()
    for path in (NORMAL, ATTACK, PDF):
        if not path.is_file():
            raise FileNotFoundError(path)

    normal = pd.read_csv(NORMAL, low_memory=False)
    normal.columns = normal.columns.str.strip()
    attack = pd.read_csv(ATTACK, header=1, low_memory=False)
    attack.columns = attack.columns.str.strip()
    attack = attack.loc[attack["Row"].notna()].reset_index(drop=True)
    schema0 = pd.read_csv(A0 / "variable_schema.csv")
    features = schema0["raw_column_name"].tolist()
    if features != [c for c in normal.columns if c not in {"Row", "Date", "Time"}]:
        raise RuntimeError("A0/Normal feature schema mismatch")
    if features != [c for c in attack.columns if c not in {"Row", "Date", "Time", LABEL}]:
        raise RuntimeError("A0/Attack feature schema mismatch")

    train = pd.concat([normal.iloc[s:e] for s, e in TRAIN_SLICES], axis=0)
    n_train = len(train)
    rows = []
    excluded = []
    for name in features:
        train_missing = int(train[name].isna().sum())
        normal_missing = int(normal[name].isna().sum())
        attack_missing = int(attack[name].isna().sum())
        if train_missing == n_train:
            category = "ALL_NAN_NORMAL_TRAIN"
            excluded.append(name)
        elif normal_missing:
            category = "PARTIAL_NAN_NORMAL"
        elif attack_missing:
            category = "TEST_ONLY_NAN"
        else:
            category = "NO_NAN"
        rows.append({
            "column": name,
            "normal_rows": len(normal),
            "normal_missing_count": normal_missing,
            "normal_missing_ratio": normal_missing / len(normal),
            "attack_rows": len(attack),
            "attack_missing_count": attack_missing,
            "attack_missing_ratio": attack_missing / len(attack),
            "normal_train_rows": n_train,
            "normal_train_missing_count": train_missing,
            "normal_train_missing_ratio": train_missing / n_train,
            "missingness_class": category,
        })
    missingness = pd.DataFrame(rows)
    kept = [c for c in features if c not in excluded]

    train_stats, invalid_train = split_stats(normal, kept, TRAIN_SLICES, "TRAIN")
    val_stats, invalid_val = split_stats(normal, kept, VAL_SLICES, "VALIDATION")
    test_stats, invalid_test = split_stats(attack, kept, ((0, len(attack)),), "TEST")
    invalid = pd.DataFrame(invalid_train + invalid_val + invalid_test)
    if invalid.empty:
        invalid = pd.DataFrame(columns=["split", "segment_id", "start_index", "end_index_exclusive", "point_count", "boundary_policy"])

    frozen = schema0.copy()
    frozen["included"] = ~frozen["raw_column_name"].isin(excluded)
    frozen["exclusion_reason"] = np.where(
        frozen["included"], "", "missing_ratio==1 across combined frozen Normal Train"
    )
    frozen["decision_data"] = "Normal Train only"

    a0mod = load_a0_module()
    attack_ts, ts_audit = a0mod.infer_audit_timestamps(attack)
    windows = {
        "canonical_00h": (pd.Timestamp("2017-10-11 00:05:00"), pd.Timestamp("2017-10-11 00:13:00")),
        "official_12h": (pd.Timestamp("2017-10-11 12:05:00"), pd.Timestamp("2017-10-11 12:13:00")),
    }
    audit_rows = []
    for name, (start, end) in windows.items():
        idx = np.flatnonzero((attack_ts >= start) & (attack_ts <= end))
        for i in idx:
            audit_rows.append({
                "audit_range": name,
                "dataframe_index": int(i),
                "raw_row_index": int(attack.iloc[i]["Row"]),
                "raw_Date": str(attack.iloc[i]["Date"]),
                "raw_Time": str(attack.iloc[i]["Time"]),
                "raw_Label": int(attack.iloc[i][LABEL]),
                "canonical_timestamp": attack_ts[i].isoformat(),
            })
    raw_audit = pd.DataFrame(audit_rows)
    raw_attack_ranges = {
        name: {
            "rows_checked": int((raw_audit.audit_range == name).sum()),
            "attack_label_points": int(((raw_audit.audit_range == name) & (raw_audit.raw_Label == -1)).sum()),
            "first_attack_timestamp": (
                raw_audit.loc[(raw_audit.audit_range == name) & (raw_audit.raw_Label == -1), "canonical_timestamp"].min()
                if ((raw_audit.audit_range == name) & (raw_audit.raw_Label == -1)).any() else None
            ),
            "last_attack_timestamp": (
                raw_audit.loc[(raw_audit.audit_range == name) & (raw_audit.raw_Label == -1), "canonical_timestamp"].max()
                if ((raw_audit.audit_range == name) & (raw_audit.raw_Label == -1)).any() else None
            ),
        }
        for name in windows
    }
    official_id13_mask = (
        (attack_ts >= pd.Timestamp("2017-10-11 12:07:30"))
        & (attack_ts <= pd.Timestamp("2017-10-11 12:10:52"))
    )
    official_id13_attack_points = int((attack.loc[official_id13_mask, LABEL] == -1).sum())

    binding = pd.read_csv(A0 / "attack_binding.csv")
    binding["final_binding_status"] = np.where(
        binding.attack_id.eq(13), "SOURCE_CONFLICT_UNRESOLVED", "BOUND"
    )
    binding["include_in_primary_event_evaluation"] = ~binding.attack_id.eq(13)
    binding["pointwise_label_source"] = ATTACK.name
    binding["event_metadata_source"] = PDF.name
    high_bound = bool(binding.loc[binding.attack_id.isin(HIGH_IDS), "final_binding_status"].eq("BOUND").all())
    bound_count = int(binding.final_binding_status.eq("BOUND").sum())

    all_ratios = [train_stats["valid_point_ratio"], val_stats["valid_point_ratio"], test_stats["valid_point_ratio"]]
    missing_status = "MISSINGNESS_PROTOCOL_READY" if min(all_ratios) >= 0.99 else "MISSINGNESS_REQUIRES_MANUAL_REVIEW"
    binding_status = "ATTACK13_SOURCE_CONFLICT"
    protocol = {
        "phase": "WADI-A0.5",
        "dataset_variant": "WADI.A2 only",
        "exact_sources": [NORMAL.name, ATTACK.name, PDF.name],
        "missingness_status": missing_status,
        "attack13_status": binding_status,
        "feature_count_raw": len(features),
        "excluded_feature_count": len(excluded),
        "excluded_features": excluded,
        "frozen_feature_count": len(kept),
        "exclusion_rule": "exclude iff missing_ratio==1 across combined frozen Normal Train",
        "attack_test_used_for_exclusion": False,
        "imputation": "NONE",
        "invalid_point_policy": "any remaining NaN makes the point invalid and creates a hard session boundary",
        "window_or_lag_crosses_invalid_point": False,
        "splits": {"train": train_stats, "validation": val_stats, "test": test_stats},
        "timestamp_parser": {
            "date_parsing": "explicit %m/%d/%Y or %m/%d/%y by raw Date block",
            "time_parsing": "raw Time retained; interpreted as minute:second-within-hour only",
            "am_pm_field_present": False,
            "twelve_hour_clock_used": False,
            "twelve_to_zero_conversion_used": False,
            "rollover": "hour/day inferred from monotonic Row and raw Date block sizes",
            "event_specific_12h_patch": False,
            "global_parser_bug_found": False,
            "source": "audit_wadi_a0.infer_audit_timestamps",
            "audit": ts_audit,
        },
        "attack13": {
            "official_pdf_start": "2017-10-11T12:07:30",
            "official_pdf_end": "2017-10-11T12:10:52",
            "official_targets": ["2_PIC_003_SP", "2_FIT_003", "2_PIT_003"],
            "official_description": "Reducing Booster set point pressure; reflected in 2FIT003 and 2PIT003",
            "pdf_visual_and_text_verified": True,
            "csv_raw_ranges": raw_attack_ranges,
            "csv_attack_points_during_exact_official_id13_interval": official_id13_attack_points,
            "official_12h_audit_range_attack_points_note": "12:05:00-12:05:03 belongs to the tail of bound ID12; exact ID13 interval has zero Attack CSV points",
            "decision": "raw CSV labels the 00h interval while the official PDF specifies 12h; preserve both sources without repair",
            "pointwise_ground_truth": "Attack CSV actual label",
            "event_binding": "SOURCE_CONFLICT_UNRESOLVED",
            "included_in_primary_event_evaluation": False,
        },
        "event_protocol": {
            "official_event_count": 15,
            "bound_primary_event_count": bound_count,
            "unresolved_ids": [13],
            "high_relevance_ids": sorted(HIGH_IDS),
            "all_high_relevance_bound": high_bound,
        },
        "attack_12_13_14_official_sequence": {
            "12": ["2017-10-11T11:59:00", "2017-10-11T12:05:00"],
            "13": ["2017-10-11T12:07:30", "2017-10-11T12:10:52"],
            "14": ["2017-10-11T12:16:00", "2017-10-11T12:25:36"],
            "chronology_consistent_in_pdf": True,
        },
        "no_model_training": True,
        "no_model_inference": True,
        "no_attack_driven_feature_exclusion": True,
        "no_event_specific_timestamp_patch": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not high_bound:
        raise RuntimeError("HIGH relevance event binding gate failed")

    OUT.mkdir(parents=True, exist_ok=True)
    missingness.to_csv(OUT / "missingness_by_column.csv", index=False)
    invalid.to_csv(OUT / "invalid_intervals.csv", index=False)
    raw_audit.to_csv(OUT / "attack13_raw_timestamp_audit.csv", index=False)
    binding.to_csv(OUT / "event_binding_final.csv", index=False)
    frozen.to_csv(OUT / "feature_schema_frozen.csv", index=False)
    (OUT / "protocol_freeze.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    doc = f"""# WADI-A0.5 Missingness + Attack13 Binding Protocol Freeze

## Scope

This protocol uses only WADI.A2: `{NORMAL.name}`, `{ATTACK.name}`, and `{PDF.name}`. It performs no model training or inference and does not modify source data.

## Frozen missingness policy

- Frozen Normal Train ranges: `[0,268799)` and `[335999,694856)` ({n_train:,} points).
- Exclude a feature iff its missing ratio is exactly 1 across the combined Normal Train. Attack/Test is never used for exclusion.
- Excluded ({len(excluded)}): {', '.join(f'`{x}`' for x in excluded)}.
- Frozen model schema: {len(kept)} process variables.
- No imputation. Any remaining NaN makes that point invalid and creates a hard segment boundary; windows/lags cannot cross it.
- Train valid ratio: {train_stats['valid_point_ratio']:.9f}; Validation: {val_stats['valid_point_ratio']:.9f}; Test: {test_stats['valid_point_ratio']:.9f}.
- Status: **{missing_status}**.

## Attack13 source adjudication

The official WADI table was visually and textually rechecked. ID13 is `2017-10-11 12:07:30` to `12:10:52`, reducing the booster set-point pressure with effects expected in `2FIT003` and `2PIT003`. The A0 parser is not a 12-hour parser: raw `Time` contains minute/second only, while hours and day rollover are reconstructed globally from monotonic `Row` and raw Date blocks. There is no AM/PM field, no `12 -> 00` conversion, and no event-specific patch.

The Attack CSV instead labels the canonical 00h interval ({raw_attack_ranges['canonical_00h']['first_attack_timestamp']} to {raw_attack_ranges['canonical_00h']['last_attack_timestamp']}). The four Attack labels at 12:05:00-12:05:03 in the requested 12:05-12:13 audit range are the tail of bound ID12; the exact official ID13 interval contains {official_id13_attack_points} Attack-labeled points. This is preserved as a source conflict; neither source is rewritten.

- Point-wise ground truth: actual Attack CSV labels.
- Official event primary evaluation: {bound_count} successfully bound events.
- ID13: excluded from the primary event aggregate and retained as `SOURCE_CONFLICT_UNRESOLVED`.
- HIGH relevance IDs 2, 3, 4, 7, 15: all `BOUND`.
- Binding status: **{binding_status}**.

## Frozen safeguards

No imputation, Attack-driven feature removal, event-specific 12-hour patch, model training, model inference, thresholding, or metric optimization was performed.
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps(protocol, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
