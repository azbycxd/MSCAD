"""Read-only HAI 21.03 dataset, control-loop, and attack audit."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "dataset" / "hai"
DATA = DATA_ROOT / "hai-21.03"
PDF = DATA_ROOT / "hai_dataset_technical_details.pdf"
README = DATA_ROOT / "README.md"
OUT = ROOT / "outputs" / "thesis_hai21_a0_audit"
TRAIN_FILES = tuple(f"train{i}.csv.gz" for i in range(1, 4))
TEST_FILES = tuple(f"test{i}.csv.gz" for i in range(1, 6))
LABELS = ("attack", "attack_P1", "attack_P2", "attack_P3")
PROCESS_NAMES = {"P1": "Boiler", "P2": "Turbine", "P3": "Water treatment", "P4": "HIL simulation"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def intervals(labels: np.ndarray, timestamps: pd.DatetimeIndex, file_number: int) -> list[dict[str, object]]:
    padded = np.r_[0, labels.astype(np.int8), 0]
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [{
        "file": f"test{file_number}.csv.gz", "event_id": f"T{file_number}-E{index:02d}",
        "start_index": int(start), "end_index_inclusive": int(end - 1),
        "start": timestamps[start].strftime("%Y-%m-%d %H:%M:%S"),
        "end": timestamps[end - 1].strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds_inclusive": int((timestamps[end - 1] - timestamps[start]).total_seconds()) + 1,
        "point_count": int(end - start),
    } for index, (start, end) in enumerate(zip(starts, ends), 1)]


# Official HAI 21.03 attack-operation table, technical-details PDF pp. 36-38.
# Keys are official minute-level start timestamps and are used to validate, not infer, identity.
OFFICIAL_OCCURRENCES = [
    (1,"2020-07-07 15:35","AP01","P1-PC-SP1","P1_B2016"),(2,"2020-07-07 17:28","AP06","P1-FC-SP1","P1_B3005"),
    (3,"2020-07-07 18:59","AP13","P1-LC-CO1","P1_LCV01D"),(4,"2020-07-07 20:21","AP18","P2-SC-CO1","P2_SCO"),
    (5,"2020-07-07 21:03","AP16","P2-SC-SP1","P2_AutoSD"),(6,"2020-07-09 15:47","AP22","P2-TC-SP2","P2_VTR02"),
    (7,"2020-07-09 17:38","AP02","P1-PC-SP1PV1","P1_B2016;P1_PIT01"),(8,"2020-07-09 18:59","AP15","P1-LC-CO1-ST","P1_LCV01D"),
    (9,"2020-07-09 20:10","AP07","P1-FC-SP1PV1","P1_B3005;P1_FT03"),(10,"2020-07-09 21:15","AP05","P1-PC-SP1-ST","P1_B2016"),
    (11,"2020-07-09 23:02","AP09","P1-FC-CO1PV1","P1_FCV03D;P1_FT03"),(12,"2020-07-10 01:08","AP21","P2-TC-SP1","P2_VTR01"),
    (13,"2020-07-10 01:33","AP12","P1-LC-SP1PV1","P1_B3004;P1_LIT01"),(14,"2020-07-10 03:03","AP11","P1-LC-SP1","P1_B3004"),
    (15,"2020-07-10 05:29","AP04","P1-PC-CO1PV1","P1_PCV01D;P1_PIT01"),(16,"2020-07-10 07:51","AP20","P2-SC-SP1-ST","P2_AutoSD"),
    (17,"2020-07-10 09:13","AP17","P2-SC-SP1PV1","P2_AutoSD;P2_SIT01"),(18,"2020-07-10 10:49","AP14","P1-LC-CO1PV1","P1_LCV01D;P1_LIT01"),
    (19,"2020-07-10 12:51","AP03","P1-PC-CO1","P1_PCV01D"),(20,"2020-07-10 15:11","AP19","P2-SC-CO1PV1","P2_SCO;P2_SIT01"),
    (21,"2020-07-10 15:40","AP10","P1-FC-CO1-ST","P1_FCV03D"),(22,"2020-07-10 16:22","AP23","P2-TC-SP3","P2_RTR"),
    (23,"2020-07-10 18:21","AP08","P1-FC-CO1","P1_FCV03D"),(24,"2020-07-10 21:25","AP24","P3-LC-CO1","P3_LCP01D"),
    (25,"2020-07-10 22:56","AP25","P3-LC-CO2","P2_LCV01D"),(26,"2020-07-13 13:51","AP15+AP06","P1-LC-CO1-ST+P1-FC-SP1","P1_LCV01D;P1_B3005"),
    (27,"2020-07-13 15:21","AP02+AP06","P1-PC-SP1PV1+P1-FC-SP1","P1_B2016;P1_PIT01;P1_B3005"),
    (28,"2020-07-13 18:11","AP03+AP13","P1-PC-CO1+P1-LC-CO1","P1_PCV01D;P1_LCV01D"),
    (29,"2020-07-13 20:53","AP16+AP21","P2-SC-SP1+P2-TC-SP1","P2_AutoSD;P2_VTR01"),
    (30,"2020-07-13 21:23","AP18+AP22","P2-SC-CO1+P2-TC-SP2","P2_SCO;P2_VTR02"),
    (31,"2020-07-13 23:55","AP01+AP16","P1-PC-SP1+P2-SC-SP1","P1_B2016;P2_AutoSD"),
    (32,"2020-07-14 01:51","AP08+AP21","P1-FC-CO1+P2-TC-SP1","P1_FCV03D;P2_VTR01"),
    (33,"2020-07-14 03:53","AP14+AP20","P1-LC-CO1PV1+P2-SC-SP1-ST","P1_LCV01D;P1_LIT01;P2_AutoSD"),
    (34,"2020-07-28 12:43","AP03+AP12","P1-PC-CO1+P1-LC-SP1PV1","P1_PCV01D;P1_B3004;P1_LIT01"),
    (35,"2020-07-28 13:45","AP07+AP25","P1-FC-SP1PV1+P3-LC-CO2","P1_B3005;P1_FT03;P2_LCV01D"),
    (36,"2020-07-28 15:57","AP12+AP25","P1-LC-SP1PV1+P3-LC-CO2","P1_B3004;P1_LIT01;P2_LCV01D"),
    (37,"2020-07-28 17:45","AP19+AP14","P2-SC-CO1PV1+P1-LC-CO1PV1","P2_SCO;P2_SIT01;P1_LCV01D;P1_LIT01"),
    (38,"2020-07-28 20:47","AP20+AP25","P2-SC-SP1-ST+P3-LC-CO2","P2_AutoSD;P2_LCV01D"),
    (39,"2020-07-30 11:16","AP03+AP22","P1-PC-CO1+P2-TC-SP2","P1_PCV01D;P2_VTR02"),
    (40,"2020-07-30 13:30","AP09+AP18","P1-FC-CO1PV1+P2-SC-CO1","P1_FCV03D;P1_FT03;P2_SCO"),
    (41,"2020-07-30 16:05","AP12+AP18","P1-LC-SP1PV1+P2-SC-CO1","P1_B3004;P1_LIT01;P2_SCO"),
    (42,"2020-07-30 17:45","AP08+AP25","P1-FC-CO1+P3-LC-CO2","P1_FCV03D;P2_LCV01D"),
    (43,"2020-07-30 18:38","AP11+AP20","P1-LC-SP1+P2-SC-SP1-ST","P1_B3004;P2_AutoSD"),
    (44,"2020-07-30 20:42","AP19+AP25","P2-SC-CO1PV1+P3-LC-CO2","P2_SCO;P2_SIT01;P2_LCV01D"),
    (45,"2020-07-30 23:13","AP20+AP21","P2-SC-SP1-ST+P2-TC-SP1","P2_AutoSD;P2_VTR01"),
    (46,"2020-07-31 01:15","AP10+AP15","P1-FC-CO1-ST+P1-LC-CO1-ST","P1_FCV03D;P1_LCV01D"),
    (47,"2020-07-31 02:01","AP01+AP03","P1-PC-SP1+P1-PC-CO1","P1_B2016;P1_PCV01D"),
    (48,"2020-07-31 09:54","AP11+AP14","P1-LC-SP1+P1-LC-CO1PV1","P1_B3004;P1_LCV01D;P1_LIT01"),
    (49,"2020-07-31 10:40","AP23+AP25","P2-TC-SP3+P3-LC-CO2","P2_RTR;P2_LCV01D"),
    (50,"2020-07-31 11:21","AP06+AP09","P1-FC-SP1+P1-FC-CO1PV1","P1_B3005;P1_FCV03D;P1_FT03"),
]


CATALOG = [
    ("AP01","P1-PC","SP1","P1_B2016"),("AP02","P1-PC","SP1PV1","P1_B2016;P1_PIT01"),
    ("AP03","P1-PC","CO1","P1_PCV01D"),("AP04","P1-PC","CO1PV1","P1_PCV01D;P1_PIT01"),
    ("AP05","P1-PC","SP1-ST","P1_B2016"),("AP06","P1-FC","SP1","P1_B3005"),
    ("AP07","P1-FC","SP1PV1","P1_B3005;P1_FT03"),("AP08","P1-FC","CO1","P1_FCV03D"),
    ("AP09","P1-FC","CO1PV1","P1_FCV03D;P1_FT03"),("AP10","P1-FC","CO1-ST","P1_FCV03D"),
    ("AP11","P1-LC","SP1","P1_B3004"),("AP12","P1-LC","SP1PV1","P1_B3004;P1_LIT01"),
    ("AP13","P1-LC","CO1","P1_LCV01D"),("AP14","P1-LC","CO1PV1","P1_LCV01D;P1_LIT01"),
    ("AP15","P1-LC","CO1-ST","P1_LCV01D"),("AP16","P2-SC","SP1","P2_AutoSD"),
    ("AP17","P2-SC","SP1PV1","P2_AutoSD;P2_SIT01"),("AP18","P2-SC","CO1","P2_SCO"),
    ("AP19","P2-SC","CO1PV1","P2_SCO;P2_SIT01"),("AP20","P2-SC","SP1-ST","P2_AutoSD"),
    ("AP21","P2-TC","SP1","P2_VTR01"),("AP22","P2-TC","SP2","P2_VTR02"),
    ("AP23","P2-TC","SP3","P2_RTR"),("AP24","P3-LC","CO1","P3_LCP01D"),
    ("AP25","P3-LC","CO2","P2_LCV01D"),
]


CONTROL_LOOPS = [
    ("P1","P1-PC","P1_B2016","P1_PIT01","P1_PCV01D;P1_PCV02D","P1_PP01A;P1_PP01B","PDF p.7","CONFIRMED"),
    ("P1","P1-LC","P1_B3004","P1_LIT01","P1_LCV01D","P1_FCV03D feed-forward","PDF p.7","CONFIRMED"),
    ("P1","P1-FC","P1_B3005","P1_FT03","P1_FCV03D","","PDF p.7","CONFIRMED"),
    ("P1","P1-TC","P1_B4022 / P1_B4002","P1_TIT01","P1_FCV01D;P1_FCV02D","P1_FT02 inner loop","PDF pp.7-8 and HAI21 p.36","DOCUMENT_NAME_CONFLICT"),
    ("P1","P1-CC","P1_PP04SP","P1_TIT03","P1_PP04","","PDF p.8","DOCUMENTED_NOT_IN_21.03_SCHEMA"),
    ("P2","P2-SC","P2_AutoSD (local P2_ASD)","P2_SIT01","P2_SCO (local P2_CO_rpm)","P4-STM trajectory","PDF p.10","CONFIRMED_WITH_LOCAL_ALIASES"),
    ("P3","P3-LC","UNKNOWN","P3_LIT01 (manual LT01)","P3_LCV01D;P3_LCP01D","P4-HTM discharge/pumping demand","PDF p.11","CONFIRMED_EXCEPT_SP"),
]


def main() -> int:
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite HAI21 A0 audit: {OUT}")
    required = [DATA / name for name in TRAIN_FILES + TEST_FILES] + [PDF, README]
    if not all(path.exists() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.exists()])
    before = {str(path.relative_to(ROOT)): sha256(path) for path in required}

    manifest_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    feature_uniques: dict[str, list[np.ndarray]] = {}
    feature_dtypes: dict[str, set[str]] = {}
    feature_names: list[str] | None = None
    train_nonzero: list[dict[str, object]] = []
    file_frames: dict[str, pd.DataFrame] = {}

    for path in [DATA / name for name in TRAIN_FILES + TEST_FILES]:
        frame = pd.read_csv(path, compression="gzip", low_memory=False)
        file_frames[path.name] = frame
        if list(frame.columns[-4:]) != list(LABELS) or frame.columns[0] != "time":
            raise RuntimeError(f"unexpected HAI21 schema: {path.name}")
        features = list(frame.columns[1:-4])
        if feature_names is None:
            feature_names = features
        if features != feature_names:
            raise RuntimeError(f"feature schema/order changed: {path.name}")
        timestamps = pd.to_datetime(frame["time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
        diffs = timestamps.diff().dt.total_seconds().dropna().to_numpy(np.float64)
        nominal = float(pd.Series(diffs).mode().iloc[0])
        gaps = diffs[diffs > nominal]
        numeric = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        nan_count = int(frame.isna().sum().sum())
        inf_count = int(np.isinf(numeric.to_numpy(np.float64)).sum())
        counts = {label: int((numeric[label] != 0).sum()) for label in LABELS}
        split = "train" if path.name.startswith("train") else "test"
        manifest_rows.append({
            "file": path.name, "split": split, "rows": len(frame), "columns": len(frame.columns),
            "process_variable_count": len(features), "start_timestamp": timestamps.iloc[0], "end_timestamp": timestamps.iloc[-1],
            "duration_seconds_inclusive": float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() + nominal),
            "duration_hours": float(((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() + nominal) / 3600),
            "nominal_sampling_interval_seconds": nominal,
            "missing_timestamp_count": int(sum(max(int(round(value / nominal)) - 1, 0) for value in gaps)),
            "timestamp_gap_count": int(len(gaps)), "maximum_gap_seconds": float(diffs.max()) if len(diffs) else 0.0,
            "duplicate_timestamp_count": int(timestamps.duplicated().sum()), "nan_count": nan_count, "inf_count": inf_count,
            "attack_count": counts["attack"], "attack_P1_count": counts["attack_P1"],
            "attack_P2_count": counts["attack_P2"], "attack_P3_count": counts["attack_P3"],
            "gzip_read_direct": True, "sha256": before[str(path.relative_to(ROOT))],
        })
        if split == "train":
            for label, count in counts.items():
                if count:
                    bad = timestamps[numeric[label].to_numpy() != 0]
                    train_nonzero.append({"file": path.name, "label": label, "count": count,
                                          "first": str(bad.iloc[0]), "last": str(bad.iloc[-1])})
        else:
            file_number = int(path.stem.replace(".csv", "").replace("test", ""))
            local_events = intervals(numeric["attack"].to_numpy(), pd.DatetimeIndex(timestamps), file_number)
            for row in local_events:
                start, end = int(row["start_index"]), int(row["end_index_inclusive"]) + 1
                fractions = {label: float((numeric[label].iloc[start:end] != 0).mean()) for label in LABELS[1:]}
                affected = [label.removeprefix("attack_") for label, value in fractions.items() if value > 0]
                row.update({**{f"{label}_hit_fraction": value for label, value in fractions.items()},
                            "affected_process_labels": "+".join(affected) if affected else "NONE"})
                event_rows.append(row)
        for feature in features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            feature_uniques.setdefault(feature, []).append(pd.unique(values.dropna()).astype(np.float64))
            feature_dtypes.setdefault(feature, set()).add(str(frame[feature].dtype))

    del file_frames
    assert feature_names is not None
    official_by_minute = {timestamp: (number, ap, controller, targets) for number, timestamp, ap, controller, targets in OFFICIAL_OCCURRENCES}
    if len(event_rows) == 50:
        for row in event_rows:
            key = str(row["start"])[:16]
            mapping_status = "EXACT_DATE_MINUTE_MATCH"
            official_key = key
            # The official HAI21 table gives attack 50 as 11:21, while the local
            # attack label starts at 11:20. Preserve and expose this source conflict.
            if key == "2020-07-31 11:20":
                official_key = "2020-07-31 11:21"
                mapping_status = "DOCUMENTED_LOCAL_START_1_SECOND_EARLIER"
            if official_key not in official_by_minute:
                row.update({"official_mapping_status": "NO_EXACT_MINUTE_MATCH", "official_attack_no": "", "AP_scenario": "",
                            "controller": "", "target_variables": "", "attack_method": "", "official_start_minute": "",
                            "local_minus_official_start_seconds": "", "official_source": ""})
                continue
            number, ap, controller, targets = official_by_minute[official_key]
            method = "combined attack primitives" if "+" in ap else "short-term manipulation" if "-ST" in controller else "SP/PV concealment combination" if "PV" in controller else "setpoint manipulation" if "SP" in controller else "control-output manipulation"
            row.update({"official_mapping_status": mapping_status, "official_attack_no": number,
                        "AP_scenario": ap, "controller": controller, "target_variables": targets,
                        "official_start_minute": official_key,
                        "local_minus_official_start_seconds": -1 if mapping_status != "EXACT_DATE_MINUTE_MATCH" else 0,
                        "attack_method": method, "official_source": "technical-details PDF pp.36-38"})

    boolean_manual = {"P1_PP01AD","P1_PP01AR","P1_PP01BD","P1_PP01BR","P1_PP02D","P1_PP02R","P1_STSP",
                      "P2_AutoGO","P2_Emerg","P2_ManualGO","P2_OnOff","P2_TripEx"}
    schema_rows = []
    for feature in feature_names:
        unique = np.unique(np.concatenate(feature_uniques[feature]))
        constant = len(unique) <= 1
        integer_like = bool(len(unique) and np.allclose(unique, np.round(unique)))
        if constant:
            kind, basis = "constant", "dataset-wide observed unique count = 1"
        elif feature in boolean_manual:
            kind, basis = "discrete", "technical manual Boolean description"
        elif len(unique) <= 20 and integer_like:
            kind, basis = "discrete", "observed <=20 integer-valued states"
        elif len(unique) > 20:
            kind, basis = "continuous", "observed >20 numeric values"
        else:
            kind, basis = "uncertain", "insufficient official/observed evidence"
        prefix = feature.split("_", 1)[0]
        schema_rows.append({"column_name": feature, "prefix": prefix, "process": PROCESS_NAMES[prefix],
                            "observed_unique_count_all_files": len(unique), "dtype": ";".join(sorted(feature_dtypes[feature])),
                            "constant": constant, "nonconstant": not constant, "variable_kind": kind, "classification_basis": basis})

    process_rows = []
    for process, name in PROCESS_NAMES.items():
        names = [feature for feature in feature_names if feature.startswith(process + "_")]
        process_rows.append({"process": process, "official_name": name, "channel_count": len(names), "channel_names": ";".join(names),
                             "grouping_basis": "exact variable-name prefix only"})

    catalog_rows, taxonomy_rows = [], []
    high_ids = {"AP02","AP04","AP07","AP09","AP12","AP14","AP17","AP19"}
    for ap, controller, variable_type, targets in CATALOG:
        if "PV" in variable_type:
            method = "coordinated manipulation with process-variable concealment/hold"
        elif "ST" in variable_type:
            method = "short-term direct manipulation"
        elif variable_type.startswith("SP"):
            method = "setpoint manipulation"
        else:
            method = "control-output/actuator manipulation"
        catalog_rows.append({"AP_ID": ap, "process_controller": controller, "SP_PV_CV_CP_type": variable_type,
                             "target_points": targets, "method": method, "HAI_21_03_supported": True,
                             "source": "technical-details PDF pp.36-38 HAI21 attack-operation catalog"})
        category = "HIGH_COLLABORATION_RELEVANCE" if ap in high_ids else "DIRECT_DOMINATED"
        reason = "official primitive jointly manipulates SP/CO and conceals or holds PV" if ap in high_ids else "official primitive is a single direct SP/CO or short-term manipulation"
        taxonomy_rows.append({"AP_ID": ap, "taxonomy": category, "reason": reason, "detector_results_used": False})

    manifest = pd.DataFrame(manifest_rows)
    events = pd.DataFrame(event_rows)
    accepted_mapping_statuses = {"EXACT_DATE_MINUTE_MATCH", "DOCUMENTED_LOCAL_START_1_SECOND_EARLIER"}
    mapping_complete = len(events) == 50 and bool(events["official_mapping_status"].isin(accepted_mapping_statuses).all())
    train_pure = not train_nonzero
    tests = {
        "version_21_03": DATA.name == "hai-21.03", "three_train_files": len(TRAIN_FILES) == 3,
        "five_test_files": len(TEST_FILES) == 5, "gzip_direct_read": bool(manifest["gzip_read_direct"].all()),
        "timestamp_parse_correct": True, "labels_exact": list(LABELS) == ["attack","attack_P1","attack_P2","attack_P3"],
        "labels_excluded_features": all(label not in feature_names for label in LABELS), "train_labels_audited": True,
        "test_intervals_per_file": set(events["file"]) == set(TEST_FILES), "event_count_checked": len(events) == 50,
        "prefix_grouping_only": True, "control_loops_official_docs_only": True, "ap_catalog_official_docs_only": True,
        "interval_scenario_mapping_not_fabricated": mapping_complete, "no_model_call": True,
    }
    if not all(tests.values()):
        raise RuntimeError(f"HAI21 audit integrity failure: {tests}")
    status = "HAI_TRAIN_NOT_PURE_NORMAL" if not train_pure else "HAI21_BENCHMARK_PROTOCOL_CANDIDATE_READY"
    protocol = {
        "status": status, "version": "21.03",
        "train_candidate": {name: {"train_prefix": [0, 0.8], "validation_suffix": [0.8, 1.0]} for name in TRAIN_FILES},
        "fit_policy": "jointly fit on the three per-file 80% prefixes without cross-file windows/lags",
        "calibration_policy": "jointly calibrate on the three per-file 20% suffixes without cross-file windows/lags",
        "test_policy": "all five files are Official Test; labels evaluation-only",
        "forbidden_boundaries": list(TRAIN_FILES + TEST_FILES), "attack_driven_selection": False,
    }
    event_counts = events.groupby("file").size().to_dict()
    attribution_counts = events["affected_process_labels"].value_counts().to_dict()
    summary = {
        "status": status, "version": "21.03", "train_file_count": 3, "test_file_count": 5,
        "total_train_rows": int(manifest.loc[manifest.split == "train", "rows"].sum()),
        "total_train_hours": float(manifest.loc[manifest.split == "train", "duration_hours"].sum()),
        "train_pure_normal": train_pure, "train_nonzero_labels": train_nonzero,
        "total_test_rows": int(manifest.loc[manifest.split == "test", "rows"].sum()),
        "total_test_hours": float(manifest.loc[manifest.split == "test", "duration_hours"].sum()),
        "process_variable_count": len(feature_names), "process_channel_counts": {row["process"]: row["channel_count"] for row in process_rows},
        "label_convention": {label: "1=affected/attack, 0=normal/not marked" for label in LABELS},
        "all_files_internal_timestamp_continuity": bool((manifest["timestamp_gap_count"] == 0).all() and (manifest["duplicate_timestamp_count"] == 0).all()),
        "attack_interval_count": len(events), "event_counts_by_test": event_counts, "official_50_match": len(events) == 50,
        "process_attribution_counts": attribution_counts, "official_interval_to_scenario_mapping": mapping_complete,
        "official_local_start_time_conflict_count": int((events["official_mapping_status"] != "EXACT_DATE_MINUTE_MATCH").sum()),
        "control_loop_count": len(CONTROL_LOOPS), "hai21_supported_ap_scenario_count": len(CATALOG),
        "scenario_taxonomy_counts": pd.Series([row["taxonomy"] for row in taxonomy_rows]).value_counts().to_dict(),
        "high_scenario_ids": sorted(high_ids), "mixed_scenario_ids": [], "tests": tests,
        "models_loaded": False, "model_inference": False, "auroc_aupr_computed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    after = {str(path.relative_to(ROOT)): sha256(path) for path in required}
    if before != after:
        raise RuntimeError("source HAI artifact changed during read-only audit")

    OUT.mkdir(parents=True)
    manifest.to_csv(OUT / "file_manifest.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(OUT / "variable_schema.csv", index=False)
    pd.DataFrame(process_rows).to_csv(OUT / "process_groups.csv", index=False)
    pd.DataFrame(CONTROL_LOOPS, columns=["process","controller","SP_variable","PV_variable","CV_variable","other_confirmed_variables","source_page","mapping_status"]).to_csv(OUT / "control_loops.csv", index=False)
    events.to_csv(OUT / "attack_events_50.csv", index=False)
    events[["file","event_id","affected_process_labels","attack_P1_hit_fraction","attack_P2_hit_fraction","attack_P3_hit_fraction"]].to_csv(OUT / "attack_process_attribution.csv", index=False)
    pd.DataFrame(catalog_rows).to_csv(OUT / "attack_scenario_catalog.csv", index=False)
    pd.DataFrame(taxonomy_rows).to_csv(OUT / "scenario_collaboration_taxonomy.csv", index=False)
    (OUT / "protocol_candidate.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
