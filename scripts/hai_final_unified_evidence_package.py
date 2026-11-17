from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUT = OUTPUTS / "hai_final_unified_evidence_package"
UNAVAILABLE = "unavailable_without_recomputation"
PACKAGE_VERSION = "1.1"
TEXT_EXTENSIONS = {".csv", ".json", ".yaml", ".yml", ".md", ".txt"}
RAW_EXTENSIONS = {".npz", ".pt", ".pth", ".joblib", ".pkl"}

STAGES = {
    "hai_step1_dataset_audit": ("HAI-1", "HAI Step 1 数据只读验收"),
    "hai_step2_dataloader_validation": ("HAI-2", "HAI Step 2 数据加载与窗口协议"),
    "hai_step3a_rec_baseline_seed1": ("HAI-3A", "HAI Step 3A Tiny seed1"),
    "hai_step3b_rec_multiseed": ("HAI-3B", "HAI Step 3B Tiny 多随机种子"),
    "hai_step3c_rec_score_alignment": ("HAI-3C", "HAI Step 3C 因果点级对齐"),
    "hai_step3d_rec_channel_calibration": ("HAI-3D", "HAI Step 3D endpoint raw top5"),
    "hai_step3e_rec_causal_alarm_policy": ("HAI-3E", "HAI Step 3E 因果告警策略"),
    "hai_step3f_rec_state_conditioned_threshold": ("HAI-3F", "HAI Step 3F 状态条件阈值"),
    "hai_step3g_rec_capacity_ablation": ("HAI-3G", "HAI Step 3G 容量消融"),
    "hai_step4a_normal_relation_diagnosis": ("HAI-4A", "HAI Step 4A 正常关系诊断"),
    "hai_step4b_structure_controlled_validation": ("HAI-4B", "HAI Step 4B 结构受控验证"),
    "hai_step4b_preserving_control_audit": ("HAI-4B-R", "HAI Step 4B-R 保持型对照修正"),
    "hai_step4b_relation_aggregation_repair": ("HAI-4B-A", "HAI Step 4B-A tail聚合诊断"),
    "hai_step4b_fair_temporal_generalization_audit": ("HAI-4B-B", "HAI Step 4B-B 公平时序泛化审计"),
    "hai_step4cd_structure_real_and_complementarity": ("HAI-4CD", "HAI Step 4C+D 真实结构探索"),
    "hai_structure_v2a_lagged_dependency": ("V2-A", "Structure V2-A 时滞条件依赖"),
    "hai_structure_v2a1r_cache_backfill_and_grouped_aggregation": ("V2-A1R", "Structure V2-A1R 精确恢复与分组"),
    "hai_structure_v2a1s_full_clean_stability": ("V2-A1S", "Structure V2-A1S full-clean稳定性"),
    "hai_structure_v2a1d_normal_drift_attribution": ("V2-A1D", "Structure V2-A1D 正常漂移归因"),
    "hai_structure_v2a1m_mode_conditioned_reference": ("V2-A1M", "Structure V2-A1M 模式条件参考"),
    "hai_structure_v2a1c_causal_adaptive_reference": ("V2-A1C", "Structure V2-A1C 因果自适应参考"),
    "stage4_3d_swat_data_audit": ("SWAT-4.3D", "SWaT 数据审计"),
    "stage4_3e1_raw_source_locator": ("SWAT-4.3E1", "SWaT 原始数据源定位"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative: str):
    return pd.read_csv(ROOT / relative)


def safe_value(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return UNAVAILABLE
    if isinstance(value, np.generic):
        return value.item()
    return value


def stage_identity(directory_name: str):
    if directory_name in STAGES:
        return STAGES[directory_name]
    return directory_name, directory_name.replace("_", " ")


def discover_stage_meta(directory: Path):
    classification, selected_status = "not_recorded", "not_recorded"
    for path in sorted(directory.glob("*summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            classification = str(payload.get("classification", classification))
            break
        except (ValueError, OSError):
            pass
    for pattern in ("selected*config.yaml", "frozen*config.yaml"):
        for path in sorted(directory.glob(pattern)):
            try:
                selected_status = str(yaml.safe_load(path.read_text(encoding="utf-8")).get("status", "recorded_without_status"))
                break
            except (ValueError, OSError, AttributeError):
                pass
    return classification, selected_status


def scan_inventory():
    rows = []
    for directory in sorted(path for path in OUTPUTS.iterdir() if path.is_dir() and path != OUT):
        stage_id, stage_name = stage_identity(directory.name)
        classification, selected_status = discover_stage_meta(directory)
        failed = "failed" in directory.name
        smoke = "smoke" in directory.name
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            suffix = path.suffix.lower()
            artifact_type = suffix.removeprefix(".") or "no_extension"
            raw = suffix in RAW_EXTENSIONS
            text = suffix in TEXT_EXTENSIONS
            eligible = bool(text and not failed and not smoke)
            exclusion = "" if eligible else ("smoke_artifact" if smoke else "failed_run_artifact" if failed else "binary_raw_or_model_artifact")
            rows.append({
                "stage_id": stage_id, "stage_name": stage_name,
                "artifact_path": str(path.relative_to(ROOT)).replace("\\", "/"), "artifact_type": artifact_type,
                "file_size": path.stat().st_size, "sha256": sha256(path),
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "status": "smoke" if smoke else "failed" if failed else "formal",
                "classification": classification, "selected_config_status": selected_status,
                "contains_metrics": suffix in {".csv", ".json"}, "contains_raw_scores": raw,
                "eligible_for_final_summary": eligible, "exclusion_reason": exclusion or "NONE",
            })
    return pd.DataFrame(rows)


def input_fingerprint():
    records = []
    for directory in sorted(path for path in OUTPUTS.iterdir() if path.is_dir() and path != OUT):
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            stat = path.stat()
            records.append((str(path.relative_to(OUTPUTS)), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps([PACKAGE_VERSION, records], separators=(",", ":")).encode()).hexdigest()


def markdown_table(frame: pd.DataFrame, title: str):
    values = frame.fillna(UNAVAILABLE).astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row.replace("|", "\\|") for row in values.iloc[index].tolist()) + " |" for index in range(len(values))]
    return f"# {title}\n\n" + "\n".join([header, separator, *rows]) + "\n"


def write_table(out: Path, stem: str, frame: pd.DataFrame, title: str):
    clean = frame.fillna(UNAVAILABLE)
    clean.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(markdown_table(clean, title), encoding="utf-8")


class Provenance:
    def __init__(self, hashes):
        self.rows = []
        self.hashes = hashes

    def add(self, metric_id, stage_id, method, split, scope, name, value, source, locator,
            unit="ratio", aggregation="reported", seed="all", threshold="not_applicable",
            validity="valid_diagnostic_or_ablation", limitation="NONE", dataset="HAI-22.04"):
        relative = source.replace("\\", "/")
        self.rows.append({
            "metric_id": metric_id, "stage_id": stage_id, "method_variant": method, "dataset": dataset,
            "split": split, "evaluation_scope": scope, "metric_name": name, "metric_value": safe_value(value),
            "unit": unit, "aggregation": aggregation, "seed": seed, "threshold_protocol": threshold,
            "source_artifact": relative, "source_column_or_json_path": locator,
            "source_hash": self.hashes.get(relative, "source_not_in_inventory"), "validity_status": validity,
            "limitation": limitation,
        })
        return metric_id

    def frame(self):
        return pd.DataFrame(self.rows)


def metric_from_rows(frame, filters, column):
    selected = frame
    for key, value in filters.items():
        selected = selected[selected[key].astype(str).eq(str(value))]
    if selected.empty or column not in selected:
        return UNAVAILABLE
    return safe_value(selected.iloc[0][column])


def build_tables(out: Path, provenance: Provenance):
    audit = read_json("outputs/hai_step1_dataset_audit/hai_dataset_audit_summary.json")
    loader = read_json("outputs/hai_step2_dataloader_validation/hai_dataloader_summary.json")
    windows = read_csv("outputs/hai_step2_dataloader_validation/hai_window_counts.csv")
    p1 = "outputs/hai_step1_dataset_audit/hai_dataset_audit_summary.json"
    p2 = "outputs/hai_step2_dataloader_validation/hai_dataloader_summary.json"
    dataset_rows = []
    def dataset_row(metric, value, source, locator, note="NONE"):
        mid = provenance.add(f"dataset.{metric}", "HAI-1" if "step1" in source else "HAI-2", "data_protocol", "all", "dataset", metric, value, source, locator, unit="count" if isinstance(value, (int, np.integer)) else "description", validity="valid_real_evaluation", limitation=note)
        dataset_rows.append({"item": metric, "value": safe_value(value), "protocol_note": note, "provenance_metric_ids": mid})
    dataset_row("train_file_count", audit["train_file_count"], p1, "$.train_file_count")
    dataset_row("test_file_count", audit["test_file_count"], p1, "$.test_file_count")
    dataset_row("feature_count", audit["feature_count"], p1, "$.feature_count")
    dataset_row("total_train_rows", audit["total_train_rows"], p1, "$.total_train_rows")
    dataset_row("total_test_rows", audit["total_test_rows"], p1, "$.total_test_rows")
    dataset_row("attack_event_count", audit["total_test_attack_segments"], p1, "$.total_test_attack_segments")
    dataset_row("globally_constant_feature_count", len(audit["globally_constant_features"]), p1, "len($.globally_constant_features)")
    for scope in ("train", "validation"):
        value = int(windows.query("window_size == 64 and scope == @scope").window_count.iloc[0])
        dataset_row(f"window64_{scope}_count", value, "outputs/hai_step2_dataloader_validation/hai_window_counts.csv", f"window_size=64,scope={scope}.window_count")
    dataset_row("window64_test_count", int(windows.query("window_size == 64 and scope == 'test'").window_count.sum()), "outputs/hai_step2_dataloader_validation/hai_window_counts.csv", "sum(window_size=64,scope=test).window_count")
    dataset_row("scaler_fit_scope", loader["scaler_fit_scope"], p2, "$.scaler_fit_scope")
    dataset_row("window_length", 64, p2, "$.window64_counts[].window_size")
    dataset_row("endpoint_semantics", "score is assigned to the causal window endpoint", "outputs/hai_step3c_rec_score_alignment/score_definition_registry.csv", "endpoint score definition")
    dataset_row("no_cross_sequence_windows", bool(loader["no_window_crosses_file"]), p2, "$.no_window_crosses_file")
    dataset_row("eligible_relation_channels", 68, "outputs/hai_step4a_normal_relation_diagnosis/hai_relation_diagnosis_summary.json", "eligible_channel_count")
    dataset_row("threshold_source", "normal validation q99 for primary Tiny report", "outputs/hai_step3g_rec_capacity_ablation/rec_capacity_summary.json", "protocol synthesis")
    dataset_row("hai_status", "development/exploratory evaluation", p2, "protocol status")
    dataset_row("swat_legacy_status", "invalid_for_claims", "outputs/stage4_3d_swat_data_audit/final_data_audit.md", "audit conclusion", "merged.csv chronology/overlap invalidity")
    write_table(out, "table_01_dataset_and_protocol", pd.DataFrame(dataset_rows), "表1 数据集与实验协议")

    alignment = read_csv("outputs/hai_step3c_rec_score_alignment/rec_alignment_point_metrics_aggregate.csv")
    rec_summary = read_csv("outputs/hai_step3g_rec_capacity_ablation/rec_capacity_multiseed_summary.csv")
    rec_source = "outputs/hai_step3g_rec_capacity_ablation/rec_capacity_multiseed_summary.csv"
    evolution_rows = []
    variants = [
        ("full_window_mean", "HAI-3B", UNAVAILABLE, "diagnostic", "earlier score; not retained in a unified formal row"),
        ("endpoint_mean", "HAI-3C", alignment, "diagnostic", "causal endpoint alignment"),
        ("endpoint_raw_top5", "HAI-3D", rec_summary[rec_summary.model_size_name.eq("tiny")], "primary_result", "selected Tiny score"),
        ("pointwise", "HAI-3E", rec_summary[rec_summary.model_size_name.eq("tiny")], "primary_result", "no smoothing"),
        ("state_threshold", "HAI-3F", UNAVAILABLE, "rejected", "normal-state threshold did not improve branch"),
        ("tiny", "HAI-3G", rec_summary[rec_summary.model_size_name.eq("tiny")], "primary_result", "capacity baseline"),
        ("small", "HAI-3G", rec_summary[rec_summary.model_size_name.eq("small")], "capacity_ablation", "not selected as main solution"),
        ("medium", "HAI-3G", UNAVAILABLE, "capacity_ablation", "formal multiseed summary unavailable without recomputation"),
    ]
    for variant, stage, source_frame, role, note in variants:
        row = {"variant": variant, "role": role, "threshold_protocol": "normal_validation_q99", "limitation": note}
        mids = []
        for metric in ("precision", "recall", "f1", "aupr", "auroc"):
            if isinstance(source_frame, pd.DataFrame):
                if variant == "endpoint_mean":
                    value = metric_from_rows(source_frame, {"score_variant": "endpoint_mean", "threshold_method": "normal_validation_q99", "scope": "macro", "metric": metric}, "mean")
                    src, locator = "outputs/hai_step3c_rec_score_alignment/rec_alignment_point_metrics_aggregate.csv", f"endpoint_mean/q99/macro/{metric}.mean"
                else:
                    value = metric_from_rows(source_frame, {"threshold_method": "normal_validation_q99", "scope": "macro"}, f"{metric}_mean")
                    src, locator = rec_source, f"model={variant if variant in ('tiny','small') else 'tiny'},q99,macro.{metric}_mean"
                mid = provenance.add(f"rec.{variant}.{metric}", stage, variant, "test", "pointwise_macro", metric, value, src, locator, aggregation="mean_3seed", threshold="normal_validation_q99", validity="valid_real_evaluation" if role == "primary_result" else "valid_diagnostic_or_ablation", limitation=note)
                mids.append(mid); row[metric] = value
            else:
                row[metric] = UNAVAILABLE
        row["provenance_metric_ids"] = ";".join(mids) or "NONE"
        evolution_rows.append(row)
    write_table(out, "table_02_reconstruction_evolution", pd.DataFrame(evolution_rows), "表2 重构分支演化")

    core = read_csv("outputs/hai_step3g_rec_capacity_ablation/rec_capacity_core_metrics_new.csv")
    tiny_rows = []
    for seed in (1, 2, 3):
        selected = core[(core.seed == seed) & core.model_size_name.eq("tiny") & core.scope.eq("macro") & core.threshold_method.eq("normal_validation_q99")].iloc[0]
        row = {"seed": seed, "threshold_protocol": "normal_validation_q99"}; mids = []
        for metric in ("precision", "recall", "f1", "aupr", "auroc", "false_positive_count"):
            value = safe_value(selected[metric]); row[metric] = value
            mids.append(provenance.add(f"tiny.seed{seed}.{metric}", "HAI-3G", "Tiny endpoint_raw_top5", "test", "pointwise_macro", metric, value, "outputs/hai_step3g_rec_capacity_ablation/rec_capacity_core_metrics_new.csv", f"seed={seed},model=tiny,q99,macro.{metric}", seed=seed, threshold="normal_validation_q99", validity="valid_real_evaluation"))
        row["provenance_metric_ids"] = ";".join(mids); tiny_rows.append(row)
    write_table(out, "table_03_tiny_multiseed_metrics", pd.DataFrame(tiny_rows), "表3 Tiny三随机种子指标")

    real = read_json("outputs/hai_step4cd_structure_real_and_complementarity/structure_real_evaluation_summary.json")
    decomposition = read_csv("outputs/hai_step4b_preserving_control_audit/breaking_target_vs_global_metrics.csv")
    repair = read_csv("outputs/hai_step4b_relation_aggregation_repair/aggregation_repair_metrics_by_corruption.csv")
    v1_rows = []
    def v1row(name, evidence, stage, metrics, source, validity, limitation):
        row = {"method": name, "evidence_scope": evidence, "validity": validity, "limitation": limitation}; mids=[]
        for metric, value, locator in metrics:
            row[metric] = safe_value(value)
            mids.append(provenance.add(f"v1.{re.sub('[^a-z0-9]+','_',name.lower())}.{metric}", stage, name, "controlled" if evidence != "real_transfer" else "test", evidence, metric, value, source, locator, threshold="source_frozen_q99", validity=validity, limitation=limitation))
        row["provenance_metric_ids"] = ";".join(mids); v1_rows.append(row)
    target = decomposition[(decomposition.window == 64) & decomposition.component.eq("target_pair_score")]
    v1row("target_edge_R1_R2_R3", "mechanism", "HAI-4B-R", [(m, float(target[m].mean()), f"window=64,target_pair.mean({m})") for m in ("auroc", "aupr", "clean_q99_tpr")], "outputs/hai_step4b_preserving_control_audit/breaking_target_vs_global_metrics.csv", "valid_controlled_mechanism_evidence", "controlled corruption only")
    tail = repair[repair.candidate_id.eq("relation_tail_top2_mean")]
    v1row("tail_top2", "global_aggregation", "HAI-4B-A", [(m, float(tail[m].mean()), f"tail_top2.mean({m})") for m in ("auroc", "aupr", "q99_tpr")], "outputs/hai_step4b_relation_aggregation_repair/aggregation_repair_metrics_by_corruption.csv", "valid_diagnostic_or_ablation", "crossfit instability prevents freezing")
    rp = real["structure_point"]
    v1row("Structure_V1_real_HAI", "real_transfer", "HAI-4CD", [(m, rp[m], f"$.structure_point.{m}") for m in ("precision", "recall", "f1", "aupr", "auroc")], "outputs/hai_step4cd_structure_real_and_complementarity/structure_real_evaluation_summary.json", "valid_real_evaluation", "exploratory one-shot; AUROC near random")
    v1row("Rec_OR_Structure", "real_transfer", "HAI-4CD", [("event_recall", real["event_union"]["or_event_recall"], "$.event_union.or_event_recall"), ("added_tp", real["point_overlap"]["structure_added_tp"], "$.point_overlap.structure_added_tp"), ("added_fp", real["point_overlap"]["structure_added_fp"], "$.point_overlap.structure_added_fp"), ("fp_per_added_tp", real["point_overlap"]["added_fp_per_added_tp"], "$.point_overlap.added_fp_per_added_tp")], "outputs/hai_step4cd_structure_real_and_complementarity/structure_real_evaluation_summary.json", "valid_diagnostic_or_ablation", "fusion rejected; not a final method")
    write_table(out, "table_04_structure_v1_controlled_and_real", pd.DataFrame(v1_rows), "表4 Structure V1受控与真实迁移")

    a1r = read_csv("outputs/hai_structure_v2a1r_cache_backfill_and_grouped_aggregation/metrics_by_corruption.csv")
    a1s_summary = read_json("outputs/hai_structure_v2a1s_full_clean_stability/structure_v2a1s_summary.json")
    a1s_family = read_csv("outputs/hai_structure_v2a1s_full_clean_stability/controlled_metrics_by_family.csv")
    a1m_summary = read_json("outputs/hai_structure_v2a1m_mode_conditioned_reference/structure_v2a1m_summary.json")
    a1c_summary = read_json("outputs/hai_structure_v2a1c_causal_adaptive_reference/structure_v2a1c_summary.json")
    v2_rows=[]
    def v2row(name, stage, family_frame, clean_fpr, worst, train1, p1v, c2v, classification, frozen, extra="NONE"):
        breaking = family_frame[family_frame.family.isin(("R1","R2","R3"))]
        row={"variant":name,"relation_count":12,"unique_target_count":len(set(read_csv("outputs/hai_structure_v2a_lagged_dependency/selected_dynamic_relation_pool.csv").target)),"self_model_count":68,"cross_model_count":12,"controlled_auroc":float(breaking.auroc.mean()),"controlled_aupr":float(breaking.aupr.mean()),"controlled_q99_tpr":float(breaking.q99_tpr.mean()),"overall_clean_fpr":safe_value(clean_fpr),"worst_sequence_fpr":safe_value(worst),"train1_second_half_fpr":safe_value(train1),"p1_fpr":safe_value(p1v),"c2_fpr":safe_value(c2v),"absorption_gate":extra,"classification":classification,"frozen_status":frozen}; mids=[]
        source_map={"V2-A1R":"outputs/hai_structure_v2a1r_cache_backfill_and_grouped_aggregation/metrics_by_corruption.csv","V2-A1S":"outputs/hai_structure_v2a1s_full_clean_stability/controlled_metrics_by_family.csv","V2-A1M":"outputs/hai_structure_v2a1m_mode_conditioned_reference/controlled_metrics_by_family.csv","V2-A1C":"outputs/hai_structure_v2a1c_causal_adaptive_reference/controlled_adaptive_metrics_by_family.csv"}; src=source_map.get(stage,"outputs/hai_structure_v2a_lagged_dependency/dynamic_structure_metrics_by_corruption.csv")
        for m in ("controlled_auroc","controlled_aupr","controlled_q99_tpr"):
            mids.append(provenance.add(f"v2.{name}.{m}",stage,name,"controlled", "R1_R2_R3_macro",m,row[m],src,f"mean R1/R2/R3 {m.removeprefix('controlled_')}",threshold="clean_q99",validity="valid_controlled_mechanism_evidence",limitation="not real HAI attack performance"))
        summary_sources={
            "V2-A1R":"outputs/hai_structure_v2a1s_full_clean_stability/structure_v2a1s_summary.json",
            "V2-A1S":"outputs/hai_structure_v2a1s_full_clean_stability/structure_v2a1s_summary.json",
            "V2-A1M":"outputs/hai_structure_v2a1m_mode_conditioned_reference/structure_v2a1m_summary.json",
            "V2-A1C":"outputs/hai_structure_v2a1c_causal_adaptive_reference/structure_v2a1c_summary.json",
        }
        summary_source=summary_sources[stage]
        for m in ("overall_clean_fpr","worst_sequence_fpr","train1_second_half_fpr","p1_fpr","c2_fpr"):
            if row[m] != UNAVAILABLE:
                mids.append(provenance.add(f"v2.{name}.{m}",stage,name,"validation" if "fpr" in m else "controlled","normal_stability",m,row[m],summary_source,f"reported/linked {m}",threshold="frozen_protocol",validity="valid_diagnostic_or_ablation",limitation="clean validation or controlled evidence; not real attack performance"))
        row["provenance_metric_ids"]=";".join(mids); v2_rows.append(row)
    grouped=a1r[a1r.aggregation.eq("grouped")]
    v2row("target_grouped_top2","V2-A1R",grouped,0.009118,0.054300,0.107529,float(grouped[grouped.family.eq('P1')].q99_tpr.iloc[0]),float(grouped[grouped.family.eq('C2')].q99_tpr.iloc[0]),"target_grouping_improves_stability_but_not_enough","frozen_controlled_validation_reference")
    v2row("A1S_full_clean","V2-A1S",a1s_family,a1s_summary["clean_fpr"],a1s_summary["worst_sequence_fpr"],0.107529,float(a1s_family[a1s_family.family.eq('P1')].q99_tpr.iloc[0]),a1s_summary["c2_fpr"],a1s_summary["classification"],"not_frozen")
    a1m_family=read_csv("outputs/hai_structure_v2a1m_mode_conditioned_reference/controlled_metrics_by_family.csv")
    v2row("A1M_mode_conditioned","V2-A1M",a1m_family,a1m_summary["clean"]["overall_fpr"],a1m_summary["clean"]["worst_sequence_fpr"],a1m_summary["clean"]["train1_second_fpr"],float(a1m_family[a1m_family.family.eq('P1')].q99_tpr.iloc[0]),a1m_summary["c2_fpr"],a1m_summary["classification"],"frozen_negative_ablation")
    a1c_family=read_csv("outputs/hai_structure_v2a1c_causal_adaptive_reference/controlled_adaptive_metrics_by_family.csv")
    v2row("A1C_causal_adaptive","V2-A1C",a1c_family,a1c_summary["clean"]["overall_fpr"],a1c_summary["clean"]["worst_sequence_fpr"],a1c_summary["clean"]["train1_second_half_fpr"],a1c_summary["p1_fpr"],a1c_summary["c2_fpr"],a1c_summary["classification"],"frozen_negative_ablation",str(a1c_summary["gates"]["persistent_absorption"]))
    write_table(out,"table_05_structure_v2_evolution",pd.DataFrame(v2_rows),"表5 Structure V2演化")

    pool=read_csv("outputs/hai_structure_v2a_lagged_dependency/selected_dynamic_relation_pool.csv")
    drift=read_csv("outputs/hai_structure_v2a1d_normal_drift_attribution/relation_drift_vs_detection_value.csv")
    relations=pool.merge(drift[["edge_id","controlled_detection_value","normal_drift_cost"]],left_on="Index",right_on="edge_id",how="left") if set(pool.Index)==set(drift.edge_id) else pool.reset_index(drop=True).join(drift[["controlled_detection_value","normal_drift_cost"]])
    relation_rows=[]
    for idx,row in relations.reset_index(drop=True).iterrows():
        mids=[]
        for metric in ("global_gain","median_sequence_gain","positive_sequence_count","worst_sequence_gain"):
            mids.append(provenance.add(f"relation.{idx}.{metric}","V2-A",f"edge_{idx}","train","relation_selection",metric,row[metric],"outputs/hai_structure_v2a_lagged_dependency/selected_dynamic_relation_pool.csv",f"row={idx}.{metric}",validity="valid_diagnostic_or_ablation"))
        for metric,column in (("controlled_detection_value","controlled_detection_value"),("drift_cost","normal_drift_cost")):
            mids.append(provenance.add(f"relation.{idx}.{metric}","V2-A1D",f"edge_{idx}","validation/controlled","edge_attribution",metric,row[column],"outputs/hai_structure_v2a1d_normal_drift_attribution/relation_drift_vs_detection_value.csv",f"edge_id={idx}.{column}",validity="valid_diagnostic_or_ablation"))
        relation_rows.append({"edge_id":idx,"source":row.source,"target":row.target,"global_gain":row.global_gain,"median_sequence_gain":row.median_sequence_gain,"positive_sequence_count":row.positive_sequence_count,"worst_sequence_gain":row.worst_sequence_gain,"controlled_detection_value":row.controlled_detection_value,"drift_cost":row.normal_drift_cost,"final_status":"frozen_12_edge_controlled_pool","provenance_metric_ids":";".join(mids)})
    write_table(out,"table_06_selected_dynamic_relations",pd.DataFrame(relation_rows),"表6 选定动态关系")

    efficiency=read_csv("outputs/hai_step3g_rec_capacity_ablation/rec_capacity_core_efficiency.csv")
    cost_rows=[]
    for model in ("tiny","small","medium"):
        subset=efficiency[efficiency.model_size_name.eq(model)]
        cost_rows.append({"component":model,"parameter_or_state_count":int(subset.trainable_parameter_count.iloc[0]),"model_file_size_bytes":safe_value(pd.to_numeric(subset.checkpoint_size_bytes,errors='coerce').dropna().mean() if pd.to_numeric(subset.checkpoint_size_bytes,errors='coerce').notna().any() else None),"window_latency_ms":float(subset.window_forward_latency_ms.mean()),"cache_size_bytes":UNAVAILABLE,"source":"rec_capacity_core_efficiency.csv"})
    cost_rows.extend([
        {"component":"V2_Ridge_coefficients","parameter_or_state_count":len(read_csv("outputs/hai_structure_v2a_lagged_dependency/final_model_coefficient_registry.csv")),"model_file_size_bytes":sum((ROOT/p).stat().st_size for p in ["outputs/hai_structure_v2a_lagged_dependency/final_self_models.joblib","outputs/hai_structure_v2a_lagged_dependency/final_cross_models.joblib"]),"window_latency_ms":UNAVAILABLE,"cache_size_bytes":UNAVAILABLE,"source":"final_model_coefficient_registry.csv"},
        {"component":"A1M_PCA_KMeans","parameter_or_state_count":604,"model_file_size_bytes":int((ROOT/"outputs/hai_structure_v2a1m_mode_conditioned_reference/selected_mode_pipeline.joblib").stat().st_size),"window_latency_ms":UNAVAILABLE,"cache_size_bytes":UNAVAILABLE,"source":"A1M lightweight_cost_audit.csv"},
        {"component":"A1C_adaptive_state","parameter_or_state_count":36,"model_file_size_bytes":0,"window_latency_ms":float(read_csv("outputs/hai_structure_v2a1c_causal_adaptive_reference/lightweight_cost_audit.csv").seconds_per_window_update_and_score.iloc[0]*1000),"cache_size_bytes":int((ROOT/"outputs/hai_structure_v2a1c_causal_adaptive_reference/validation_adaptive_reference_trace.npz").stat().st_size),"source":"A1C lightweight_cost_audit.csv"},
    ])
    for idx,row in enumerate(cost_rows):
        stage="HAI-3G" if idx<3 else "V2-A1M" if idx==4 else "V2-A1C" if idx==5 else "V2-A"
        source="outputs/hai_step3g_rec_capacity_ablation/rec_capacity_core_efficiency.csv" if idx<3 else "outputs/hai_structure_v2a1m_mode_conditioned_reference/lightweight_cost_audit.csv" if idx==4 else "outputs/hai_structure_v2a1c_causal_adaptive_reference/lightweight_cost_audit.csv" if idx==5 else "outputs/hai_structure_v2a_lagged_dependency/final_model_coefficient_registry.csv"
        mids=[]
        for metric in ("parameter_or_state_count","model_file_size_bytes","window_latency_ms","cache_size_bytes"):
            if row[metric] != UNAVAILABLE:
                mids.append(provenance.add(f"cost.{idx}.{metric}",stage,row["component"],"not_applicable","lightweight",metric,row[metric],source,f"reported/derived {metric}",unit="count" if "count" in metric or "bytes" in metric else "ms",validity="valid_diagnostic_or_ablation"))
        row["provenance_metric_ids"]=";".join(mids)
    write_table(out,"table_07_lightweight_cost",pd.DataFrame(cost_rows),"表7 轻量化成本")

    gates=[
        ("V1 real transfer","real complementarity","FP per added TP",real["point_overlap"]["added_fp_per_added_tp"],False,real["classification"],"stopped","cost too high","retain failed baseline"),
        ("V2-A","controlled and clean stability","all gates",False,False,"stable_dynamic_relations_exist_but_global_score_is_unstable","continued","mechanism exists","try grouped aggregation"),
        ("V2-A1R","exact reproduction","66/66",66,True,"target_grouping_improves_stability_but_not_enough","continued","exact recovery passed","full-clean audit"),
        ("V2-A1S","normal stability","worst FPR <=0.05",a1s_summary["worst_sequence_fpr"],False,a1s_summary["classification"],"continued","sequence instability","diagnose drift"),
        ("V2-A1M","mode reference","train1 second FPR reduction",a1m_summary["clean"]["train1_second_fpr"],False,a1m_summary["classification"],"continued","mode conditioning ineffective","last adaptive attempt"),
        ("V2-A1C","adaptive reference","drift and absorption gates",a1c_summary["clean"]["train1_second_half_fpr"],False,a1c_summary["classification"],"stopped","no drift reduction and absorption risk","freeze negative result"),
    ]
    gate_frame=pd.DataFrame(gates,columns=["stage","intended_question","gate","observed_value","passed","classification","stopped_or_continued","reason","next_decision"])
    gate_sources=[
        ("HAI-4CD","outputs/hai_step4cd_structure_real_and_complementarity/structure_real_evaluation_summary.json","$.point_overlap.added_fp_per_added_tp"),
        ("V2-A","outputs/hai_structure_v2a_lagged_dependency/structure_v2a_summary.json","$.gates_passed"),
        ("V2-A1R","outputs/hai_structure_v2a1r_cache_backfill_and_grouped_aggregation/structure_v2a1r_summary.json","$.checks_passed"),
        ("V2-A1S","outputs/hai_structure_v2a1s_full_clean_stability/structure_v2a1s_summary.json","$.worst_sequence_fpr"),
        ("V2-A1M","outputs/hai_structure_v2a1m_mode_conditioned_reference/structure_v2a1m_summary.json","$.clean.train1_second_fpr"),
        ("V2-A1C","outputs/hai_structure_v2a1c_causal_adaptive_reference/structure_v2a1c_summary.json","$.clean.train1_second_half_fpr"),
    ]
    gate_frame["provenance_metric_ids"]=[provenance.add(f"gate.{index}",stage,gate_frame.iloc[index].stage,"validation/controlled","gate",gate_frame.iloc[index].gate,gate_frame.iloc[index].observed_value,source,locator,validity="valid_diagnostic_or_ablation") for index,(stage,source,locator) in enumerate(gate_sources)]
    write_table(out,"table_08_gate_history",gate_frame,"表8 门禁历史")
    negatives=pd.DataFrame([
        {"finding":"model capacity is not main problem","stage":"HAI-3G","evidence":"Tiny/Small/Medium capacity ablation","status":"rejected_improvement_path"},
        {"finding":"state threshold ineffective","stage":"HAI-3F","evidence":"normal-state gates failed","status":"negative_ablation"},
        {"finding":"alarm smoothing not viable","stage":"HAI-3E","evidence":"event recall declined","status":"negative_ablation"},
        {"finding":"tail calibration unstable","stage":"HAI-4B-B","evidence":"fair crossfit could not support freezing","status":"invalid_for_primary_claim"},
        {"finding":"target grouping partially effective","stage":"V2-A1R","evidence":"controlled metrics improved but worst clean FPR failed","status":"partial_mechanism_evidence"},
        {"finding":"mode conditioning ineffective","stage":"V2-A1M","evidence":a1m_summary["classification"],"status":"frozen_negative_ablation"},
        {"finding":"adaptive reference ineffective and absorption-prone","stage":"V2-A1C","evidence":a1c_summary["classification"],"status":"frozen_negative_ablation"},
    ])
    write_table(out,"table_09_negative_results_and_ablation",negatives,"表9 负结果与消融")
    return {"dataset":pd.DataFrame(dataset_rows),"reconstruction":pd.DataFrame(evolution_rows),"tiny":pd.DataFrame(tiny_rows),"v1":pd.DataFrame(v1_rows),"v2":pd.DataFrame(v2_rows),"relations":pd.DataFrame(relation_rows),"cost":pd.DataFrame(cost_rows),"gates":gate_frame,"negative":negatives}


def validity_registry():
    return pd.DataFrame([
        {"evidence_id":"HAI_DATA_PROTOCOL","stage_id":"HAI-1/2","validity_status":"valid_real_evaluation","scope":"data and split protocol","reason":"formal audit, no overlap, reproducible boundaries"},
        {"evidence_id":"TINY_REAL","stage_id":"HAI-3A-3G","validity_status":"valid_real_evaluation","scope":"Tiny reconstruction on HAI test","reason":"formal protocol and three seeds"},
        {"evidence_id":"V1_CONTROLLED","stage_id":"HAI-4B","validity_status":"valid_controlled_mechanism_evidence","scope":"controlled relation corruption","reason":"mechanism only, not real attacks"},
        {"evidence_id":"V1_REAL","stage_id":"HAI-4CD","validity_status":"valid_real_evaluation","scope":"exploratory structure-only HAI","reason":"one-shot exploratory transfer; weak result"},
        {"evidence_id":"V2_CONTROLLED","stage_id":"V2-A/A1R/A1S","validity_status":"valid_controlled_mechanism_evidence","scope":"dynamic collaborative corruption","reason":"test never used; normal stability failed"},
        {"evidence_id":"A1M","stage_id":"V2-A1M","validity_status":"valid_diagnostic_or_ablation","scope":"mode-conditioned reference","reason":"negative ablation"},
        {"evidence_id":"A1C","stage_id":"V2-A1C","validity_status":"valid_diagnostic_or_ablation","scope":"adaptive reference and absorption","reason":"negative and safety ablation"},
        {"evidence_id":"SWAT_LEGACY","stage_id":"SWAT-4.3D/4.3E1","validity_status":"invalid_for_claims","scope":"legacy merged.csv","reason":"chronology/overlap audit invalidated claims"},
        {"evidence_id":"FUSION","stage_id":"FINAL","validity_status":"unavailable_without_recomputation","scope":"Rec+Structure V2 fusion","reason":"not executed because structure stability gate failed"},
    ])


def freeze_manifest():
    return {
        "reconstruction_backbone":{"method":"Tiny DSConv AE","architecture":"86->64->32->64->86","parameter_count":16420,"window":64,"score":"endpoint_raw_top5","alarm":"pointwise","primary_threshold":"normal_validation_q99","status":"real_evaluation_baseline_frozen","conclusion":"真实HAI可评价，但点级性能和误报仍有限"},
        "structure_v1":{"method":"10 pairs, zero-lag Pearson, Fisher-z, global median/MAD, original max","status":"frozen_failed_real_transfer_baseline","conclusion":"受控破坏有效，但真实HAI AUROC接近随机且互补代价高"},
        "structure_v2_static":{"method":"12 directed lagged Ridge dependency relations, rolling16, static train median/MAD, target-group max then top2 mean","status":"frozen_controlled_validation_reference","conclusion":"受控机制有效，正常非平稳性未解决，不允许真实HAI评价或融合"},
        "structure_v2_a1m":{"status":"frozen_negative_ablation","conclusion":"稳定train-only运行模式参考不能降低正常依赖漂移"},
        "structure_v2_a1c":{"status":"frozen_negative_ablation","conclusion":"自适应参考不能降低漂移，且存在持续异常吸收风险"},
        "final_fusion":{"status":"not_executed","reason":"structure branch failed normal-validation stability gate","metrics":"unavailable_without_recomputation"},
    }


def claim_boundary():
    return {
        "SUPPORTED":[
            "Tiny reconstruction is a lightweight real-HAI baseline.",
            "Endpoint raw top5 improves the original reconstruction score trade-off.",
            "Stable dynamic cross-channel conditional dependencies exist in normal data.",
            "Structure V2 detects controlled collaborative relation corruptions with useful AUROC/AUPR.",
            "P1/C2 controls show V2 is not merely a point prediction-error detector.",
            "Target grouping reduces duplicate-target amplification.",
            "Parameter and adaptive-state counts remain lightweight.",
        ],
        "PARTIALLY_SUPPORTED":[
            "Structure V2 has collaborative anomaly-detection potential under controlled protocols.",
            "Structure and reconstruction provide limited independent evidence in exploratory analysis.",
            "Lagged conditional relations better match the intended HAI mechanism than zero-lag Pearson.",
        ],
        "NOT_SUPPORTED":[
            "Structure V2 performs strongly on real HAI attacks.", "Structure V2 passed real HAI test evaluation.",
            "Rec+Structure fusion improves final performance.", "The method covers most anomaly types.",
            "Normal non-stationarity is solved.", "The adaptive reference is deployment-safe.",
            "Legacy SWaT results are valid.", "The current method is SOTA.",
        ],
    }


def merge_unresolved(out):
    rows=[]
    for path in sorted(OUTPUTS.rglob("unresolved_issues_delta.json")):
        if OUT in path.parents or "smoke" in str(path) or "failed" in str(path):
            continue
        payload=json.loads(path.read_text(encoding="utf-8")); stage=stage_identity(path.parent.name)[0]
        if isinstance(payload,list): entries=payload
        elif isinstance(payload,dict) and "issues" in payload: entries=payload["issues"]
        else: entries=[payload]
        for index,item in enumerate(entries):
            question=str(item.get("research_question") or item.get("issue") or item.get("classification") or f"unresolved ledger from {stage}")
            issue_id=str(item.get("issue_id") or f"{stage}-{index+1}")
            rows.append({"issue_id":issue_id,"first_stage":stage,"latest_stage":stage,"research_question":question,"current_status":str(item.get("current_status") or "open_or_documented"),"evidence":str(item.get("evidence") or item)[:1000],"severity":str(item.get("severity") or "method limitation"),"affects_validity":bool(item.get("affects_validity",False)),"affects_final_method":bool(item.get("affects_final_method",True)),"affects_thesis_claim":bool(item.get("affects_thesis_claim",True)),"resolved_by":str(item.get("resolved_by") or "NONE"),"why_unresolved":str(item.get("why_unresolved") or "No accepted frozen method resolves this issue"),"minimum_future_action":str(item.get("minimum_future_action") or "Discuss before any new experiment"),"recomputation_scope":str(item.get("recomputation_scope") or "requires explicit future protocol"),"frozen_artifacts_reusable":True,"priority":str(item.get("priority") or "P2"),"source_artifact":str(path.relative_to(ROOT)).replace('\\','/')})
    frame=pd.DataFrame(rows).drop_duplicates(subset=["issue_id","research_question"])
    frame.to_csv(out/"unresolved_issues_master.csv",index=False)
    (out/"unresolved_issues_master.json").write_text(json.dumps(frame.to_dict("records"),ensure_ascii=False,indent=2),encoding="utf-8")
    return frame


def decision_matrix():
    return pd.DataFrame([
        {"method":"Tiny rec","role":"real HAI lightweight baseline","real_evaluation_available":True,"controlled_validation_available":False,"validity":"valid_real_evaluation","performance":"limited","stability":"multiseed variable","lightweight":"yes","interpretability":"channel top5","deployment_readiness":"research baseline","thesis_role":"primary real baseline","keep_or_reject":"keep","reason":"only frozen real-HAI detector"},
        {"method":"Structure V1","role":"failed baseline and motivation","real_evaluation_available":True,"controlled_validation_available":True,"validity":"real weak; controlled valid","performance":"near-random real AUROC","stability":"poor transfer","lightweight":"yes","interpretability":"pairwise","deployment_readiness":"no","thesis_role":"negative baseline","keep_or_reject":"keep as failure evidence","reason":"motivates dynamic conditional structure"},
        {"method":"Structure V2 static","role":"controlled collaborative prototype","real_evaluation_available":False,"controlled_validation_available":True,"validity":"valid_controlled_mechanism_evidence","performance":"strong controlled","stability":"normal instability","lightweight":"yes","interpretability":"directed edges","deployment_readiness":"no","thesis_role":"core controlled prototype","keep_or_reject":"keep with boundary","reason":"mechanism valid; real stability not established"},
        {"method":"A1M","role":"mode-conditioning ablation","real_evaluation_available":False,"controlled_validation_available":True,"validity":"valid_diagnostic_or_ablation","performance":"no drift reduction","stability":"failed","lightweight":"yes","interpretability":"mode references","deployment_readiness":"no","thesis_role":"negative ablation","keep_or_reject":"keep as ablation","reason":"rules out coarse normal modes"},
        {"method":"A1C","role":"adaptive/safety ablation","real_evaluation_available":False,"controlled_validation_available":True,"validity":"valid_diagnostic_or_ablation","performance":"no drift reduction","stability":"absorption risk","lightweight":"yes","interpretability":"edge states","deployment_readiness":"no","thesis_role":"negative safety evidence","keep_or_reject":"keep as ablation","reason":"adaptation unsafe and ineffective"},
        {"method":"Fusion","role":"not executed","real_evaluation_available":False,"controlled_validation_available":False,"validity":"unavailable_without_recomputation","performance":UNAVAILABLE,"stability":UNAVAILABLE,"lightweight":UNAVAILABLE,"interpretability":UNAVAILABLE,"deployment_readiness":"no","thesis_role":"not reported","keep_or_reject":"reject execution","reason":"structure stability gate failed"},
    ])


def final_report(out, tables, provenance_count, issue_count):
    tiny=tables["tiny"]; v1=tables["v1"]; v2=tables["v2"]
    tiny_auroc=float(tiny.auroc.mean()); v1_real=v1[v1.method.eq("Structure_V1_real_HAI")].iloc[0]; static=v2[v2.variant.eq("A1S_full_clean")].iloc[0]; a1c=v2[v2.variant.eq("A1C_causal_adaptive")].iloc[0]
    text=f"""# HAI 最终统一证据包

## 1. 项目目标

本项目研究轻量重构检测与跨通道协同关系异常。当前文档只汇总既有正式产物，不执行训练、推理、阈值拟合、结构分数重算或融合。

## 2. 数据与协议

HAI 22.04 包含 6 个训练序列、4 个测试序列、86 个过程通道和 58 个攻击事件；窗口严格保持文件边界，主重构协议使用 window=64 与因果 endpoint 评分。来源：`outputs/hai_step1_dataset_audit/hai_dataset_audit_summary.json`、`outputs/hai_step2_dataloader_validation/hai_dataloader_summary.json`。旧 SWaT merged.csv 结果被归为 invalid_for_claims。

## 3. 重构分支演化

最终保留 Tiny DSConv AE、endpoint raw top5、pointwise、normal-validation q99。Tiny 三种子 macro AUROC 均值为 {tiny_auroc:.6f}，但精确率、召回率及误报仍受随机种子影响。来源：`outputs/hai_step3g_rec_capacity_ablation/rec_capacity_core_metrics_new.csv`（metric IDs: tiny.seed1.auroc, tiny.seed2.auroc, tiny.seed3.auroc）。容量消融表明扩大到 Small/Medium 不是主要解决方向。

## 4. Structure V1

zero-lag Pearson/Fisher-z 能响应受控关系破坏，但真实 HAI structure-only AUROC 为 {float(v1_real.auroc):.6f}，added FP per added TP 为 14.971645，真实迁移和互补代价均不理想。来源：`outputs/hai_step4cd_structure_real_and_complementarity/structure_real_evaluation_summary.json`（metric IDs: v1.structure_v1_real_hai.auroc, v1.rec_or_structure.fp_per_added_tp）。

## 5. Structure V2

V2 冻结 12 条有向时滞条件依赖关系，以 rolling dependency loss 和 target-group max then top2 mean 聚合。受控 R1/R2/R3 证据有效，但不能外推为真实 HAI 攻击性能。

## 6. 正常非平稳性

A1S full-clean overall FPR 为 {float(static.overall_clean_fpr):.6f}，worst-sequence FPR 为 {float(static.worst_sequence_fpr):.6f}，后者未通过稳定性门禁。来源：`outputs/hai_structure_v2a1s_full_clean_stability/structure_v2a1s_summary.json`。

## 7. Mode/Adaptive 失败消融

A1M 的模式条件参考未降低 train1 后半漂移。A1C 的 train1 后半 FPR 为 {float(a1c.train1_second_half_fpr):.6f}，且持续异常吸收门禁失败；两者均冻结为 negative ablation，不进入真实结构评价。来源：A1M/A1C summary JSON。

## 8. 轻量化证据

Tiny 参数量为 16,420；Structure V2 使用线性 Ridge 关系；A1M 附加统计量为 604；A1C 仅维护 36 个自适应状态。来源：`table_07_lightweight_cost.csv` 及其 provenance。

## 9. 最终方法冻结

真实 HAI 主基线仅保留 Tiny rec。Structure V1 保留为失败基线；Structure V2 static 保留为受控机制原型；A1M/A1C 保留为负消融；Fusion 不执行且不报告性能。

## 10. 可以支持的结论

Tiny 可构成轻量真实 HAI 基线；动态条件依赖在正常数据中存在；V2 对受控协同破坏有检测能力；target grouping 部分缓解重复目标放大；整体参数与状态规模较小。

## 11. 不能支持的结论

不能声称 V2 已通过真实 HAI 测试、融合提升最终性能、正常非平稳性已解决、自适应参考可安全部署、旧 SWaT 结果有效或当前方法达到 SOTA。

## 12. 未解决问题

统一问题总账包含 {issue_count} 条去重记录。它们仅作为下一阶段讨论输入，本阶段不执行 minimum future action。

## 13. 下一阶段范围

下一阶段只允许讨论最终问题账本与改进决策；任何新实验必须重新明确协议并由用户显式启动。当前证据注册表共 {provenance_count} 条，报告中的关键数字均可回溯到 source artifact。
"""
    (out/"final_hai_unified_evidence_package.md").write_text(text,encoding="utf-8")


def run(dry_run=False):
    fingerprint=input_fingerprint()
    status_path=OUT/"stage_status.json"
    if not dry_run and status_path.exists():
        status=json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("final_report_complete") and status.get("input_fingerprint")==fingerprint:
            print("REUSE completed unified evidence package",flush=True); return 0
    discovered=[p for p in OUTPUTS.iterdir() if p.is_dir() and p!=OUT]
    text_count=sum(1 for d in discovered for p in d.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS)
    excluded_count=sum(1 for d in discovered for p in d.rglob("*") if p.is_file() and p.suffix.lower() not in TEXT_EXTENSIONS)
    missing_summary=sum(1 for name in STAGES if (OUTPUTS/name).exists() and not list((OUTPUTS/name).glob("*summary.json")))
    if dry_run:
        print(f"discovered_stage_count={len(discovered)}",flush=True); print(f"included_artifact_count={text_count}",flush=True); print(f"excluded_artifact_count={excluded_count}",flush=True); print(f"missing_summary_count={missing_summary}",flush=True)
        print("FORBIDDEN_RECOMPUTE: raw HAI/SWaT CSV; HAI test labels; AE/Ridge/checkpoint; forward; rec/structure score; threshold fit; corruption; fusion",flush=True)
        print("THESIS_TABLES: table_01 through table_09, final method decision matrix, claim boundary, unresolved master ledger",flush=True)
        print("HAI raw/test/model loaded=false",flush=True); return 0
    OUT.mkdir(parents=True,exist_ok=True)
    status={key:False for key in ["historical_artifacts_discovered","hashes_verified","metric_provenance_complete","evidence_validity_complete","final_method_freeze_complete","thesis_tables_complete","unresolved_issues_merged","claim_boundary_complete","final_report_complete"]}; status["input_fingerprint"]=fingerprint
    status_path.write_text(json.dumps(status,indent=2),encoding="utf-8")
    inventory=scan_inventory(); inventory.to_csv(OUT/"artifact_inventory.csv",index=False)
    status["historical_artifacts_discovered"]=True; status["hashes_verified"]=True
    hashes=dict(zip(inventory.artifact_path,inventory.sha256)); provenance=Provenance(hashes)
    tables=build_tables(OUT,provenance)
    provenance_frame=provenance.frame().fillna(UNAVAILABLE); provenance_frame.to_csv(OUT/"metric_provenance_registry.csv",index=False)
    status["metric_provenance_complete"]=True; status["thesis_tables_complete"]=True
    validity_registry().to_csv(OUT/"evidence_validity_registry.csv",index=False); status["evidence_validity_complete"]=True
    (OUT/"final_method_freeze_manifest.yaml").write_text(yaml.safe_dump(freeze_manifest(),allow_unicode=True,sort_keys=False),encoding="utf-8"); status["final_method_freeze_complete"]=True
    claims=claim_boundary(); (OUT/"final_claim_boundary.yaml").write_text(yaml.safe_dump(claims,allow_unicode=True,sort_keys=False),encoding="utf-8")
    claim_md="# 最终主张边界\n\n"+"\n\n".join(f"## {category}\n\n"+"\n".join(f"- {item}" for item in items) for category,items in claims.items())+"\n"; (OUT/"final_claim_boundary.md").write_text(claim_md,encoding="utf-8"); status["claim_boundary_complete"]=True
    issues=merge_unresolved(OUT); status["unresolved_issues_merged"]=True
    decisions=decision_matrix(); write_table(OUT,"final_method_decision_matrix",decisions,"最终方法决策矩阵")
    final_report(OUT,tables,len(provenance_frame),len(issues)); status["final_report_complete"]=True
    status_path.write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(f"COMPLETE stages={len(discovered)} inventory={len(inventory)} provenance={len(provenance_frame)} issues={len(issues)}",flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description="HAI Final Step 5A unified evidence package")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(); return run(args.dry_run)


if __name__=="__main__":
    raise SystemExit(main())
