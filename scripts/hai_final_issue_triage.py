from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "hai_final_unified_evidence_package"
OUT = ROOT / "outputs" / "hai_final_issue_triage_and_improvement_decision"
VERSION = "1.0"

REQUIRED_INPUTS = [
    "unresolved_issues_master.csv", "unresolved_issues_master.json", "final_method_freeze_manifest.yaml",
    "final_claim_boundary.yaml", "final_method_decision_matrix.csv", "evidence_validity_registry.csv",
    "metric_provenance_registry.csv", "table_01_dataset_and_protocol.csv",
    "table_02_reconstruction_evolution.csv", "table_04_structure_v1_controlled_and_real.csv",
    "table_05_structure_v2_evolution.csv", "table_08_gate_history.csv",
    "table_09_negative_results_and_ablation.csv", "final_hai_unified_evidence_package.md",
]

DECISIONS = {
    "V2-A1C-1": {
        "problem": "因果自适应reference未降低train1漂移，且会吸收持续协同异常。",
        "severity": "P1", "action_class": "retain_as_explicit_limitation",
        "why": "A1C已是结构分支最后一次预注册改进；继续调参会提高validation过拟合风险。",
        "effect_results": "不推翻受控机制证据，但阻止Structure V2真实HAI评价、融合和部署主张。",
        "effect_thesis": "核心结构贡献必须限定为受控机制原型，自适应版本作为安全性负结果。",
        "minimum_action": "无需新实验；在方法限制、负结果和结论中报告漂移及吸收门禁失败。",
        "success_gate": "论文所有Structure V2表述均包含controlled-only和not deployment-ready限定。",
        "fallback": "若无法维持限定，删除Structure V2真实检测器相关主张。",
        "claim": "Structure V2 real/deployment claims", "method": "Structure V2/A1C",
    },
    "V2A1D-001": {
        "problem": "train1正常误报究竟是局部关系漂移还是运行模式级变化。",
        "severity": "P2", "action_class": "already_resolved_or_superseded",
        "why": "A1D完成归因，A1M进一步证明train-only稳定模式覆盖存在但不能降低漂移。",
        "effect_results": "形成失败原因解释，不再构成待运行实验。",
        "effect_thesis": "用于解释正常非平稳性，而非支持模式条件方法。",
        "minimum_action": "引用A1D与A1M负结果并关闭该实验方向。",
        "success_gate": "归因章节同时引用A1D和A1M artifact。",
        "fallback": "保留为未来更细工况建模问题。",
        "claim": "normal dependency drift explanation", "method": "A1D/A1M",
    },
    "V2A1D-002": {
        "problem": "缺少可复用的通道、残差和dependency-loss离线缓存。",
        "severity": "P3", "action_class": "already_resolved_or_superseded",
        "why": "A1M已持久化train/validation/controlled raw dependency cache及self/cross residual。",
        "effect_results": "后续A1M/A1C已基于缓存完成，不影响现有证据有效性。",
        "effect_thesis": "仅影响工程复用说明。",
        "minimum_action": "在复现清单中指向A1M三类raw cache。",
        "success_gate": "缓存路径和hash进入复现附录。",
        "fallback": "标注逐通道原始值不属于最终证据包。",
        "claim": "reproducibility", "method": "A1M cache",
    },
    "V2A1M-001": {
        "problem": "validation正常模式是否被normal train覆盖。",
        "severity": "P2", "action_class": "already_resolved_or_superseded",
        "why": "A1M显示overall OOD很低且train1后半OOD为0，但模式条件reference仍无效。",
        "effect_results": "排除模式未覆盖作为主要失败原因。",
        "effect_thesis": "支持将A1M写为负消融。",
        "minimum_action": "报告OOD审计与represented_modes_do_not_reduce_drift分类。",
        "success_gate": "不得把低OOD误写为检测性能提升。",
        "fallback": "保留为模式粒度可能过粗的局限性。",
        "claim": "mode coverage", "method": "A1M",
    },
    "V2A1R-FIX-001": {
        "problem": "精确scale恢复是否使target-grouped聚合达到稳定门禁。",
        "severity": "P2", "action_class": "already_resolved_or_superseded",
        "why": "A1R完成66/66精确复现；分组改善受控指标，但worst clean FPR仍失败。",
        "effect_results": "确认分组部分有效，不足以冻结真实检测器。",
        "effect_thesis": "可作为聚合消融和缓存恢复审计。",
        "minimum_action": "保留exact reproduction与partial improvement结论。",
        "success_gate": "论文不把partial improvement表述为稳定性通过。",
        "fallback": "删除分组优越性强主张，仅报告诊断。",
        "claim": "target grouping", "method": "V2-A1R",
    },
    "V2A1S-001": {
        "problem": "target-grouped结构分数在完整normal validation上是否稳定。",
        "severity": "P1", "action_class": "retain_as_explicit_limitation",
        "why": "A1S已确认sequence-specific instability；A1M和A1C均未解决，继续修改会造成反复适配validation。",
        "effect_results": "不影响受控协同破坏证据，但禁止真实HAI结构性能与融合主张。",
        "effect_thesis": "必须明确结构方法的真实迁移和部署边界。",
        "minimum_action": "在摘要、实验和结论中区分controlled mechanism与real detector。",
        "success_gate": "所有真实性能表仅使用Tiny或明确标为V1 exploratory结果。",
        "fallback": "将Structure V2降为未来工作，不作为论文主体方法。",
        "claim": "Structure V2 stability", "method": "V2-A1S",
    },
    "V2A-001": {
        "problem": "动态结构分数是否同时具备特异性和正常稳定性。",
        "severity": "P2", "action_class": "already_resolved_or_superseded",
        "why": "A1R确认分组作用，A1S确认不稳定，A1D/A1M/A1C解释并验证改进失败。",
        "effect_results": "问题已得到分解答案：受控特异性存在，正常稳定性不足。",
        "effect_thesis": "形成Structure V2的最终双重边界。",
        "minimum_action": "引用完整V2-A至A1C证据链，不再新增聚合或reference。",
        "success_gate": "结论同时报告机制有效与稳定性失败。",
        "fallback": "仅保留V2为受控实验。",
        "claim": "dynamic structure specificity and stability", "method": "Structure V2",
    },
}

IMPACT_FIELDS = [
    "affects_data_validity", "affects_metric_validity", "affects_method_validity",
    "affects_real_world_claim", "affects_controlled_claim", "affects_lightweight_claim",
    "affects_novelty_claim", "affects_thesis_completeness", "affects_deployment_claim",
    "affects_reproducibility",
]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint():
    return hashlib.sha256(json.dumps([VERSION, [(name, sha256(INPUT / name)) for name in REQUIRED_INPUTS]], separators=(",", ":")).encode()).hexdigest()


def markdown_table(frame: pd.DataFrame, title: str):
    values = frame.astype(str)
    lines = [f"# {title}", "", "| " + " | ".join(values.columns) + " |", "| " + " | ".join(["---"] * len(values.columns)) + " |"]
    lines.extend("| " + " | ".join(value.replace("|", "\\|") for value in values.iloc[index]) + " |" for index in range(len(values)))
    return "\n".join(lines) + "\n"


def write_table(stem: str, frame: pd.DataFrame, title: str):
    frame.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(markdown_table(frame, title), encoding="utf-8")


def load_inputs():
    missing = [name for name in REQUIRED_INPUTS if not (INPUT / name).exists()]
    if missing:
        raise RuntimeError(f"invalid_issue_triage_input missing={missing}")
    issues = pd.read_csv(INPUT / "unresolved_issues_master.csv")
    issue_json = json.loads((INPUT / "unresolved_issues_master.json").read_text(encoding="utf-8"))
    freeze = yaml.safe_load((INPUT / "final_method_freeze_manifest.yaml").read_text(encoding="utf-8"))
    claims = yaml.safe_load((INPUT / "final_claim_boundary.yaml").read_text(encoding="utf-8"))
    provenance = pd.read_csv(INPUT / "metric_provenance_registry.csv")
    inventory = pd.read_csv(INPUT / "artifact_inventory.csv")
    return issues, issue_json, freeze, claims, provenance, inventory


def input_audit(issues, issue_json, provenance, inventory):
    rows = []
    inventory_paths = set(inventory.artifact_path.astype(str))
    provenance_stages = set(provenance.stage_id.astype(str))
    for issue in issues.itertuples():
        source_exists = str(issue.source_artifact) in inventory_paths and (ROOT / str(issue.source_artifact)).exists()
        stage_tokens = {str(issue.first_stage), str(issue.latest_stage)}
        provenance_exists = bool(stage_tokens & provenance_stages)
        evidence_exists = isinstance(issue.evidence, str) and bool(issue.evidence.strip())
        claim_affected = bool(issue.affects_thesis_claim) or DECISIONS[issue.issue_id]["claim"] != "NONE"
        method_affected = bool(issue.affects_final_method) or DECISIONS[issue.issue_id]["method"] != "NONE"
        complete = source_exists and evidence_exists and provenance_exists and claim_affected and method_affected
        rows.append({"issue_id": issue.issue_id, "source_exists": source_exists, "evidence_exists": evidence_exists,
                     "provenance_exists": provenance_exists, "claim_affected": claim_affected,
                     "method_affected": method_affected, "input_complete": complete,
                     "audit_note": "complete" if complete else "invalid_issue_triage_input"})
    frame = pd.DataFrame(rows)
    json_ids = {str(row["issue_id"]) for row in issue_json}
    if len(issues) != 7 or issues.issue_id.duplicated().any() or set(issues.issue_id) != json_ids or set(issues.issue_id) != set(DECISIONS):
        frame["input_complete"] = False
        frame["audit_note"] = "invalid_issue_triage_input count/id mismatch"
    return frame


def severity_frame(issues):
    rows = []
    for issue in issues.itertuples():
        decision = DECISIONS[issue.issue_id]
        impacts = {field: "none" for field in IMPACT_FIELDS}
        if issue.issue_id == "V2-A1C-1":
            impacts.update(affects_method_validity="high", affects_real_world_claim="critical", affects_controlled_claim="low", affects_novelty_claim="medium", affects_thesis_completeness="high", affects_deployment_claim="critical")
        elif issue.issue_id == "V2A1S-001":
            impacts.update(affects_metric_validity="medium", affects_method_validity="high", affects_real_world_claim="critical", affects_controlled_claim="low", affects_novelty_claim="medium", affects_thesis_completeness="high", affects_deployment_claim="critical")
        elif issue.issue_id == "V2A1D-002":
            impacts.update(affects_reproducibility="low", affects_thesis_completeness="low")
        else:
            impacts.update(affects_method_validity="medium", affects_real_world_claim="medium", affects_thesis_completeness="medium", affects_reproducibility="low")
        rows.append({"issue_id": issue.issue_id, "severity": decision["severity"], "action_class": decision["action_class"], **impacts,
                     "evidence_explanation": decision["why"][:100], "source_artifact": issue.source_artifact})
    return pd.DataFrame(rows)


def decision_frame(issues):
    rows = []
    for issue in issues.itertuples():
        d = DECISIONS[issue.issue_id]
        rows.append({"issue_id": issue.issue_id, "problem": d["problem"], "evidence": str(issue.evidence)[:500],
                     "severity": d["severity"], "action_class": d["action_class"],
                     "must_fix": d["action_class"] == "must_resolve_before_thesis", "why": d["why"],
                     "effect_on_current_results": d["effect_results"], "effect_on_thesis": d["effect_thesis"],
                     "minimum_action": d["minimum_action"], "success_gate": d["success_gate"],
                     "failure_fallback": d["fallback"], "reusable_artifacts": issue.source_artifact,
                     "final_decision": "close_with_evidence" if d["action_class"] == "already_resolved_or_superseded" else "retain_and_disclose_limitation",
                     "artifact_source": issue.source_artifact})
    return pd.DataFrame(rows)


def recomputation_matrix(issues):
    rows = []
    for issue in issues.itertuples():
        d = DECISIONS[issue.issue_id]
        future = d["action_class"] in {"must_resolve_before_thesis", "should_improve_if_time_allows"}
        rows.append({"issue_id": issue.issue_id, "action_class": d["action_class"],
                     "reuse_artifacts": issue.source_artifact, "new_data_required": "none" if not future else "explicitly authorized data only",
                     "new_compute_required": "none; reporting only" if not future else "one bounded preregistered evaluation",
                     "stages_to_rerun": "none", "stages_not_to_rerun": "HAI Step1-5A; Tiny; Structure V1/V2; A1M; A1C",
                     "estimated_scope": "document-only" if not future else "one stage maximum", "leakage_risk": "none" if not future else "must freeze before evaluation",
                     "external_dependency": "none", "problem_statement": d["problem"],
                     "improvement_objective": "accurate bounded thesis reporting", "scientific_reason": d["why"],
                     "minimum_data_needed": "existing evidence package", "minimum_artifacts_reused": issue.source_artifact,
                     "minimum_new_computation": "none", "forbidden_recomputation": "raw data, model forward, score, threshold, corruption, fusion",
                     "expected_outputs": "thesis limitation/ablation text", "success_gate": d["success_gate"],
                     "failure_stop_rule": d["fallback"], "estimated_stage_count": 0,
                     "dependency_on_external_data": False, "whether_user_confirmation_required": False})
    return pd.DataFrame(rows)


def claim_revision(claims):
    rows = [
        ("C1", "Tiny is a lightweight real-HAI baseline.", "5A Tiny multiseed and freeze manifest", "NONE", True, False, False, "保留Tiny为真实HAI轻量基线。", "性能与误报有限", "方法与真实实验"),
        ("C2", "Structure V2 detects collaborative anomalies.", "V2 controlled R1/R2/R3", "V2-A1C-1;V2A1S-001", True, True, False, "Structure V2在受控协同关系破坏协议中表现出检测潜力。", "仅限controlled mechanism evidence", "结构受控实验"),
        ("C3", "Structure V2 is strong on real HAI attacks.", "no valid V2 real evaluation", "V2-A1C-1;V2A1S-001", False, False, True, "删除。", "NOT_SUPPORTED", "结论"),
        ("C4", "Rec+Structure fusion improves final performance.", "fusion not executed", "V2-A1C-1;V2A1S-001", False, False, True, "删除。", "NOT_SUPPORTED", "结论"),
        ("C5", "The method remains lightweight.", "table_07 and freeze manifest", "NONE", True, False, False, "保留参数量与状态量轻量化主张。", "不等同于部署就绪", "复杂度"),
        ("C6", "Normal non-stationarity has been solved.", "A1S/A1M/A1C negative evidence", "V2-A1C-1;V2A1S-001", False, False, True, "删除，并改写为主要局限性。", "NOT_SUPPORTED", "局限性"),
        ("C7", "Dynamic conditional relations are meaningful.", "V2-A controlled evidence", "V2A-001;V2A1R-FIX-001", True, True, False, "时滞条件依赖在正常数据和受控破坏中具有机制意义。", "不外推真实攻击性能", "结构方法"),
    ]
    return pd.DataFrame(rows, columns=["claim_id", "current_claim", "supporting_evidence", "affected_issue_ids", "retain", "weaken", "remove", "revised_claim", "required_qualifier", "thesis_section"])


def priority_frame(decisions):
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    rows = []
    for row in decisions.itertuples():
        rows.append({"issue_id": row.issue_id, "severity": row.severity, "action_class": row.action_class,
                     "thesis_impact": "high" if row.severity == "P1" else "medium" if row.severity == "P2" else "low",
                     "scientific_value": "high" if row.severity == "P1" else "medium", "compute_cost": "none",
                     "time_cost": "low", "leakage_risk": "none", "expected_benefit": "claim accuracy and completeness",
                     "recommended_decision": row.final_decision})
    frame = pd.DataFrame(rows).sort_values("severity", key=lambda s: s.map(order)).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def experiments_to_stop():
    rows = [
        ("increase Tiny capacity", "Step3G showed capacity is not the main problem", "HAI-3G", "compute without resolving score separability", "only under a newly motivated future method"),
        ("rec state-specific thresholds", "Step3F gates failed", "HAI-3F", "validation overfitting", "no"),
        ("repeat alarm smoothing search", "Step3E reduced event recall", "HAI-3E", "policy overfitting", "no"),
        ("continue Structure V1 threshold/aggregation repair", "4B-B/4CD showed unstable transfer", "HAI-4B-B/4CD", "post-hoc tuning", "only as a new preregistered study"),
        ("add V2 half-life candidates", "A1C final adaptive attempt failed", "V2-A1C", "validation adaptation and anomaly absorption", "no"),
        ("increase KMeans mode count", "A1M stable modes did not reduce drift", "V2-A1M", "unbounded mode search", "no"),
        ("sequence-specific validation thresholds", "violates shared frozen protocol", "A1S/A1M/A1C", "leakage and non-transferable thresholds", "no"),
        ("select edges using HAI test", "test must remain evaluation-only", "5A claim boundary", "direct leakage", "no"),
        ("run fusion before structure stability", "structure gate failed", "5A freeze manifest", "large FP cost and unsupported claims", "no"),
        ("use legacy SWaT merged.csv", "SWaT chronology/overlap audit invalidated it", "SWAT-4.3D/4.3E1", "invalid data claims", "only after obtaining a legal independent source"),
    ]
    return pd.DataFrame(rows, columns=["experiment_direction", "why_stop", "existing_evidence", "risk_if_continued", "future_restart_allowed"])


def write_yaml_md(stem, payload, title):
    (OUT / f"{stem}.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        lines.append(f"## {key}\n")
        if isinstance(value, list): lines.extend(f"- {item}" for item in value)
        elif isinstance(value, dict): lines.extend(f"- {subkey}: {subvalue}" for subkey, subvalue in value.items())
        else: lines.append(str(value))
        lines.append("")
    (OUT / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


def run(dry_run=False):
    issues, issue_json, freeze, claims, provenance, inventory = load_inputs()
    counts = pd.Series([DECISIONS[i]["severity"] for i in issues.issue_id]).value_counts()
    if dry_run:
        print(f"unresolved_issue_count={len(issues)}", flush=True)
        print("preliminary_severity=" + ",".join(f"{p}={int(counts.get(p,0))}" for p in ("P0","P1","P2","P3")), flush=True)
        print("FORBIDDEN: all training/inference/score/threshold/corruption/fusion/SWaT download/external evaluation", flush=True)
        print("DECISION_FILES: input audit, severity, final decision, recomputation, claim revision, readiness, external confirmation, priority, stop list, two roadmaps, final report", flush=True)
        print("raw data/model/score loaded=false", flush=True)
        return 0
    current_fingerprint = fingerprint()
    status_path = OUT / "stage_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("final_report_complete") and status.get("input_fingerprint") == current_fingerprint:
            print("REUSE completed issue triage and improvement decision", flush=True)
            return 0
    OUT.mkdir(parents=True, exist_ok=True)
    status = {"inputs_verified": False, "issue_triage_complete": False, "minimum_recomputation_complete": False,
              "claim_revision_complete": False, "readiness_assessment_complete": False,
              "external_confirmation_decision_complete": False, "roadmaps_complete": False,
              "historical_hashes_unchanged": False, "final_report_complete": False,
              "input_fingerprint": current_fingerprint}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    before = {name: sha256(INPUT / name) for name in REQUIRED_INPUTS}
    audit = input_audit(issues, issue_json, provenance, inventory)
    audit.to_csv(OUT / "unresolved_issue_input_audit.csv", index=False)
    if not audit.input_complete.all():
        print("classification=invalid_issue_triage_input", flush=True)
        return 2
    status["inputs_verified"] = True
    severity = severity_frame(issues); severity.to_csv(OUT / "issue_severity_classification.csv", index=False)
    decisions = decision_frame(issues); write_table("final_unresolved_issue_decision", decisions, "最终未解决问题决策")
    status["issue_triage_complete"] = True
    recompute = recomputation_matrix(issues); recompute.to_csv(OUT / "minimum_recomputation_matrix.csv", index=False)
    status["minimum_recomputation_complete"] = True
    revisions = claim_revision(claims); write_table("claim_revision_matrix", revisions, "论文主张修订矩阵")
    status["claim_revision_complete"] = True
    readiness = {"level": "Level 1", "decision": "当前证据即可完成受限且可信的论文",
                 "basis": ["无未解决P0", "Tiny真实HAI基线可追溯", "Structure V2主张可限定为受控机制证据", "所有越界主张已在5A边界中删除"],
                 "must_resolve_before_thesis_count": 0, "new_experiment_required": False}
    write_yaml_md("thesis_readiness_assessment", readiness, "论文主体可行性判断")
    status["readiness_assessment_complete"] = True
    external = {"decision": "strongly_recommended", "required_for_current_bounded_thesis": False,
                "reason": "HAI已参与开发，Structure V2没有真实冻结评价，旧SWaT无效；独立数据可增强外部有效性，但不是受限论文主张的硬门禁。",
                "frozen_method_point": "Tiny及Structure V2 static全部配置、关系、阈值和聚合在看新数据前冻结",
                "allowed_adaptation": "仅数据schema、通道映射和无标签scaler适配",
                "forbidden_tuning": "不得按新数据标签调整关系、阈值、聚合、half-life或模型",
                "metrics": ["Precision", "Recall", "F1", "AUPR", "AUROC", "event recall", "false positives"],
                "failure_reporting": "失败和不可适配结果必须报告", "no_redevelopment": True,
                "execution_status": "protocol_only_not_executed"}
    write_yaml_md("external_confirmation_decision", external, "外部数据确认决策")
    status["external_confirmation_decision_complete"] = True
    priority = priority_frame(decisions); write_table("improvement_priority_matrix", priority, "改进优先级矩阵")
    stopped = experiments_to_stop(); write_table("experiments_not_to_continue", stopped, "停止实验清单")
    minimum_plan = """# Minimum Thesis Completion Plan

| step | task | required | input | output | estimated_scope | stop_condition |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 锁定5A证据包与5B决策 | yes | 5A/5B artifacts | hash manifest | 分钟级 | hash一致 |
| 2 | 按claim revision撰写方法与实验 | yes | tables 01-09 | 论文实验章节 | 文档工作 | 不新增无provenance数字 |
| 3 | 写入A1S/A1M/A1C负结果和局限 | yes | final decision table | 局限性章节 | 文档工作 | controlled不写成real |
| 4 | 完成摘要、结论和复现附录 | yes | freeze manifest/claim boundary | 可提交初稿 | 文档工作 | 无越界主张 |
| 5 | 全文证据链校对 | yes | metric provenance | 最终定稿 | 一轮校对 | 所有关键数字可追溯 |
"""
    (OUT / "minimum_thesis_completion_plan.md").write_text(minimum_plan, encoding="utf-8")
    stronger_plan = """# Stronger Thesis Improvement Plan

最多新增一个实验阶段，并且必须由用户另行确认：

1. 获取合法、完整、未参与当前方法设计的工业时序数据。
2. 在查看标签和结果前冻结Tiny与Structure V2 static；只允许schema、通道映射和无标签scaler适配。
3. 执行一次确认性评价，报告全部预注册指标和失败结果。若无法无歧义适配或需调参，立即停止并保持当前Level 1论文。

禁止把新数据再次变为开发集，也不允许追加第二轮关系、阈值或聚合搜索。
"""
    (OUT / "stronger_thesis_improvement_plan.md").write_text(stronger_plan, encoding="utf-8")
    status["roadmaps_complete"] = True
    report = f"""# HAI Final Step 5B 最终问题决策报告

## 1. 当前阶段

项目已完成方法冻结和统一证据打包，本阶段只做问题分级与论文决策，没有运行实验。来源：`outputs/hai_final_unified_evidence_package/stage_status.json`。

## 2. 统一证据包可信性

5A输入完整，7条issue ID唯一，均有历史artifact和阶段provenance；本阶段输入审计全部通过。来源：`unresolved_issue_input_audit.csv`。

## 3. 七条问题概览与分级

最终分级为P0=0、P1=2、P2=4、P3=1。P1是A1S确认的正常序列非平稳性和A1C确认的漂移未改善/持续异常吸收风险。来源：`issue_severity_classification.csv`。

## 4. 必须解决的问题

没有must_resolve_before_thesis问题。当前所有关键数字可追溯，且论文可以通过降低Structure主张范围避免无证据结论。

## 5. 建议改善的问题

七条历史问题中没有仍需自动补实验的A/B类。若时间允许，唯一建议是一次完全冻结的独立外部数据确认，但必须由用户另行授权。来源：`external_confirmation_decision.yaml`。

## 6. 必须保留的局限性

V2A1S-001和V2-A1C-1归入retain_as_explicit_limitation：Structure V2仅能作为受控机制原型，不能作为稳定真实检测器或部署方案。来源：`final_unresolved_issue_decision.csv`。

## 7. 已解决或被取代的问题

V2A1D-001、V2A1D-002、V2A1M-001、V2A1R-FIX-001和V2A-001均由后续正式阶段回答或取代，保留解决证据但不再进入未来实验。来源：同上。

## 8. 不再继续的实验

停止扩容、状态阈值、alarm smoothing、V1继续修补、V2新增half-life/KMeans、sequence-specific阈值、test选边、未过门禁融合和旧SWaT merged.csv。来源：`experiments_not_to_continue.csv`。

## 9. 最小论文完成计划

无需新实验；按5A表格、冻结清单、主张边界和5B修订矩阵完成正文及证据校对。来源：`minimum_thesis_completion_plan.md`。

## 10. 增强版计划

最多一个新增实验阶段：使用合法独立数据做一次冻结确认；不可调参，失败也必须报告。来源：`stronger_thesis_improvement_plan.md`。

## 11. 外部确认决策

结论为strongly_recommended，而非required_before_final_claim。受限论文可直接完成；若标题声称通用工业异常检测，则应先取得独立确认。来源：`external_confirmation_decision.yaml`。

## 12. 论文可行性等级

唯一判断为Level 1：当前证据即可完成受限且可信的论文。来源：`thesis_readiness_assessment.yaml`。

## 13. 用户下一步唯一决策

选择其一：直接按Level 1受限主张完成论文；或另行授权一个冻结的独立外部确认阶段。未得到授权前，不启动任何新实验。
"""
    (OUT / "final_hai_issue_triage_and_improvement_decision.md").write_text(report, encoding="utf-8")
    after = {name: sha256(INPUT / name) for name in REQUIRED_INPUTS}
    status["historical_hashes_unchanged"] = before == after
    status["final_report_complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("COMPLETE classification=valid_issue_triage thesis_readiness=Level_1", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description="HAI Final Step 5B issue triage")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
