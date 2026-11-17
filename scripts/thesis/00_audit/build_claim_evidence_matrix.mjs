import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "file:///C:/Users/jiusi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = "D:/workspace/python/mscad-thesis";
const output = path.join(root, "outputs/thesis_registry/claim_evidence_matrix.csv");

const rows = [
  {
    claim_id: "C1",
    claim: "局部滞后双视角预测分歧能够识别跨通道关系破坏",
    claim_type: "CORE_METHOD_MECHANISM",
    supporting_dataset: "Historical CoMuT; SWaT controlled; HAI21",
    supporting_experiment: "A1 D0 mechanism; HAI final DUAL",
    metric: "CoMuT median AUROC=0.995989; controlled SWaT median AUROC=0.996913; HAI AUROC/AUPR/F1=0.916334/0.570193/0.610556",
    evidence_strength: "STRONG_MULTI_LEVEL",
    boundary: "SWaT official real-attack DUAL is limited; mechanism evidence is not an industrial leaderboard",
    allowed_wording: "双视角协同检测获得合成机制、受控真实轨迹和HAI真实攻击的分层支持",
    forbidden_wording: "双视角方法在所有数据集和所有攻击类型上均有效",
  },
  {
    claim_id: "C2",
    claim: "多尺度滞后对真实协同异常的贡献具有数据集依赖性",
    claim_type: "COMPONENT_ABLATION",
    supporting_dataset: "HAI21; SWaT",
    supporting_experiment: "A1 D0 versus D1 Short-lag",
    metric: "HAI AUPR 0.570193→0.426064; SWaT AUPR 0.167510→0.565012; SWaT event hits remain 2/36",
    evidence_strength: "STRONG_DATASET_DEPENDENT",
    boundary: "Short-lag improves SWaT ranking but not event coverage; no universal lag claim",
    allowed_wording: "较长滞后在HAI中提供正向贡献，滞后作用随数据语义变化",
    forbidden_wording: "长滞后在所有数据集都必要或总能提高性能",
  },
  {
    claim_id: "C3",
    claim: "Normal-only Reliability Gate抑制不可靠target污染max aggregation",
    claim_type: "COMPONENT_MECHANISM",
    supporting_dataset: "SWaT controlled; HAI21; SWaT",
    supporting_experiment: "A1 D2 No-Gate; D3 gate decomposition",
    metric: "Controlled AUROC 0.996913→0.630637 and AUPR 0.996931→0.550076; D3-D reproduces D0",
    evidence_strength: "STRONG_CONTROLLED",
    boundary: "HAI and SWaT real-test D2 are close to D0; Gate is a normal-only reliability constraint, not a Test optimizer",
    allowed_wording: "受控关系破坏实验支持Gate对不可靠候选聚合污染的抑制作用",
    forbidden_wording: "Gate保证每个真实数据集的测试性能提升",
  },
  {
    claim_id: "C4",
    claim: "正常尾概率校准改善REC与DUAL异构证据的尾部可比性",
    claim_type: "CORE_CALIBRATION_METHOD",
    supporting_dataset: "SWaT; HAI21; WADI.A2",
    supporting_experiment: "A2 F3 Robust-Z versus F4 Tail-Calibrated",
    metric: "Robust-Z top-1% DUAL share=100% on all three; Tail REC/DUAL=65.19/34.81,59.15/40.85,51.78/48.22; q99 balance≈1",
    evidence_strength: "STRONG_CROSS_DATASET_CALIBRATION",
    boundary: "Evidence balance does not guarantee higher Test AUROC/AUPR",
    allowed_wording: "尾概率映射显著改善正常尾部证据可比性",
    forbidden_wording: "尾校准在所有数据集都提高检测性能",
  },
  {
    claim_id: "C5",
    claim: "Tail-calibrated fusion在HAI上体现REC与DUAL互补性",
    claim_type: "POSITIVE_DETECTION_ASSOCIATION",
    supporting_dataset: "HAI21",
    supporting_experiment: "A2 F0/F1/F4 fusion ablation",
    metric: "Best single AUPR=0.570193; Tail AUPR=0.617897; delta=+0.047703",
    evidence_strength: "MODERATE_SINGLE_DATASET",
    boundary: "Association is dataset-specific and does not establish universal causality",
    allowed_wording: "HAI结果显示校准后的异构证据具有正向互补关联",
    forbidden_wording: "融合机制已被证明普遍优于所有单分支",
  },
  {
    claim_id: "C6",
    claim: "Depthwise-separable REC支持万级参数的轻量化整体框架",
    claim_type: "LIGHTWEIGHT_DESIGN",
    supporting_dataset: "SWaT; HAI21",
    supporting_experiment: "A3 R0/R1/R2; Efficiency E1",
    metric: "R0 params=11730/15482; R2/R0=2.192/2.126 with lower AUPR; R1 has fewer params and higher AUROC/AUPR",
    evidence_strength: "STRONG_EFFICIENCY_WITH_ARCHITECTURE_BOUNDARY",
    boundary: "R1 is retrospective; fixed k=3 is not optimal; GDN has fewer params than Ours-FULL",
    allowed_wording: "Depthwise-separable design有效控制参数规模，框架强调性能—效率权衡",
    forbidden_wording: "R0或k=3是最优结构，或Ours对所有基线都更轻更快",
  },
  {
    claim_id: "C7",
    claim: "SWaT official real attacks构成DUAL与TAIL的明确适用边界",
    claim_type: "NEGATIVE_BOUNDARY",
    supporting_dataset: "SWaT 2015 Official Physical",
    supporting_experiment: "Final strict evaluation; A2 fusion ablation",
    metric: "DUAL AUROC/AUPR=0.675631/0.167510; Tail AUPR=0.206747; REC AUPR=0.743353",
    evidence_strength: "STRONG_BOUNDARY",
    boundary: "Do not hide or relabel this negative result; REC ranking remains stronger",
    allowed_wording: "SWaT表明协同证据与尾校准收益依赖异常语义和阈值迁移",
    forbidden_wording: "Ours-DUAL或Ours-TAIL在SWaT上优于REC",
  },
  {
    claim_id: "C8",
    claim: "WADI.A2显示normal-domain shift下证据平衡不等于检测有效",
    claim_type: "DOMAIN_SHIFT_BOUNDARY",
    supporting_dataset: "WADI.A2",
    supporting_experiment: "A2 domain-shift stress test",
    metric: "F0/F1/F3/F4 AUPR all approximately 0.052–0.055; Tail top-1% REC/DUAL=51.78/48.22",
    evidence_strength: "STRESS_TEST_ONLY",
    boundary: "WADI is not a Primary dataset and is not used to retune gate, lag, alpha, score, or fusion",
    allowed_wording: "WADI用于揭示域偏移与证据平衡的局限",
    forbidden_wording: "WADI结果证明跨数据集泛化，或可用于主性能排名",
  },
];

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
}

const headers = [
  "claim_id",
  "claim",
  "claim_type",
  "supporting_dataset",
  "supporting_experiment",
  "metric",
  "evidence_strength",
  "boundary",
  "allowed_wording",
  "forbidden_wording",
];
const csv = [
  headers.map(csvCell).join(","),
  ...rows.map((row) => headers.map((key) => csvCell(row[key])).join(",")),
].join("\r\n") + "\r\n";

const workbook = await Workbook.fromCSV(csv, { sheetName: "ClaimEvidence" });
workbook.recalculate();
await workbook.inspect({
  kind: "table",
  range: "ClaimEvidence!A1:J9",
  include: "values,formulas",
  tableMaxRows: 9,
  tableMaxCols: 10,
  maxChars: 16000,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 50 },
  summary: "claim evidence formula error scan",
});
const errorText = errors.ndjson ?? String(errors);
if (errorText.includes('"matchCount":') && !errorText.includes('"matchCount":0')) {
  throw new Error("Spreadsheet error token found");
}

await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, csv, "utf8");
console.log(JSON.stringify({ output, rows: rows.length, columns: headers.length }, null, 2));
