import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "file:///C:/Users/jiusi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = "D:/workspace/python/mscad-thesis";
const out = path.join(root, "outputs", "thesis_ablation", "innovation1");
const payload = JSON.parse(await fs.readFile(path.join(out, "abl3b_csv_payload.json"), "utf8"));

const specs = [
  {
    key: "innovation1_hai_ablation",
    file: "innovation1_hai_ablation.csv",
    headers: [
      "variant", "architecture_change", "params", "param_ratio_vs_R0", "auroc", "aupr", "precision", "recall", "f1", "fpr",
      "event_hit", "high_median_hit", "high_hit", "direct_median_hit", "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
      "peak_vram_mib", "state_dict_bytes", "train_time_seconds", "best_epoch", "best_validation_loss", "stop_reason",
      "delta_auroc_vs_R0", "delta_aupr_vs_R0", "delta_f1_vs_R0", "delta_fpr_vs_R0", "delta_params_vs_R0", "delta_latency_mean_ms_vs_R0",
    ],
  },
  {
    key: "innovation1_cross_dataset_ablation",
    file: "innovation1_cross_dataset_ablation.csv",
    headers: [
      "dataset", "variant", "params", "auroc", "aupr", "f1", "fpr", "event_hit", "latency_mean_ms", "latency_p95_ms", "peak_vram_mib",
      "delta_auroc_vs_R0", "delta_aupr_vs_R0", "delta_f1_vs_R0", "delta_fpr_vs_R0", "delta_params_vs_R0", "delta_latency_mean_ms_vs_R0",
    ],
  },
  {
    key: "innovation1_cross_dataset_plot_data",
    file: "innovation1_cross_dataset_plot_data.csv",
    headers: [
      "panel", "dataset", "variant", "params", "auroc", "aupr", "f1", "fpr", "event_hit", "latency_mean_ms", "latency_p95_ms", "peak_vram_mib",
      "delta_auroc_vs_R0", "delta_aupr_vs_R0", "delta_f1_vs_R0", "delta_fpr_vs_R0", "delta_params_vs_R0", "delta_latency_mean_ms_vs_R0",
      "x_metric", "x_value", "y_metric", "y_value",
    ],
  },
];

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "number" ? String(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function columnName(count) {
  let value = count;
  let name = "";
  while (value > 0) {
    value -= 1;
    name = String.fromCharCode(65 + (value % 26)) + name;
    value = Math.floor(value / 26);
  }
  return name;
}

for (const spec of specs) {
  const records = payload[spec.key];
  if (!Array.isArray(records) || records.length === 0) throw new Error(`Missing records: ${spec.key}`);
  const matrix = [spec.headers, ...records.map((row) => spec.headers.map((header) => row[header] ?? null))];
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("Data");
  sheet.getRangeByIndexes(0, 0, matrix.length, spec.headers.length).values = matrix;
  workbook.recalculate();
  const inspected = await workbook.inspect({
    kind: "table",
    sheetId: "Data",
    range: `A1:${columnName(spec.headers.length)}${Math.min(matrix.length, 10)}`,
    include: "values,formulas",
    tableMaxRows: 10,
    tableMaxCols: spec.headers.length,
    maxChars: 12000,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 100 },
    summary: `formula error scan: ${spec.file}`,
  });
  const csv = matrix.map((row) => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
  await fs.writeFile(path.join(out, spec.file), csv, "utf8");
  const roundTrip = await Workbook.fromCSV(csv, { sheetName: "Data" });
  const roundTripInspect = await roundTrip.inspect({ kind: "table", sheetId: "Data", range: `A1:B${matrix.length}`, include: "values", tableMaxRows: matrix.length, tableMaxCols: 2, maxChars: 5000 });
  process.stdout.write(JSON.stringify({ file: spec.file, rows: records.length, columns: spec.headers.length, inspected: inspected.ndjson, errors: errors.ndjson, roundTrip: roundTripInspect.ndjson }) + "\n");
}
