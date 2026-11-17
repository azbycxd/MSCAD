import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "D:/workspace/python/mscad-thesis";
const outDir = `${root}/outputs/thesis_efficiency_final`;
const payload = JSON.parse(await fs.readFile(`${outDir}/tables_payload.json`, "utf8"));
const summary = JSON.parse(await fs.readFile(`${outDir}/efficiency_summary.json`, "utf8"));
const workbook = Workbook.create();

const schemas = {
  "Ours parameters": ["row_type", "dataset", "group", "target", "target_count", "cross_feature_dimension", "self_params", "cross_params", "dual_params"],
  "Ours model sizes": ["dataset", "method", "model_size_bytes", "trainable_params", "calibration_state_elements"],
  "Ours latency": ["dataset", "method", "device", "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "peak_vram_mib", "warmup", "timed_endpoints", "batch_size", "score_equivalence_max_abs_diff", "rec_time_fraction", "dual_time_fraction", "calibration_fusion_time_fraction"],
  "Efficiency comparison": ["dataset", "method", "params", "model_size_bytes", "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "peak_vram_mib", "device"],
  "Performance efficiency": ["dataset", "method", "params", "aupr", "auroc", "f1", "mean_latency_ms"],
};
const data = {
  "Ours parameters": payload.parameter_breakdown,
  "Ours model sizes": payload.model_sizes,
  "Ours latency": payload.ours_latency,
  "Efficiency comparison": payload.efficiency_comparison,
  "Performance efficiency": payload.performance_efficiency,
};

for (const [sheetName, headers] of Object.entries(schemas)) {
  const sheet = workbook.worksheets.add(sheetName);
  const rows = [headers, ...data[sheetName].map(record => headers.map(key => record[key] ?? null))];
  sheet.getRangeByIndexes(0, 0, rows.length, headers.length).values = rows;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.getRangeByIndexes(0, 0, rows.length, headers.length).format.font = { name: "Arial", size: 10, color: "#1F2937" };
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
    fill: "#1F4E78",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRangeByIndexes(0, 0, rows.length, headers.length).format.autofitColumns();
  for (let c = 0; c < headers.length; c++) {
    const name = headers[c];
    const range = sheet.getRangeByIndexes(1, c, Math.max(rows.length - 1, 1), 1);
    if (name.includes("latency") || name.includes("fraction") || name.includes("auroc") || name.includes("aupr") || name === "f1" || name.includes("diff") || name.includes("vram")) range.format.numberFormat = "0.000000";
    if (name.includes("params") || name.includes("bytes") || name.includes("count") || name.includes("dimension") || name.includes("elements") || name === "warmup" || name === "timed_endpoints" || name === "batch_size") range.format.numberFormat = "#,##0";
  }
}
const comparisonSheet = workbook.worksheets.getItem("Efficiency comparison");
comparisonSheet.getRange("A:A").format.columnWidth = 12;
comparisonSheet.getRange("B:B").format.columnWidth = 18;
comparisonSheet.getRange("C:D").format.columnWidth = 20;
comparisonSheet.getRange("E:G").format.columnWidth = 22;
comparisonSheet.getRange("H:H").format.columnWidth = 18;
comparisonSheet.getRange("I:I").format.columnWidth = 12;

const ratioSheet = workbook.worksheets.add("Parameter ratios");
const ratioRows = [["Dataset", "Comparator", "Comparator params / Ours-FULL", "Ours-FULL params / Comparator", "Comparator latency / Ours-FULL", "Ours-FULL latency / Comparator"]];
for (const row of summary.parameter_ratios) ratioRows.push([row.dataset, row.comparator, row.comparator_params_over_ours_full, row.ours_full_params_over_comparator, row.comparator_latency_over_ours_full, row.ours_full_latency_over_comparator]);
ratioSheet.getRangeByIndexes(0, 0, ratioRows.length, 6).values = ratioRows;
ratioSheet.showGridLines = false;
ratioSheet.freezePanes.freezeRows(1);
ratioSheet.getRange(`A1:F${ratioRows.length}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
ratioSheet.getRange("A1:F1").format = { fill: "#1F4E78", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
ratioSheet.getRange(`C2:F${ratioRows.length}`).format.numberFormat = "0.000000";
ratioSheet.getRange(`A1:F${ratioRows.length}`).format.autofitColumns();

const csvEscape = value => {
  if (value == null) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const toCsv = rows => rows.map(row => row.map(csvEscape).join(",")).join("\n") + "\n";
const csvOutputs = {
  "Ours parameters": "ours_parameter_breakdown.csv",
  "Ours model sizes": "ours_model_sizes.csv",
  "Ours latency": "ours_latency.csv",
  "Efficiency comparison": "efficiency_comparison.csv",
  "Performance efficiency": "performance_efficiency.csv",
};

workbook.recalculate();
for (const [sheetName, fileName] of Object.entries(csvOutputs)) {
  const check = await workbook.inspect({ kind: "table", range: `${sheetName}!A1:${String.fromCharCode(64 + schemas[sheetName].length)}${data[sheetName].length + 1}`, include: "values,formulas", tableMaxRows: 12, tableMaxCols: 16 });
  console.log(check.ndjson);
  const rows = [schemas[sheetName], ...data[sheetName].map(record => schemas[sheetName].map(key => record[key] ?? null))];
  await fs.writeFile(`${outDir}/${fileName}`, toCsv(rows), "utf8");
}
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });
console.log(errors.ndjson);
const preview = await workbook.render({ sheetName: "Efficiency comparison", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(`${outDir}/efficiency_comparison_preview.png`, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outDir}/efficiency_audit.xlsx`);
