import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "D:/workspace/python/mscad-thesis";
const baselinePath = `${root}/outputs/thesis_baselines/baseline_table_template.csv`;
const outDir = `${root}/outputs/thesis_baselines/gdn`;
const tableText = await fs.readFile(baselinePath, "utf8");
const workbook = await Workbook.fromCSV(tableText, { sheetName: "Unified baseline" });
const sheet = workbook.worksheets.getItem("Unified baseline");
const values = sheet.getUsedRange().values;
const headers = values[0];
const col = Object.fromEntries(headers.map((name, index) => [name, index]));

const swat = JSON.parse(await fs.readFile(`${outDir}/swat_metrics.json`, "utf8"));
const hai = JSON.parse(await fs.readFile(`${outDir}/hai_metrics.json`, "utf8"));
const efficiency = JSON.parse(await fs.readFile(`${outDir}/efficiency.json`, "utf8"));
const resultByDataset = { SWaT: [swat, efficiency.SWaT], HAI21: [hai, efficiency.HAI21] };
for (let r = 1; r < values.length; r++) {
  const dataset = values[r][col.Dataset];
  if (values[r][col.Method] !== "GDN" || !(dataset in resultByDataset)) continue;
  const [metrics, eff] = resultByDataset[dataset];
  const eventHit = dataset === "SWaT" ? metrics.events.event_hit_count : metrics.events.ALL.event_hit_count;
  const updates = {
    Params: eff.trainable_parameters,
    AUROC: metrics.AUROC,
    AUPR: metrics.AUPR,
    Precision: metrics.Precision,
    Recall: metrics.Recall,
    F1: metrics.F1,
    FPR: metrics.FPR,
    EventHitCount: eventHit,
    Latency: eff.latency_ms_mean,
  };
  for (const [name, value] of Object.entries(updates)) values[r][col[name]] = value;
}
// Preserve identifiers as text and make all available numeric table fields typed numbers.
for (let r = 1; r < values.length; r++) {
  for (let c = 2; c < headers.length; c++) {
    if (values[r][c] !== "" && values[r][c] != null) values[r][c] = Number(values[r][c]);
  }
}
sheet.getUsedRange().values = values;
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.getRange(`A1:K${values.length}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
sheet.getRange("A1:K1").format = { fill: "#1F4E78", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center" };
sheet.getRange(`C2:C${values.length}`).format.numberFormat = "#,##0";
sheet.getRange(`D2:I${values.length}`).format.numberFormat = "0.000000";
sheet.getRange(`J2:J${values.length}`).format.numberFormat = "0";
sheet.getRange(`K2:K${values.length}`).format.numberFormat = "0.000";
sheet.getRange(`A1:K${values.length}`).format.autofitColumns();
sheet.getRange("A:A").format.columnWidth = 12;
sheet.getRange("B:B").format.columnWidth = 16;

const ours = (await fs.readFile(`${root}/outputs/thesis_innovation3b_tail_fusion_final/cross_dataset_metrics.csv`, "utf8"))
  .trim().split(/\r?\n/).slice(1).map(line => line.split(","));
const oursDual = Object.fromEntries(ours.filter(row => row[1] === "DUAL").map(row => [row[0], row]));
const comparison = workbook.worksheets.add("GDN vs Ours-DUAL");
const comparisonRows = [
  ["Dataset", "Method", "AUROC", "AUPR", "F1", "FPR", "HIGH median hit", "Params", "Latency ms"],
  ["SWaT", "GDN", swat.AUROC, swat.AUPR, swat.F1, swat.FPR, null, efficiency.SWaT.trainable_parameters, efficiency.SWaT.latency_ms_mean],
  ["SWaT", "Ours-DUAL", Number(oursDual.SWaT[3]), Number(oursDual.SWaT[4]), Number(oursDual.SWaT[7]), Number(oursDual.SWaT[8]), null, null, null],
  ["HAI21", "GDN", hai.AUROC, hai.AUPR, hai.F1, hai.FPR, hai.events.HIGH.median_predicted_anomaly_fraction, efficiency.HAI21.trainable_parameters, efficiency.HAI21.latency_ms_mean],
  ["HAI21", "Ours-DUAL", Number(oursDual.HAI21[3]), Number(oursDual.HAI21[4]), Number(oursDual.HAI21[7]), Number(oursDual.HAI21[8]), 0.8410852713178295, null, null],
];
comparison.getRange("A1:I5").values = comparisonRows;
comparison.showGridLines = false;
comparison.freezePanes.freezeRows(1);
comparison.getRange("A1:I5").format.font = { name: "Arial", size: 10, color: "#1F2937" };
comparison.getRange("A1:I1").format = { fill: "#1F4E78", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center" };
comparison.getRange("C2:G5").format.numberFormat = "0.000000";
comparison.getRange("H2:H5").format.numberFormat = "#,##0";
comparison.getRange("I2:I5").format.numberFormat = "0.000";
comparison.getRange("A1:I5").format.autofitColumns();

const csvEscape = value => {
  if (value == null) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const toCsv = rows => rows.map(row => row.map(csvEscape).join(",")).join("\n") + "\n";

workbook.recalculate();
const unifiedCheck = await workbook.inspect({ kind: "table", range: `Unified baseline!A1:K${values.length}`, include: "values,formulas", tableMaxRows: 20, tableMaxCols: 12 });
const comparisonCheck = await workbook.inspect({ kind: "table", range: "GDN vs Ours-DUAL!A1:I5", include: "values,formulas", tableMaxRows: 10, tableMaxCols: 10 });
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });
console.log(unifiedCheck.ndjson);
console.log(comparisonCheck.ndjson);
console.log(errors.ndjson);

await fs.mkdir(outDir, { recursive: true });
for (const [sheetName, fileName] of [["Unified baseline", "baseline_table_preview.png"], ["GDN vs Ours-DUAL", "gdn_vs_ours_dual_preview.png"]]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(baselinePath, toCsv(values), "utf8");
await fs.writeFile(`${outDir}/gdn_vs_ours_dual.csv`, toCsv(comparisonRows), "utf8");
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outDir}/gdn_unified_comparison.xlsx`);
