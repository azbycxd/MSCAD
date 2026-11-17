import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "file:///C:/Users/jiusi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = "D:/workspace/python/mscad-thesis";
const targets = [
  {
    payload: "outputs/thesis_figures/ablation/ablation_figure_data_audit_payload.json",
    output: "outputs/thesis_figures/ablation/ablation_figure_data_audit.csv",
    sheet: "FigureDataAudit",
  },
  {
    payload: "outputs/thesis_ablation/ablation_git_inventory_payload.json",
    output: "outputs/thesis_ablation/ablation_git_inventory.csv",
    sheet: "GitInventory",
  },
];

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function columnName(count) {
  let n = count;
  let result = "";
  while (n > 0) {
    n -= 1;
    result = String.fromCharCode(65 + (n % 26)) + result;
    n = Math.floor(n / 26);
  }
  return result;
}

async function createCsv(target) {
  const rows = JSON.parse(await fs.readFile(path.join(root, target.payload), "utf8"));
  if (!Array.isArray(rows) || rows.length === 0) throw new Error(`${target.payload}: no rows`);
  const headers = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const csv = `${[
    headers.map(csvCell).join(","),
    ...rows.map((row) => headers.map((key) => csvCell(row[key])).join(",")),
  ].join("\r\n")}\r\n`;
  const workbook = await Workbook.fromCSV(csv, { sheetName: target.sheet });
  workbook.recalculate();
  await workbook.inspect({
    kind: "table",
    range: `${target.sheet}!A1:${columnName(headers.length)}${Math.min(rows.length + 1, 40)}`,
    include: "values,formulas",
    tableMaxRows: 40,
    tableMaxCols: Math.min(headers.length, 30),
    maxChars: 12000,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 50 },
    summary: `${target.output} formula error scan`,
  });
  const errorText = errors.ndjson ?? String(errors);
  if (errorText.includes('"matchCount":') && !errorText.includes('"matchCount":0')) {
    throw new Error(`${target.output}: spreadsheet error token found`);
  }
  await fs.writeFile(path.join(root, target.output), csv, "utf8");
  return { output: target.output, rows: rows.length, columns: headers.length };
}

const results = [];
for (const target of targets) results.push(await createCsv(target));
console.log(JSON.stringify(results, null, 2));
