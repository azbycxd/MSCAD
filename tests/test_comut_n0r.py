from pathlib import Path
import csv, hashlib, json, yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_comut_n0r"


def summary(): return json.loads((OUT / "audit_summary.json").read_text())
def manifest(): return json.loads((OUT / "generation_manifest.json").read_text())
def test_01_official_repo_url(): assert manifest()["official_repository_url"] == "https://github.com/HPI-Information-Systems/s2gpp_experiments"
def test_02_commit_recorded(): assert len(manifest()["official_repository_commit"]) == 40
def test_03_config_hash_recorded(): assert len(manifest()["original_config_sha256"]) == 64
def test_04_seed_42(): assert summary()["seed"] == 42
def test_05_only_compat_change(): assert manifest()["removed_amplitude_count"] == 13 and manifest()["other_changed_key_count"] == 0
def test_06_dataset_count(): assert summary()["dataset_count"] == 13
def test_07_rows_valid(): assert summary()["row_count_per_case"] == [10000]
def test_08_labels_parse(): assert summary()["label_column"] == "is_anomaly"
def test_09_intervals_recovered(): assert sum(1 for _ in csv.DictReader((OUT / "anomaly_intervals.csv").open())) == 19
def test_10_no_nan_inf(): assert summary()["nan_count"] == summary()["inf_count"] == 0
def test_11_gitignored():
    import subprocess
    result=subprocess.run(["git","check-ignore","dataset/CoMuT_compat/probe"],cwd=ROOT,capture_output=True,text=True); assert result.returncode == 0
def test_12_reference_only():
    rows=list(csv.DictReader((OUT / "official_reference_results.csv").open())); assert len(rows)==13 and all(r["locally_reproduced"]=="false" for r in rows)
