from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage4_3e1_locate_raw_swat_sources.py"
SPEC = importlib.util.spec_from_file_location("raw_locator", SCRIPT)
locator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(locator)


def profile(labels, monotonic=True):
    stats = locator.label_sequence_stats(labels)
    return {"readable": True, "possible_timestamp_column": "Timestamp", "timestamp_monotonic": monotonic,
            "row_count": len(labels), **stats}


def test_normal_attack_normal_candidate_is_recognized(tmp_path):
    path = tmp_path / "raw_swat_test.csv"
    current = {"normal.csv": "n", "attack.csv": "a", "merged.csv": "m"}
    assert locator.chronological_attack_candidate(profile(["Normal", "Attack", "Normal"]), path, "raw", current)


def test_attack_only_is_not_complete_test(tmp_path):
    current = {"normal.csv": "n", "attack.csv": "a", "merged.csv": "m"}
    assert not locator.chronological_attack_candidate(profile(["Attack", "Attack"]), tmp_path / "raw.csv", "x", current)


def test_normal_plus_attack_suffix_is_rejected(tmp_path):
    current = {"normal.csv": "n", "attack.csv": "a", "merged.csv": "derived"}
    assert not locator.chronological_attack_candidate(profile(["Normal", "Normal", "Attack"]), tmp_path / "candidate.csv", "derived", current)


def test_label_strip_and_case():
    assert locator.normalize_label(" Normal ") == "Normal"
    assert locator.normalize_label("ATTACK") == "Attack"


def test_transition_and_positive_segment_counts():
    stats = locator.label_sequence_stats(["Normal", "Attack", "Attack", "Normal", "Attack", "Normal"])
    assert stats["label_transition_count"] == 4
    assert stats["positive_segment_count"] == 2
    assert stats["contains_normal_attack_normal"]


def test_file_hash_comparison(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"abc")
    assert locator.sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_search_scope_never_expands_to_drive_root(tmp_path):
    project = tmp_path / "workspace" / "mscad"
    project.mkdir(parents=True)
    roots = locator.safe_search_roots(project)
    assert locator.search_scope_is_safe(roots, project)
    assert all(root != Path(root.drive + "\\") for root in roots)


def test_profile_is_read_only(tmp_path):
    path = tmp_path / "raw_swat.csv"
    pd.DataFrame({"Timestamp": ["01/01/2020 00:00:00"], "F1": [1], "Normal/Attack": ["Normal"]}).to_csv(path, index=False)
    before = locator.sha256_file(path)
    locator.profile_csv(path)
    assert locator.sha256_file(path) == before


def test_script_does_not_import_or_run_models():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "load_state_dict" not in text
    assert "model(" not in text


def test_script_has_no_data_mutation_operations():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "to_excel(" not in text
    assert "sort_values(" not in text
    assert "drop_duplicates(" not in text
    assert "inplace=True" not in text
    assert ".unlink(" not in text
