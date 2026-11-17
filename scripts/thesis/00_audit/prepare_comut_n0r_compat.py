"""Create and audit the sole authorized CoMuT compatibility adaptation."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REPO = Path(r"D:\workspace\python\external\s2gpp_experiments")
ORIGINAL = OFFICIAL_REPO / "dataset_configs" / "comut.yaml"
GENERATION_DIR = ROOT / "dataset" / "CoMuT_generation"
COMPAT = GENERATION_DIR / "comut_compat.yaml"
OUTPUT = ROOT / "outputs" / "thesis_comut_n0r"
EXPECTED_ORIGINAL_SHA = "D9F6EB03B8E7F81CD8FC555180C4419DA448F6040FC29768A832EC7DFBF4175B"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if COMPAT.exists() or OUTPUT.exists():
        raise FileExistsError("refusing to overwrite prior N0R artifacts")
    original = ORIGINAL.read_text(encoding="utf-8")
    if sha(ORIGINAL) != EXPECTED_ORIGINAL_SHA:
        raise RuntimeError("official comut.yaml hash changed")
    parsed = yaml.safe_load(original)
    cases = parsed["timeseries"]
    targets = [case for case in cases if case["base-oscillation"]["kind"] == "random-mode-jump"]
    if len(cases) != 13 or len(targets) != 13 or any(case["base-oscillation"].get("amplitude") != 1.0 for case in targets):
        raise RuntimeError("official RMJ amplitude contract changed")
    needle = "    amplitude: 1.0\n"
    removed = original.count(needle)
    compat_text = original.replace(needle, "")
    if removed != 13:
        raise RuntimeError(f"expected 13 exact amplitude lines, got {removed}")
    compat_parsed = yaml.safe_load(compat_text)
    expected = yaml.safe_load(original)
    for case in expected["timeseries"]:
        del case["base-oscillation"]["amplitude"]
    if expected != compat_parsed:
        raise RuntimeError("compat config contains a non-amplitude structural change")
    GENERATION_DIR.mkdir(parents=True); OUTPUT.mkdir(parents=True)
    COMPAT.write_text(compat_text, encoding="utf-8", newline="")
    diff = "".join(difflib.unified_diff(original.splitlines(True), compat_text.splitlines(True), fromfile="official/comut.yaml", tofile="compat/comut_compat.yaml"))
    (OUTPUT / "comut_compat_diff.txt").write_text(
        f"changed_key_count=13\nremoved_amplitude_count=13\nother_changed_key_count=0\noriginal_config_sha256={sha(ORIGINAL)}\ncompat_config_sha256={sha(COMPAT)}\n\n{diff}", encoding="utf-8"
    )
    (OUTPUT / "generation_manifest.json").write_text(json.dumps({
        "official_repository_url": "https://github.com/HPI-Information-Systems/s2gpp_experiments",
        "official_repository_commit": "5cf6d8f42fae319a8237111e196218dff77ad107",
        "author_artifact_found": False,
        "original_config_sha256": sha(ORIGINAL), "compat_config_sha256": sha(COMPAT),
        "changed_key_count": 13, "removed_amplitude_count": 13, "other_changed_key_count": 0,
        "seed": 42, "compatibility_reproduction": True, "exact_historical_generator": False,
        "adaptation": "remove random-mode-jump.amplitude only for current schema compatibility",
        "generation_status": "PENDING",
    }, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "source_search_report.md").write_text(
        "# CoMuT author artifact search\n\nNo generated CoMuT dataset was found in any local branch, tag, Git LFS entry, or historical `data/` tree of official repository commit `5cf6d8f`. Historical trees contained configuration files only.\n", encoding="utf-8"
    )
    print(json.dumps({"removed_amplitude_count": removed, "other_changed_key_count": 0, "compat_sha256": sha(COMPAT)}))


if __name__ == "__main__":
    main()
