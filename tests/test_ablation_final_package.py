from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_ablation"
FIG = ROOT / "outputs" / "thesis_figures" / "ablation"
SCRIPT = ROOT / "scripts" / "thesis" / "03_ablation" / "build_ablation_final_package.py"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_all_figure_values_are_bound_to_final_csvs() -> None:
    audit = rows(FIG / "ablation_figure_data_audit.csv")
    allowed = {
        "outputs/thesis_ablation/innovation1/innovation1_cross_dataset_ablation.csv",
        "outputs/thesis_ablation/innovation2/summary/innovation2_real_ablation.csv",
        "outputs/thesis_ablation/innovation2/summary/innovation2_mechanism_ablation.csv",
        "outputs/thesis_ablation/innovation3/normal_tail_balance.csv",
        "outputs/thesis_ablation/innovation3/fusion_detection_ablation.csv",
    }
    assert audit
    assert {row["source_csv"] for row in audit} <= allowed
    for row in audit:
        source_rows = rows(ROOT / row["source_csv"])
        matches = [
            candidate for candidate in source_rows
            if candidate["dataset"] == row["dataset"]
            and candidate["variant"] == row["variant"]
            and row["metric"] in candidate
        ]
        assert matches
        assert abs(float(matches[0][row["metric"]]) - float(row["value"])) <= 1e-12


def test_r0_r1_r2_numbers_and_direction_are_consistent() -> None:
    data = rows(OUT / "innovation1" / "innovation1_cross_dataset_ablation.csv")
    assert len(data) == 6
    for dataset in ("SWaT", "HAI21"):
        subset = {row["variant"]: row for row in data if row["dataset"] == dataset}
        assert set(subset) == {"R0 Final", "R1", "R2"}
        assert int(subset["R1"]["params"]) < int(subset["R0 Final"]["params"])
        assert float(subset["R1"]["auroc"]) > float(subset["R0 Final"]["auroc"])
        assert float(subset["R1"]["aupr"]) > float(subset["R0 Final"]["aupr"])
        assert int(subset["R2"]["params"]) > 2 * int(subset["R0 Final"]["params"])
        assert float(subset["R2"]["aupr"]) < float(subset["R0 Final"]["aupr"])


def test_a1_results_are_consistent() -> None:
    real = rows(OUT / "innovation2" / "summary" / "innovation2_real_ablation.csv")
    mechanism = rows(OUT / "innovation2" / "summary" / "innovation2_mechanism_ablation.csv")
    hai = {row["variant"]: row for row in real if row["dataset"] == "HAI"}
    assert float(hai["D1 Short-lag"]["aupr"]) < float(hai["D0 Final"]["aupr"])
    assert int(hai["D1 Short-lag"]["event_hit_count"]) == int(hai["D0 Final"]["event_hit_count"])
    controlled = {row["variant"]: row for row in mechanism if row["dataset"] == "SWaT controlled"}
    assert float(controlled["D2 No-Gate"]["auroc"]) < float(controlled["D0 Final"]["auroc"])
    assert float(controlled["D2 No-Gate"]["aupr"]) < float(controlled["D0 Final"]["aupr"])


def test_a2_results_are_consistent() -> None:
    balance = rows(OUT / "innovation3" / "normal_tail_balance.csv")
    fusion = rows(OUT / "innovation3" / "fusion_detection_ablation.csv")
    robust = [row for row in balance if row["variant"] == "F3 Robust-Z"]
    assert len(robust) == 3
    assert all(float(row["top1_dual_share"]) == 1.0 for row in robust)
    hai = {row["variant"]: row for row in fusion if row["dataset"] == "HAI21"}
    swat = {row["variant"]: row for row in fusion if row["dataset"] == "SWaT"}
    assert abs(float(hai["F4 Tail-Calibrated"]["aupr"]) - 0.6178966926696478) < 1e-12
    assert float(swat["F4 Tail-Calibrated"]["aupr"]) < float(swat["F0 REC"]["aupr"])


def test_no_not_extracted_or_tbd_is_filled_into_figures() -> None:
    audit_text = (FIG / "ablation_figure_data_audit.csv").read_text(encoding="utf-8")
    assert "NOT_EXTRACTED" not in audit_text
    assert "TBD" not in audit_text


def test_package_is_presentation_only_and_performed_no_prohibited_action() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    banned = ("import torch", ".fit(", ".predict(", "DataLoader(", "optimizer.step(", "backward(")
    assert all(token not in source for token in banned)
    summary = json.loads((OUT / "ablation_final_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ABLATION_FINAL_PACKAGE_READY"
    assert all(value is False for value in summary["prohibited_actions"].values())


def test_no_source_file_deleted_and_no_git_operation_encoded() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".unlink(" not in source
    assert ".rmdir(" not in source
    assert "git add" not in source
    assert "git commit" not in source
    assert "git push" not in source


def test_final_outputs_exist() -> None:
    for stem in (
        "fig_ablation_lightweight_rec",
        "fig_ablation_dual_view",
        "fig_ablation_tail_calibration",
    ):
        assert (FIG / f"{stem}.png").stat().st_size > 0
        assert (FIG / f"{stem}.pdf").stat().st_size > 0
    assert (ROOT / "docs" / "thesis" / "ablation_tables_latex.tex").stat().st_size > 0


def test_statuses_are_frozen() -> None:
    required = {
        "ablation_a1_innovation2_results.md": "ABLATION_A1_INNOVATION2_COMPLETED",
        "ablation_a2_innovation3_results.md": "ABLATION_A2_INNOVATION3_COMPLETED",
        "ablation_a3b_innovation1_hai_results.md": "ABLATION_A3_INNOVATION1_COMPLETED",
    }
    for name, status in required.items():
        text = (ROOT / "docs" / "thesis" / name).read_text(encoding="utf-8")
        assert status in text
