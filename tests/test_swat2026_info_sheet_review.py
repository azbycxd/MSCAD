from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from zipfile import ZipFile

from scripts.swat2026.extract_and_review_info_sheet import (
    KNOWN_SCHEDULE_IMAGE_SHA256,
    TRANSCRIPTION,
    extract_docx,
    label_policy_candidates,
    make_schedule,
    phase_rows,
    scan_bad_inputs,
    validate_schedule,
)


def make_docx(path: Path) -> bytes:
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (100).to_bytes(4, "big") + (20).to_bytes(4, "big") + b"fixture"
    document = b'''<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
 <w:body><w:p><w:r><w:t>Dataset A.12</w:t></w:r></w:p>
 <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Header</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
 <w:p><w:r><w:drawing><wp:inline><wp:docPr id="1" name="Attack schedule"/>
 <a:graphic><a:graphicData><a:blip r:embed="rId5"/></a:graphicData></a:graphic>
 </wp:inline></w:drawing></w:r></w:p></w:body></w:document>'''
    rels = b'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
</Relationships>'''
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", rels)
        archive.writestr("word/media/image1.png", png)
    return png


def known_image() -> dict[str, str]:
    return {"sha256": KNOWN_SCHEDULE_IMAGE_SHA256, "extracted_path": "media/image1.png"}


def test_stdlib_docx_xml_media_table_and_source_unchanged(tmp_path: Path) -> None:
    docx = tmp_path / "fixture.docx"
    png = make_docx(docx)
    before = hashlib.sha256(docx.read_bytes()).hexdigest()
    result = extract_docx(docx, tmp_path / "out")
    assert any(row["text"] == "Dataset A.12" for row in result["paragraphs"])
    assert any(row["content"] == "Header" for row in result["tables"])
    assert len(result["media"]) == 1
    media = result["media"][0]
    assert media["width"] == 100 and media["height"] == 20 and media["relationship_ids"] == "rId5"
    assert Path(media["extracted_path"]).read_bytes() == png
    assert hashlib.sha256(docx.read_bytes()).hexdigest() == before


def test_transcription_is_candidate_never_approved() -> None:
    rows = make_schedule(known_image(), Path("official.docx"))
    assert len(rows) == 11
    assert all(row["human_review_status"] == "needs_human_confirmation" for row in rows)
    assert all(row["transcription_method"] == "manual_visual_transcription_by_codex" for row in rows)
    assert not any("approved" in row["human_review_status"] for row in rows)


def test_unknown_image_produces_blank_review_template() -> None:
    rows = make_schedule({"sha256": "unknown", "extracted_path": "media/unknown.png"}, Path("official.docx"))
    assert len(rows) == 11 and all(not row["attack_name"] for row in rows)
    assert all(row["human_review_status"] == "needs_human_confirmation" for row in rows)


def test_eleven_records_validate_and_remain_pending_review() -> None:
    validation, summary = validate_schedule(make_schedule(known_image(), Path("official.docx")))
    assert len(validation) == 11 and summary["missing_field_count"] == 0
    assert summary["invalid_start_count"] == summary["invalid_end_count"] == 0
    assert summary["outside_attack_phase_count"] == 0
    assert summary["duplicate_attack_ids"] == []


def test_invalid_reversed_and_out_of_phase_are_reported() -> None:
    rows = make_schedule(known_image(), Path("official.docx"))
    rows[0]["start_time"] = "not-a-time"
    rows[1]["start_time"], rows[1]["end_time"] = "14:00:00", "13:59:00"
    rows[2]["start_time"], rows[2]["end_time"] = "12:00:00", "12:01:00"
    _, summary = validate_schedule(rows)
    assert summary["invalid_start_count"] == 1
    assert summary["reversed_interval_count"] == 1
    assert summary["outside_attack_phase_count"] >= 2


def test_overlap_is_reported_not_deleted() -> None:
    rows = make_schedule(known_image(), Path("official.docx"))
    rows[1]["start_time"], rows[1]["end_time"] = "13:04:00", "13:06:00"
    validation, summary = validate_schedule(rows)
    assert summary["positive_duration_overlap_pairs"]
    assert len(validation) == 11


def test_boundary_policy_counts_actual_seconds() -> None:
    rows = make_schedule(known_image(), Path("official.docx"))
    result = label_policy_candidates(rows)
    assert result["closed_interval"]["interval_sum_point_count"] == 3131
    assert result["closed_interval"]["unique_session_point_count"] == 3130
    assert result["left_closed_right_open"]["unique_session_point_count"] == 3120
    assert result["unique_point_count_difference"] == 10
    assert result["formal_labels_generated"] is False


def test_phase_is_not_point_label_and_february_remains_unresolved() -> None:
    rows = phase_rows(Path("official.docx"))
    assert rows[0]["phase_type"] == "normal_phase"
    assert "not a point label" in rows[0]["interpretation"]
    assert rows[1]["phase_type"] == "attack_experiment_phase" and "intermittent" in rows[1]["interpretation"]
    assert [row["phase_type"] for row in rows[2:]] == ["unresolved", "unresolved"]


def test_bad_input_is_located_without_source_mutation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "11-Mar-2026_0900_1700.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_stamp", "sensor"])
        writer.writerow(["11-Mar-2026 13:01:00", "Bad Input"])
        writer.writerow(["11-Mar-2026 13:10:00", "1"])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "bad.csv"
    summary = scan_bad_inputs(data_dir, make_schedule(known_image(), Path("official.docx")), output)
    assert summary["occurrence_count"] == 1 and summary["values_modified"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert "Bad Input" in output.read_text(encoding="utf-8-sig")


def test_source_has_no_model_hai_n0_n1_label_or_split_mutation() -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "swat2026" / "extract_and_review_info_sheet.py"
    text = source.read_text(encoding="utf-8").lower()
    assert "import torch" not in text and "torch.load" not in text and "load_state_dict" not in text
    assert "data/hai.py" not in text and "data/swat2026.py" not in text
    assert "labels.csv" not in text and "split.csv" not in text and "to_parquet" not in text
    assert len(TRANSCRIPTION) == 11
