"""
End-to-end regression test: builds the synthetic sample PDF (cover + MD&A
filler + real digital financial statements + one OCR-only scanned page)
and asserts the pipeline finds the right sections and pulls real table data.

This is the main regression check — if a future change to the locator or
table extraction breaks something, this is what catches it.
"""
import subprocess
import sys
from pathlib import Path

from extractor.pipeline import extract

SAMPLE_PATH = Path(__file__).parent.parent / "samples" / "sample_annual_report.pdf"


def _ensure_sample_exists():
    if not SAMPLE_PATH.exists():
        subprocess.run(
            [sys.executable, "samples/make_sample.py"],
            cwd=SAMPLE_PATH.parent.parent,
            check=True,
        )


def test_pipeline_finds_all_statement_sections():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH))

    categories_found = set()
    for s in result["sections_found"]:
        categories_found.update(s["categories"])

    assert {"balance_sheet", "income_statement", "cash_flow", "equity"} <= categories_found
    # notes pages should NOT be flagged by default
    assert "notes" not in categories_found


def test_pipeline_reports_document_metadata_key():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH))
    # synthetic sample has no cover-page unit/FY markers, so values may be
    # None — the point of this test is that the key exists with the right
    # shape, not that detection succeeds on a doc that doesn't have the info.
    assert set(result["document_metadata"].keys()) == {
        "company_name", "financial_year", "currency", "unit_scale",
        "standalone_or_consolidated",
    }


def test_pipeline_debug_mode_includes_page_scores():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH), debug=True)
    assert "_debug" in result
    assert len(result["_debug"]["page_scores"]) >= 4


def test_pipeline_extracts_real_table_data():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH))

    balance_sheet_page = next(
        p for p in result["pages"] if "Balance Sheet" in p["raw_text"]
    )
    flat_cells = [cell for table in balance_sheet_page["tables"] for row in table for cell in row]
    assert any("1,245" in cell for cell in flat_cells)  # cash and cash equivalents FY2025


def test_pipeline_ocr_fallback_produces_text():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH))

    scanned_pages = [p for p in result["pages"] if p["extraction_method"] == "scanned"]
    assert len(scanned_pages) == 1
    assert "Stockholders" in scanned_pages[0]["raw_text"] or \
           "Stockholder" in scanned_pages[0]["raw_text"]


def test_pipeline_does_not_flag_mda_only_pages():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH))
    flagged_human_pages = {p["page_number_human"] for p in result["pages"]}
    # page 5 is deep in the MD&A filler section, nowhere near any statement
    assert 5 not in flagged_human_pages


def test_pipeline_rejects_padded_prose_as_financial_table():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH), debug=True)
    candidate_pages = {t["page_number_human"] for t in result["_debug"]["table_candidates"]}
    # The MD&A page is padded into the located section but must not be a table-search target.
    assert 11 not in candidate_pages


def test_pipeline_returns_separate_evidence_layers():
    _ensure_sample_exists()
    import tempfile
    with tempfile.TemporaryDirectory() as out:
        result = extract(str(SAMPLE_PATH), out_dir=out)
        assert set(result["artifacts"]) >= {"document_json", "tables_json", "visuals_json"}
        assert Path(result["artifacts"]["document_json"]).exists()
        assert Path(result["artifacts"]["tables_json"]).exists()
        assert Path(result["artifacts"]["visuals_json"]).exists()
        image_dir = Path(result["artifacts"]["page_image_dir"])
        assert any(image_dir.glob("*.png"))
