"""End-to-end regression tests for the financial PDF evidence pipeline."""
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
    assert "notes" not in categories_found


def test_pipeline_reports_document_metadata_key():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH))
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
    balance_sheet_page = next(p for p in result["pages"] if "Balance Sheet" in p["raw_text"])
    flat_cells = [cell for table in balance_sheet_page["tables"] for row in table for cell in row]
    assert any("1,245" in cell for cell in flat_cells)


def test_pipeline_ocr_fallback_produces_text():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH))
    scanned_pages = [p for p in result["pages"] if p["extraction_method"] == "scanned"]
    assert len(scanned_pages) == 1
    assert "Stockholders" in scanned_pages[0]["raw_text"] or "Stockholder" in scanned_pages[0]["raw_text"]


def test_pipeline_does_not_flag_mda_only_pages():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH))
    flagged_human_pages = {p["page_number_human"] for p in result["pages"]}
    assert 5 not in flagged_human_pages


def test_pipeline_rejects_padded_prose_as_financial_table():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH), debug=True)
    candidate_pages = {t["page_number_human"] for t in result["_debug"]["table_candidates"]}
    assert 11 not in candidate_pages


def test_pipeline_isolates_core_statement_tables():
    _ensure_sample_exists()
    result = extract(str(SAMPLE_PATH))
    statement_tables = result["statement_tables"]
    assert set(statement_tables) == {"balance_sheet", "income_statement", "cash_flow"}
    for statement_type, payload in statement_tables.items():
        assert payload["statement_type"] == statement_type
        assert "tables" in payload
        for table in payload["tables"]:
            assert table["statement_type"] == statement_type
            assert table["statement_confidence"] >= 0.5


def test_pipeline_returns_separate_evidence_layers_and_statement_files():
    _ensure_sample_exists()
    import tempfile
    with tempfile.TemporaryDirectory() as out:
        result = extract(str(SAMPLE_PATH), out_dir=out)
        assert set(result["artifacts"]) >= {
            "document_json", "tables_json", "visuals_json", "statement_files"
        }
        assert Path(result["artifacts"]["document_json"]).exists()
        assert Path(result["artifacts"]["tables_json"]).exists()
        assert Path(result["artifacts"]["visuals_json"]).exists()
        for path in result["artifacts"]["statement_files"].values():
            assert Path(path).exists()
        image_dir = Path(result["artifacts"]["page_image_dir"])
        assert any(image_dir.glob("*.png"))
