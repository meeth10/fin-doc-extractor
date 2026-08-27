from pathlib import Path

from extractor.pipeline import extract


def test_real_report_smoke(tmp_path):
    pdf = Path(__file__).resolve().parents[1] / "samples" / "report.pdf"
    assert pdf.exists(), f"missing real-report fixture: {pdf}"

    result = extract(
        str(pdf),
        out_dir=str(tmp_path),
        include_notes=False,
        debug=False,
        render_images=False,
    )

    assert result["total_pages"] > 1
    assert result["flagged_page_count"] > 0
    assert result["table_summary"]["candidate_count"] > 0

    statement_categories = {
        "balance_sheet",
        "income_statement",
        "cash_flow",
        "equity",
    }
    found_categories = {
        category
        for section in result["sections_found"]
        for category in section["categories"]
    }
    assert found_categories & statement_categories, (
        f"No financial statement section detected; found={found_categories}"
    )

    artifacts = result.get("artifacts", {})
    extracted_json = artifacts.get("document_json")
    assert extracted_json and Path(extracted_json).exists()
