from extractor.financial_resolver import build_evidence
from agent.financial_agent import _select_reported_candidate


def test_malformed_revenue_table_is_rescued_by_titled_aggregate_text():
    data = {
        "summary": {"source_name": "Apple.pdf", "metadata": {"currency": "USD", "unit": "millions"}},
        "document": {"pages": [{
            "page_number_human": 32,
            "raw_text": "Apple Inc.\nCONSOLIDATED STATEMENTS OF OPERATIONS\nYears ended September 27, 2025 September 28, 2024 September 30, 2023\nProducts 307,003 294,866 298,085\nServices 109,158 96,169 85,200\nTotal net sales 416,161 391,035 383,285",
        }]},
        "statement_tables": {
            "income_statement": {"tables": [{
                "page_number_human": 32,
                "table_title": "Consolidated Statements of Operations",
                "source": "pymupdf_layout",
                "score": 1.0,
                "validated": True,
                "statement_assignment": "title",
                "table": [["Particulars", "2025", "2024", "2023"], ["Revenue", "307,003", "294,866", "298,085"]],
            }]},
            "balance_sheet": {"tables": []},
            "cash_flow": {"tables": []},
        },
    }
    evidence = build_evidence("What was revenue?", data)
    selected = _select_reported_candidate("What was revenue?", evidence)
    assert selected is not None
    assert selected[0] == 416161.0
    assert selected[2]["matched_alias"] == "total net sales"
    assert selected[2]["source"] == "raw_text"
