from agent.financial_agent import _deterministic_computation, _total_debt
from agent.financial_facts import build_fact_store


def sample_data():
    return {
        "summary": {"source_name": "test.pdf", "metadata": {"currency": "INR", "unit": "crore"}},
        "document": {"pages": []},
        "statement_tables": {
            "balance_sheet": {
                "tables": [{
                    "page_number": 86, "page_number_human": 87, "table_title": "Consolidated Balance Sheet",
                    "source": "pymupdf_layout", "score": 1.0, "validated": True, "statement_assignment": "title",
                    "table": [
                        ["Particulars", "2025", "2024"],
                        ["Cash and cash equivalents", "1,245", "980"],
                        ["Commercial paper", "300", "250"],
                        ["Term debt", "700", "600"],
                        ["Term debt", "1,000", "900"],
                        ["Total liabilities", "7,000", "6,000"],
                    ],
                }]
            },
            "income_statement": {
                "tables": [{
                    "page_number": 88, "page_number_human": 89, "table_title": "Consolidated Profit and Loss",
                    "source": "pymupdf_layout", "score": 1.0, "validated": True, "statement_assignment": "title",
                    "table": [
                        ["Particulars", "2025", "2024"],
                        ["Revenue", "12,000", "10,000"],
                        ["Operating profit", "3,000", "2,500"],
                        ["Depreciation and amortisation", "500", "450"],
                        ["Total expenses", "9,000", "7,500"],
                    ],
                }]
            },
            "cash_flow": {"tables": []},
        },
    }


def test_fact_store_preserves_numeric_column_identity():
    store = build_fact_store(sample_data())
    revenue = [f for f in store["facts"] if f["metric"] == "revenue" and f["page"] == 89]
    assert [(f["column_index"], f["period"], f["value"]) for f in revenue] == [(1, "2025", 12000.0), (2, "2024", 10000.0)]


def test_total_debt_reconstruction_uses_balance_sheet_components_only():
    store = build_fact_store(sample_data())
    debt = _total_debt(store["facts"])
    assert debt["answer"] == 2000.0
    assert {item["value"] for item in debt["inputs"]} == {300.0, 700.0, 1000.0}


def test_revenue_yoy_change_is_deterministic_and_period_aligned():
    store = build_fact_store(sample_data())
    plan = {
        "metrics": ["revenue"],
        "operation": "yoy_percent",
        "target_period": "2025",
        "comparison_period": "2024",
        "scope": "unknown",
        "basis": "reported",
        "needs_narrative": False,
        "definition": "Revenue growth",
        "confidence": "high",
    }
    result = _deterministic_computation("What was revenue growth?", sample_data(), store["facts"], plan)
    assert result["answer"] == 20.0
    assert result["formula"] == "(latest − prior) / prior × 100"
