from extractor.financial_facts import build_fact_store


def sample_data():
    return {
        "summary": {"source_name": "test.pdf", "metadata": {"currency": "INR", "unit": "crore"}},
        "document": {"pages": []},
        "statement_tables": {
            "balance_sheet": {
                "tables": [
                    {
                        "page_number": 86,
                        "page_number_human": 87,
                        "table_title": "Balance Sheet",
                        "source": "pymupdf_layout",
                        "score": 1.0,
                        "validated": True,
                        "statement_assignment": "title",
                        "table": [
                            ["Particulars", "2025", "2024"],
                            ["Cash and cash equivalents", "1,245", "980"],
                            ["Total debt", "2,000", "2,200"],
                        ],
                    }
                ]
            },
            "income_statement": {
                "tables": [
                    {
                        "page_number": 88,
                        "page_number_human": 89,
                        "table_title": "Profit and Loss Statement",
                        "source": "pymupdf_layout",
                        "score": 1.0,
                        "validated": True,
                        "statement_assignment": "title",
                        "table": [
                            ["Particulars", "2025", "2024"],
                            ["Revenue", "12,000", "10,000"],
                            ["Operating profit", "3,000", "2,500"],
                            ["Depreciation and amortisation", "500", "450"],
                        ],
                    }
                ]
            },
            "cash_flow": {"tables": []},
        },
    }


def test_fact_store_is_periodized_and_source_linked():
    store = build_fact_store(sample_data())
    cash = [f for f in store["facts"] if f["metric"] == "cash_and_equivalents"]
    assert {f["period"] for f in cash} == {"2025", "2024"}
    assert all(f["page"] == 87 for f in cash)


def test_fact_store_includes_derived_ebitda():
    store = build_fact_store(sample_data())
    ebitda = [f for f in store["facts"] if f["metric"] == "ebitda" and f["status"] == "derived"]
    assert ebitda
    assert ebitda[0]["value"] == 3500.0
    assert ebitda[0]["formula"] == "EBIT + depreciation & amortisation"
