from extractor.financial_resolver import build_evidence, metric_from_question, resolve_metric


def sample_data():
    return {
        "summary": {
            "source_name": "test.pdf",
            "metadata": {"currency": "INR", "unit": "crore"},
        },
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
                        ],
                    }
                ]
            }
        },
    }


def test_metric_from_question():
    assert metric_from_question("What was the cash balance?") == "cash_and_equivalents"
    assert metric_from_question("What was EBITDA?") == "ebitda"


def test_resolve_cash_uses_balance_sheet():
    candidates = resolve_metric("cash_and_equivalents", sample_data())
    assert candidates
    assert candidates[0]["values"] == [1245.0, 980.0]
    assert candidates[0]["page"] == 87


def test_build_evidence_is_small_and_targeted():
    evidence = build_evidence("What was the cash balance?", sample_data())
    assert evidence["metric"] == "cash_and_equivalents"
    assert len(evidence["candidates"]) == 1
    assert evidence["candidates"][0]["table_title"] == "Balance Sheet"
