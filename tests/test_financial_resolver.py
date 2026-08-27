from extractor.financial_resolver import build_evidence, metric_from_question, question_intent, resolve_metric


def sample_data():
    return {
        "summary": {"source_name": "test.pdf", "metadata": {"currency": "INR", "unit": "crore"}},
        "document": {"pages": []},
        "statement_tables": {
            "balance_sheet": {"tables": [{"page_number": 86, "page_number_human": 87, "table_title": "Balance Sheet", "source": "pymupdf_layout", "score": 1.0, "validated": True, "statement_assignment": "title", "table": [["Particulars", "2025", "2024"], ["Cash and cash equivalents", "1,245", "980"], ["Total debt", "2,000", "2,200"], ["Market capitalization", "10,000", "9,000"]]}]},
            "income_statement": {"tables": [{"page_number": 88, "page_number_human": 89, "table_title": "Profit and Loss Statement", "source": "pymupdf_layout", "score": 1.0, "validated": True, "statement_assignment": "title", "table": [["Particulars", "2025", "2024"], ["Revenue", "12,000", "10,000"], ["Operating profit", "3,000", "2,500"], ["Depreciation and amortisation", "500", "450"], ["Profit before tax", "2,300", "1,900"], ["Finance costs", "700", "650"], ["Total expenses", "9,000", "7,500"]]}]},
            "cash_flow": {"tables": []},
        },
    }


def test_metric_from_question():
    assert metric_from_question("What was the cash balance?") == "cash_and_equivalents"
    assert metric_from_question("What was EBITDA?") == "ebitda"
    assert metric_from_question("What was EV?") == "enterprise_value"
    assert metric_from_question("What was operational income?") == "ebit"
    assert metric_from_question("What was operating income?") == "ebit"
    assert metric_from_question("What was the expense?") == "total_expenses"
    assert metric_from_question("What were total expenses?") == "total_expenses"


def test_change_intent():
    assert question_intent("Did EBITDA increase or decrease?") == "yoy_change"
    assert question_intent("What was the percentage increase in EBITDA?") == "yoy_percent"


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


def test_ebitda_is_deterministically_derived_from_ebit_and_depreciation():
    evidence = build_evidence("What was EBITDA?", sample_data())
    assert evidence["computed"]["answer"] == 3500.0
    assert evidence["computed"]["status"] == "derived"


def test_enterprise_value_is_market_cap_plus_debt_minus_cash():
    evidence = build_evidence("What was enterprise value?", sample_data())
    assert evidence["computed"]["answer"] == 10755.0
    assert evidence["computed"]["formula"] == "market capitalization + total debt − cash"


def test_sum_and_difference_are_supported():
    add = build_evidence("What is the sum of revenue and EBITDA?", sample_data())
    diff = build_evidence("What is the difference between revenue and EBITDA?", sample_data())
    assert add["computed"]["answer"] == 15500.0
    assert diff["computed"]["answer"] == 8500.0
