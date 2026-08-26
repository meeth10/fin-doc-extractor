from extractor.financial_resolver import build_evidence, metric_from_question, resolve_metric


def sample_data():
    return {
        "summary": {"source_name": "test.pdf", "metadata": {"currency": "INR", "unit": "crore"}},
        "document": {"pages": []},
        "statement_tables": {
            "balance_sheet": {"tables": [{"page_number": 86, "page_number_human": 87, "table_title": "Balance Sheet", "source": "pymupdf_layout", "score": 1.0, "validated": True, "statement_assignment": "title", "table": [["Particulars", "2025", "2024"], ["Cash and cash equivalents", "1,245", "980"], ["Total debt", "2,000", "2,200"], ["Market capitalization", "10,000", "9,000"]]}]},
            "income_statement": {"tables": [{"page_number": 88, "page_number_human": 89, "table_title": "Profit and Loss Statement", "source": "pymupdf_layout", "score": 1.0, "validated": True, "statement_assignment": "title", "table": [["Particulars", "2025", "2024"], ["Revenue", "12,000", "10,000"], ["Operating profit", "3,000", "2,500"], ["Depreciation and amortisation", "500", "450"], ["Profit before tax", "2,300", "1,900"], ["Finance costs", "700", "650"]]}]},
            "cash_flow": {"tables": []},
        },
    }


def apple_style_data():
    data = sample_data()
    data["statement_tables"]["income_statement"]["tables"][0]["table"] = [
        ["Particulars", "September 27, 2025", "September 28, 2024", "September 30, 2023"],
        ["Products", "307,003", "294,866", "298,085"],
        ["Services", "109,158", "96,169", "85,200"],
        ["Total net sales", "416,161", "391,035", "383,285"],
        ["Operating income", "133,050", "123,216", "114,301"],
        ["Net income", "112,010", "93,736", "96,995"],
    ]
    data["statement_tables"]["cash_flow"]["tables"] = [{
        "page_number": 90,
        "page_number_human": 91,
        "table_title": "Consolidated Statements of Cash Flows",
        "source": "pymupdf_layout",
        "score": 1.0,
        "validated": True,
        "statement_assignment": "title",
        "table": [
            ["Particulars", "September 27, 2025", "September 28, 2024", "September 30, 2023"],
            ["Depreciation and amortization", "11,698", "11,519", "11,519"],
        ],
    }]
    return data


def test_metric_from_question():
    assert metric_from_question("What was the cash balance?") == "cash_and_equivalents"
    assert metric_from_question("What was EBITDA?") == "ebitda"
    assert metric_from_question("What was EV?") == "enterprise_value"


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


def test_total_net_sales_resolves_as_revenue_not_year_number():
    evidence = build_evidence("What was revenue at FY2025?", apple_style_data())
    assert evidence["computed"] is None
    assert evidence["candidates"]
    assert evidence["candidates"][0]["matched_alias"] == "total net sales"
    assert evidence["candidates"][0]["values"] == [416161.0, 391035.0, 383285.0]
    assert evidence["candidates"][0]["page"] == 89


def test_ebitda_growth_derives_two_aligned_periods():
    evidence = build_evidence("What was EBITDA growth?", apple_style_data())
    assert evidence["computed"] is not None
    assert evidence["computed"]["latest_value"] == 144748.0
    assert evidence["computed"]["prior_value"] == 134735.0
    assert evidence["computed"]["change"] == 10013.0
