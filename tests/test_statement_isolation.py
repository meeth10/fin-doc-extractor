from extractor.pipeline import _primary_title, _statement_sequence_candidates


def test_primary_title_detects_profit_and_loss():
    text = "Frontier Springs Limited\nProfit and Loss Statement\nFor the year ended 31st March, 2025\n(Amount in Lacs)\nRevenue from operations 123.00 100.00"
    title = _primary_title(text)
    assert title is not None
    assert title["statement_type"] == "income_statement"
    assert title["is_primary_title"] is True
    assert title["has_year_header"] is True


def test_primary_title_ignores_buried_note_reference():
    lines = "\n".join([
        "Notes to the Financial Statements",
        "Note 25 Trade Receivables",
        *(["Some explanatory prose"] * 8),
        "The amount recognised in the balance sheet is disclosed below.",
    ])
    title = _primary_title(lines)
    assert title is None


def test_statement_sequence_prefers_bs_income_cf_successession():
    pages = [
        {"page_number": 86, "statement_title": {"statement_type": "balance_sheet", "is_primary_title": True}},
        {"page_number": 89, "statement_title": {"statement_type": "income_statement", "is_primary_title": True}},
        {"page_number": 90, "statement_title": {"statement_type": "cash_flow", "is_primary_title": True}},
        {"page_number": 120, "statement_title": {"statement_type": "balance_sheet", "is_primary_title": True}},
    ]
    tables = [
        {"page_number": 86, "validated": True, "score": 0.95},
        {"page_number": 89, "validated": True, "score": 0.98},
        {"page_number": 90, "validated": True, "score": 0.99},
        {"page_number": 120, "validated": True, "score": 0.90},
    ]
    selected = _statement_sequence_candidates(pages, tables)
    assert selected == {
        "balance_sheet": [86],
        "income_statement": [89],
        "cash_flow": [90],
    }
