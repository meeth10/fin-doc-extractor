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


def _pages(bs, income, cf):
    return [
        {"page_number": bs, "statement_title": {"statement_type": "balance_sheet", "is_primary_title": True, "has_year_header": True}},
        {"page_number": income, "statement_title": {"statement_type": "income_statement", "is_primary_title": True, "has_year_header": True}},
        {"page_number": cf, "statement_title": {"statement_type": "cash_flow", "is_primary_title": True, "has_year_header": True}},
    ]


def _tables(bs, income, cf):
    return [
        {"page_number": bs, "validated": True, "score": 0.95},
        {"page_number": income, "validated": True, "score": 0.98},
        {"page_number": cf, "validated": True, "score": 0.99},
    ]


def test_statement_sequence_prefers_bs_income_cf_successession():
    selected = _statement_sequence_candidates(_pages(86, 89, 90), _tables(86, 89, 90))
    assert selected == {
        "balance_sheet": [86],
        "income_statement": [89],
        "cash_flow": [90],
    }


def test_statement_sequence_accepts_full_cluster_with_five_page_span():
    selected = _statement_sequence_candidates(_pages(86, 88, 91), _tables(86, 88, 91))
    assert selected == {
        "balance_sheet": [86],
        "income_statement": [88],
        "cash_flow": [91],
    }


def test_statement_sequence_rejects_cluster_beyond_five_pages():
    pages = _pages(86, 90, 92)
    tables = _tables(86, 90, 92)
    selected = _statement_sequence_candidates(pages, tables)
    assert selected["balance_sheet"] == [86]
    assert selected["income_statement"] == [90]
    assert selected["cash_flow"] == [92]
    # No complete succession can be used across a >5-page span.
