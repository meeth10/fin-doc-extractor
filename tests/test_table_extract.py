from extractor.table_extract import detect_statement_title


def test_statement_title_detects_balance_sheet():
    statement_type, title = detect_statement_title(
        "Frontier Springs Limited\nBalance Sheet\nFor the year ended 31st March, 2025\n"
    )
    assert statement_type == "balance_sheet"
    assert title == "Balance Sheet"


def test_statement_title_detects_profit_and_loss():
    statement_type, title = detect_statement_title(
        "Frontier Springs Limited | Annual Report 2024-25\nProfit and Loss Statement\n"
    )
    assert statement_type == "income_statement"
    assert title == "Profit and Loss Statement"


def test_statement_title_detects_cash_flow():
    statement_type, title = detect_statement_title(
        "Statement of Cash Flows\nFor the year ended 31 March 2025\n"
    )
    assert statement_type == "cash_flow"
    assert title == "Statement of Cash Flows"


def test_statement_title_handles_ocr_spacing_and_case():
    statement_type, title = detect_statement_title(
        "CONSOLIDATED  STATEMENT OF PROFIT AND LOSS\nFigures in Rs. Crores\n"
    )
    assert statement_type == "income_statement"
    assert "STATEMENT OF PROFIT AND LOSS" in title.upper()
