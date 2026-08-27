from scraper.page_classifier import classify_page


def test_balance_sheet_structure_scores_high():
    text = "Balance Sheet\nAssets\nCurrent Assets\nTotal Assets\nLiabilities\nEquity\nTotal Liabilities"
    scores = classify_page(text)
    assert scores["balance_sheet"] > scores["income_statement"]
    assert scores["balance_sheet"] > scores["cash_flow"]


def test_cash_flow_structure_scores_high():
    text = "Statement of Cash Flows\nCash Flows from Operating Activities\nCash Flows from Investing Activities\nCash Flows from Financing Activities"
    scores = classify_page(text)
    assert scores["cash_flow"] > scores["balance_sheet"]
    assert scores["cash_flow"] > scores["income_statement"]
