from scraper.page_classifier import classify_page
from scraper.statement_locator import locate_statements


def test_adjacent_statements_form_one_neighborhood():
    texts = [
        "cover",
        "Balance Sheet\nAssets\nLiabilities\nEquity\nTotal Assets",
        "Balance Sheet continued\nCurrent Assets\nTotal Liabilities",
        "Statement of Operations\nSales\nGross Profit\nNet Income",
        "Statement of Cash Flows\nOperating Activities\nInvesting Activities\nFinancing Activities",
        "Notes",
    ]
    pages = [
        {"number": i, "statement_scores": classify_page(text)}
        for i, text in enumerate(texts, start=1)
    ]
    regions = locate_statements(pages)
    assert {r.statement_type for r in regions} == {
        "balance_sheet",
        "income_statement",
        "cash_flow",
    }
    assert max(r.end_page for r in regions) - min(r.start_page for r in regions) <= 8
