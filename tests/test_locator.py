from extractor.locator import (
    score_page, score_pages, merge_into_sections, locate_financial_statements,
    extract_year_labels,
)

MDA = ("Management's Discussion and Analysis. Lorem ipsum outlook and market "
       "conditions commentary, no statement content here at all whatsoever.")

BALANCE_SHEET_PAGE = """
Consolidated Balance Sheets
As at 31 March 2025 and 31 March 2024 (Rs. in millions)
                                          2025-26        2024-25
Non-current assets
Property, plant and equipment              1,200.00       1,050.00
Investments                                  300.00         250.00
Current assets
Inventories                                  450.00         400.00
Trade receivables                            600.00         540.00
Cash and cash equivalents                  1,245.00         980.00
Total assets                               9,884.00       9,010.00
Borrowings                                 2,000.00       1,800.00
Trade payables                               900.00         850.00
Total equity                               5,664.00       5,000.00
"""

INCOME_STATEMENT_PAGE = """
Consolidated Statements of Operations
For the year ended 2025-26 and 2024-25 (Rs. in millions)
Revenue from operations                    12,400.00      11,200.00
Other income                                  120.00         100.00
Total income                               12,520.00      11,300.00
Cost of materials                           7,300.00       6,800.00
Employee benefits                           1,200.00       1,050.00
Finance costs                                 210.00         190.00
Depreciation                                  400.00         360.00
Other expenses                                800.00         700.00
Profit before tax                           2,610.00       2,200.00
Tax expense                                   505.00         510.00
Profit for the year                         2,105.00       1,690.00
"""

CASH_FLOW_PAGE = """
Consolidated Statements of Cash Flows
For the year ended 2025-26 and 2024-25 (Rs. in millions)
Cash flow from operating activities
Working capital changes                       200.00         180.00
Income tax paid                               505.00         480.00
Cash generated from operations              2,950.00       2,410.00
Investing activities
Purchase of property                        1,100.00         980.00
Financing activities
Net increase                                1,150.00         780.00
Net decrease                                    0.00           0.00
Cash and cash equivalents                   1,245.00         980.00
"""

EQUITY_PAGE = """
Statement of Changes in Equity
For the year ended 2025-26 and 2024-25 (Rs. in millions)
Equity share capital                          500.00         500.00
Other equity                                4,900.00       4,300.00
Securities premium                          1,200.00       1,000.00
General reserve                               300.00         300.00
Retained earnings                           3,964.00       3,500.00
Total comprehensive income                  2,105.00       1,690.00
"""

CONTENTS_PAGE = """
Contents

Directors' Report ............................. 12
Management Discussion and Analysis ............ 25
Consolidated Balance Sheet ..................... 69
Statement of Profit and Loss ................... 71
Notes to Financial Statements .................. 85
"""

AUDITOR_REPORT_PAGE = """
Independent Auditor's Report

We have audited the accompanying consolidated Balance Sheet of the Company
as at 31 March 2025. Basis for opinion: our audit was conducted in
accordance with the standards on auditing. Auditor's responsibility is
described further below.
"""

CROSS_REFERENCE_PAGE = """
Management Discussion and Analysis

For details on capital expenditure, refer note 14 to the financial
statements regarding Property, Plant and Equipment. See note 22 for the
Borrowings breakdown. As per note 5, provisions are disclosed separately.
"""


# --- Phase 1 required tests (spec section 17) --------------------------

def test_contents_page_not_statement():
    score = score_page(CONTENTS_PAGE)
    assert score.status == "none"
    assert score.signals["is_contents_page"] is True


def test_income_statement_detection():
    score = score_page(INCOME_STATEMENT_PAGE)
    assert score.status == "confident"
    assert score.best_category == "income_statement"


def test_balance_sheet_detection():
    score = score_page(BALANCE_SHEET_PAGE)
    assert score.status == "confident"
    assert score.best_category == "balance_sheet"


def test_cash_flow_detection():
    score = score_page(CASH_FLOW_PAGE)
    assert score.status == "confident"
    assert score.best_category == "cash_flow"


def test_equity_detection():
    score = score_page(EQUITY_PAGE)
    assert score.status == "confident"
    assert score.best_category == "equity"


def test_year_alignment_does_not_assume_first_is_latest():
    assert extract_year_labels("2025-26      2024-25") == ["FY2026", "FY2025"]
    # reversed column order — must still map correctly, not just by position
    assert extract_year_labels("2024-25      2025-26") == ["FY2025", "FY2026"]


# --- additional locator behavior ----------------------------------------

def test_auditor_report_page_scores_low_despite_mentioning_balance_sheet():
    score = score_page(AUDITOR_REPORT_PAGE)
    assert score.status != "confident"
    assert score.signals["auditor_hits"] >= 1


def test_cross_reference_page_is_penalized():
    score = score_page(CROSS_REFERENCE_PAGE)
    assert score.status != "confident"
    assert score.signals["cross_reference_hits"] >= 2


def test_score_pages_ignores_notes_by_default():
    hits = score_pages(["Notes to Consolidated Financial Statements\n" + BALANCE_SHEET_PAGE])
    assert "notes" not in hits[0].category_scores


def test_score_pages_includes_notes_when_opted_in():
    hits = score_pages(["Notes to Consolidated Financial Statements\nblah blah"], include_notes=True)
    assert "notes" in hits[0].category_scores


def test_ocr_mangled_text_still_matches():
    # OCR frequently drops the space between adjacent words
    mangled = EQUITY_PAGE.replace("Statement of Changes in Equity", "Statementof Changesin Equity")
    score = score_page(mangled)
    assert score.best_category == "equity"


def test_merge_contiguous_confident_pages_into_one_section():
    pages = [MDA, BALANCE_SHEET_PAGE, INCOME_STATEMENT_PAGE, MDA]
    sections, ambiguous, scores = locate_financial_statements(pages, pad_pages=0)
    assert len(sections) == 1
    assert sections[0]["start_page"] == 1
    assert sections[0]["end_page"] == 2


def test_gap_beyond_tolerance_creates_two_sections():
    pages = [BALANCE_SHEET_PAGE, MDA, MDA, MDA, MDA, CASH_FLOW_PAGE]
    sections, ambiguous, scores = locate_financial_statements(pages, gap_tolerance=1, pad_pages=0)
    assert len(sections) == 2


def test_padding_does_not_exceed_document_bounds():
    sections, ambiguous, scores = locate_financial_statements(
        [BALANCE_SHEET_PAGE], gap_tolerance=2, pad_pages=3
    )
    assert sections[0]["start_page"] == 0
    assert sections[0]["end_page"] == 0


def test_ambiguous_page_does_not_anchor_a_section_alone():
    # a page with only a heading and nothing else lands in "ambiguous",
    # not "confident" — it should not appear as its own section.
    weak_page = "Consolidated Balance Sheets\n(continued on next page)"
    sections, ambiguous, scores = locate_financial_statements([MDA, weak_page, MDA])
    assert sections == []
    assert 1 in ambiguous


def test_mda_only_page_gets_no_signal():
    score = score_page(MDA)
    assert score.status == "none"
