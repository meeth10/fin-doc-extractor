from extractor.metadata import detect_metadata

COVER_PAGE_INR_LAKHS = """
Acme Global Holdings Limited
Annual Report — Financial Year 2025-26
(All amounts in Rs. in Lakhs unless otherwise stated)
Standalone Financial Statements
"""

COVER_PAGE_USD_MILLIONS = """
Northwind Manufacturing Corporation
Annual Report — FY2025
(Amounts in USD in millions, unless otherwise indicated)
Consolidated Financial Statements
"""


def test_unit_detection_inr_lakhs():
    meta = detect_metadata([COVER_PAGE_INR_LAKHS])
    assert meta.currency == "INR"
    assert meta.unit_scale == "lakhs"


def test_unit_detection_usd_millions():
    meta = detect_metadata([COVER_PAGE_USD_MILLIONS])
    assert meta.currency == "USD"
    assert meta.unit_scale == "millions"


def test_currency_symbol_crores():
    meta = detect_metadata(["Figures are in ₹ Crores throughout this report."])
    assert meta.currency == "INR"
    assert meta.unit_scale == "crores"


def test_financial_year_detection():
    meta = detect_metadata([COVER_PAGE_INR_LAKHS])
    assert meta.financial_year == "FY2026"


def test_standalone_detection():
    meta = detect_metadata([COVER_PAGE_INR_LAKHS])
    assert meta.standalone_or_consolidated == "standalone"


def test_consolidated_detection():
    meta = detect_metadata([COVER_PAGE_USD_MILLIONS])
    assert meta.standalone_or_consolidated == "consolidated"


def test_company_name_detection():
    meta = detect_metadata([COVER_PAGE_INR_LAKHS])
    assert meta.company_name == "Acme Global Holdings Limited"


def test_no_metadata_found_returns_none_fields_not_garbage():
    meta = detect_metadata(["just some generic prose with no report markers at all"])
    assert meta.currency is None
    assert meta.unit_scale is None
    assert meta.financial_year is None
    assert meta.standalone_or_consolidated is None


def test_both_standalone_and_consolidated_mentioned_is_left_ambiguous():
    text = "This report includes both Standalone and Consolidated financial statements."
    meta = detect_metadata([text])
    assert meta.standalone_or_consolidated is None
