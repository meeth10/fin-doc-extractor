from extractor.classify import classify_page, MIN_CHARS_FOR_DIGITAL


class FakePage:
    """Stand-in for a PyMuPDF page — classify_page only calls get_text()."""
    def __init__(self, text):
        self._text = text

    def get_text(self, _mode):
        return self._text


def test_digital_page_with_real_text():
    assert classify_page(FakePage("Consolidated Balance Sheets " * 5)) == "digital"


def test_empty_page_is_scanned():
    assert classify_page(FakePage("")) == "scanned"


def test_whitespace_only_page_is_scanned():
    assert classify_page(FakePage("   \n\n   ")) == "scanned"


def test_boundary_just_under_threshold_is_scanned():
    assert classify_page(FakePage("x" * (MIN_CHARS_FOR_DIGITAL - 1))) == "scanned"


def test_boundary_at_threshold_is_digital():
    assert classify_page(FakePage("x" * MIN_CHARS_FOR_DIGITAL)) == "digital"
