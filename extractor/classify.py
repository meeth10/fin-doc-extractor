"""
Per-page classification: does this page have a usable digital text layer,
or does it need OCR (scanned image, embedded-image financial table, etc.)?

Deterministic rule, no ML/LLM: if the digital text layer for a page is
shorter than MIN_CHARS_FOR_DIGITAL (after stripping whitespace), we treat
the page as "scanned" and route it to OCR instead.
"""

MIN_CHARS_FOR_DIGITAL = 20


def classify_page(fitz_page) -> str:
    """Return 'digital' or 'scanned' for a PyMuPDF page object."""
    text = fitz_page.get_text("text") or ""
    if len(text.strip()) >= MIN_CHARS_FOR_DIGITAL:
        return "digital"
    return "scanned"
