from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from .models import EvidenceDocument, Page
from .page_classifier import classify_page
from .statement_locator import locate_statements
from .table_extractor import extract_tables


def _text_quality(text: str) -> float:
    if not text.strip():
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text)
    return round(
        min(
            1.0,
            0.5 * printable / len(text)
            + 0.5 * min(1.0, letters / max(20, len(text) * 0.15)),
        ),
        3,
    )


def scrape_pdf(pdf_path: str) -> EvidenceDocument:
    """Extract raw page text, statement regions, and tables as neutral evidence."""
    source = str(Path(pdf_path).expanduser().resolve())
    with fitz.open(source) as doc:
        page_dicts: list[dict[str, Any]] = []
        pages: list[Page] = []
        page_count = doc.page_count
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            scores = classify_page(text)
            page_dicts.append({"number": number, "statement_scores": scores})
            pages.append(
                Page(
                    number=number,
                    text=text,
                    extraction_method="digital",
                    text_quality=_text_quality(text),
                    statement_scores=scores,
                )
            )

    regions = locate_statements(page_dicts)
    region_pages = [
        page.number
        for page in pages
        if any(region.start_page <= page.number <= region.end_page for region in regions)
    ]
    tables = extract_tables(source, region_pages)

    for table in tables:
        containing = [
            region
            for region in regions
            if region.start_page <= table.page <= region.end_page
        ]
        if containing:
            table.statement_type = min(
                containing,
                key=lambda region: abs(region.start_page - table.page),
            ).statement_type

    return EvidenceDocument(
        source_file=source,
        page_count=page_count,
        metadata={},
        pages=pages,
        statement_regions=regions,
        tables=tables,
    )
