from __future__ import annotations

from typing import Any

import fitz
import pdfplumber

from .models import TableEvidence


def _clean(rows: Any) -> list[list[str]]:
    result: list[list[str]] = []
    if not rows:
        return result
    for row in rows:
        if row is None:
            continue
        cleaned = ["" if cell is None else str(cell).strip() for cell in row]
        if any(cell for cell in cleaned):
            result.append(cleaned)
    return result


def _confidence(rows: list[list[str]]) -> float:
    if len(rows) < 2:
        return 0.0
    widths = [len(row) for row in rows if row]
    if not widths:
        return 0.0
    consistency = max(0.0, 1.0 - (max(widths) - min(widths)) / max(1, max(widths)))
    numeric_cells = sum(1 for row in rows for cell in row if any(ch.isdigit() for ch in cell))
    numeric_density = min(1.0, numeric_cells / max(4, sum(widths) * 0.35))
    return round(min(1.0, 0.45 * consistency + 0.55 * numeric_density), 3)


def _pymupdf_tables(page: fitz.Page) -> list[TableEvidence]:
    tables: list[TableEvidence] = []
    try:
        finder = page.find_tables()
    except Exception:
        return tables
    for table in finder.tables:
        rows = _clean(table.extract())
        confidence = _confidence(rows)
        if rows:
            bbox = tuple(float(x) for x in table.bbox)
            tables.append(TableEvidence(page.number + 1, rows, bbox=bbox, extraction_method="pymupdf", confidence=confidence))
    return tables


def _pdfplumber_tables(pdf: pdfplumber.PDF, page_number: int) -> list[TableEvidence]:
    page = pdf.pages[page_number - 1]
    tables: list[TableEvidence] = []
    settings_variants = (
        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
        {"vertical_strategy": "text", "horizontal_strategy": "text"},
        {"vertical_strategy": "lines", "horizontal_strategy": "text"},
    )
    for settings in settings_variants:
        try:
            found = page.find_tables(table_settings=settings)
        except Exception:
            continue
        for table in found:
            rows = _clean(table.extract())
            confidence = _confidence(rows)
            if not rows:
                continue
            bbox = tuple(float(x) for x in table.bbox)
            candidate = TableEvidence(page_number, rows, bbox=bbox, extraction_method="pdfplumber", confidence=confidence)
            if not any(existing.rows == candidate.rows and existing.bbox == candidate.bbox for existing in tables):
                tables.append(candidate)
        if tables:
            break
    return tables


def extract_tables(pdf_path: str, pages: list[int]) -> list[TableEvidence]:
    """Extract candidate tables from statement pages, preferring PyMuPDF then pdfplumber."""
    results: list[TableEvidence] = []
    seen: set[tuple[int, tuple[tuple[str, ...], ...]]] = set()

    with fitz.open(pdf_path) as doc:
        for page_number in pages:
            if page_number < 1 or page_number > doc.page_count:
                continue
            candidates = _pymupdf_tables(doc[page_number - 1])
            for table in candidates:
                key = (table.page, tuple(tuple(row) for row in table.rows))
                if key not in seen:
                    seen.add(key)
                    results.append(table)

    with pdfplumber.open(pdf_path) as pdf:
        for page_number in pages:
            if page_number < 1 or page_number > len(pdf.pages):
                continue
            if any(t.page == page_number and t.confidence >= 0.65 for t in results):
                continue
            candidates = _pdfplumber_tables(pdf, page_number)
            for table in candidates:
                key = (table.page, tuple(tuple(row) for row in table.rows))
                if key not in seen:
                    seen.add(key)
                    results.append(table)

    return sorted(results, key=lambda t: (t.page, -t.confidence))
