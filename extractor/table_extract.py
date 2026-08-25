"""Hybrid financial table extraction and validation.

The extractor intentionally returns *candidates*, not facts. PDF table libraries can
mistake positioned prose for tables, so every candidate is scored using financial
shape signals before it is promoted to ``validated``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

TEXT_STRATEGY_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "intersection_tolerance": 5,
    "snap_tolerance": 3,
    "join_tolerance": 3,
}

NUM_RE = re.compile(
    r"(?:\(?[₹$€£]?\s*[-−]?\s*[\d,]+(?:\.\d+)?%?\)?|[-−]?\s*[\d,]+(?:\.\d+)?%?)"
)
YEAR_RE = re.compile(r"\b(?:FY\s*)?20\d{2}(?:[-/–]\d{2})?\b", re.I)
LABEL_RE = re.compile(r"[A-Za-z][A-Za-z&().,/\- ]{2,}")


def _clean(raw_table: Sequence[Sequence[Any]]) -> List[List[str]]:
    out: List[List[str]] = []
    for row in raw_table:
        if row is None:
            continue
        cleaned = ["" if cell is None else str(cell).strip() for cell in row]
        if any(cleaned):
            out.append(cleaned)
    return out


def _numeric_count(text: str) -> int:
    return len(NUM_RE.findall(text))


def _financial_row_like(row: Sequence[str]) -> bool:
    text = " ".join(cell for cell in row if cell).strip()
    return bool(LABEL_RE.search(text)) and _numeric_count(text) >= 1


def score_table(table: Sequence[Sequence[str]]) -> Dict[str, Any]:
    """Score a table candidate for *financial usefulness*, not visual existence."""
    rows = [_clean([row])[0] if row else [] for row in table]
    rows = [r for r in rows if any(r)]
    if not rows:
        return {"score": 0.0, "validated": False, "reason": "empty"}

    nonempty_cells = [c for r in rows for c in r if c]
    numeric_cells = sum(_numeric_count(c) > 0 for c in nonempty_cells)
    numeric_tokens = sum(_numeric_count(c) for c in nonempty_cells)
    label_cells = sum(bool(LABEL_RE.search(c)) for c in nonempty_cells)
    year_cells = sum(bool(YEAR_RE.search(c)) for c in nonempty_cells)
    width = max(len(r) for r in rows)
    row_count = len(rows)
    shape_rows = sum(_financial_row_like(r) for r in rows)
    widths = [len(r) for r in rows]
    consistent = sum(1 for w in widths if abs(w - width) <= 1) / max(1, len(widths))
    avg_cell_len = sum(len(c) for c in nonempty_cells) / max(1, len(nonempty_cells))

    numeric_density = numeric_cells / max(1, len(nonempty_cells))
    score = 0.0
    score += min(0.45, numeric_density * 0.55)
    score += min(0.25, shape_rows / max(1, row_count) * 0.30)
    score += min(0.12, consistent * 0.12)
    score += 0.08 if width >= 2 else 0
    score += 0.05 if year_cells else 0
    score += 0.10 if row_count >= 4 else 0
    score += 0.05 if label_cells >= 3 else 0

    # Strong negative signals for prose fragmented into many tiny cells.
    if width >= 10 and numeric_tokens == 0:
        score -= 0.45
    if width >= 10 and numeric_density < 0.20:
        score -= 0.35
    if width >= 15 and row_count >= 20 and numeric_density < 0.30:
        score -= 0.45
    if avg_cell_len < 3 and numeric_tokens == 0:
        score -= 0.30
    if row_count <= 2 and numeric_tokens == 0:
        score -= 0.15

    score = round(max(0.0, min(1.0, score)), 3)
    validated = bool(
        score >= 0.52
        and row_count >= 3
        and numeric_tokens >= 3
        and shape_rows >= 2
    )
    if validated:
        reason = "financial_shape"
    elif width >= 10 and numeric_tokens == 0:
        reason = "fragmented_prose"
    elif numeric_tokens < 3:
        reason = "too_few_numeric_values"
    else:
        reason = "weak_financial_shape"

    return {
        "score": score,
        "validated": validated,
        "reason": reason,
        "rows": row_count,
        "columns": width,
        "numeric_tokens": numeric_tokens,
        "numeric_density": round(numeric_density, 3),
        "shape_rows": shape_rows,
        "year_cells": year_cells,
        "avg_cell_length": round(avg_cell_len, 2),
    }


def _pdfplumber_candidates(pdfplumber_page) -> List[Tuple[str, List[List[str]]]]:
    found: List[Tuple[str, List[List[str]]]] = []
    try:
        for table in pdfplumber_page.extract_tables() or []:
            found.append(("pdfplumber_grid", _clean(table)))
    except Exception:
        pass
    try:
        for table in pdfplumber_page.extract_tables(table_settings=TEXT_STRATEGY_SETTINGS) or []:
            found.append(("pdfplumber_text", _clean(table)))
    except Exception:
        pass
    return found


def _token_is_numeric(text: str) -> bool:
    text = text.strip()
    return bool(NUM_RE.fullmatch(text)) or bool(NUM_RE.search(text))


def _layout_candidates(fitz_page) -> List[List[List[str]]]:
    """Reconstruct borderless tables from word coordinates as a fallback."""
    words = fitz_page.get_text("words")
    if not words:
        return []

    lines: List[List[Tuple[float, float, str]]] = []
    y_tol = 3.5
    for x0, y0, x1, y1, word, *_ in sorted(words, key=lambda w: (w[1], w[0])):
        if not word.strip():
            continue
        placed = False
        cy = (y0 + y1) / 2
        for line in lines:
            if abs(cy - line[0][1]) <= y_tol:
                line.append((x0, cy, word.strip()))
                placed = True
                break
        if not placed:
            lines.append([(x0, cy, word.strip())])

    rows: List[List[Tuple[float, str]]] = []
    for raw in lines:
        raw.sort(key=lambda t: t[0])
        rows.append([(x, text) for x, _, text in raw])

    financial_rows: List[List[str]] = []
    for row in rows:
        nums = [i for i, (_, text) in enumerate(row) if _token_is_numeric(text)]
        if len(nums) < 1 or not any(LABEL_RE.search(text) for _, text in row):
            continue
        cells: List[str] = []
        current: List[str] = []
        last_x = None
        for x, text in row:
            if last_x is not None and x - last_x > 18 and current:
                cells.append(" ".join(current).strip())
                current = []
            current.append(text)
            last_x = x + max(4, len(text) * 4.2)
        if current:
            cells.append(" ".join(current).strip())
        if len(cells) >= 2:
            financial_rows.append(cells)

    if len(financial_rows) < 3:
        return []
    return [financial_rows]


def extract_table_candidates(pdfplumber_page, fitz_page=None) -> List[Dict[str, Any]]:
    """Return ranked table candidates from multiple extraction strategies."""
    candidates: List[Dict[str, Any]] = []
    for source, table in _pdfplumber_candidates(pdfplumber_page):
        if table:
            metrics = score_table(table)
            candidates.append({"source": source, "table": table, **metrics})

    if fitz_page is not None:
        for table in _layout_candidates(fitz_page):
            metrics = score_table(table)
            candidates.append({"source": "pymupdf_layout", "table": table, **metrics})

    # De-duplicate exact tables, keeping the strongest source/score.
    unique: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        key = repr(candidate["table"])
        if key not in unique or candidate["score"] > unique[key]["score"]:
            unique[key] = candidate

    return sorted(unique.values(), key=lambda c: c["score"], reverse=True)


def extract_tables_from_page(pdfplumber_page) -> List[List[List[str]]]:
    """Backward-compatible API: return only raw pdfplumber tables."""
    return [_clean(t) for _, t in _pdfplumber_candidates(pdfplumber_page)]
