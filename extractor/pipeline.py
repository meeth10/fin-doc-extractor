"""Financial PDF evidence extraction pipeline."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pymupdf as fitz
import pdfplumber

from .classify import classify_page
from .ocr import ocr_page
from .locator import locate_financial_statements
from .table_extract import extract_table_candidates
from .metadata import detect_metadata

RENDER_DPI = 120
ProgressCallback = Optional[Callable[[int, str], None]]
TARGET_STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")
SEQUENCE_MAX_SPAN = 5
MAX_CONTINUATION_PAGES = 2

TITLE_PATTERNS = {
    "balance_sheet": [
        r"^(?:consolidated\s+)?balance\s+sheets?(?:\s+(?:as\s+(?:at|on|of)\b|for\s+the\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+financial\s+position(?:\s+(?:as\s+(?:at|on|of)\b|for\s+the\b).*)?$",
    ],
    "income_statement": [
        r"^(?:consolidated\s+)?profit\s+and\s+loss\s+statements?(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+profit\s+and\s+loss(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?income\s+statements?(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+comprehensive\s+income(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+income(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
    ],
    "cash_flow": [
        r"^(?:consolidated\s+)?cash\s+flow\s+statements?(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+cash\s+flows?(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
    ],
}
_TITLE_RE = {k: [re.compile(p, re.I) for p in v] for k, v in TITLE_PATTERNS.items()}
YEAR_RE = re.compile(r"\b(?:FY\s*)?20\d{2}(?:[-/–]\d{2})?\b", re.I)


def _progress(callback: ProgressCallback, percent: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, percent)), message)


def get_all_page_text(doc: "fitz.Document", progress_callback: ProgressCallback = None) -> List[Dict]:
    results = []
    total = max(1, doc.page_count)
    for i, page in enumerate(doc):
        method = classify_page(page)
        text = page.get_text("text") if method == "digital" else ocr_page(page)
        results.append({"page_number": i, "method": method, "text": text})
        if (i + 1) == total or (i + 1) % max(1, total // 20) == 0:
            _progress(progress_callback, 5 + int(((i + 1) / total) * 35), f"Reading pages ({i + 1}/{total})")
    return results


def _flagged_pages_from_sections(sections: List[Dict]) -> set[int]:
    flagged = set()
    for section in sections:
        flagged.update(range(section["start_page"], section["end_page"] + 1))
    return flagged


def _render_pages(doc: "fitz.Document", page_numbers: set[int], image_dir: Path, progress_callback: ProgressCallback = None) -> List[Dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    page_list = sorted(page_numbers)
    total = max(1, len(page_list))
    for i, pn in enumerate(page_list, start=1):
        path = image_dir / f"page_{pn + 1:04d}.png"
        doc[pn].get_pixmap(matrix=mat, alpha=False).save(str(path))
        rendered.append({"page_number_human": pn + 1, "path": str(path), "dpi": RENDER_DPI, "format": "png"})
        _progress(progress_callback, 80 + int((i / total) * 15), f"Rendering page images ({i}/{total})")
    return rendered


def _nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _looks_title_like(normalized: str) -> bool:
    words = normalized.split()
    if len(words) > 24:
        return False
    narrative_markers = (
        "the amount", "the company", "is disclosed", "is recognised", "are disclosed",
        "recognised in", "refer to", "see note", "as per note", "during the year",
        "following", "pursuant", "thereof", "below", "above",
    )
    return not any(marker in normalized.lower() for marker in narrative_markers)


def _primary_title(page_text: str) -> Optional[Dict]:
    lines = _nonempty_lines(page_text)
    for idx, line in enumerate(lines[:18]):
        normalized = re.sub(r"\s+", " ", line).strip()
        if not normalized or len(normalized) > 140 or not _looks_title_like(normalized):
            continue
        for statement_type, patterns in _TITLE_RE.items():
            if any(p.fullmatch(normalized) for p in patterns):
                context = " ".join(lines[idx:idx + 5])
                return {
                    "statement_type": statement_type,
                    "title": normalized,
                    "line_index": idx,
                    "has_year_header": bool(YEAR_RE.search(context)),
                    "is_primary_title": idx <= 8,
                }
    return None


def _annotate_page_titles(page_records: List[Dict]) -> None:
    for record in page_records:
        record["statement_title"] = _primary_title(record["text"])


def _statement_sequence_candidates(page_records: List[Dict], table_records: List[Dict]) -> Dict[str, List[int]]:
    """Select a high-confidence BS -> IS -> CF cluster, using a <=5 page window."""
    titles: Dict[str, List[Dict]] = {t: [] for t in TARGET_STATEMENTS}
    for record in page_records:
        title = record.get("statement_title")
        if title and title["statement_type"] in TARGET_STATEMENTS and title["is_primary_title"]:
            titles[title["statement_type"]].append({"page": record["page_number"], "year": title["has_year_header"]})

    table_score: Dict[int, float] = {}
    table_validated: Dict[int, bool] = {}
    for table in table_records:
        if table["validated"]:
            table_score[table["page_number"]] = max(table_score.get(table["page_number"], 0.0), float(table["score"]))
            table_validated[table["page_number"]] = True

    combos = []
    for bs in titles["balance_sheet"]:
        for inc in titles["income_statement"]:
            for cf in titles["cash_flow"]:
                if not bs["page"] < inc["page"] < cf["page"]:
                    continue
                span = cf["page"] - bs["page"]
                if span > SEQUENCE_MAX_SPAN:
                    continue
                gaps = (inc["page"] - bs["page"], cf["page"] - inc["page"])
                proximity = 1.0 / (1.0 + span) + 1.0 / (1.0 + gaps[0]) + 1.0 / (1.0 + gaps[1])
                year_support = 0.4 * sum(int(x["year"]) for x in (bs, inc, cf))
                table_support = sum(table_score.get(p, 0.0) for p in (bs["page"], inc["page"], cf["page"]))
                validation_support = sum(int(table_validated.get(p, False)) for p in (bs["page"], inc["page"], cf["page"]))
                score = 10.0 + proximity + year_support + table_support + 1.5 * validation_support - 0.75 * span
                combos.append((score, bs["page"], inc["page"], cf["page"]))

    if combos:
        _, bs, inc, cf = max(combos, key=lambda x: x[0])
        return {"balance_sheet": [bs], "income_statement": [inc], "cash_flow": [cf]}

    # Outside a complete cluster, keep only one title candidate per statement.
    # These are deliberately low-confidence fallbacks and do not unlock broad
    # continuation assignment.
    return {
        t: ([max(pages, key=lambda x: (x["year"], -x["page"]))["page"]] if pages else [])
        for t, pages in titles.items()
    }


def _attach_statement_types(page_records: List[Dict], table_records: List[Dict]) -> Dict[str, List[int]]:
    """High-precision statement classification with three-statement clustering."""
    _annotate_page_titles(page_records)
    selected = _statement_sequence_candidates(page_records, table_records)
    title_by_page = {r["page_number"]: r.get("statement_title") for r in page_records}
    complete_cluster = all(selected.get(t) for t in TARGET_STATEMENTS)

    for table in table_records:
        table["statement_type"] = None
        table["statement_assignment"] = "unassigned"
        table["statement_confidence"] = 0.0
        title = title_by_page.get(table["page_number"])
        table["table_title"] = title["title"] if title else None

    for statement_type, pages in selected.items():
        for table in table_records:
            if table["validated"] and table["page_number"] in pages:
                table["statement_type"] = statement_type
                table["statement_assignment"] = "title" if complete_cluster else "title_candidate"
                table["statement_confidence"] = 1.0 if complete_cluster else 0.82

    if complete_cluster:
        ordered = sorted((pages[0], kind) for kind, pages in selected.items())
        cluster_start = ordered[0][0]
        cluster_end = ordered[-1][0]
        for table in table_records:
            if table["statement_type"] is not None or not table["validated"]:
                continue
            pn = table["page_number"]
            for idx, (start, kind) in enumerate(ordered):
                next_start = ordered[idx + 1][0] if idx + 1 < len(ordered) else cluster_end + MAX_CONTINUATION_PAGES + 1
                if start < pn < next_start and pn <= start + MAX_CONTINUATION_PAGES:
                    table["statement_type"] = kind
                    table["statement_assignment"] = "continuation"
                    table["statement_confidence"] = 0.92
                    break
    else:
        # Isolated title pages remain evidence, but we do not spread their
        # classification across neighboring tables.
        for table in table_records:
            if table["statement_type"] is None:
                title = title_by_page.get(table["page_number"])
                if title and title["statement_type"] in TARGET_STATEMENTS and table["validated"]:
                    table["statement_type"] = title["statement_type"]
                    table["statement_assignment"] = "title_candidate"
                    table["statement_confidence"] = 0.82

    return selected
