"""
Statement-page detection — Phase 1 of the v3 pipeline.

A page mentioning "Balance Sheet" is not necessarily a Balance Sheet page
(contents pages, auditor's reports, and cross-references all mention it
too). This module scores each page on multiple independent signals and
only commits to a classification above a confidence threshold; pages that
land in between get marked "ambiguous" rather than forced into a category.

Still fully deterministic — no LLM involvement. That's Phase 6.

Signals per page, per candidate statement category:
  + heading evidence       (e.g. "Consolidated Balance Sheets")
  + line-item evidence     (e.g. "Trade receivables", "Finance costs")
  + tabular-shape evidence (lines that look like "label  123.45  108.20")
  + monetary density       (how many number-like tokens overall)
  + year-header evidence   (e.g. "2025-26", "FY2026")

Page-level penalties (apply to every category, since they indicate this
page probably isn't a genuine statement page regardless of what it
mentions):
  - contents-page pattern  (dot-leaders / "Title ......... 69")
  - auditor's-report language
  - heavy cross-referencing ("refer note 14", "see page 71")
  - strong "Notes to Financial Statements" heading with weak line-item
    evidence (penalizes the *statement* categories on a notes page, not
    the notes category itself)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

STATEMENT_CATEGORIES = ["balance_sheet", "income_statement", "cash_flow", "equity"]

HEADING_PATTERNS = {
    "balance_sheet": [
        r"consolidated balance sheets?",
        r"balance sheets?",
        r"statements? of financial position",
    ],
    "income_statement": [
        r"consolidated statements? of operations",
        r"statements? of income",
        r"income statements?",
        r"statements? of comprehensive income",
        r"statement of profit and loss",
        r"profit and loss account",
    ],
    "cash_flow": [
        r"consolidated statements? of cash flows?",
        r"statements? of cash flows?",
        r"cash flow statements?",
    ],
    "equity": [
        r"statements? of stockholders'? equity",
        r"statements? of changes in equity",
        r"statement of shareholders'? funds",
    ],
    "notes": [
        r"notes to (the )?(consolidated )?financial statements",
        r"notes to accounts",
    ],
}

# Section 4: statement-specific line-item signatures. Plain phrases (not
# regex) — matched with the same whitespace-tolerant treatment as headings
# since these come from the same OCR'd/digital text.
LINE_ITEM_SIGNATURES = {
    "balance_sheet": [
        "non-current assets", "current assets", "property, plant and equipment",
        "capital work-in-progress", "investments", "inventories",
        "trade receivables", "cash and cash equivalents", "borrowings",
        "trade payables", "provisions", "total assets", "total equity",
    ],
    "income_statement": [
        "revenue from operations", "other income", "total income",
        "cost of materials", "employee benefits", "finance costs",
        "depreciation", "other expenses", "profit before tax", "tax expense",
        "profit for the year", "other comprehensive income", "earnings per share",
    ],
    "cash_flow": [
        "cash flow from operating activities", "working capital changes",
        "cash generated from operations", "income tax paid",
        "investing activities", "purchase of property", "financing activities",
        "net increase", "net decrease", "cash and cash equivalents",
    ],
    "equity": [
        "equity share capital", "other equity", "capital reserve",
        "securities premium", "general reserve", "retained earnings",
        "total comprehensive income",
    ],
}

# page-level penalty signals
CONTENTS_LINE_PATTERNS = [
    r"\.{3,}\s*\d{1,4}\s*$",        # "Balance Sheet .......... 69"
    r"^.{3,80}?[-\u2013\u2014]\s*page\s*\d{1,4}\s*$",  # "IND AS Balance Sheet — page 69"
]
AUDITOR_PATTERNS = [
    r"independent auditor'?s? report",
    r"we have audited",
    r"basis for (qualified )?opinion",
    r"auditor'?s? responsibilit",
]
CROSS_REFERENCE_PATTERNS = [
    r"refer(?:red|s)? to note",
    r"refer note \d",
    r"see note \d",
    r"as per note \d",
]
YEAR_PATTERN = re.compile(r"\b(20\d{2})\s*[-\u2013/]\s*(\d{2})\b|\bFY\s*[- ]?\s*(20\d{2})\b")
NUMERIC_TOKEN_PATTERN = re.compile(r"[\d,]+\.\d{2}|\(\s*[\d,]+\.\d{2}\s*\)|[\d,]{4,}")

# weights — tuned against the synthetic sample + contents-page cases;
# adjust here first if real filings need different balance, not by hacking
# the underlying signal functions.
W_HEADING = 4
W_LINE_ITEM = 1        # per distinct line item, capped
LINE_ITEM_CAP = 6
W_TABLE_SHAPE = 2       # flat bonus if >=3 label+number lines present
TABLE_SHAPE_MIN_LINES = 3
W_MONETARY = 1          # per 5 monetary tokens, capped
MONETARY_CAP = 3
W_YEAR_HEADER = 2

P_CONTENTS = 20          # page-level, effectively zeroes any category
P_AUDITOR = 6
P_CROSS_REFERENCE = 2    # per hit, capped
P_CROSS_REFERENCE_CAP = 6
P_NOTES_WEAK_EVIDENCE = 4

CONFIDENT_THRESHOLD = 8
AMBIGUOUS_THRESHOLD = 4
CONFIDENCE_SATURATION = 16  # score at/above this maps to confidence 1.0

DEFAULT_GAP_TOLERANCE = 2
DEFAULT_PAD_PAGES = 1


def _loosen(pattern: str) -> str:
    """OCR drops/merges spacing ('Statement of' -> 'Statementof'); make
    literal spaces in patterns tolerant instead of requiring exact spacing."""
    return pattern.replace(" ", r"\s*")


def _compile_group(patterns: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    return {cat: [re.compile(_loosen(p)) for p in pats] for cat, pats in patterns.items()}


_HEADING_RE = _compile_group(HEADING_PATTERNS)
_LINE_ITEM_RE = {
    # re.escape() also escapes literal spaces (for verbose-mode safety),
    # which breaks a naive _loosen(re.escape(phrase)) — the space becomes
    # "\ " first, then _loosen turns that into "\\s*" (an escaped literal
    # backslash), which can never match. Escape each word individually and
    # join with \s* directly instead.
    cat: [re.compile(r"\s*".join(re.escape(w) for w in li.split()))
          for li in items]
    for cat, items in LINE_ITEM_SIGNATURES.items()
}
_CONTENTS_RE = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in CONTENTS_LINE_PATTERNS]
_AUDITOR_RE = [re.compile(p, re.IGNORECASE) for p in AUDITOR_PATTERNS]
_CROSS_REF_RE = [re.compile(p, re.IGNORECASE) for p in CROSS_REFERENCE_PATTERNS]


@dataclass
class PageScore:
    page_number: int  # 0-indexed
    category_scores: Dict[str, float] = field(default_factory=dict)
    best_category: Optional[str] = None
    confidence: float = 0.0
    status: str = "none"  # "confident" | "ambiguous" | "none"
    signals: Dict[str, object] = field(default_factory=dict)  # debug detail


def _looks_like_contents_page(text: str) -> bool:
    hit_lines = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for pat in _CONTENTS_RE:
            if pat.search(line):
                hit_lines += 1
                break
    return hit_lines >= 3


def _count_table_shape_lines(text: str) -> int:
    """Lines that look like 'label   123.45   108.20' — a text label
    followed by two-or-more monetary-looking tokens on the same line,
    which is the shape of an actual statement row (vs. narrative prose)."""
    count = 0
    for line in text.split("\n"):
        tokens = NUMERIC_TOKEN_PATTERN.findall(line)
        if len(tokens) >= 2 and re.search(r"[A-Za-z]{3,}", line):
            count += 1
    return count


def score_page(text: str, include_notes: bool = False) -> PageScore:
    lower = " ".join(text.lower().split())

    is_contents = _looks_like_contents_page(text)
    auditor_hits = sum(1 for pat in _AUDITOR_RE if pat.search(lower))
    cross_ref_hits = sum(len(pat.findall(lower)) for pat in _CROSS_REF_RE)
    table_shape_lines = _count_table_shape_lines(text)
    monetary_tokens = len(NUMERIC_TOKEN_PATTERN.findall(text))
    has_year_header = bool(YEAR_PATTERN.search(text))
    notes_heading_hit = any(pat.search(lower) for pat in _HEADING_RE["notes"])

    categories = STATEMENT_CATEGORIES + (["notes"] if include_notes else [])
    category_scores = {}

    for category in categories:
        heading_hit = any(pat.search(lower) for pat in _HEADING_RE.get(category, []))
        heading_score = W_HEADING if heading_hit else 0

        line_item_hits = 0
        if category in _LINE_ITEM_RE:
            line_item_hits = sum(1 for pat in _LINE_ITEM_RE[category] if pat.search(lower))
        line_item_score = min(line_item_hits, LINE_ITEM_CAP) * W_LINE_ITEM

        table_score = W_TABLE_SHAPE if table_shape_lines >= TABLE_SHAPE_MIN_LINES else 0
        monetary_score = min(monetary_tokens // 5, MONETARY_CAP) * W_MONETARY
        year_score = W_YEAR_HEADER if has_year_header else 0

        raw = heading_score + line_item_score + table_score + monetary_score + year_score

        # page-level penalties, applied per category
        penalty = 0
        if is_contents:
            penalty += P_CONTENTS
        if auditor_hits:
            penalty += P_AUDITOR
        penalty += min(cross_ref_hits * P_CROSS_REFERENCE, P_CROSS_REFERENCE_CAP)
        if category != "notes" and notes_heading_hit and line_item_hits < 2:
            penalty += P_NOTES_WEAK_EVIDENCE

        category_scores[category] = max(0, raw - penalty)

    best_category = max(category_scores, key=category_scores.get) if category_scores else None
    best_score = category_scores.get(best_category, 0) if best_category else 0

    if best_score >= CONFIDENT_THRESHOLD:
        status = "confident"
    elif best_score >= AMBIGUOUS_THRESHOLD:
        status = "ambiguous"
    else:
        status = "none"
        best_category = None

    confidence = round(min(1.0, best_score / CONFIDENCE_SATURATION), 2) if best_score else 0.0

    return PageScore(
        page_number=-1,  # filled in by caller
        category_scores=category_scores,
        best_category=best_category,
        confidence=confidence,
        status=status,
        signals={
            "is_contents_page": is_contents,
            "auditor_hits": auditor_hits,
            "cross_reference_hits": cross_ref_hits,
            "table_shape_lines": table_shape_lines,
            "monetary_tokens": monetary_tokens,
            "has_year_header": has_year_header,
        },
    )


def score_pages(pages_text: List[str], include_notes: bool = False) -> List[PageScore]:
    scores = []
    for i, text in enumerate(pages_text):
        s = score_page(text, include_notes=include_notes)
        s.page_number = i
        scores.append(s)
    return scores


def merge_into_sections(scores: List[PageScore], total_pages: int,
                         gap_tolerance: int = DEFAULT_GAP_TOLERANCE,
                         pad_pages: int = DEFAULT_PAD_PAGES) -> List[Dict]:
    """
    Only "confident" pages anchor a section — "ambiguous" pages never start
    or extend a section on their own, but do get swept in by padding if
    they happen to sit right next to a confident run (a genuine statement
    continuation page is a very plausible ambiguous neighbor; a random
    ambiguous page elsewhere in the document is not).
    """
    confident = [s for s in scores if s.status == "confident"]
    if not confident:
        return []

    confident.sort(key=lambda s: s.page_number)
    sections = []
    cur_start = cur_end = confident[0].page_number
    cur_categories = {confident[0].best_category}
    cur_conf = [confident[0].confidence]

    for s in confident[1:]:
        if s.page_number - cur_end <= gap_tolerance + 1:
            cur_end = s.page_number
            cur_categories.add(s.best_category)
            cur_conf.append(s.confidence)
        else:
            sections.append((cur_start, cur_end, cur_categories, cur_conf))
            cur_start, cur_end = s.page_number, s.page_number
            cur_categories, cur_conf = {s.best_category}, [s.confidence]
    sections.append((cur_start, cur_end, cur_categories, cur_conf))

    padded = []
    for start, end, cats, confs in sections:
        p_start = max(0, start - pad_pages)
        p_end = min(total_pages - 1, end + pad_pages)
        padded.append({
            "start_page": p_start,
            "end_page": p_end,
            "categories": sorted(cats),
            "confidence": round(sum(confs) / len(confs), 2),
        })
    return padded


def extract_year_labels(text: str) -> List[str]:
    """
    Detect fiscal-year labels present on a page (e.g. "2025-26" -> "FY2026",
    "FY2024" -> "FY2024"), in the order they appear in the text.

    Deliberately does NOT assume the first column/label found is the most
    recent fiscal year — some filings show oldest-first, some newest-first.
    This only reports which years are present and in what order they were
    written; mapping a specific numeric *column* to a specific *value* is
    table-reconstruction work (later phase), not page-level detection.
    """
    labels = []
    for m in YEAR_PATTERN.finditer(text):
        if m.group(1) and m.group(2):
            # "2025-26" style: the second, shorter number is the FY-ending
            # year suffix (Indian FY convention: 2025-26 = FY2026).
            labels.append(f"FY20{m.group(2)}")
        elif m.group(3):
            labels.append(f"FY{m.group(3)}")
    seen, ordered = set(), []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def locate_financial_statements(pages_text: List[str], include_notes: bool = False,
                                 gap_tolerance: int = DEFAULT_GAP_TOLERANCE,
                                 pad_pages: int = DEFAULT_PAD_PAGES):
    scores = score_pages(pages_text, include_notes=include_notes)
    sections = merge_into_sections(scores, len(pages_text), gap_tolerance, pad_pages)
    ambiguous_pages = [s.page_number for s in scores if s.status == "ambiguous"]
    return sections, ambiguous_pages, scores
