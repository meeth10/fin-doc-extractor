from __future__ import annotations

import re


_PATTERNS: dict[str, tuple[str, ...]] = {
    "balance_sheet": (
        r"\bbalance sheets?\b",
        r"\bstatement(?:s)? of financial position\b",
        r"\bassets\b.*\bliabilities\b.*\bequity\b",
        r"\bcurrent assets\b",
        r"\bnon[- ]current assets\b",
        r"\btotal assets\b",
        r"\btotal liabilities\b",
    ),
    "income_statement": (
        r"\bprofit and loss\b",
        r"\bstatement(?:s)? of operations\b",
        r"\bstatement(?:s)? of income\b",
        r"\bprofit for the (?:year|period)\b",
        r"\bprofit before tax\b",
        r"\bcost of sales\b",
        r"\bgross profit\b",
        r"\boperating income\b",
    ),
    "cash_flow": (
        r"\bcash flow statement\b",
        r"\bstatement(?:s)? of cash flows\b",
        r"\bcash flows? from operating activities\b",
        r"\bcash flows? from investing activities\b",
        r"\bcash flows? from financing activities\b",
        r"\bnet cash generated from operating activities\b",
    ),
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def classify_page(text: str) -> dict[str, float]:
    """Return structural statement scores in [0, 1]. No accounting-label mapping."""
    normalized = _normalise(text)
    if not normalized:
        return {kind: 0.0 for kind in _PATTERNS}

    scores: dict[str, float] = {}
    words = max(1, len(normalized.split()))
    for kind, patterns in _PATTERNS.items():
        hits = sum(1 for pattern in patterns if re.search(pattern, normalized))
        density = min(1.0, hits / max(3, len(patterns) * 0.45))
        scores[kind] = round(min(1.0, 0.7 * density + 0.3 * min(1.0, hits / max(8, words / 8))), 3)
    return scores
