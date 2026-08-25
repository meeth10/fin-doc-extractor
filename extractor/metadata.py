"""
Document-level metadata detection: company name, financial year, currency,
unit/scale, standalone-vs-consolidated.

Deterministic regex over the first few pages (cover + early front matter,
where this stuff almost always lives) rather than the whole document.
Does NOT convert monetary values — only records what scale the source
document is already in, so later stages can keep numbers in their native
unit instead of silently guessing.
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Optional

FRONT_MATTER_PAGE_COUNT = 5

# unit/scale phrasing, ordered roughly by specificity
UNIT_PATTERNS = [
    (r"(?:inr|rs\.?|₹)\s*in\s*lakhs?", ("INR", "lakhs")),
    (r"(?:inr|rs\.?|₹)\s*in\s*crores?", ("INR", "crores")),
    (r"₹\s*crores?", ("INR", "crores")),
    (r"₹\s*lakhs?", ("INR", "lakhs")),
    (r"(?:inr|rs\.?|₹)\s*in\s*millions?", ("INR", "millions")),
    (r"(?:usd|\$)\s*in\s*millions?", ("USD", "millions")),
    (r"(?:usd|\$)\s*in\s*thousands?", ("USD", "thousands")),
    (r"(?:in\s*)?millions?\s*of\s*(?:us\s*)?dollars", ("USD", "millions")),
    (r"amounts?\s*in\s*(?:rs\.?|inr|₹)\s*(lakhs?|crores?|millions?|thousands?)", None),  # handled specially below
]

FY_PATTERNS = [
    # "year ended 31 March 2025" — the year given IS the FY (India convention)
    (r"(?:for the )?year ended\s+(?:31st?|31)\s+march[,]?\s+(20\d{2})", 1),
    # "FY2025" / "FY 2025" / "FY-2025"
    (r"\bFY\s*[- ]?\s*(20\d{2})\b", 1),
    # "2025-26" — ending-year convention (matches locator.extract_year_labels)
    (r"\b(20\d{2})\s*[-\u2013]\s*(\d{2})\b", 2),
    # bare "financial year 2025" with no trailing "-26"
    (r"financial\s+year\s*(20\d{2})\b(?!\s*[-\u2013])", 1),
]

STANDALONE_PATTERN = re.compile(r"\bstandalone\b", re.IGNORECASE)
CONSOLIDATED_PATTERN = re.compile(r"\bconsolidated\b", re.IGNORECASE)

# Very rough company-name heuristic: look for a line ending in a common
# corporate suffix near the top of the first page. This is intentionally
# conservative — wrong company name is worse than a missing one.
COMPANY_SUFFIX_PATTERN = re.compile(
    r"^([A-Z][A-Za-z0-9&.,'\-\s]{2,80}?"
    r"(?:Limited|Ltd\.?|Inc\.?|Corporation|Corp\.?|LLC|LLP|Company|Co\.?|"
    r"Holdings|Group|plc|PLC))\s*$",
    re.MULTILINE,
)


@dataclass
class DocumentMetadata:
    company_name: Optional[str] = None
    financial_year: Optional[str] = None
    currency: Optional[str] = None
    unit_scale: Optional[str] = None
    standalone_or_consolidated: Optional[str] = None  # "standalone" | "consolidated" | None

    def as_dict(self):
        return asdict(self)


def _detect_unit(text: str):
    norm = text.lower()
    for pattern, result in UNIT_PATTERNS:
        m = re.search(pattern, norm)
        if m:
            if result is not None:
                return result
            # the generic "amounts in Rs. <scale>" pattern — scale is group 1
            scale = m.group(1)
            return ("INR", scale.rstrip("s") + "s")
    return (None, None)


def _detect_financial_year(text: str) -> Optional[str]:
    for pattern, fy_group in FY_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        if fy_group == 2:
            # "2025-26" style: second (shorter) number is the ending year
            return f"FY20{m.group(2)}"
        year = m.group(1)
        if len(year) == 2:
            year = "20" + year
        return f"FY{year}"
    return None


def _detect_company_name(text: str) -> Optional[str]:
    m = COMPANY_SUFFIX_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return None


def _detect_standalone_or_consolidated(text: str) -> Optional[str]:
    has_standalone = bool(STANDALONE_PATTERN.search(text))
    has_consolidated = bool(CONSOLIDATED_PATTERN.search(text))
    if has_standalone and not has_consolidated:
        return "standalone"
    if has_consolidated and not has_standalone:
        return "consolidated"
    # both or neither present in the front matter — genuinely ambiguous at
    # this stage (many reports contain both sets of statements); leave for
    # per-statement-page detection rather than guessing at the doc level.
    return None


def detect_metadata(pages_text: List[str]) -> DocumentMetadata:
    front_matter = "\n".join(pages_text[:FRONT_MATTER_PAGE_COUNT])
    currency, unit_scale = _detect_unit(front_matter)
    return DocumentMetadata(
        company_name=_detect_company_name(front_matter),
        financial_year=_detect_financial_year(front_matter),
        currency=currency,
        unit_scale=unit_scale,
        standalone_or_consolidated=_detect_standalone_or_consolidated(front_matter),
    )
