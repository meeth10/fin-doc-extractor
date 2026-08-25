from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .financial_schema import EvidenceRef

METRIC_ALIASES: Dict[str, List[str]] = {
    "cash_and_equivalents": [
        "cash and cash equivalents", "cash & cash equivalents", "cash equivalents",
        "cash and bank balances", "cash balances", "cash and balances with banks",
    ],
    "revenue": ["revenue from operations", "revenue", "total revenue", "sales", "turnover"],
    "ebitda": ["ebitda", "operating ebitda", "adjusted ebitda"],
    "ebit": ["ebit", "operating profit", "profit from operations", "operating income"],
    "depreciation": ["depreciation", "depreciation and amortisation", "depreciation & amortisation"],
    "pat": ["profit for the year", "profit for the period", "profit after tax", "net profit", "profit attributable to owners", "net income"],
    "pbt": ["profit before tax", "profit before income tax", "profit before taxes", "pre-tax profit"],
    "finance_costs": ["finance costs", "finance cost", "interest expense", "interest costs"],
    "total_debt": ["total debt", "borrowings", "total borrowings", "debt", "non-current borrowings", "current borrowings"],
    "cfo": ["cash flow from operating activities", "net cash generated from operating activities", "cash generated from operations"],
    "capex": ["capital expenditure", "purchase of property", "purchase of property, plant and equipment", "purchase of fixed assets", "additions to property, plant and equipment"],
    "total_assets": ["total assets"],
    "total_equity": ["total equity", "equity attributable to owners", "shareholders' equity"],
    "eps": ["earnings per share", "basic earnings per share", "diluted earnings per share"],
}

QUESTION_ALIASES = {
    "cash balance": "cash_and_equivalents", "cash position": "cash_and_equivalents", "cash": "cash_and_equivalents",
    "revenue": "revenue", "sales": "revenue", "ebitda": "ebitda", "ebit": "ebit",
    "profit after tax": "pat", "pat": "pat", "net profit": "pat", "pbt": "pbt",
    "debt": "total_debt", "total debt": "total_debt", "operating cash flow": "cfo", "cfo": "cfo",
    "capex": "capex", "free cash flow": "fcf",
}

NUMBER_RE = re.compile(r"(?:₹|\$|€|£)?\s*[-−(]?\s*\d[\d,]*(?:\.\d+)?\s*\)?%?")
YEAR_RE = re.compile(r"(?:FY\s*)?20\d{2}(?:[-/–]\d{2})?", re.I)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _number(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    negative = "(" in text and ")" in text
    cleaned = re.sub(r"[^0-9.\-]", "", text.replace("−", "-"))
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -abs(number) if negative else number


def _periods(rows: List[List[str]]) -> List[str]:
    for row in rows[:6]:
        found = YEAR_RE.findall(" ".join(row))
        if found:
            return found
    return []


def _iter_tables(data: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for statement_type, bucket in (data.get("statement_tables") or {}).items():
        for table in bucket.get("tables", []):
            yield statement_type, table


def _row_match(row: List[Any], aliases: List[str]) -> Optional[str]:
    cells = [_norm(c) for c in row]
    label = " ".join(c for c in cells if c)
    for alias in aliases:
        a = _norm(alias)
        if a and (label == a or label.startswith(a + " ") or a in label):
            return alias
    return None


def resolve_metric(metric: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    aliases = METRIC_ALIASES.get(metric, [])
    candidates: List[Dict[str, Any]] = []
    for statement_type, table in _iter_tables(data):
        rows = table.get("table") or []
        if not rows or not aliases:
            continue
        periods = _periods(rows)
        title = table.get("table_title")
        page = table.get("page_number_human")
        for row in rows:
            if not isinstance(row, list):
                continue
            matched = _row_match(row, aliases)
            if not matched:
                continue
            values = []
            for cell in row[1:]:
                value = _number(cell)
                if value is not None:
                    values.append(value)
            if not values:
                continue
            candidates.append({
                "metric": metric, "matched_alias": matched, "values": values,
                "periods": periods, "page": page, "statement": statement_type,
                "table_title": title, "source": table.get("source"),
                "score": float(table.get("score", 0) or 0),
                "validated": bool(table.get("validated")),
                "assignment": table.get("statement_assignment"),
            })
    return sorted(candidates, key=lambda x: (not x["validated"], -x["score"], x["page"] or 10**9))


def resolve_raw_text(metric: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    aliases = METRIC_ALIASES.get(metric, [])
    results: List[Dict[str, Any]] = []
    for page in (data.get("document", {}).get("pages") or []):
        text = str(page.get("raw_text") or "")
        lower = _norm(text)
        matched = next((a for a in aliases if _norm(a) in lower), None)
        if not matched:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for i, line in enumerate(lines):
            if _norm(matched) not in _norm(line):
                continue
            window = lines[i:i + 8]
            values: List[float] = []
            for item in window[1:]:
                numbers = NUMBER_RE.findall(item)
                for token in numbers:
                    number = _number(token)
                    if number is not None:
                        values.append(number)
                if len(values) >= 4:
                    break
            if values:
                period_matches = YEAR_RE.findall(" ".join(lines[max(0, i - 6):i + 2]))
                results.append({
                    "metric": metric,
                    "matched_alias": matched,
                    "values": values[:4],
                    "periods": period_matches,
                    "page": page.get("page_number_human"),
                    "statement": None,
                    "table_title": None,
                    "source": "raw_text",
                    "score": 0.6,
                    "validated": True,
                    "assignment": "raw_text_fallback",
                    "snippet": "\n".join(window),
                })
                break
    return results


def metric_from_question(question: str) -> Optional[str]:
    text = _norm(question)
    for phrase, metric in sorted(QUESTION_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if phrase in text:
            return metric
    return None


def build_evidence(question: str, data: Dict[str, Any]) -> Dict[str, Any]:
    metric = metric_from_question(question)
    candidates = resolve_metric(metric, data) if metric else []
    raw_evidence = resolve_raw_text(metric, data)[:5] if metric else []
    related: Dict[str, Any] = {}

    if metric == "ebitda":
        related["ebit"] = resolve_metric("ebit", data)[:3] or resolve_raw_text("ebit", data)[:3]
        related["depreciation"] = resolve_metric("depreciation", data)[:3] or resolve_raw_text("depreciation", data)[:3]
        related["pbt"] = resolve_metric("pbt", data)[:3] or resolve_raw_text("pbt", data)[:3]
        related["finance_costs"] = resolve_metric("finance_costs", data)[:3] or resolve_raw_text("finance_costs", data)[:3]

    return {
        "question": question,
        "metric": metric,
        "document": {
            "source_name": data.get("summary", {}).get("source_name"),
            "metadata": data.get("summary", {}).get("metadata", {}),
        },
        "candidates": candidates[:5],
        "raw_evidence": raw_evidence,
        "related": related,
    }


def evidence_sources(candidates: List[Dict[str, Any]], raw_evidence: Optional[List[Dict[str, Any]]] = None) -> List[EvidenceRef]:
    refs: List[EvidenceRef] = []
    seen = set()
    for item in list(candidates) + list(raw_evidence or []):
        key = (item.get("page"), item.get("statement"), item.get("table_title"), item.get("source"))
        if key in seen:
            continue
        seen.add(key)
        refs.append(EvidenceRef(page=item.get("page"), statement=item.get("statement"), table_title=item.get("table_title"), source=item.get("source")))
    return refs
