from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

YEAR_RE = re.compile(r"(?:FY\s*)?(20\d{2})(?:[-/–]\d{2})?", re.I)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


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


def _periods(rows: List[List[Any]]) -> List[str]:
    for row in rows[:8]:
        found = YEAR_RE.findall(" ".join(str(x) for x in row))
        if found:
            return found
    return []


def _tables(data: Dict[str, Any], statement: str) -> Iterable[Dict[str, Any]]:
    for table in (data.get("statement_tables") or {}).get(statement, {}).get("tables", []):
        yield table


def _row_label(row: List[Any]) -> str:
    return _norm(row[0]) if row else ""


def _row_values(row: List[Any]) -> List[float]:
    values = []
    for cell in row[1:]:
        value = _number(cell)
        if value is not None:
            values.append(value)
    return values


def _best_table(data: Dict[str, Any], statement: str) -> Optional[Dict[str, Any]]:
    tables = list(_tables(data, statement))
    if not tables:
        return None
    tables.sort(key=lambda t: (
        0 if t.get("statement_assignment") in {"title", "continuation"} else 1,
        -(float(t.get("score", 0) or 0)),
        int(t.get("page_number_human", 10**9)),
    ))
    return tables[0]


def _latest_period_index(periods: List[str], values: List[float]) -> Tuple[int, Optional[str]]:
    if not values:
        return 0, None
    years = []
    for idx, period in enumerate(periods[: len(values)]):
        match = YEAR_RE.search(str(period))
        if match:
            years.append((int(match.group(1)), idx, period))
    if years:
        _, idx, period = max(years)
        return idx, period
    return 0, periods[0] if periods else None


def _metric_row(data: Dict[str, Any], statement: str, labels: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    table = _best_table(data, statement)
    if not table:
        return None
    rows = table.get("table") or []
    periods = _periods(rows)
    candidates = []
    for row in rows:
        if not isinstance(row, list):
            continue
        label = _row_label(row)
        if label in labels:
            values = _row_values(row)
            if values:
                idx, period = _latest_period_index(periods, values)
                if idx < len(values):
                    candidates.append({
                        "value": values[idx],
                        "period": period,
                        "page": table.get("page_number_human"),
                        "table_title": table.get("table_title"),
                        "source": table.get("source"),
                        "row_label": row[0],
                        "periods": periods,
                        "values": values,
                    })
    return candidates[0] if candidates else None


def trusted_answer(question: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = _norm(question)

    if "cash" in q:
        fact = _metric_row(data, "balance_sheet", (
            "cash and cash equivalents",
            "cash & cash equivalents",
        ))
        if fact:
            return {
                "metric": "cash_and_equivalents",
                "answer": fact["value"],
                "period": fact["period"],
                "status": "reported",
                "confidence": "high",
                "formula": None,
                "inputs": [],
                "source_page": fact["page"],
                "explanation": f"Cash and cash equivalents reported on page {fact['page']} for {fact['period']}.",
            }

    if "debt" in q and "ebit" not in q:
        table = _best_table(data, "balance_sheet")
        if table:
            rows = table.get("table") or []
            periods = _periods(rows)
            components = (
                "commercial paper",
                "term debt",
            )
            by_period: Dict[str, float] = {p: 0.0 for p in periods}
            matched = []
            for row in rows:
                if not isinstance(row, list):
                    continue
                label = _row_label(row)
                if label not in components:
                    continue
                values = _row_values(row)
                matched.append((label, values))
                for idx, value in enumerate(values[: len(periods)]):
                    by_period[periods[idx]] += value
            if matched and by_period:
                latest_period = max(by_period, key=lambda p: int(YEAR_RE.search(p).group(1)) if YEAR_RE.search(p) else -1)
                return {
                    "metric": "total_debt",
                    "answer": round(by_period[latest_period], 2),
                    "period": latest_period,
                    "status": "derived",
                    "confidence": "high",
                    "formula": "commercial paper + term debt",
                    "inputs": [
                        {"name": label, "value": next(v for i, v in enumerate(vals) if i < len(periods) and periods[i] == latest_period), "page": table.get("page_number_human")}
                        for label, vals in matched
                    ],
                    "source_page": table.get("page_number_human"),
                    "explanation": f"Total debt is derived from commercial paper plus term debt on page {table.get('page_number_human')} for {latest_period}.",
                }

    if "revenue" in q or "sales" in q:
        fact = _metric_row(data, "income_statement", (
            "total net sales",
            "net sales",
            "revenue from operations",
            "total revenue",
            "revenue",
        ))
        if fact:
            return {
                "metric": "revenue",
                "answer": fact["value"],
                "period": fact["period"],
                "status": "reported",
                "confidence": "high",
                "formula": None,
                "inputs": [],
                "source_page": fact["page"],
                "explanation": f"Total revenue is reported on page {fact['page']} for {fact['period']}.",
            }

    return None
