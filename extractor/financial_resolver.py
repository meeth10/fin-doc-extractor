from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .financial_schema import EvidenceRef

METRIC_ALIASES: Dict[str, List[str]] = {
    "cash_and_equivalents": [
        "cash and cash equivalents", "cash & cash equivalents", "cash equivalents",
        "cash and bank balances", "cash balances", "cash and balances with banks",
    ],
    "revenue": [
        "total net sales", "net sales", "revenue from operations", "total revenue",
        "revenue", "sales", "turnover",
    ],
    "ebitda": ["ebitda", "operating ebitda", "adjusted ebitda"],
    "ebit": ["operating income", "operating profit", "profit from operations", "ebit"],
    "depreciation": [
        "depreciation and amortisation", "depreciation & amortisation",
        "depreciation and amortization", "depreciation & amortization", "depreciation",
    ],
    "pat": ["net income", "profit for the year", "profit for the period", "profit after tax", "net profit", "profit attributable to owners"],
    "pbt": ["income before provision for income taxes", "profit before tax", "profit before income tax", "profit before taxes", "pre-tax profit"],
    "finance_costs": ["finance costs", "finance cost", "interest expense", "interest costs"],
    "total_debt": ["total debt", "total borrowings", "borrowings", "debt", "non-current borrowings", "current borrowings"],
    "market_capitalization": ["market capitalization", "market capitalisation", "market cap"],
    "enterprise_value": ["enterprise value", "enterprise valuation", "ev"],
    "cfo": ["cash flow from operating activities", "net cash generated from operating activities", "cash generated from operations"],
    "capex": ["capital expenditure", "purchase of property", "purchase of property, plant and equipment", "purchase of fixed assets", "additions to property, plant and equipment"],
    "total_assets": ["total assets"],
    "total_equity": ["total equity", "equity attributable to owners", "shareholders' equity"],
    "eps": ["earnings per share", "basic earnings per share", "diluted earnings per share"],
}

QUESTION_ALIASES = {
    "cash balance": "cash_and_equivalents", "cash position": "cash_and_equivalents", "cash": "cash_and_equivalents",
    "revenue": "revenue", "sales": "revenue", "net sales": "revenue", "ebitda": "ebitda", "ebit": "ebit",
    "profit after tax": "pat", "pat": "pat", "net profit": "pat", "pbt": "pbt",
    "debt": "total_debt", "total debt": "total_debt", "market cap": "market_capitalization",
    "market capitalization": "market_capitalization", "enterprise value": "enterprise_value", "ev": "enterprise_value",
    "operating cash flow": "cfo", "cfo": "cfo", "capex": "capex", "free cash flow": "fcf",
}

NUMBER_RE = re.compile(r"(?:₹|\$|€|£)?\s*[-−(]?\s*\d[\d,]*(?:\.\d+)?\s*\)?%?")
YEAR_RE = re.compile(r"(?:FY\s*)?20\d{2}(?:[-/–]\d{2})?", re.I)

PREFERRED_STATEMENTS = {
    "revenue": ("income_statement",),
    "ebit": ("income_statement",),
    "pat": ("income_statement",),
    "pbt": ("income_statement",),
    "finance_costs": ("income_statement", "cash_flow"),
    "depreciation": ("cash_flow", "income_statement"),
    "cash_and_equivalents": ("balance_sheet",),
    "total_assets": ("balance_sheet",),
    "total_equity": ("balance_sheet",),
    "total_debt": ("balance_sheet",),
    "market_capitalization": ("balance_sheet",),
    "eps": ("income_statement",),
    "cfo": ("cash_flow",),
    "capex": ("cash_flow",),
}

FINANCIAL_PAGE_MARKERS = (
    "balance sheet", "balance sheets", "financial position", "profit and loss",
    "statements of operations", "statement of operations", "income statement",
    "statements of income", "statement of income", "cash flow", "cash flows",
    "statements of cash flows", "statement of cash flows",
)


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
    for row in rows[:8]:
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
    # Prefer a match against the descriptive/label portion of the row. This
    # prevents short aliases from matching numeric or narrative cells.
    for alias in aliases:
        a = _norm(alias)
        if not a:
            continue
        if cells and (cells[0] == a or cells[0].startswith(a + " ")):
            return alias
    label_match = re.match(r"^(.*?)(?=\s+(?:[$€£₹]?[-(]?\d|\d))", label)
    label_prefix = _norm(label_match.group(1)) if label_match else label
    for alias in aliases:
        a = _norm(alias)
        if a and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", label_prefix):
            return alias
    return None


def _candidate_sort_key(metric: str, item: Dict[str, Any]) -> tuple:
    preferred = 0 if item.get("statement") in PREFERRED_STATEMENTS.get(metric, ()) else 1
    assignment = item.get("assignment")
    assignment_rank = {"title": 0, "continuation": 1, "title_candidate": 2, "raw_text_fallback": 4}.get(assignment, 3)
    validated_rank = 0 if item.get("validated") else 1
    return (preferred, validated_rank, assignment_rank, -float(item.get("score", 0) or 0), item.get("page") or 10**9)


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
    candidates.sort(key=lambda x: _candidate_sort_key(metric, x))
    return candidates


def _page_has_financial_statement(page: Dict[str, Any]) -> bool:
    text = _norm(page.get("raw_text") or "")
    if not text or not any(marker in text for marker in FINANCIAL_PAGE_MARKERS):
        return False
    return bool(YEAR_RE.search(text))


def resolve_raw_text(metric: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    aliases = METRIC_ALIASES.get(metric, [])
    results: List[Dict[str, Any]] = []
    for page in (data.get("document", {}).get("pages") or []):
        if not _page_has_financial_statement(page):
            continue
        text = str(page.get("raw_text") or "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for i, line in enumerate(lines):
            normalized_line = _norm(line)
            matched = next((a for a in aliases if re.search(rf"(?<!\w){re.escape(_norm(a))}(?!\w)", normalized_line)), None)
            if not matched:
                continue
            # First consume numbers on the same line. This handles flattened
            # financial tables such as: Total net sales 416,161 391,035 383,285.
            tail = re.split(rf"(?<!\w){re.escape(_norm(matched))}(?!\w)", normalized_line, maxsplit=1)[-1]
            values = [_number(t) for t in NUMBER_RE.findall(tail)]
            values = [v for v in values if v is not None]
            if len(values) < 2:
                window = lines[i + 1:i + 6]
                for item in window:
                    values.extend(v for v in (_number(t) for t in NUMBER_RE.findall(item)) if v is not None)
                    if len(values) >= 4:
                        break
            if len(values) < 2:
                continue
            period_matches = YEAR_RE.findall(" ".join(lines[max(0, i - 8):i + 2]))
            if len(period_matches) < 2:
                continue
            results.append({
                "metric": metric,
                "matched_alias": matched,
                "values": values[:len(period_matches)],
                "periods": period_matches,
                "page": page.get("page_number_human"),
                "statement": None,
                "table_title": None,
                "source": "raw_text",
                "score": 0.55,
                "validated": False,
                "assignment": "raw_text_fallback",
                "snippet": "\n".join(lines[max(0, i - 2):i + 6]),
            })
            break
    return results


def metric_from_question(question: str) -> Optional[str]:
    text = _norm(question)
    for phrase, metric in sorted(QUESTION_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"(?<!\w){re.escape(_norm(phrase))}(?!\w)", text):
            return metric
    return None


def metrics_from_question(question: str) -> List[str]:
    text = _norm(question)
    found: List[str] = []
    for metric, aliases in METRIC_ALIASES.items():
        if metric == "enterprise_value":
            continue
        for alias in aliases:
            a = _norm(alias)
            if a and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text):
                if metric not in found:
                    found.append(metric)
                break
    return found


def question_intent(question: str) -> str:
    text = _norm(question)
    if any(p in text for p in ("% increase", "percentage increase", "percent increase", "growth rate", "% growth", "grew by")):
        return "yoy_percent"
    if any(p in text for p in ("increase", "decrease", "change", "grew", "decline", "growth", "versus", "vs", "year over year", "yoy")):
        return "yoy_change"
    if any(p in text for p in ("sum of", "add ", "plus", "combined", "total of")):
        return "sum"
    if any(p in text for p in ("difference between", "difference in", "minus", "subtract", "less")):
        return "difference"
    return "value"


def _pick_candidate(question: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    requested = None
    m = re.search(r"\b(?:fy\s*)?(20\d{2}(?:[-/–]\d{2})?)\b", question, re.I)
    if m:
        requested = m.group(1)
    for candidate in candidates:
        values = candidate.get("values") or []
        periods = candidate.get("periods") or []
        if len(values) < 1:
            continue
        if requested and periods:
            for idx, period in enumerate(periods):
                if requested in period and idx < len(values):
                    return candidate
            continue
        return candidate
    return None


def _select_latest(candidate: Dict[str, Any]) -> Optional[Tuple[float, Optional[str]]]:
    values = candidate.get("values") or []
    periods = candidate.get("periods") or []
    if not values:
        return None
    return float(values[0]), periods[0] if periods else None


def _same_period(*periods: Optional[str]) -> bool:
    known = {p for p in periods if p}
    return len(known) <= 1


def compute_change(question: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidate = _pick_candidate(question, candidates)
    if not candidate:
        return None
    values = candidate.get("values") or []
    periods = candidate.get("periods") or []
    if len(values) < 2:
        return None
    latest = float(values[0])
    prior = float(values[1])
    if prior == 0:
        return None
    delta = latest - prior
    pct = (delta / prior) * 100.0
    latest_period = periods[0] if len(periods) > 0 else None
    prior_period = periods[1] if len(periods) > 1 else None
    if question_intent(question) == "yoy_percent":
        answer = round(pct, 2)
        formula = "(latest − prior) / prior × 100"
    else:
        answer = round(delta, 2)
        formula = "latest − prior"
    return {
        "metric": candidate.get("metric"), "status": "derived", "answer": answer,
        "latest_value": latest, "prior_value": prior, "change": round(delta, 2),
        "percent_change": round(pct, 2), "latest_period": latest_period,
        "prior_period": prior_period,
        "period": f"{latest_period} vs {prior_period}" if latest_period and prior_period else None,
        "formula": formula, "source": candidate,
        "inputs": [
            {"name": latest_period or "Latest period", "value": latest, "page": candidate.get("page")},
            {"name": prior_period or "Prior period", "value": prior, "page": candidate.get("page")},
        ],
    }


def _series_from_candidates(candidates: List[Dict[str, Any]]) -> Optional[Tuple[List[str], List[float], Dict[str, Any]]]:
    if not candidates:
        return None
    for candidate in candidates:
        values = candidate.get("values") or []
        periods = candidate.get("periods") or []
        if len(values) >= 2 and len(periods) >= 2:
            n = min(len(values), len(periods))
            return periods[:n], [float(v) for v in values[:n]], candidate
    return None


def _derived_ebitda_series(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    explicit = resolve_metric("ebitda", data)
    direct = _series_from_candidates(explicit)
    if direct:
        periods, values, source = direct
        return {"periods": periods, "values": values, "status": "reported", "source": source}

    ebit_candidates = resolve_metric("ebit", data)
    dep_candidates = resolve_metric("depreciation", data)
    if not ebit_candidates or not dep_candidates:
        # Raw text is a genuine last resort, not a peer to validated tables.
        ebit_candidates = ebit_candidates or resolve_raw_text("ebit", data)
        dep_candidates = dep_candidates or resolve_raw_text("depreciation", data)
    if not ebit_candidates or not dep_candidates:
        return None

    ebit = _series_from_candidates(ebit_candidates)
    dep = _series_from_candidates(dep_candidates)
    if not ebit or not dep:
        return None
    e_periods, e_values, e_source = ebit
    d_periods, d_values, d_source = dep
    common = [(p, e_values[i], d_values[d_periods.index(p)]) for i, p in enumerate(e_periods) if p in d_periods]
    if len(common) < 2:
        return None
    periods = [x[0] for x in common]
    values = [round(x[1] + x[2], 2) for x in common]
    return {
        "periods": periods,
        "values": values,
        "status": "derived",
        "source": {"items": [e_source, d_source]},
        "formula": "EBIT + depreciation & amortisation",
    }


def _two_metric_ebitda(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    series = _derived_ebitda_series(data)
    if not series or len(series["values"]) < 1:
        return None
    value = series["values"][0]
    period = series["periods"][0] if series["periods"] else None
    source = series["source"]
    inputs: List[Dict[str, Any]] = []
    if isinstance(source, dict) and source.get("items"):
        for item in source["items"]:
            inputs.append({"name": item.get("metric", "input"), "value": (item.get("values") or [None])[0], "page": item.get("page")})
    elif isinstance(source, dict):
        inputs = [{"name": "EBITDA", "value": value, "page": source.get("page")}]
    return {
        "metric": "ebitda", "status": series.get("status", "derived"), "answer": round(value, 2),
        "period": period, "formula": series.get("formula"), "inputs": inputs,
        "source": source, "confidence": "high" if series.get("status") == "reported" else "medium",
    }


def _three_metric_ebitda(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pbt = resolve_metric("pbt", data) or resolve_raw_text("pbt", data)
    finance = resolve_metric("finance_costs", data) or resolve_raw_text("finance_costs", data)
    dep = resolve_metric("depreciation", data) or resolve_raw_text("depreciation", data)
    if not pbt or not finance or not dep:
        return None
    p = _select_latest(pbt[0]); f = _select_latest(finance[0]); d = _select_latest(dep[0])
    if not p or not f or not d or not _same_period(p[1], f[1], d[1]):
        return None
    pv, pp = p; fv, fp = f; dv, dp = d
    return {
        "metric": "ebitda", "status": "derived", "answer": round(pv + fv + dv, 2),
        "period": pp or fp or dp, "formula": "PBT + finance costs + depreciation & amortisation",
        "inputs": [
            {"name": "PBT", "value": pv, "page": pbt[0].get("page")},
            {"name": "Finance costs", "value": fv, "page": finance[0].get("page")},
            {"name": "Depreciation & amortisation", "value": dv, "page": dep[0].get("page")},
        ], "source": {"items": [pbt[0], finance[0], dep[0]]}, "confidence": "medium",
    }


def compute_ebitda(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    explicit = resolve_metric("ebitda", data)
    if explicit:
        latest = _select_latest(explicit[0])
        if latest:
            value, period = latest
            return {"metric": "ebitda", "status": "reported", "answer": round(value, 2), "period": period, "formula": None, "inputs": [], "source": explicit[0], "confidence": "high"}
    return _two_metric_ebitda(data) or _three_metric_ebitda(data)


def _series_change(question: str, periods: List[str], values: List[float], source: Any, metric: str) -> Optional[Dict[str, Any]]:
    if len(values) < 2 or not periods:
        return None
    latest = float(values[0])
    prior = float(values[1])
    if prior == 0:
        return None
    delta = latest - prior
    pct = delta / prior * 100.0
    is_pct = question_intent(question) == "yoy_percent"
    return {
        "metric": metric,
        "status": "derived",
        "answer": round(pct if is_pct else delta, 2),
        "latest_value": latest,
        "prior_value": prior,
        "change": round(delta, 2),
        "percent_change": round(pct, 2),
        "latest_period": periods[0] if periods else None,
        "prior_period": periods[1] if len(periods) > 1 else None,
        "period": f"{periods[0]} vs {periods[1]}" if len(periods) > 1 else periods[0],
        "formula": "(latest − prior) / prior × 100" if is_pct else "latest − prior",
        "source": source,
        "inputs": [
            {"name": periods[0] if periods else "Latest period", "value": latest, "page": None},
            {"name": periods[1] if len(periods) > 1 else "Prior period", "value": prior, "page": None},
        ],
    }


def compute_arithmetic(question: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    intent = question_intent(question)
    if intent not in {"sum", "difference"}:
        return None
    metrics = metrics_from_question(question)
    if len(metrics) < 2:
        return None
    selected = []
    for metric in metrics[:2]:
        if metric == "ebitda":
            derived = compute_ebitda(data)
            if derived:
                items = (derived.get("source") or {}).get("items") or [] if isinstance(derived.get("source"), dict) else []
                page = items[0].get("page") if items else None
                selected.append((metric, float(derived["answer"]), derived.get("period"), {"page": page, "source": "derived"}))
                continue
        candidates = resolve_metric(metric, data)
        if not candidates:
            candidates = resolve_raw_text(metric, data)
        if not candidates:
            return None
        latest = _select_latest(candidates[0])
        if latest is None:
            return None
        value, period = latest
        selected.append((metric, value, period, candidates[0]))
    (m1, v1, p1, c1), (m2, v2, p2, c2) = selected
    if not _same_period(p1, p2):
        return None
    answer = v1 + v2 if intent == "sum" else v1 - v2
    formula = f"{m1} + {m2}" if intent == "sum" else f"{m1} − {m2}"
    return {
        "metric": f"{m1}_{'plus' if intent == 'sum' else 'minus'}_{m2}", "status": "derived",
        "answer": round(answer, 2), "period": p1 or p2, "formula": formula,
        "inputs": [
            {"name": m1, "value": v1, "page": c1.get("page") if c1 else None},
            {"name": m2, "value": v2, "page": c2.get("page") if c2 else None},
        ], "source": {"items": [c for c in (c1, c2) if c]},
    }


def compute_enterprise_value(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    explicit = resolve_metric("enterprise_value", data)
    if explicit:
        latest = _select_latest(explicit[0])
        if latest:
            value, period = latest
            return {"metric": "enterprise_value", "status": "reported", "answer": round(value, 2), "period": period, "formula": None, "inputs": [], "source": explicit[0]}

    market = resolve_metric("market_capitalization", data)
    debt = resolve_metric("total_debt", data)
    cash = resolve_metric("cash_and_equivalents", data)
    if not market or not debt or not cash:
        return None
    m = _select_latest(market[0]); d = _select_latest(debt[0]); c = _select_latest(cash[0])
    if not m or not d or not c or not _same_period(m[1], d[1], c[1]):
        return None
    mv, mp = m; dv, dp = d; cv, cp = c
    return {
        "metric": "enterprise_value", "status": "derived", "answer": round(mv + dv - cv, 2),
        "period": mp or dp or cp, "formula": "market capitalization + total debt − cash",
        "inputs": [
            {"name": "Market capitalization", "value": mv, "page": market[0].get("page")},
            {"name": "Total debt", "value": dv, "page": debt[0].get("page")},
            {"name": "Cash & equivalents", "value": cv, "page": cash[0].get("page")},
        ], "source": {"items": [market[0], debt[0], cash[0]]},
    }


def build_evidence(question: str, data: Dict[str, Any]) -> Dict[str, Any]:
    metric = metric_from_question(question)
    candidates = resolve_metric(metric, data) if metric else []
    raw_evidence = []
    if metric:
        raw_evidence = resolve_raw_text(metric, data)[:3]
    related: Dict[str, Any] = {}
    if metric == "ebitda":
        related["ebit"] = resolve_metric("ebit", data)[:3] or resolve_raw_text("ebit", data)[:3]
        related["depreciation"] = resolve_metric("depreciation", data)[:3] or resolve_raw_text("depreciation", data)[:3]
        related["pbt"] = resolve_metric("pbt", data)[:3] or resolve_raw_text("pbt", data)[:3]
        related["finance_costs"] = resolve_metric("finance_costs", data)[:3] or resolve_raw_text("finance_costs", data)[:3]

    computed = None
    intent = question_intent(question)
    if metric == "enterprise_value":
        computed = compute_enterprise_value(data)
    elif metric == "ebitda":
        if intent == "value":
            computed = compute_ebitda(data)
        else:
            series = _derived_ebitda_series(data)
            if series:
                computed = _series_change(question, series["periods"], series["values"], series.get("source"), "ebitda")
    elif metric and intent in {"yoy_percent", "yoy_change"}:
        computed = compute_change(question, candidates or raw_evidence)
    elif metric and intent in {"sum", "difference"}:
        computed = compute_arithmetic(question, data)

    if computed is None and intent in {"sum", "difference"}:
        computed = compute_arithmetic(question, data)

    return {
        "question": question, "metric": metric, "intent": intent,
        "document": {"source_name": data.get("summary", {}).get("source_name"), "metadata": data.get("summary", {}).get("metadata", {})},
        "candidates": candidates[:5], "raw_evidence": raw_evidence, "related": related, "computed": computed,
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
