from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .financial_resolver import (
    METRIC_ALIASES,
    compute_ebitda,
    compute_enterprise_value,
    resolve_metric,
    resolve_raw_text,
)

CORE_METRICS = (
    "revenue",
    "ebit",
    "ebitda",
    "pat",
    "pbt",
    "cash_and_equivalents",
    "total_debt",
    "cfo",
    "capex",
    "total_assets",
    "total_equity",
    "market_capitalization",
)


def _period_value_pairs(candidate: Dict[str, Any]) -> Iterable[tuple[str | None, float]]:
    values = candidate.get("values") or []
    periods = candidate.get("periods") or []
    for idx, value in enumerate(values):
        period = periods[idx] if idx < len(periods) else None
        yield period, float(value)


def _candidate_facts(metric: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = resolve_metric(metric, data) or resolve_raw_text(metric, data)
    facts: List[Dict[str, Any]] = []
    for candidate in candidates:
        for period, value in _period_value_pairs(candidate):
            facts.append({
                "metric": metric,
                "value": value,
                "period": period,
                "status": "reported",
                "confidence": "high" if candidate.get("validated") else "medium",
                "page": candidate.get("page"),
                "statement": candidate.get("statement"),
                "table_title": candidate.get("table_title"),
                "source": candidate.get("source"),
                "matched_alias": candidate.get("matched_alias"),
            })
    return facts


def _dedupe(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for fact in facts:
        key = (fact.get("metric"), fact.get("period"), fact.get("value"), fact.get("page"), fact.get("source"))
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


def build_fact_store(data: Dict[str, Any]) -> Dict[str, Any]:
    metadata = data.get("summary", {}).get("metadata", {}) or {}
    facts: List[Dict[str, Any]] = []

    for metric in CORE_METRICS:
        facts.extend(_candidate_facts(metric, data))

    derived: List[Dict[str, Any]] = []
    ebitda = compute_ebitda(data)
    if ebitda and ebitda.get("status") == "derived":
        derived.append(ebitda)

    enterprise_value = compute_enterprise_value(data)
    if enterprise_value and enterprise_value.get("status") == "derived":
        derived.append(enterprise_value)

    facts = _dedupe(facts)
    for item in derived:
        facts.append({
            "metric": item.get("metric"),
            "value": item.get("answer"),
            "period": item.get("period"),
            "status": "derived",
            "confidence": item.get("confidence", "medium"),
            "formula": item.get("formula"),
            "inputs": item.get("inputs", []),
            "page": None,
            "statement": None,
            "table_title": None,
            "source": "calculation_engine",
        })

    facts.sort(key=lambda x: (x.get("metric") or "", str(x.get("period") or ""), x.get("page") or 10**9))
    return {
        "schema_version": "1.0",
        "document": {
            "source_name": data.get("summary", {}).get("source_name"),
            "currency": metadata.get("currency"),
            "unit": metadata.get("unit") or metadata.get("currency_unit"),
        },
        "fact_count": len(facts),
        "metrics": sorted(set(f["metric"] for f in facts if f.get("metric"))),
        "facts": facts,
    }
