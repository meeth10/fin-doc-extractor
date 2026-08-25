from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from extractor.financial_resolver import build_evidence, evidence_sources
from extractor.financial_schema import FinancialAnswer
from .ollama_client import chat_json

SYSTEM_PROMPT = """You are a financial analyst answering questions about one company using supplied evidence.
Rules:
1. Use only the supplied evidence. Never invent a number.
2. Prefer a directly reported line item over a derived value.
3. If the requested metric is absent, return status=not_available and answer=null.
4. If the evidence conflicts or the period cannot be identified, return status=ambiguous.
5. A derived metric must show the formula and its input values.
6. Preserve the document's currency and unit.
7. Return JSON only with this shape:
{
  "metric": string,
  "answer": number|string|null,
  "period": string|null,
  "currency": string|null,
  "unit": string|null,
  "status": "reported"|"derived"|"ambiguous"|"not_available",
  "confidence": "high"|"medium"|"low",
  "formula": string|null,
  "inputs": [{"name": string, "value": number|string, "page": number|null}],
  "explanation": string|null
}
"""


def _requested_year(question: str) -> Optional[str]:
    m = re.search(r"\b(?:fy\s*)?(20\d{2}(?:[-/–]\d{2})?)\b", question, re.I)
    return m.group(1) if m else None


def _select_reported_candidate(question: str, evidence: Dict[str, Any]) -> Optional[Tuple[float | str, Optional[str], Dict[str, Any]]]:
    requested = _requested_year(question)
    pools = list(evidence.get("candidates", [])) + list(evidence.get("raw_evidence", []))
    for candidate in pools:
        values = candidate.get("values") or []
        if not values:
            continue
        periods = candidate.get("periods") or []
        if requested:
            for idx, period in enumerate(periods):
                if requested in period and idx < len(values):
                    return values[idx], period, candidate
            continue
        return values[0], periods[0] if periods else None, candidate
    return None


def _find_first(evidence: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    items = evidence.get("related", {}).get(name) or []
    return items[0] if items else None


def _select_value(question: str, candidate: Dict[str, Any]) -> Tuple[Optional[float | str], Optional[str]]:
    requested = _requested_year(question)
    values = candidate.get("values") or []
    periods = candidate.get("periods") or []
    if not values:
        return None, None
    if requested:
        for idx, period in enumerate(periods):
            if requested in period and idx < len(values):
                return values[idx], period
    return values[0], periods[0] if periods else None


def _derived_ebitda(question: str, evidence: Dict[str, Any]) -> Optional[FinancialAnswer]:
    # Prefer an explicitly reported EBITDA line.
    explicit = _select_reported_candidate(question, evidence)
    if explicit:
        value, period, source = explicit
        metadata = evidence.get("document", {}).get("metadata", {}) or {}
        return FinancialAnswer(
            metric="ebitda",
            answer=value,
            period=period,
            currency=metadata.get("currency"),
            unit=metadata.get("unit") or metadata.get("currency_unit"),
            status="reported",
            confidence="high" if source.get("validated") else "medium",
            formula=None,
            inputs=[],
            sources=evidence_sources(evidence.get("candidates", []), evidence.get("raw_evidence", []))[:3],
            explanation="EBITDA is explicitly reported in the source evidence.",
        )

    ebit = _find_first(evidence, "ebit")
    depreciation = _find_first(evidence, "depreciation")
    if not ebit or not depreciation:
        return None

    ebit_value, ebit_period = _select_value(question, ebit)
    dep_value, dep_period = _select_value(question, depreciation)
    if not isinstance(ebit_value, (int, float)) or not isinstance(dep_value, (int, float)):
        return None
    if ebit_period and dep_period and ebit_period != dep_period:
        return None

    period = ebit_period or dep_period
    metadata = evidence.get("document", {}).get("metadata", {}) or {}
    answer = float(ebit_value) + float(dep_value)
    input_pages = [
        {"name": "EBIT", "value": ebit_value, "page": ebit.get("page")},
        {"name": "Depreciation", "value": dep_value, "page": depreciation.get("page")},
    ]
    refs = evidence_sources(
        evidence.get("candidates", []) + (evidence.get("related", {}).get("ebit") or []) + (evidence.get("related", {}).get("depreciation") or []),
        evidence.get("raw_evidence", []),
    )
    return FinancialAnswer(
        metric="ebitda",
        answer=answer,
        period=period,
        currency=metadata.get("currency"),
        unit=metadata.get("unit") or metadata.get("currency_unit"),
        status="derived",
        confidence="high",
        formula="EBIT + depreciation & amortisation",
        inputs=input_pages,
        sources=refs[:5],
        explanation="No explicit EBITDA line was used; EBITDA was derived from EBIT plus depreciation/amortisation for the same reporting period.",
    )


def _direct_answer(question: str, evidence: Dict[str, Any]) -> Optional[FinancialAnswer]:
    metric = evidence.get("metric")
    if not metric:
        return None
    if metric == "ebitda":
        return _derived_ebitda(question, evidence)
    selected = _select_reported_candidate(question, evidence)
    if not selected:
        return None
    value, period, source = selected
    metadata = evidence.get("document", {}).get("metadata", {}) or {}
    return FinancialAnswer(
        metric=metric,
        answer=value,
        period=period,
        currency=metadata.get("currency"),
        unit=metadata.get("unit") or metadata.get("currency_unit"),
        status="reported",
        confidence="high" if source.get("validated") else "medium",
        formula=None,
        inputs=[],
        sources=evidence_sources(evidence.get("candidates", []), evidence.get("raw_evidence", []))[:3],
        explanation=f"Directly reported under {source.get('table_title') or source.get('matched_alias') or metric}.",
    )


def _validate(payload: Dict[str, Any], evidence: Dict[str, Any]) -> FinancialAnswer:
    required = {"metric", "answer", "period", "currency", "unit", "status", "confidence", "formula", "inputs", "explanation"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Ollama response missing fields: {sorted(missing)}")
    if payload["status"] not in {"reported", "derived", "ambiguous", "not_available"}:
        raise RuntimeError("Ollama returned an invalid financial status.")
    if payload["confidence"] not in {"high", "medium", "low"}:
        raise RuntimeError("Ollama returned an invalid confidence.")
    if payload["status"] != "not_available" and payload["answer"] is None:
        raise RuntimeError("Non-available answer cannot have a null value.")
    if payload["status"] == "not_available" and payload["answer"] is not None:
        raise RuntimeError("A not_available answer must have a null value.")
    return FinancialAnswer(
        metric=evidence.get("metric") or payload["metric"],
        answer=payload["answer"],
        period=payload["period"],
        currency=payload["currency"],
        unit=payload["unit"],
        status=payload["status"],
        confidence=payload["confidence"],
        formula=payload["formula"],
        inputs=payload["inputs"] or [],
        sources=evidence_sources(evidence.get("candidates", []), evidence.get("raw_evidence", [])),
        explanation=payload["explanation"],
    )


def _needs_llm(question: str, metric: str) -> bool:
    q = question.lower()
    analytical = ("why", "how", "explain", "driver", "changed", "change", "grew", "growth", "margin", "compare", "versus", "vs", "trend")
    if any(word in q for word in analytical):
        return True
    return metric not in {"cash_and_equivalents", "revenue", "ebitda", "ebit", "pat", "pbt", "finance_costs", "total_debt", "cfo", "capex", "total_assets", "total_equity", "eps"}


def answer_question(question: str, data: Dict[str, Any]) -> Dict[str, Any]:
    evidence = build_evidence(question, data)
    if not evidence.get("metric"):
        return {
            "metric": None,
            "answer": None,
            "status": "ambiguous",
            "confidence": "low",
            "message": "I could not map that question to a supported financial metric yet.",
            "evidence": evidence,
        }

    direct = _direct_answer(question, evidence)
    if direct is not None and not _needs_llm(question, evidence["metric"]):
        return {**direct.as_dict(), "evidence": evidence}

    user_prompt = (
        "Answer the question using ONLY this evidence.\n\n"
        f"{evidence}\n\n"
        f"Question: {question}"
    )
    payload = chat_json(SYSTEM_PROMPT, user_prompt)
    answer = _validate(payload, evidence)
    return {**answer.as_dict(), "evidence": evidence}
