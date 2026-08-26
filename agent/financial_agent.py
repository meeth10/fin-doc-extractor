from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from extractor.financial_resolver import build_evidence, evidence_sources
from extractor.financial_schema import FinancialAnswer
from .ollama_client import chat_json

SYSTEM_PROMPT = """You are a financial analyst answering questions about one company using supplied evidence.
Rules:
1. Use only the supplied evidence. Never invent a number.
2. Prefer a directly reported aggregate line item over a component line item.
3. If the resolver supplies a deterministic calculation under `computed`, use it exactly.
4. Raw OCR/text evidence is supporting evidence only; do not call it high-confidence reported data.
5. If the requested metric is absent from reliable evidence, return status=not_available and answer=null.
6. If the evidence conflicts or the period cannot be identified, return status=ambiguous.
7. A derived metric must show the formula and input values.
8. Preserve the document's currency and unit.
9. Return JSON only with the requested schema.
"""


def _requested_year(question: str) -> Optional[str]:
    m = re.search(r"\b(?:fy\s*)?(20\d{2}(?:[-/–]\d{2})?)\b", question, re.I)
    return m.group(1) if m else None


def _candidate_priority(candidate: Dict[str, Any]) -> tuple:
    alias = str(candidate.get("matched_alias") or "").lower()
    aggregate = {
        "total net sales", "net sales", "revenue from operations", "total revenue",
        "total debt", "total borrowings", "cash and cash equivalents", "cash & cash equivalents",
    }
    is_aggregate = 0 if alias in aggregate else 1
    is_structured = 0 if candidate.get("source") != "raw_text" else 1
    validated = 0 if candidate.get("validated") else 1
    return is_aggregate, is_structured, validated, -float(candidate.get("score", 0) or 0)


def _select_reported_candidate(question: str, evidence: Dict[str, Any]) -> Optional[Tuple[float | str, Optional[str], Dict[str, Any]]]:
    requested = _requested_year(question)
    candidates = sorted(list(evidence.get("candidates", [])), key=_candidate_priority)
    raw_candidates = sorted(list(evidence.get("raw_evidence", [])), key=_candidate_priority)

    # Prefer a statement-table aggregate row. Only use raw text to rescue a
    # missing/malformed table when it contains an explicit aggregate label.
    pools = candidates[:]
    for raw in raw_candidates:
        alias = str(raw.get("matched_alias") or "").lower()
        if alias in {"total net sales", "net sales", "revenue from operations", "total revenue", "total debt", "total borrowings", "cash and cash equivalents", "cash & cash equivalents"}:
            pools.append(raw)
    pools.sort(key=_candidate_priority)

    for candidate in pools:
        values = candidate.get("values") or []
        periods = candidate.get("periods") or []
        if not values:
            continue
        if requested:
            for idx, period in enumerate(periods):
                if requested in period and idx < len(values):
                    return values[idx], period, candidate
            continue
        return values[0], periods[0] if periods else None, candidate
    return None


def _source_truth(question: str, evidence: Dict[str, Any]) -> Optional[FinancialAnswer]:
    metric = evidence.get("metric")
    if not metric:
        return None
    selected = _select_reported_candidate(question, evidence)
    if not selected:
        return None
    value, period, source = selected
    metadata = evidence.get("document", {}).get("metadata", {}) or {}
    is_raw = source.get("source") == "raw_text"
    confidence = "medium" if is_raw else ("high" if source.get("validated") else "medium")
    return FinancialAnswer(
        metric=metric,
        answer=value,
        period=period,
        currency=metadata.get("currency"),
        unit=metadata.get("unit") or metadata.get("currency_unit"),
        status="reported",
        confidence=confidence,
        formula=None,
        inputs=[],
        sources=evidence_sources([source], []),
        explanation=f"Source-selected under {source.get('table_title') or source.get('matched_alias') or metric}.",
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


def _ground_with_computed(answer: FinancialAnswer, evidence: Dict[str, Any]) -> FinancialAnswer:
    computed = evidence.get("computed")
    if not computed:
        return answer
    metadata = evidence.get("document", {}).get("metadata", {}) or {}
    source = computed.get("source") or {}
    source_candidates = source.get("items") or ([source] if source else [])
    return FinancialAnswer(
        metric=computed.get("metric") or answer.metric,
        answer=computed.get("answer"),
        period=computed.get("period"),
        currency=metadata.get("currency"),
        unit=metadata.get("unit") or metadata.get("currency_unit"),
        status=computed.get("status", "derived"),
        confidence=computed.get("confidence", "high"),
        formula=computed.get("formula"),
        inputs=computed.get("inputs") or [],
        sources=evidence_sources(source_candidates, []),
        explanation=answer.explanation or "Calculated deterministically from source evidence.",
    )


def _ground_with_source_truth(llm_answer: FinancialAnswer, source_truth: Optional[FinancialAnswer]) -> FinancialAnswer:
    if source_truth is None:
        return llm_answer
    return FinancialAnswer(
        metric=source_truth.metric,
        answer=source_truth.answer,
        period=source_truth.period,
        currency=source_truth.currency,
        unit=source_truth.unit,
        status=source_truth.status,
        confidence=source_truth.confidence,
        formula=source_truth.formula,
        inputs=llm_answer.inputs,
        sources=source_truth.sources,
        explanation=llm_answer.explanation or source_truth.explanation,
    )


def answer_question(question: str, data: Dict[str, Any]) -> Dict[str, Any]:
    evidence = build_evidence(question, data)
    if not evidence.get("metric"):
        return {"metric": None, "answer": None, "status": "ambiguous", "confidence": "low", "message": "I could not map that question to a supported financial metric yet.", "llm_used": False, "evidence": evidence}

    source_truth = _source_truth(question, evidence)
    user_prompt = (
        "Answer the question using ONLY this evidence.\n\n"
        "If `computed` is present, it is authoritative.\n"
        "Do not convert a component line into an aggregate metric.\n\n"
        f"{evidence}\n\nQuestion: {question}"
    )
    payload = chat_json(SYSTEM_PROMPT, user_prompt)
    answer = _validate(payload, evidence)
    if evidence.get("computed"):
        answer = _ground_with_computed(answer, evidence)
    elif source_truth is not None:
        answer = _ground_with_source_truth(answer, source_truth)
    return {**answer.as_dict(), "llm_used": True, "llm_model": "qwen3:8b", "evidence": evidence}
