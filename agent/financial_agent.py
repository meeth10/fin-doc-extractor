from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from extractor.financial_resolver import build_evidence, evidence_sources
from extractor.financial_schema import FinancialAnswer
from .ollama_client import chat_json
from .retrieval_agent import retrieve

REASONING_MODEL = "deepseek-r1:14b"
RETRIEVAL_MODEL = "ibm/granite4.2:3b"

SYSTEM_PROMPT = """You are the senior financial reasoning analyst.
You receive one question and a compact evidence packet selected by a separate retrieval model.

Rules:
1. Use only the supplied evidence packet and deterministic computed evidence.
2. Never invent a number, period, unit, or source.
3. Prefer directly reported aggregate line items over components.
4. If `computed` is present, it is authoritative and must be used exactly.
5. Distinguish reported from derived values.
6. For derived metrics, show the formula and actual input values.
7. If evidence is insufficient or conflicts, return ambiguous/not_available rather than guessing.
8. Keep the explanation concise: 2-5 sentences unless the question asks for analysis.
9. Return exactly one valid JSON object and nothing else:
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
        raise RuntimeError(f"DeepSeek response missing fields: {sorted(missing)}")
    if payload["status"] not in {"reported", "derived", "ambiguous", "not_available"}:
        raise RuntimeError("DeepSeek returned an invalid financial status.")
    if payload["confidence"] not in {"high", "medium", "low"}:
        raise RuntimeError("DeepSeek returned an invalid confidence.")
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
        sources=evidence_sources(source_candidates, evidence.get("raw_evidence", [])),
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
        return {
            "metric": None,
            "answer": None,
            "status": "ambiguous",
            "confidence": "low",
            "message": "I could not map that question to a supported financial metric yet.",
            "llm_used": False,
            "evidence": evidence,
        }

    retrieval_packet = retrieve(question, evidence)
    reasoning_input = {
        "question": question,
        "computed": evidence.get("computed"),
        "document": evidence.get("document"),
        "selected_sources": retrieval_packet.get("selected_sources", []),
        "retrieval_warnings": retrieval_packet.get("warnings", []),
    }
    user_prompt = (
        "Reason through the question using ONLY this compact evidence packet. "
        "Do not use prior knowledge or unsupported facts. "
        "If `computed` is present, it is authoritative and must be used exactly.\n\n"
        f"{json.dumps(reasoning_input, ensure_ascii=False)}\n\n"
        f"Question: {question}"
    )
    payload = chat_json(
        SYSTEM_PROMPT,
        user_prompt,
        model=REASONING_MODEL,
        think="low",
        num_ctx=32768,
        num_predict=768,
    )
    answer = _validate(payload, evidence)
    if evidence.get("computed"):
        answer = _ground_with_computed(answer, evidence)
    else:
        source_truth = _source_truth(question, evidence)
        if source_truth is not None:
            answer = _ground_with_source_truth(answer, source_truth)
    return {
        **answer.as_dict(),
        "llm_used": True,
        "llm_model": REASONING_MODEL,
        "retrieval_model": RETRIEVAL_MODEL,
        "retrieval": retrieval_packet,
        "evidence": evidence,
    }
