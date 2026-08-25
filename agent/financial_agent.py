from __future__ import annotations

from typing import Any, Dict

from extractor.financial_resolver import build_evidence, evidence_sources
from extractor.financial_schema import FinancialAnswer, EvidenceRef
from .ollama_client import chat_json

SYSTEM_PROMPT = """You are a financial analyst answering questions about one company using supplied evidence.
Rules:
1. Use only the supplied evidence. Never invent a number.
2. Prefer a directly reported line item over a derived value.
3. If the requested metric is absent, return status=not_available.
4. If the evidence conflicts or the period cannot be identified, return status=ambiguous.
5. A derived metric must show the formula and its input values.
6. Preserve the document's currency and unit.
7. Return JSON only with this shape:
{
  \"metric\": string,
  \"answer\": number|string|null,
  \"period\": string|null,
  \"currency\": string|null,
  \"unit\": string|null,
  \"status\": \"reported\"|\"derived\"|\"ambiguous\"|\"not_available\",
  \"confidence\": \"high\"|\"medium\"|\"low\",
  \"formula\": string|null,
  \"inputs\": [{\"name\": string, \"value\": number|string, \"page\": number|null}],
  \"explanation\": string|null
}
"""


def _validate(payload: Dict[str, Any], evidence: Dict[str, Any]) -> FinancialAnswer:
    required = {"metric", "answer", "period", "currency", "unit", "status", "confidence", "formula", "inputs", "explanation"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Ollama response missing fields: {sorted(missing)}")
    if payload["status"] not in {"reported", "derived", "ambiguous", "not_available"}:
        raise RuntimeError("Ollama returned an invalid financial status.")
    if payload["confidence"] not in {"high", "medium", "low"}:
        raise RuntimeError("Ollama returned an invalid confidence.")
    metric = evidence.get("metric") or payload["metric"]
    if payload["status"] != "not_available" and payload["answer"] is None:
        raise RuntimeError("Non-available answer cannot have a null value.")
    return FinancialAnswer(
        metric=metric,
        answer=payload["answer"],
        period=payload["period"],
        currency=payload["currency"],
        unit=payload["unit"],
        status=payload["status"],
        confidence=payload["confidence"],
        formula=payload["formula"],
        inputs=payload["inputs"] or [],
        sources=evidence_sources(evidence.get("candidates", [])),
        explanation=payload["explanation"],
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
            "evidence": evidence,
        }

    user_prompt = (
        "Answer the question using ONLY this evidence.\n\n"
        f"{evidence}\n\n"
        f"Question: {question}"
    )
    payload = chat_json(SYSTEM_PROMPT, user_prompt)
    answer = _validate(payload, evidence)
    return {**answer.as_dict(), "evidence": evidence}
