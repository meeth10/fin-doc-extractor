from __future__ import annotations

import json
from typing import Any, Dict, List

from .ollama_client import chat_json_with_trace

RETRIEVAL_MODEL = "ibm/granite4.2:3b"

SYSTEM_PROMPT = """You are the retrieval specialist for a financial analysis system.
Your job is NOT to answer the user's question.
Select the smallest set of supplied source records that directly supports the answer.

Rules:
1. Choose only from supplied records. Never invent values, pages, periods, or rows.
2. Prefer validated statement tables over raw text.
3. Prefer primary statement pages over indexes, notes, narrative pages, and references.
4. Prefer aggregate rows for aggregate metrics: Total net sales over Products/Services; total debt over debt issuance.
5. Preserve the exact reported values and periods.
6. Select at most 6 source records unless the query genuinely requires more.
7. Return exactly one JSON object matching the requested schema. No markdown.
"""


def _record_key(record: Dict[str, Any]) -> str:
    return "|".join(str(record.get(k, "")) for k in ("metric", "page", "statement", "table_title", "matched_alias"))


def _candidate_index(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in evidence.get("candidates", []):
        if item.get("values"):
            records.append({
                "metric": item.get("metric"), "page": item.get("page"), "statement": item.get("statement"),
                "table_title": item.get("table_title"), "matched_alias": item.get("matched_alias"),
                "values": item.get("values", [])[:4], "periods": item.get("periods", [])[:4],
                "validated": bool(item.get("validated")), "assignment": item.get("assignment"),
                "score": item.get("score", 0),
            })
    for item in evidence.get("raw_evidence", []):
        if item.get("values"):
            records.append({
                "metric": item.get("metric"), "page": item.get("page"), "statement": item.get("statement"),
                "table_title": item.get("table_title"), "matched_alias": item.get("matched_alias"),
                "values": item.get("values", [])[:4], "periods": item.get("periods", [])[:4],
                "validated": False, "assignment": "raw_text_fallback", "score": item.get("score", 0),
            })
    seen = set(); unique = []
    for item in records:
        key = _record_key(item)
        if key in seen:
            continue
        seen.add(key); unique.append(item)
    unique.sort(key=lambda x: (not x["validated"], -(float(x.get("score", 0) or 0))))
    return unique[:40]


def _deterministic_fallback(evidence: Dict[str, Any], reason: str) -> Dict[str, Any]:
    candidates = _candidate_index(evidence)
    selected = candidates[:1]
    computed = evidence.get("computed")
    if computed:
        source = computed.get("source") or {}
        selected = source.get("items") or ([source] if source else selected)
    return {
        "query_type": evidence.get("intent", "value"),
        "selected_metrics": [evidence.get("metric")] if evidence.get("metric") else [],
        "selected_sources": selected[:6],
        "warnings": [reason],
        "fallback": True,
    }


def retrieve(question: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    index = _candidate_index(evidence)
    payload = {
        "question": question, "metric": evidence.get("metric"), "intent": evidence.get("intent"),
        "computed": evidence.get("computed"), "candidate_index": index,
    }
    user = "Select evidence only from candidate_index. Never answer the question.\n\n" + json.dumps(payload, ensure_ascii=False)
    try:
        result, raw_output = chat_json_with_trace(
            SYSTEM_PROMPT, user, model=RETRIEVAL_MODEL, think=False, num_ctx=8192, num_predict=384
        )
        selected = result.get("selected_sources")
        if not isinstance(selected, list):
            raise RuntimeError("Granite retrieval JSON missing selected_sources")
        allowed = {_record_key(x): x for x in index}
        clean = []
        for item in selected[:6]:
            if isinstance(item, dict):
                key = _record_key(item)
                if key in allowed:
                    clean.append(allowed[key])
        if not clean and index:
            fallback = _deterministic_fallback(evidence, "Granite selected no valid candidate records.")
            fallback["raw_model_output"] = raw_output
            fallback["model"] = RETRIEVAL_MODEL
            return fallback
        result["selected_sources"] = clean
        result["raw_model_output"] = raw_output
        result["model"] = RETRIEVAL_MODEL
        result["fallback"] = False
        return result
    except (RuntimeError, TypeError, ValueError) as exc:
        fallback = _deterministic_fallback(evidence, f"Granite retrieval fallback used: {exc}")
        fallback["raw_model_output"] = None
        fallback["model"] = RETRIEVAL_MODEL
        return fallback
