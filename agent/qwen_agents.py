from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from .ollama_client import chat_json_with_trace

# Sized for a 24 GB RAM workstation. Ollama unloads models between calls by default,
# so the 4B/8B/4B sequence does not require three large resident models.
PLANNER_MODEL = "qwen3:4b"
ANALYST_MODEL = "qwen3:8b"
VERIFIER_MODEL = "qwen3:4b"

ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_text": {"type": "string"},
        "metric": {"type": ["string", "null"]},
        "status": {"type": "string", "enum": ["reported", "derived", "reconstructed", "inferred", "ambiguous", "not_available"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "period": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "scope": {"type": ["string", "null"]},
        "formula": {"type": ["string", "null"]},
        "inputs": {"type": "array", "items": {"type": "object"}},
        "explanation": {"type": "string"},
    },
    "required": ["answer_text", "metric", "status", "confidence", "period", "currency", "unit", "scope", "formula", "inputs", "explanation"],
}

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "severity": {"type": "string", "enum": ["none", "minor", "major"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "required_changes": {"type": "array", "items": {"type": "string"}},
        "source_support": {"type": "string", "enum": ["direct", "calculated", "partial", "unsupported"]},
    },
    "required": ["approved", "severity", "issues", "required_changes", "source_support"],
}

ANALYST_PROMPT = """You are the senior financial analyst in an evidence-first filing system.
Use ONLY the supplied fact records, deterministic calculations, and narrative excerpts.
Never invent a number, period, source, or definition.
Rules:
1. A reported fact outranks an inference.
2. A deterministic calculation is authoritative only when its inputs are explicitly supplied and period/scope aligned.
3. Distinguish stock values from flows and consolidated from standalone.
4. For analytical questions, explain the evidence and clearly label inference.
5. If evidence is insufficient, say so rather than guessing.
6. Use the exact unit and currency from the evidence.
Return exactly one JSON object matching the supplied schema."""

VERIFIER_PROMPT = """You are an independent financial controller reviewing an answer.
Audit ONLY against the supplied evidence and calculation graph.
Check: metric identity, period alignment, scope, units/currency, stock-vs-flow, arithmetic, formula validity, aggregate double-counting, and unsupported causal claims.
Approve only when the answer is fully supported. A correct number with a misleading definition should be rejected.
Return exactly one JSON object matching the supplied schema."""


def _call(prompt: str, payload: Dict[str, Any], model: str, ctx: int, out: int, think: bool | str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    return chat_json_with_trace(
        prompt,
        json.dumps(payload, ensure_ascii=False),
        model=model,
        think=think,
        num_ctx=ctx,
        num_predict=out,
        format_schema=schema,
    )


def analyze(question: str, retrieval: Dict[str, Any], computation: Dict[str, Any] | None) -> Dict[str, Any]:
    result, raw = _call(
        ANALYST_PROMPT,
        {
            "question": question,
            "plan": retrieval.get("plan", {}),
            "deterministic_calculation": computation,
            "selected_facts": retrieval.get("selected_facts", []),
            "selected_chunks": retrieval.get("selected_chunks", []),
            "warnings": retrieval.get("warnings", []),
        },
        ANALYST_MODEL,
        12288,
        700,
        True,
        ANALYST_SCHEMA,
    )
    result["_raw_model_output"] = raw
    return result


def critique(question: str, retrieval: Dict[str, Any], computation: Dict[str, Any] | None, analysis: Dict[str, Any]) -> Dict[str, Any]:
    result, raw = _call(
        VERIFIER_PROMPT,
        {
            "question": question,
            "plan": retrieval.get("plan", {}),
            "deterministic_calculation": computation,
            "selected_facts": retrieval.get("selected_facts", []),
            "analysis": analysis,
        },
        VERIFIER_MODEL,
        8192,
        360,
        False,
        VERIFIER_SCHEMA,
    )
    result["_raw_model_output"] = raw
    return result
