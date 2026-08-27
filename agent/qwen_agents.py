from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from .ollama_client import chat_json_with_trace

DOCUMENT_MODEL = "qwen3:8b"
REASONING_MODEL = "qwen3:14b"
CRITIC_MODEL = "qwen3:8b"

REASONING_PROMPT = """You are the senior financial analyst.
Use ONLY the supplied facts, calculations, and source excerpts.
Do not invent numbers or sources. Deterministic calculations are authoritative.
Explain what the evidence means financially; distinguish fact from inference.
Return exactly one JSON object with:
{"answer_text":string,"metric":string|null,"status":"reported|derived|ambiguous|not_available","confidence":"high|medium|low","period":string|null,"currency":string|null,"unit":string|null,"formula":string|null,"inputs":[{"name":string,"value":number|string,"page":number|null}],"explanation":string}
"""

CRITIC_PROMPT = """You are a financial controller auditing an analyst answer.
Check period alignment, units, stock-vs-flow, aggregate-vs-component logic, arithmetic, source support, and unsupported claims.
Return exactly:
{"approved":true|false,"severity":"none|minor|major","issues":[string],"required_changes":[string]}
"""


def _call(prompt: str, payload: Dict[str, Any], model: str, ctx: int, out: int, think: bool | str) -> Tuple[Dict[str, Any], str]:
    return chat_json_with_trace(prompt, json.dumps(payload, ensure_ascii=False), model=model, think=think, num_ctx=ctx, num_predict=out)


def analyze(question: str, retrieval: Dict[str, Any], computation: Dict[str, Any] | None) -> Dict[str, Any]:
    result, raw = _call(
        REASONING_PROMPT,
        {"question": question, "deterministic_calculation": computation, "selected_facts": retrieval.get("selected_facts", []), "selected_chunks": retrieval.get("selected_chunks", []), "warnings": retrieval.get("warnings", [])},
        REASONING_MODEL, 12288, 640, True,
    )
    result["_raw_model_output"] = raw
    return result


def critique(question: str, retrieval: Dict[str, Any], computation: Dict[str, Any] | None, analysis: Dict[str, Any]) -> Dict[str, Any]:
    result, raw = _call(
        CRITIC_PROMPT,
        {"question": question, "deterministic_calculation": computation, "selected_facts": retrieval.get("selected_facts", []), "analysis": analysis},
        CRITIC_MODEL, 8192, 320, False,
    )
    result["_raw_model_output"] = raw
    return result
