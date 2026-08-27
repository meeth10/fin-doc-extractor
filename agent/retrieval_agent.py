from __future__ import annotations

import json
from typing import Any, Dict

from .ollama_client import chat_json

RETRIEVAL_MODEL = "ibm/granite4.2:3b"

SYSTEM_PROMPT = """You are the retrieval specialist for a financial analysis system.
Your job is NOT to answer the user's question.
Your job is to identify the smallest, strongest evidence packet needed for another reasoning model.

Rules:
1. Prefer validated statement tables over raw text.
2. Prefer the titled primary financial statement over notes, indexes, narrative pages, and references.
3. Prefer aggregate rows for aggregate metrics (for example Total net sales over Products/Services; Total debt over debt issuance).
4. Preserve periods and page numbers.
5. Select evidence, do not invent or calculate values.
6. Return JSON only:
{
  "query_type": string,
  "selected_metrics": [string],
  "selected_sources": [
    {"metric": string, "page": number|null, "statement": string|null, "table_title": string|null, "matched_alias": string|null, "values": [number], "periods": [string]}
  ],
  "warnings": [string]
}
"""


def retrieve(question: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "Select the strongest evidence for this question. Do not answer it.\n\n"
        f"Question: {question}\n\n"
        f"Evidence: {json.dumps(evidence, ensure_ascii=False)}"
    )
    return chat_json(SYSTEM_PROMPT, prompt, model=RETRIEVAL_MODEL, think=False)
