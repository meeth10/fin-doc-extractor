from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from agent.financial_facts import build_fact_store, total_debt_candidates
from .ollama_client import chat_json_with_trace, embed_texts

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DOCUMENT_MODEL = "qwen3:8b"
CACHE_FILENAME = "qwen_embedding_cache.json"

DOCUMENT_SYSTEM_PROMPT = """You are the document analyst in a financial analysis system.
You select and organize evidence. You do NOT calculate or answer the user's question.

Rules:
1. Select only supplied fact IDs or chunk IDs.
2. Prefer validated primary financial-statement evidence.
3. Distinguish balance/stock values from cash-flow movements.
4. For an aggregate metric represented by components, select all required components.
5. Prefer the requested reporting period.
6. Never invent values, pages, labels, periods, or IDs.
7. Return exactly one JSON object:
{
  "selected_fact_ids": ["f1"],
  "selected_chunk_ids": ["c1"],
  "warnings": ["..."]
}
"""


_VALUE_INTENTS = {"value", "yoy_percent", "yoy_change", "sum", "difference"}
_ANALYTICAL_HINTS = {
    "why", "explain", "reason", "driver", "cause", "changed", "impact",
    "trend", "quality", "risk", "outlook", "strategy", "margin", "growth",
}


def _simple_value_question(question: str) -> bool:
    lowered = question.lower()
    return not any(word in lowered for word in _ANALYTICAL_HINTS)


def _tokens(text: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _page_chunks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page in data.get("document", {}).get("pages") or []:
        page_number = page.get("page_number_human") or page.get("page_number")
        text = str(page.get("raw_text") or "").strip()
        if not text:
            continue
        chunks.append({
            "chunk_id": f"p{page_number}",
            "page": page_number,
            "text": text[:5000],
        })
    return chunks


def _fingerprint_chunks(chunks: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(str(chunk.get("page")).encode())
        digest.update(str(chunk.get("text")).encode())
    return digest.hexdigest()


def _load_or_build_embeddings(chunks: List[Dict[str, Any]], out_dir: str | None) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    cache_path = Path(out_dir or "output") / CACHE_FILENAME
    fingerprint = _fingerprint_chunks(chunks)
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache.get("fingerprint") == fingerprint and cache.get("model") == EMBEDDING_MODEL:
                return cache.get("chunks", [])
        except (OSError, ValueError, TypeError):
            pass

    embeddings = embed_texts([chunk["text"] for chunk in chunks], EMBEDDING_MODEL)
    cached = []
    for chunk, vector in zip(chunks, embeddings):
        cached.append({**chunk, "embedding": vector})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"fingerprint": fingerprint, "model": EMBEDDING_MODEL, "chunks": cached}, ensure_ascii=False),
        encoding="utf-8",
    )
    return cached


def _rank_facts(question: str, facts: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    q = _tokens(question)
    debt_query = "debt" in q or "borrowings" in q
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for fact in facts:
        text = " ".join(str(fact.get(k) or "") for k in (
            "metric", "label", "section_context", "statement", "table_title", "period"
        ))
        overlap = len(q & _tokens(text))
        score = float(overlap)
        if fact.get("validated"):
            score += 0.5
        if fact.get("statement") == "balance_sheet":
            score += 0.5
        if debt_query:
            label = str(fact.get("label", "")).lower()
            if "commercial paper" in label or "term debt" in label or "borrowings" in label:
                score += 3.0
            if fact.get("is_flow_candidate") or fact.get("statement") == "cash_flow":
                score -= 4.0
        scored.append((score, fact))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("page") or 10**9))
    return [fact for score, fact in scored[:limit] if score > 0]


def retrieve(question: str, data: Dict[str, Any], *, out_dir: str = "output") -> Dict[str, Any]:
    store = build_fact_store(data)
    facts = store.get("facts", [])
    ranked = _rank_facts(question, facts)

    # Strong source-aware fast path for ordinary financial facts.
    if _simple_value_question(question):
        selected = ranked
        lowered = question.lower()
        if "total debt" in lowered or lowered.strip() == "debt":
            debt = total_debt_candidates(facts)
            if debt:
                period = debt[0].get("period")
                selected = total_debt_candidates(facts, period=period)
        return {
            "model": DOCUMENT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "mode": "deterministic_fact_first",
            "raw_model_output": None,
            "model_response": None,
            "selected_facts": selected[:12],
            "selected_chunks": [],
            "candidate_facts": ranked[:20],
            "candidate_chunks": [],
            "warnings": [],
            "fallback": False,
        }

    chunks = _page_chunks(data)
    embedded = _load_or_build_embeddings(chunks, out_dir)
    query_embedding = embed_texts([question], EMBEDDING_MODEL)[0] if embedded else []
    ranked_chunks = sorted(
        embedded,
        key=lambda chunk: _cosine(query_embedding, chunk.get("embedding", [])),
        reverse=True,
    )[:8]

    fact_payload = [
        {
            "id": fact["fact_id"],
            **{key: value for key, value in fact.items() if key != "fact_id"},
        }
        for fact in ranked[:20]
    ]
    chunk_payload = [
        {"id": chunk["chunk_id"], "page": chunk["page"], "text": chunk["text"]}
        for chunk in ranked_chunks
    ]

    raw_output = None
    model_response = None
    warnings: List[str] = []
    try:
        model_response, raw_output = chat_json_with_trace(
            DOCUMENT_SYSTEM_PROMPT,
            json.dumps({
                "question": question,
                "facts": fact_payload,
                "narrative_chunks": chunk_payload,
            }, ensure_ascii=False),
            model=DOCUMENT_MODEL,
            think=False,
            num_ctx=8192,
            num_predict=384,
        )
        fact_by_id = {fact["fact_id"]: fact for fact in ranked}
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in ranked_chunks}
        selected_facts = [fact_by_id[fid] for fid in model_response.get("selected_fact_ids", []) if fid in fact_by_id]
        selected_chunks = [chunk_by_id[cid] for cid in model_response.get("selected_chunk_ids", []) if cid in chunk_by_id]
        if not selected_facts:
            selected_facts = ranked[:8]
            warnings.append("Qwen 8B selected no valid fact IDs; retained deterministic candidates.")
        return {
            "model": DOCUMENT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "mode": "semantic",
            "raw_model_output": raw_output,
            "model_response": model_response,
            "selected_facts": selected_facts,
            "selected_chunks": selected_chunks,
            "candidate_facts": ranked[:20],
            "candidate_chunks": ranked_chunks,
            "warnings": warnings + list(model_response.get("warnings") or []),
            "fallback": False,
        }
    except (RuntimeError, TypeError, ValueError) as exc:
        warnings.append(f"Qwen retrieval fallback used: {exc}")
        return {
            "model": DOCUMENT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "mode": "semantic_fallback",
            "raw_model_output": raw_output,
            "model_response": model_response,
            "selected_facts": ranked[:8],
            "selected_chunks": ranked_chunks[:4],
            "candidate_facts": ranked[:20],
            "candidate_chunks": ranked_chunks,
            "warnings": warnings,
            "fallback": True,
        }
