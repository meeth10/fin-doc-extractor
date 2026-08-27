from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .financial_facts import build_fact_store, fact_text
from .ollama_client import chat_json_with_trace, embed_texts

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
PLANNER_MODEL = "qwen3:4b"

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "metrics": {"type": "array", "items": {"type": "string"}},
        "operation": {"type": "string", "enum": ["value", "yoy_change", "yoy_percent", "sum", "difference", "explain"]},
        "target_period": {"type": ["string", "null"]},
        "comparison_period": {"type": ["string", "null"]},
        "scope": {"type": "string", "enum": ["consolidated", "standalone", "unknown"]},
        "basis": {"type": "string", "enum": ["reported", "derived", "unknown"]},
        "needs_narrative": {"type": "boolean"},
        "definition": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["metrics", "operation", "target_period", "comparison_period", "scope", "basis", "needs_narrative", "definition", "confidence"],
}

PLANNER_PROMPT = """You are the query planner for a financial filing analysis engine.
Convert the user's question into a retrieval/calculation plan. Do not answer the question.
Use the financial concept vocabulary supplied by the application.
Rules:
- If the wording says adjusted EBITDA, select adjusted_ebitda and NOT ebitda.
- Distinguish reported metrics from derived metrics.
- Treat why/driver/cause/impact/trend/risk/quality/strategy questions as analytical.
- Resolve explicit years into target_period/comparison_period.
- Default scope to unknown; never invent consolidated/standalone status.
Return exactly one JSON object matching the schema."""

_METRIC_TERMS = {
    "adjusted_ebitda": ("adjusted ebitda",),
    "enterprise_value": ("enterprise value", "ev"),
    "total_debt": ("total debt", "debt", "borrowings"),
    "cash_and_equivalents": ("cash and cash equivalents", "cash balance", "cash"),
    "revenue": ("revenue", "sales", "turnover"),
    "ebitda": ("ebitda",),
    "ebit": ("ebit", "operating profit", "operating income"),
    "depreciation": ("depreciation", "amortisation", "amortization"),
    "pat": ("profit after tax", "net profit", "net income", "pat"),
    "pbt": ("profit before tax", "pbt"),
    "finance_costs": ("finance cost", "finance costs", "interest expense"),
    "total_assets": ("total assets",),
    "total_equity": ("total equity", "shareholders' equity"),
    "cfo": ("operating cash flow", "cash flow from operating activities", "cfo"),
    "capex": ("capital expenditure", "capex", "purchase of property"),
    "total_expenses": ("total expenses", "total expense", "expenditure", "costs"),
}
_ANALYTICAL_HINTS = {"why", "explain", "reason", "driver", "drivers", "cause", "causes", "impact", "trend", "quality", "risk", "outlook", "strategy", "what drove"}
YEAR_RE = re.compile(r"(?:FY\s*)?(?:19|20)\d{2}(?:[-/–]\d{2})?", re.I)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _simple_value_question(question: str) -> bool:
    return not any(hint in question.lower() for hint in _ANALYTICAL_HINTS)


def _normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(plan)
    metrics = list(dict.fromkeys(str(m) for m in result.get("metrics") or []))
    if "adjusted_ebitda" in metrics and "ebitda" in metrics:
        metrics.remove("ebitda")
    result["metrics"] = metrics
    return result


def _heuristic_plan(question: str) -> Dict[str, Any]:
    lowered = question.lower()
    metrics: List[str] = []
    for metric, aliases in _METRIC_TERMS.items():
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lowered) for alias in aliases):
            metrics.append(metric)
    if "adjusted ebitda" in lowered:
        metrics = [m for m in metrics if m != "ebitda"]
        if "adjusted_ebitda" not in metrics:
            metrics.insert(0, "adjusted_ebitda")
    if not metrics and "expense" in lowered:
        metrics = ["total_expenses"]
    if any(x in lowered for x in ("percentage increase", "percent increase", "% increase", "% growth", "growth rate", "percentage change")):
        operation = "yoy_percent"
    elif any(x in lowered for x in ("increase", "decrease", "change", "growth", "decline", "yoy", "year over year", "versus")):
        operation = "yoy_change"
    elif any(x in lowered for x in ("sum of", "plus", "combined", "add ")):
        operation = "sum"
    elif any(x in lowered for x in ("difference between", "minus", "subtract")):
        operation = "difference"
    elif not _simple_value_question(question):
        operation = "explain"
    else:
        operation = "value"
    years = YEAR_RE.findall(question)
    return _normalize_plan({
        "metrics": metrics,
        "operation": operation,
        "target_period": years[0] if years else None,
        "comparison_period": years[1] if len(years) > 1 else None,
        "scope": "consolidated" if "consolidated" in lowered else ("standalone" if "standalone" in lowered else "unknown"),
        "basis": "reported" if operation == "value" else "unknown",
        "needs_narrative": operation == "explain",
        "definition": "user-requested financial metric",
        "confidence": "medium" if metrics else "low",
    })


def _valid_plan(plan: Dict[str, Any]) -> bool:
    required = {"metrics", "operation", "target_period", "comparison_period", "scope", "basis", "needs_narrative", "definition", "confidence"}
    return isinstance(plan, dict) and required.issubset(plan) and isinstance(plan.get("metrics"), list)


def plan_question(question: str) -> Tuple[Dict[str, Any], bool, str | None]:
    heuristic = _heuristic_plan(question)
    try:
        result, raw = chat_json_with_trace(PLANNER_PROMPT, json.dumps({"question": question, "heuristic_plan": heuristic}, ensure_ascii=False), model=PLANNER_MODEL, think=False, num_ctx=6144, num_predict=320, format_schema=PLAN_SCHEMA)
        result = _normalize_plan(result)
        if _valid_plan(result):
            return result, True, raw
    except RuntimeError:
        pass
    return heuristic, False, None


def _rank_facts(question: str, facts: List[Dict[str, Any]], plan: Dict[str, Any] | None = None, limit: int = 40) -> List[Dict[str, Any]]:
    plan = plan or _heuristic_plan(question)
    query_tokens = _tokens(question)
    metrics = set(plan.get("metrics") or [])
    target_period = plan.get("target_period")
    scope = plan.get("scope")
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for fact in facts:
        score = float(len(query_tokens & _tokens(fact_text(fact))))
        if fact.get("metric") in metrics:
            score += 6.0
        if fact.get("validated"):
            score += 1.5
        if float(fact.get("statement_confidence", 0) or 0) >= 0.9:
            score += 1.0
        if target_period and str(fact.get("period")) == str(target_period):
            score += 4.0
        if scope and scope != "unknown" and fact.get("scope") == scope:
            score += 3.0
        if "total_debt" in metrics and fact.get("statement") == "balance_sheet":
            score += 1.5
        if "total_debt" in metrics and fact.get("is_flow_candidate"):
            score -= 6.0
        if plan.get("basis") == "reported" and fact.get("status") != "reported":
            score -= 3.0
        if score > 0:
            scored.append((score, fact))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("page") or 10**9, pair[1].get("row_index") or 10**9))
    return [fact for _, fact in scored[:limit]]


def _chunk_pages(data: Dict[str, Any], chunk_size: int = 1400, overlap: int = 180) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page in data.get("document", {}).get("pages") or []:
        page_number = page.get("page_number_human") or page.get("page_number")
        text = str(page.get("raw_text") or "").strip()
        if not text:
            continue
        start = 0
        part = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            fragment = text[start:end].strip()
            if fragment:
                chunks.append({"chunk_id": f"p{page_number}-c{part}", "page": page_number, "text": fragment})
            if end >= len(text):
                break
            start = max(0, end - overlap)
            part += 1
    return chunks


def _fingerprint(items: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item.get("fact_id") or item.get("chunk_id")).encode())
        digest.update(str(item.get("text") or fact_text(item)).encode())
        digest.update(str(item.get("value")).encode())
        digest.update(str(item.get("period")).encode())
    return digest.hexdigest()


def _embed_cached(items: List[Dict[str, Any]], out_dir: str, filename: str) -> List[Dict[str, Any]]:
    if not items:
        return []
    path = Path(out_dir) / filename
    fp = _fingerprint(items)
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            if cache.get("fingerprint") == fp and cache.get("model") == EMBEDDING_MODEL:
                cached = cache.get("items", [])
                if isinstance(cached, list) and len(cached) == len(items):
                    return cached
        except (OSError, ValueError, TypeError):
            pass
    vectors = embed_texts([str(x.get("text") or fact_text(x)) for x in items], EMBEDDING_MODEL)
    enriched = [{**item, "embedding": vector} for item, vector in zip(items, vectors)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": fp, "model": EMBEDDING_MODEL, "items": enriched}, ensure_ascii=False), encoding="utf-8")
    return enriched


def _without_embedding(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in item.items() if k != "embedding"}


def retrieve(question: str, data: Dict[str, Any], *, out_dir: str = "output", plan: Dict[str, Any] | None = None) -> Dict[str, Any]:
    resolved_plan = _normalize_plan(plan or _heuristic_plan(question))
    planner_used = False
    planner_raw = None
    if plan is None:
        resolved_plan, planner_used, planner_raw = plan_question(question)
    store = build_fact_store(data)
    facts = store.get("facts", [])
    lexical = _rank_facts(question, facts, resolved_plan)
    semantic_ranked: List[Dict[str, Any]] = []
    warnings: List[str] = []
    query_embedding: List[float] = []
    try:
        embedded = _embed_cached([{**f, "text": fact_text(f)} for f in facts], out_dir, "qwen_fact_embedding_cache.json")
        if embedded:
            query_embedding = embed_texts([question], EMBEDDING_MODEL)[0]
            semantic_ranked = sorted(embedded, key=lambda f: _cosine(query_embedding, f.get("embedding", [])), reverse=True)
            semantic_scores = {f.get("fact_id"): _cosine(query_embedding, f.get("embedding", [])) for f in embedded}
            lexical_rank = {f.get("fact_id"): i for i, f in enumerate(lexical)}
            merged: List[Tuple[float, Dict[str, Any]]] = []
            for fact in facts:
                fid = fact.get("fact_id")
                lexical_component = max(0.0, 1.0 - lexical_rank.get(fid, 80) / 80.0)
                semantic_component = max(0.0, semantic_scores.get(fid, 0.0))
                structured_component = 0.75 if fid in lexical_rank else 0.0
                merged.append((lexical_component + semantic_component + structured_component, fact))
            merged.sort(key=lambda pair: (-pair[0], pair[1].get("page") or 10**9, pair[1].get("row_index") or 10**9))
            ranked_facts = [fact for _, fact in merged[:40]]
        else:
            ranked_facts = lexical
    except RuntimeError as exc:
        warnings.append(f"Embedding retrieval unavailable; using structured ranking only: {exc}")
        ranked_facts = lexical

    selected_facts = ranked_facts[:16]
    selected_chunks: List[Dict[str, Any]] = []
    candidate_chunks: List[Dict[str, Any]] = []
    if resolved_plan.get("needs_narrative") or not selected_facts or resolved_plan.get("operation") == "explain":
        chunks = _chunk_pages(data)
        try:
            embedded_chunks = _embed_cached(chunks, out_dir, "qwen_narrative_embedding_cache.json")
            chunk_ranked = sorted(embedded_chunks, key=lambda c: _cosine(query_embedding, c.get("embedding", [])), reverse=True) if query_embedding else embedded_chunks[:8]
            candidate_chunks = [_without_embedding(c) for c in chunk_ranked[:16]]
            selected_chunks = candidate_chunks[:8]
        except RuntimeError as exc:
            warnings.append(f"Narrative embeddings unavailable: {exc}")
            candidate_chunks = chunks[:12]
            selected_chunks = candidate_chunks[:6]

    return {
        "model": PLANNER_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "mode": "hybrid",
        "plan": resolved_plan,
        "planner_used": planner_used,
        "planner_raw_output": planner_raw,
        "selected_facts": selected_facts,
        "selected_chunks": selected_chunks,
        "candidate_facts": ranked_facts[:40],
        "semantic_fact_candidates": [_without_embedding(f) for f in semantic_ranked[:24]],
        "candidate_chunks": candidate_chunks,
        "warnings": warnings,
        "fallback": bool(warnings),
    }
