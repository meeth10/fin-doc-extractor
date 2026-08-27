from __future__ import annotations

from typing import Any, Dict, List, Optional

from extractor.financial_schema import FinancialAnswer
from extractor.financial_resolver import build_evidence, evidence_sources, resolve_metric, resolve_raw_text
from .financial_facts import build_fact_store, total_debt_candidates
from .qwen_agents import CRITIC_MODEL, DOCUMENT_MODEL, REASONING_MODEL, analyze, critique
from .qwen_retrieval import retrieve
from .query_semantics import compute_ebitda_change, expense_candidates, normalize_question


def _metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    return data.get("summary", {}).get("metadata", {}) or {}


def _total_debt(selected_facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    facts = [f for f in selected_facts if f.get("statement") == "balance_sheet" and f.get("status", "reported") == "reported" and not f.get("is_flow_candidate", False)]
    if not facts:
        return None
    periods = [f.get("period") for f in facts if f.get("period")]
    period = periods[0] if periods else None
    if period:
        facts = [f for f in facts if f.get("period") == period]
    explicit = [f for f in facts if "total debt" in str(f.get("label", "")).lower()]
    if explicit:
        f = explicit[0]
        return {"metric": "total_debt", "status": "reported", "answer": f["value"], "period": period, "formula": None, "inputs": [{"name": f.get("label"), "value": f.get("value"), "page": f.get("page")}], "source": {"items": [f]}, "confidence": "high"}
    components = [f for f in facts if any(token in str(f.get("label", "")).lower() or token in str(f.get("section_context", "")).lower() for token in ("commercial paper", "term debt", "borrowings"))]
    unique: List[Dict[str, Any]] = []
    seen = set()
    for fact in components:
        key = (str(fact.get("label", "")).lower(), str(fact.get("section_context", "")).lower(), fact.get("value"), fact.get("page"), fact.get("period"))
        if key not in seen:
            seen.add(key); unique.append(fact)
    commercial = [f for f in unique if "commercial paper" in str(f.get("label", "")).lower()]
    term = [f for f in unique if "term debt" in str(f.get("label", "")).lower() or "term debt" in str(f.get("section_context", "")).lower()]
    if commercial and len(term) >= 2:
        chosen = commercial[:1] + term[:2]
        return {"metric": "total_debt", "status": "derived", "answer": round(sum(float(f["value"]) for f in chosen), 2), "period": period, "formula": "commercial paper + current term debt + non-current term debt", "inputs": [{"name": f.get("label"), "value": f.get("value"), "page": f.get("page")} for f in chosen], "source": {"items": chosen}, "confidence": "high"}
    return None


def _reported_result(metric: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for candidate in candidates:
        values = candidate.get("values") or []
        if not values:
            continue
        periods = candidate.get("periods") or []
        return {"metric": metric, "status": "reported", "answer": values[0], "period": periods[0] if periods else None, "formula": None, "inputs": [], "source": candidate, "confidence": "high" if candidate.get("validated") else "medium"}
    return None


def _expense_result(question: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = expense_candidates(data)
    if not candidates:
        return None
    lowered = question.lower()
    if any(term in lowered for term in ("percentage", "percent", "%")):
        intent = "yoy_percent"
    elif any(term in lowered for term in ("increase", "decrease", "change", "growth", "decline", "yoy", "year over year", "vs", "versus")):
        intent = "yoy_change"
    else:
        intent = "value"
    metric = "total_expenses"
    if intent == "value":
        return _reported_result(metric, candidates)
    from extractor.financial_resolver import compute_change
    result = compute_change(question, candidates)
    if result:
        result["metric"] = metric
    return result


def _deterministic_computation(question: str, data: Dict[str, Any], selected_facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    lowered = question.lower()
    if "total debt" in lowered or lowered.strip() == "debt":
        result = _total_debt(selected_facts)
        if result:
            return result
    if any(term in lowered for term in ("total expenses", "total expense", "total expenditure", "total costs", "expense", "expenses", "expenditure", "expenditures")):
        result = _expense_result(question, data)
        if result:
            return result
    metric_evidence = build_evidence(question, data)
    metric = metric_evidence.get("metric")
    intent = metric_evidence.get("intent")
    if metric == "ebitda" and intent in {"yoy_change", "yoy_percent"}:
        change = compute_ebitda_change(question, data)
        if change:
            return change
    if metric_evidence.get("computed"):
        return metric_evidence["computed"]
    candidates = resolve_metric(metric, data) if metric else []
    if not candidates and metric:
        candidates = resolve_raw_text(metric, data)
    return _reported_result(metric, candidates) if metric else None


def _from_computation(computation: Dict[str, Any], data: Dict[str, Any]) -> FinancialAnswer:
    metadata = _metadata(data)
    source = computation.get("source") or {}
    items = source.get("items") or ([source] if source else [])
    return FinancialAnswer(metric=computation.get("metric") or "unknown", answer=computation.get("answer"), period=computation.get("period"), currency=metadata.get("currency"), unit=metadata.get("unit") or metadata.get("currency_unit"), status=computation.get("status", "derived"), confidence=computation.get("confidence", "high"), formula=computation.get("formula"), inputs=computation.get("inputs") or [], sources=evidence_sources(items, []), explanation=None)


def answer_question(question: str, data: Dict[str, Any]) -> Dict[str, Any]:
    normalized_question = normalize_question(question)
    retrieval = retrieve(normalized_question, data)
    computation = _deterministic_computation(normalized_question, data, retrieval.get("selected_facts", []))
    if computation is not None and retrieval.get("mode") == "deterministic_fact_first":
        answer = _from_computation(computation, data)
        if computation.get("status") == "derived":
            inputs = "; ".join(f"{i.get('name')}={i.get('value')}" for i in computation.get("inputs", []))
            answer.explanation = f"Calculated deterministically using {computation.get('formula')}: {inputs}."
        else:
            answer.explanation = "Reported from the cited financial statement."
        return {**answer.as_dict(), "llm_used": False, "models": {"embedding": retrieval.get("embedding_model"), "document": DOCUMENT_MODEL, "reasoning": REASONING_MODEL, "critic": CRITIC_MODEL}, "normalized_question": normalized_question, "retrieval": retrieval, "deterministic_computation": computation}
    analysis = analyze(normalized_question, retrieval, computation)
    controller = critique(normalized_question, retrieval, computation, analysis)
    if not controller.get("approved", False) and computation is not None:
        revision = analyze(normalized_question, {**retrieval, "warnings": retrieval.get("warnings", []) + controller.get("issues", [])}, computation)
        if revision.get("answer_text"):
            analysis = revision
    if computation is not None:
        answer = _from_computation(computation, data)
        answer.explanation = analysis.get("answer_text") or ""
    else:
        answer = FinancialAnswer(metric=analysis.get("metric") or "unknown", answer=analysis.get("answer"), period=analysis.get("period"), currency=analysis.get("currency"), unit=analysis.get("unit"), status=analysis.get("status", "ambiguous"), confidence=analysis.get("confidence", "low"), formula=analysis.get("formula"), inputs=analysis.get("inputs") or [], sources=evidence_sources(retrieval.get("selected_facts", []), []), explanation=analysis.get("answer_text") or analysis.get("explanation"))
    return {**answer.as_dict(), "llm_used": True, "models": {"embedding": retrieval.get("embedding_model"), "document": DOCUMENT_MODEL, "reasoning": REASONING_MODEL, "critic": CRITIC_MODEL}, "normalized_question": normalized_question, "retrieval": retrieval, "deterministic_computation": computation, "analysis": analysis, "controller": controller}
