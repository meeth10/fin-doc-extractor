from __future__ import annotations

from typing import Any, Dict, List, Optional

from extractor.financial_schema import FinancialAnswer
from extractor.financial_resolver import build_evidence, evidence_sources, resolve_metric, resolve_raw_text
from .financial_facts import build_fact_store, total_debt_candidates
from .qwen_agents import CRITIC_MODEL, DOCUMENT_MODEL, REASONING_MODEL, analyze, critique
from .qwen_retrieval import retrieve


def _metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    return data.get("summary", {}).get("metadata", {}) or {}


def _total_debt(selected_facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    facts = [f for f in selected_facts if f.get("statement") == "balance_sheet" and f.get("status") == "reported" and not f.get("is_flow_candidate")]
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

    components = [f for f in facts if any(token in str(f.get("label", "")).lower() for token in ("commercial paper", "term debt", "borrowings"))]
    # Keep one row for each distinct reported component.
    unique = []
    seen = set()
    for f in components:
        key = (str(f.get("label", "")).lower(), f.get("value"), f.get("page"), f.get("period"))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    commercial = [f for f in unique if "commercial paper" in str(f.get("label", "")).lower()]
    term = [f for f in unique if "term debt" in str(f.get("label", "")).lower()]
    if commercial and len(term) >= 2:
        chosen = commercial + term[:2]
        return {
            "metric": "total_debt",
            "status": "derived",
            "answer": round(sum(float(f["value"]) for f in chosen), 2),
            "period": period,
            "formula": "commercial paper + current term debt + non-current term debt",
            "inputs": [{"name": f.get("label"), "value": f.get("value"), "page": f.get("page")} for f in chosen],
            "source": {"items": chosen},
            "confidence": "high",
        }
    return None


def _deterministic_computation(question: str, data: Dict[str, Any], selected_facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    lowered = question.lower()
    if "total debt" in lowered or lowered.strip() == "debt":
        result = _total_debt(selected_facts)
        if result:
            return result

    evidence = build_evidence(question, data)
    if evidence.get("computed"):
        return evidence["computed"]

    metric = evidence.get("metric")
    candidates = resolve_metric(metric, data) if metric else []
    if not candidates and metric:
        candidates = resolve_raw_text(metric, data)
    if candidates and candidates[0].get("values"):
        c = candidates[0]
        return {
            "metric": metric,
            "status": "reported",
            "answer": c["values"][0],
            "period": (c.get("periods") or [None])[0],
            "formula": None,
            "inputs": [],
            "source": c,
            "confidence": "high" if c.get("validated") else "medium",
        }
    return None


def _from_computation(computation: Dict[str, Any], data: Dict[str, Any]) -> FinancialAnswer:
    metadata = _metadata(data)
    source = computation.get("source") or {}
    items = source.get("items") or ([source] if source else [])
    return FinancialAnswer(
        metric=computation.get("metric") or "unknown",
        answer=computation.get("answer"),
        period=computation.get("period"),
        currency=metadata.get("currency"),
        unit=metadata.get("unit") or metadata.get("currency_unit"),
        status=computation.get("status", "derived"),
        confidence=computation.get("confidence", "high"),
        formula=computation.get("formula"),
        inputs=computation.get("inputs") or [],
        sources=evidence_sources(items, []),
        explanation=None,
    )


def answer_question(question: str, data: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = retrieve(question, data)
    computation = _deterministic_computation(question, data, retrieval.get("selected_facts", []))

    # Factual questions should not pay the cost of a reasoning model. The
    # deterministic answer plus source provenance is already the authoritative
    # answer; Qwen is reserved for interpretation-heavy work.
    if computation is not None and retrieval.get("mode") == "deterministic_fact_first":
        answer = _from_computation(computation, data)
        if computation.get("status") == "derived":
            explanation = "Derived from the reported balance-sheet components: " + "; ".join(
                f"{i.get('name')}={i.get('value')}" for i in computation.get("inputs", [])
            ) + "."
        else:
            explanation = "Reported from the cited financial statement."
        answer.explanation = explanation
        return {
            **answer.as_dict(),
            "llm_used": False,
            "models": {
                "embedding": retrieval.get("embedding_model"),
                "document": DOCUMENT_MODEL,
                "reasoning": REASONING_MODEL,
                "critic": CRITIC_MODEL,
            },
            "retrieval": retrieval,
            "deterministic_computation": computation,
        }

    analysis = analyze(question, retrieval, computation)
    controller = critique(question, retrieval, computation, analysis)

    if not controller.get("approved", False) and computation is not None:
        revision = analyze(
            question,
            {**retrieval, "warnings": retrieval.get("warnings", []) + controller.get("issues", [])},
            computation,
        )
        if revision.get("answer_text"):
            analysis = revision

    if computation is not None:
        answer = _from_computation(computation, data)
        answer.explanation = analysis.get("answer_text") or ""
    else:
        answer = FinancialAnswer(
            metric=analysis.get("metric") or "unknown",
            answer=analysis.get("answer"),
            period=analysis.get("period"),
            currency=analysis.get("currency"),
            unit=analysis.get("unit"),
            status=analysis.get("status", "ambiguous"),
            confidence=analysis.get("confidence", "low"),
            formula=analysis.get("formula"),
            inputs=analysis.get("inputs") or [],
            sources=evidence_sources(retrieval.get("selected_facts", []), []),
            explanation=analysis.get("answer_text") or analysis.get("explanation"),
        )

    return {
        **answer.as_dict(),
        "llm_used": True,
        "models": {"embedding": retrieval.get("embedding_model"), "document": DOCUMENT_MODEL, "reasoning": REASONING_MODEL, "critic": CRITIC_MODEL},
        "retrieval": retrieval,
        "deterministic_computation": computation,
        "analysis": analysis,
        "controller": controller,
    }
