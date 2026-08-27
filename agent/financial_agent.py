from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from extractor.financial_schema import EvidenceRef, FinancialAnswer
from .financial_facts import build_fact_store, total_debt_candidates
from .qwen_agents import ANALYST_MODEL, PLANNER_MODEL, VERIFIER_MODEL, analyze, critique
from .qwen_retrieval import plan_question, retrieve
from .query_semantics import normalize_question


def _metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    return data.get("summary", {}).get("metadata", {}) or {}


def _year(period: Any) -> int:
    m = re.search(r"(?:19|20)\d{2}", str(period or ""))
    return int(m.group(0)) if m else -1


def _scope_ok(fact: Dict[str, Any], requested: Optional[str]) -> bool:
    return not requested or requested == "unknown" or fact.get("scope") in {requested, "unknown"}


def _candidate_facts(facts: Sequence[Dict[str, Any]], metric: str, plan: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    plan = plan or {}
    result = []
    for fact in facts:
        if fact.get("metric") != metric or fact.get("is_flow_candidate") or not _scope_ok(fact, plan.get("scope")):
            continue
        if plan.get("basis") == "reported" and fact.get("status") != "reported":
            continue
        result.append(fact)
    result.sort(key=lambda f: (
        0 if f.get("status") == "reported" else 1,
        0 if f.get("validated") else 1,
        -float(f.get("statement_confidence", 0) or 0),
        -float(f.get("score", 0) or 0),
        -_year(f.get("period")),
        f.get("page") or 10**9,
        f.get("row_index") or 10**9,
    ))
    return result


def _best(facts: Sequence[Dict[str, Any]], metric: str, period: Optional[str], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = _candidate_facts(facts, metric, plan)
    if period:
        candidates = [f for f in candidates if str(f.get("period")) == str(period)]
    return candidates[0] if candidates else None


def _latest_period(facts: Sequence[Dict[str, Any]], preferred: Optional[str] = None) -> Optional[str]:
    periods = {str(f.get("period")) for f in facts if f.get("period")}
    if preferred and preferred in periods:
        return preferred
    return max(periods, key=_year) if periods else None


def _input(name: str, fact: Dict[str, Any], value: Any = None) -> Dict[str, Any]:
    return {
        "name": name,
        "value": fact.get("value") if value is None else value,
        "period": fact.get("period"),
        "page": fact.get("page"),
        "fact_id": fact.get("fact_id"),
    }


def _items(source: Any) -> List[Dict[str, Any]]:
    if isinstance(source, dict) and isinstance(source.get("items"), list):
        return [x for x in source["items"] if isinstance(x, dict)]
    return [source] if isinstance(source, dict) else []


def _refs(items: Sequence[Dict[str, Any]]) -> List[EvidenceRef]:
    result: List[EvidenceRef] = []
    seen = set()
    for item in items:
        key = (item.get("fact_id"), item.get("page"), item.get("row_index"), item.get("column_index"), item.get("label"), item.get("value"))
        if key in seen:
            continue
        seen.add(key)
        result.append(EvidenceRef(
            page=item.get("page"), statement=item.get("statement"), table_title=item.get("table_title"),
            source=item.get("source"), fact_id=item.get("fact_id"), row_index=item.get("row_index"), column_index=item.get("column_index"),
        ))
    return result


def _total_debt(facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Resolve gross debt from a reported total or non-flow balance-sheet components."""
    bs = [f for f in facts if f.get("statement") == "balance_sheet" and f.get("status", "reported") == "reported" and not f.get("is_flow_candidate")]
    if not bs:
        return None
    period = _latest_period(bs)
    if period:
        bs = [f for f in bs if str(f.get("period")) == period]
    explicit = [f for f in bs if re.search(r"(?<!\w)(total debt|total borrowings)(?!\w)", str(f.get("label", "")).lower())]
    if explicit:
        fact = max(explicit, key=lambda f: (bool(f.get("validated")), float(f.get("statement_confidence", 0) or 0), float(f.get("score", 0) or 0)))
        return {
            "metric": "total_debt", "status": "reported", "answer": fact.get("value"), "period": fact.get("period"),
            "formula": None, "inputs": [_input(fact.get("label") or "Total debt", fact)], "source": {"items": [fact]},
            "confidence": "high", "scope": fact.get("scope"), "definition": "Reported gross debt / total borrowings.",
        }
    candidates = total_debt_candidates(bs, period=period)
    if not candidates:
        return None
    def label(f: Dict[str, Any]) -> str:
        return str(f.get("label", "")).lower()
    granular = [f for f in candidates if any(x in label(f) for x in ("term debt", "commercial paper", "notes payable", "bank loan", "senior note", "debenture", "revolving credit"))]
    working = granular or candidates
    if granular:
        generic = [f for f in working if "borrowings" in label(f) and not any(x in label(f) for x in ("current", "non-current"))]
        if generic:
            working = [f for f in working if f not in generic]
    chosen: List[Dict[str, Any]] = []
    seen = set()
    for f in working:
        key = (label(f), f.get("section_context"), f.get("value"), f.get("page"), f.get("column_index"))
        if key not in seen:
            seen.add(key)
            chosen.append(f)
    commercial = [f for f in chosen if "commercial paper" in label(f)]
    term = [f for f in chosen if "term debt" in label(f)]
    other = [f for f in chosen if f not in commercial and f not in term]
    chosen = commercial[:1] + term[:2] + other
    if not chosen:
        return None
    return {
        "metric": "total_debt", "status": "derived", "derivation_type": "reconstructed", "answer": round(sum(float(f.get("value", 0)) for f in chosen), 2),
        "period": period, "formula": "sum of non-flow balance-sheet debt components",
        "inputs": [_input(f.get("label") or "Debt component", f) for f in chosen], "source": {"items": chosen},
        "confidence": "high" if all(f.get("validated") for f in chosen) else "medium",
        "scope": next((f.get("scope") for f in chosen if f.get("scope") != "unknown"), "unknown"),
        "definition": "Gross debt reconstructed from balance-sheet debt components without cash-flow movements.",
    }


def _reported(metric: str, facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = _candidate_facts(facts, metric, plan)
    period = _latest_period(candidates, plan.get("target_period"))
    fact = _best(candidates, metric, period, plan)
    if not fact:
        return None
    return {
        "metric": metric, "status": "reported", "answer": fact.get("value"), "period": fact.get("period"),
        "formula": None, "inputs": [_input(fact.get("label") or metric, fact)], "source": {"items": [fact]},
        "confidence": "high" if fact.get("validated") else "medium", "scope": fact.get("scope"),
        "definition": "Directly reported financial-statement value.",
    }


def _change(metric: str, facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = _candidate_facts(facts, metric, plan)
    by_period: Dict[str, Dict[str, Any]] = {}
    for fact in candidates:
        p = str(fact.get("period")) if fact.get("period") else None
        if p and p not in by_period:
            by_period[p] = fact
    ordered = sorted(by_period, key=_year, reverse=True)
    if not ordered:
        return None
    target = plan.get("target_period") if plan.get("target_period") in by_period else ordered[0]
    idx = ordered.index(target)
    prior = plan.get("comparison_period") if plan.get("comparison_period") in by_period else (ordered[idx + 1] if idx + 1 < len(ordered) else None)
    if not prior:
        return None
    latest, previous = by_period[target], by_period[prior]
    delta = float(latest.get("value")) - float(previous.get("value"))
    pct = None if float(previous.get("value")) == 0 else delta / float(previous.get("value")) * 100.0
    percentage = plan.get("operation") == "yoy_percent"
    return {
        "metric": metric, "status": "derived", "answer": round(pct, 2) if percentage and pct is not None else (None if percentage else round(delta, 2)),
        "period": f"{target} vs {prior}", "formula": "(latest − prior) / prior × 100" if percentage else "latest − prior",
        "inputs": [_input("Latest period", latest), _input("Prior period", previous)], "source": {"items": [latest, previous]},
        "confidence": "high" if latest.get("validated") and previous.get("validated") else "medium",
        "scope": latest.get("scope") or previous.get("scope"), "definition": "Period-over-period change using aligned reported values.",
        "latest_value": float(latest.get("value")), "prior_value": float(previous.get("value")), "change": round(delta, 2), "percent_change": round(pct, 2) if pct is not None else None,
    }


def _ebitda(facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    explicit = _reported("ebitda", facts, {**plan, "basis": "reported"})
    if explicit:
        explicit["definition"] = "Reported EBITDA as presented in the filing."
        return explicit
    target = _latest_period(_candidate_facts(facts, "ebit", plan), plan.get("target_period"))
    ebit = _best(facts, "ebit", target, plan)
    dep = _best(facts, "depreciation", target, plan)
    if not ebit or not dep or str(ebit.get("period")) != str(dep.get("period")):
        return None
    value = float(ebit.get("value")) + abs(float(dep.get("value")))
    return {
        "metric": "ebitda", "status": "reconstructed", "answer": round(value, 2), "period": ebit.get("period"),
        "formula": "EBIT + depreciation & amortisation", "inputs": [_input("EBIT", ebit), _input("Depreciation & amortisation", dep)],
        "source": {"items": [ebit, dep]}, "confidence": "medium", "scope": ebit.get("scope"),
        "definition": "Reconstructed EBITDA from reported EBIT plus depreciation & amortisation.",
    }


def _enterprise_value(facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    period = plan.get("target_period") or _latest_period(facts)
    market = _best(facts, "market_capitalization", period, plan)
    cash = _best(facts, "cash_and_equivalents", period, plan)
    debt = _total_debt([f for f in facts if not period or str(f.get("period")) == str(period)])
    if not market or not cash or not debt:
        return None
    answer = round(float(market.get("value")) + float(debt.get("answer")) - float(cash.get("value")), 2)
    return {
        "metric": "enterprise_value", "status": "derived", "answer": answer, "period": period,
        "formula": "market capitalization + total debt − cash",
        "inputs": [_input("Market capitalization", market), *debt.get("inputs", []), _input("Cash", cash)],
        "source": {"items": [market, *_items(debt.get("source")), cash]}, "confidence": "medium", "scope": market.get("scope") or cash.get("scope"),
        "definition": "Enterprise value reconstructed as market capitalization plus gross debt less cash and cash equivalents.",
    }


def _arithmetic(metrics: List[str], operation: str, facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    results = [_reported(metric, facts, plan) for metric in metrics]
    if any(r is None for r in results) or not results:
        return None
    values = [float(r["answer"]) for r in results if r is not None]
    if operation == "sum":
        answer, formula = sum(values), " + ".join(metrics)
    elif len(values) >= 2:
        answer, formula = values[0] - values[1], " − ".join(metrics[:2])
    else:
        return None
    return {
        "metric": "arithmetic_result", "status": "derived", "answer": round(answer, 2), "period": results[0].get("period"),
        "formula": formula, "inputs": [r["inputs"][0] for r in results], "source": {"items": [i for r in results for i in _items(r["source"]) ]},
        "confidence": "high" if all(r.get("confidence") == "high" for r in results) else "medium", "scope": results[0].get("scope"),
        "definition": "Deterministic arithmetic over aligned reported facts.",
    }


def _deterministic_computation(first: Any, second: Any, third: Any, fourth: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Compute from the normalized fact store.

    Supports the current three-argument API `(data, selected_facts, plan)` and the
    previous four-argument test/extension API `(question, data, selected_facts, plan)`.
    The question argument is retained only for compatibility; the plan is authoritative.
    """
    if isinstance(first, str):
        data = second
        selected = third
        plan = fourth or {}
    else:
        data = first
        selected = second
        plan = third

    facts = build_fact_store(data).get("facts", [])
    selected_ids = {f.get("fact_id") for f in selected}
    working = list(selected) + [f for f in facts if f.get("fact_id") not in selected_ids]
    metrics = list(plan.get("metrics") or [])
    operation = plan.get("operation", "value")
    if "adjusted_ebitda" in metrics:
        if operation in {"value", "yoy_change", "yoy_percent"}:
            return _change("adjusted_ebitda", working, plan) if operation != "value" else _reported("adjusted_ebitda", working, plan)
    if "total_debt" in metrics:
        debt = _total_debt(working)
        if debt and operation == "value":
            return debt
        if operation in {"yoy_change", "yoy_percent"}:
            return _change_debt(working, plan)
    if metrics == ["ebitda"]:
        if operation in {"yoy_change", "yoy_percent"}:
            return _change_derived_ebitda(working, plan)
        return _ebitda(working, plan)
    if metrics == ["enterprise_value"]:
        return _enterprise_value(working, plan)
    if operation in {"sum", "difference"} and len(metrics) >= 2:
        return _arithmetic(metrics, operation, working, plan)
    if metrics:
        return _change(metrics[0], working, plan) if operation in {"yoy_change", "yoy_percent"} else _reported(metrics[0], working, plan)
    return None


def _change_debt(facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    periods = sorted({str(f.get("period")) for f in facts if f.get("period") and f.get("statement") == "balance_sheet"}, key=_year, reverse=True)
    if len(periods) < 2:
        return None
    target = plan.get("target_period") or periods[0]
    prior = plan.get("comparison_period") or (periods[periods.index(target) + 1] if target in periods and periods.index(target) + 1 < len(periods) else periods[1])
    a = _total_debt([f for f in facts if str(f.get("period")) == str(target)])
    b = _total_debt([f for f in facts if str(f.get("period")) == str(prior)])
    if not a or not b:
        return None
    delta = float(a["answer"]) - float(b["answer"])
    pct = None if float(b["answer"]) == 0 else delta / float(b["answer"]) * 100
    percentage = plan.get("operation") == "yoy_percent"
    return {
        "metric": "total_debt", "status": "derived", "answer": round(pct, 2) if percentage and pct is not None else (None if percentage else round(delta, 2)),
        "period": f"{target} vs {prior}", "formula": "(latest − prior) / prior × 100" if percentage else "latest − prior",
        "inputs": [{"name": f"Total debt {target}", "value": a["answer"], "page": a["inputs"][0].get("page") if a.get("inputs") else None}, {"name": f"Total debt {prior}", "value": b["answer"], "page": b["inputs"][0].get("page") if b.get("inputs") else None}],
        "source": {"items": _items(a.get("source")) + _items(b.get("source"))}, "confidence": "high" if a.get("confidence") == b.get("confidence") == "high" else "medium", "scope": a.get("scope") or b.get("scope"),
        "definition": "Period-over-period movement in reconstructed gross debt.",
    }


def _change_derived_ebitda(facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    periods = sorted({str(f.get("period")) for f in facts if f.get("period") and f.get("statement") in {"income_statement", None}}, key=_year, reverse=True)
    if len(periods) < 2:
        return None
    target = plan.get("target_period") or periods[0]
    idx = periods.index(target) if target in periods else 0
    prior = plan.get("comparison_period") or (periods[idx + 1] if idx + 1 < len(periods) else None)
    if not prior:
        return None
    a = _ebitda(facts, {**plan, "target_period": target, "operation": "value"})
    b = _ebitda(facts, {**plan, "target_period": prior, "operation": "value"})
    if not a or not b:
        return None
    delta = float(a["answer"]) - float(b["answer"])
    pct = None if float(b["answer"]) == 0 else delta / float(b["answer"]) * 100
    percentage = plan.get("operation") == "yoy_percent"
    return {
        "metric": "ebitda", "status": "derived", "answer": round(pct, 2) if percentage and pct is not None else (None if percentage else round(delta, 2)),
        "period": f"{target} vs {prior}", "formula": "(latest − prior) / prior × 100" if percentage else "latest − prior",
        "inputs": [{"name": f"EBITDA {target}", "value": a["answer"], "page": a["inputs"][0].get("page") if a.get("inputs") else None}, {"name": f"EBITDA {prior}", "value": b["answer"], "page": b["inputs"][0].get("page") if b.get("inputs") else None}],
        "source": {"items": _items(a.get("source")) + _items(b.get("source"))}, "confidence": "medium", "scope": a.get("scope") or b.get("scope"),
        "definition": "Period-over-period movement in reported or reconstructed EBITDA.",
    }


def _answer_from_computation(computation: Dict[str, Any], data: Dict[str, Any], verification: Optional[Dict[str, Any]]) -> FinancialAnswer:
    meta = _metadata(data)
    return FinancialAnswer(
        metric=computation.get("metric") or "unknown", answer=computation.get("answer"), period=computation.get("period"),
        currency=meta.get("currency"), unit=meta.get("unit") or meta.get("currency_unit"), status=computation.get("status", "derived"),
        confidence=computation.get("confidence", "medium"), formula=computation.get("formula"), inputs=computation.get("inputs") or [],
        sources=_refs(_items(computation.get("source"))), explanation=None, scope=computation.get("scope") or meta.get("standalone_or_consolidated"),
        definition=computation.get("definition"), warnings=[], verification=verification,
    )


def answer_question(question: str, data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_question(question)
    plan, planner_used, planner_raw = plan_question(normalized)
    retrieval = retrieve(normalized, data, plan=plan)
    computation = _deterministic_computation(data, retrieval.get("selected_facts", []), plan)
    analysis = None
    verification = None
    warnings = list(retrieval.get("warnings") or [])
    llm_used = bool(planner_used)

    if computation is None or plan.get("needs_narrative") or plan.get("operation") == "explain":
        try:
            analysis = analyze(normalized, retrieval, computation)
            llm_used = True
        except RuntimeError as exc:
            warnings.append(f"Analyst unavailable: {exc}")

    draft = analysis or computation
    if draft:
        try:
            verification = critique(normalized, retrieval, computation, draft)
            llm_used = True
        except RuntimeError as exc:
            warnings.append(f"Verifier unavailable: {exc}")

    if verification and not verification.get("approved", False):
        issues = list(verification.get("issues") or []) + list(verification.get("required_changes") or [])
        try:
            retry_retrieval = {**retrieval, "warnings": warnings + [f"Verifier: {x}" for x in issues]}
            analysis = analyze(normalized, retry_retrieval, computation)
            verification = critique(normalized, retry_retrieval, computation, analysis)
            llm_used = True
        except RuntimeError as exc:
            warnings.append(f"Correction pass unavailable: {exc}")

    if computation is not None:
        answer = _answer_from_computation(computation, data, verification)
        if verification and not verification.get("approved", False):
            answer.explanation = "The deterministic calculation is shown, but independent verification did not approve the final claim."
        elif analysis:
            answer.explanation = analysis.get("answer_text") or analysis.get("explanation") or ""
        else:
            answer.explanation = "Reported from the cited financial statement." if computation.get("status") == "reported" else f"Calculated using {computation.get('formula')}."
    elif analysis:
        answer = FinancialAnswer(
            metric=analysis.get("metric") or ((plan.get("metrics") or ["unknown"])[0]),
            answer=analysis.get("answer"), period=analysis.get("period"), currency=analysis.get("currency") or _metadata(data).get("currency"),
            unit=analysis.get("unit") or _metadata(data).get("unit") or _metadata(data).get("currency_unit"), status=analysis.get("status", "ambiguous"),
            confidence=analysis.get("confidence", "low"), formula=analysis.get("formula"), inputs=analysis.get("inputs") or [],
            sources=_refs(retrieval.get("selected_facts", [])), explanation=analysis.get("answer_text") or analysis.get("explanation"),
            scope=analysis.get("scope") or plan.get("scope"), definition=plan.get("definition"), warnings=warnings, verification=verification,
        )
    else:
        answer = FinancialAnswer(
            metric=((plan.get("metrics") or ["unknown"])[0]), answer=None, period=plan.get("target_period"),
            currency=_metadata(data).get("currency"), unit=_metadata(data).get("unit") or _metadata(data).get("currency_unit"),
            status="ambiguous", confidence="low", formula=None, inputs=[], sources=_refs(retrieval.get("selected_facts", [])),
            explanation="The available evidence was insufficient to produce a defensible answer.", scope=plan.get("scope"),
            definition=plan.get("definition"), warnings=warnings, verification=verification,
        )

    if verification and not verification.get("approved", False):
        answer.confidence = "low"
        answer.warnings.append("Final model verification did not approve the answer; treat it as provisional.")
    return {
        **answer.as_dict(), "llm_used": llm_used,
        "models": {"embedding": retrieval.get("embedding_model"), "planner": PLANNER_MODEL, "analyst": ANALYST_MODEL, "verifier": VERIFIER_MODEL},
        "normalized_question": normalized, "plan": plan, "retrieval": retrieval, "deterministic_computation": computation,
        "analysis": analysis, "controller": verification, "planner_raw_output": planner_raw,
    }
