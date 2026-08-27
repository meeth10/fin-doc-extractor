from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from extractor.financial_schema import EvidenceRef, FinancialAnswer
from extractor.financial_resolver import resolve_metric, resolve_raw_text
from .financial_facts import build_fact_store, total_debt_candidates
from .qwen_agents import ANALYST_MODEL, PLANNER_MODEL, VERIFIER_MODEL, analyze, critique
from .qwen_retrieval import plan_question, retrieve
from .query_semantics import normalize_question


# -----------------------------
# Evidence helpers
# -----------------------------


def _metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    return data.get("summary", {}).get("metadata", {}) or {}


def _period_year(period: Any) -> int:
    match = re.search(r"(?:19|20)\d{2}", str(period or ""))
    return int(match.group(0)) if match else -1


def _latest_period(facts: Sequence[Dict[str, Any]], preferred: Optional[str] = None) -> Optional[str]:
    periods = {str(f.get("period")) for f in facts if f.get("period")}
    if preferred and preferred in periods:
        return preferred
    return max(periods, key=_period_year) if periods else None


def _scope_ok(fact: Dict[str, Any], requested: Optional[str]) -> bool:
    if not requested or requested == "unknown":
        return True
    return fact.get("scope") in {requested, "unknown"}


def _candidate_facts(facts: Sequence[Dict[str, Any]], metric: str, plan: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    plan = plan or {}
    candidates = [
        f for f in facts
        if f.get("metric") == metric and _scope_ok(f, plan.get("scope")) and not f.get("is_flow_candidate", False)
    ]
    candidates.sort(
        key=lambda f: (
            0 if f.get("validated") else 1,
            -float(f.get("statement_confidence", 0) or 0),
            -float(f.get("score", 0) or 0),
            -_period_year(f.get("period")),
            f.get("page") or 10**9,
        )
    )
    return candidates


def _best_fact(facts: Sequence[Dict[str, Any]], metric: str, period: Optional[str], plan: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    candidates = [f for f in _candidate_facts(facts, metric, plan) if not period or str(f.get("period")) == str(period)]
    return candidates[0] if candidates else None


def _evidence_items(source: Any) -> List[Dict[str, Any]]:
    if not source:
        return []
    if isinstance(source, dict) and isinstance(source.get("items"), list):
        return [x for x in source["items"] if isinstance(x, dict)]
    if isinstance(source, dict):
        return [source]
    return []


def _evidence_refs(items: Sequence[Dict[str, Any]]) -> List[EvidenceRef]:
    refs: List[EvidenceRef] = []
    seen = set()
    for item in items:
        key = (item.get("fact_id"), item.get("page"), item.get("row_index"), item.get("column_index"), item.get("label"), item.get("value"))
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            EvidenceRef(
                page=item.get("page"),
                statement=item.get("statement"),
                table_title=item.get("table_title"),
                source=item.get("source"),
                fact_id=item.get("fact_id"),
                row_index=item.get("row_index"),
                column_index=item.get("column_index"),
            )
        )
    return refs


def _input(name: str, fact: Dict[str, Any], value: Any = None) -> Dict[str, Any]:
    return {
        "name": name,
        "value": fact.get("value") if value is None else value,
        "period": fact.get("period"),
        "page": fact.get("page"),
        "fact_id": fact.get("fact_id"),
    }


# -----------------------------
# Deterministic accounting layer
# -----------------------------


def _total_debt(selected_facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Resolve gross debt from an explicit total or non-flow balance-sheet components.

    This intentionally does not treat cash-flow issuance/repayment rows as debt.
    Component selection prefers the most specific rows and avoids adding an aggregate
    borrowings row on top of current/non-current components.
    """
    facts = [
        f for f in selected_facts
        if f.get("statement") == "balance_sheet"
        and f.get("status", "reported") == "reported"
        and not f.get("is_flow_candidate", False)
    ]
    if not facts:
        return None
    requested_periods = [str(f.get("period")) for f in facts if f.get("period")]
    period = max(set(requested_periods), key=_period_year) if requested_periods else None
    if period:
        facts = [f for f in facts if str(f.get("period")) == period]

    explicit = [
        f for f in facts
        if re.search(r"(?<!\w)(total debt|total borrowings)(?!\w)", str(f.get("label", "")).lower())
    ]
    if explicit:
        f = max(explicit, key=lambda x: (bool(x.get("validated")), float(x.get("statement_confidence", 0) or 0), float(x.get("score", 0) or 0)))
        return {
            "metric": "total_debt", "status": "reported", "answer": f.get("value"), "period": f.get("period"),
            "formula": None, "inputs": [_input(f.get("label") or "Total debt", f)],
            "source": {"items": [f]}, "confidence": "high", "scope": f.get("scope"),
            "definition": "Reported gross debt / total borrowings as stated in the balance sheet.",
        }

    candidates = total_debt_candidates(facts, period=period)
    if not candidates:
        return None

    def label(f: Dict[str, Any]) -> str:
        return str(f.get("label", "")).lower()

    # Remove obvious subtotal rows when more granular current/non-current debt exists.
    granular = [
        f for f in candidates
        if any(x in label(f) for x in ("term debt", "commercial paper", "notes payable", "bank loan", "senior note", "debenture", "revolving credit"))
    ]
    generic = [f for f in candidates if "borrowings" in label(f) and not any(x in label(f) for x in ("current", "non-current"))]
    working = granular or candidates
    if granular and generic:
        # A single generic borrowings line is assumed to be an aggregate, so do not double count it.
        working = granular

    chosen: List[Dict[str, Any]] = []
    seen = set()
    for fact in working:
        key = (label(fact), str(fact.get("section_context", "")).lower(), fact.get("value"), fact.get("page"), fact.get("column_index"))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(fact)

    # Prefer one commercial-paper row and retain distinct term-debt sections (e.g. current/non-current).
    commercial = [f for f in chosen if "commercial paper" in label(f)]
    term = [f for f in chosen if "term debt" in label(f)]
    other = [f for f in chosen if f not in commercial and f not in term]
    chosen = commercial[:1] + term[:2] + other
    if not chosen:
        return None

    answer = round(sum(float(f.get("value", 0)) for f in chosen), 2)
    return {
        "metric": "total_debt", "status": "reconstructed", "answer": answer, "period": period,
        "formula": "sum of non-flow balance-sheet debt components", "inputs": [_input(f.get("label") or "Debt component", f) for f in chosen],
        "source": {"items": chosen}, "confidence": "high" if all(f.get("validated") for f in chosen) else "medium",
        "scope": next((f.get("scope") for f in chosen if f.get("scope") != "unknown"), "unknown"),
        "definition": "Gross debt reconstructed from balance-sheet debt components; cash and lease liabilities are not added unless explicitly presented as debt components.",
    }


def _reported(metric: str, facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = _candidate_facts(facts, metric, plan)
    period = _latest_period(candidates, plan.get("target_period"))
    fact = _best_fact(candidates, metric, period, plan)
    if not fact:
        return None
    return {
        "metric": metric,
        "status": "reported",
        "answer": fact.get("value"),
        "period": fact.get("period"),
        "formula": None,
        "inputs": [_input(fact.get("label") or metric, fact)],
        "source": {"items": [fact]},
        "confidence": "high" if fact.get("validated") else "medium",
        "scope": fact.get("scope"),
        "definition": "Directly reported financial-statement value.",
    }


def _pair(metric: str, facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    candidates = _candidate_facts(facts, metric, plan)
    by_period: Dict[str, Dict[str, Any]] = {}
    for fact in candidates:
        period = str(fact.get("period")) if fact.get("period") else None
        if not period or period in by_period:
            continue
        by_period[period] = fact
    periods = sorted(by_period, key=_period_year, reverse=True)
    if plan.get("target_period") in by_period:
        latest_period = plan["target_period"]
    elif periods:
        latest_period = periods[0]
    else:
        return None
    if plan.get("comparison_period") in by_period:
        prior_period = plan["comparison_period"]
    else:
        idx = periods.index(latest_period) if latest_period in periods else 0
        prior_period = periods[idx + 1] if idx + 1 < len(periods) else None
    if not prior_period or latest_period not in by_period:
        return None
    return by_period[latest_period], by_period[prior_period]


def _change(metric: str, facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pair = _pair(metric, facts, plan)
    if not pair:
        return None
    latest, prior = pair
    if latest.get("scope") not in {"unknown", prior.get("scope"), None} and prior.get("scope") not in {"unknown", None}:
        return None
    latest_value = float(latest.get("value"))
    prior_value = float(prior.get("value"))
    delta = latest_value - prior_value
    percent = None if prior_value == 0 else delta / prior_value * 100.0
    percentage = plan.get("operation") == "yoy_percent"
    return {
        "metric": metric,
        "status": "derived",
        "answer": round(percent, 2) if percentage and percent is not None else (None if percentage else round(delta, 2)),
        "period": f"{latest.get('period')} vs {prior.get('period')}",
        "formula": "(latest − prior) / prior × 100" if percentage else "latest − prior",
        "inputs": [_input("Latest period", latest), _input("Prior period", prior)],
        "source": {"items": [latest, prior]},
        "confidence": "high" if latest.get("validated") and prior.get("validated") else "medium",
        "scope": latest.get("scope") or prior.get("scope"),
        "definition": "Period-over-period change using aligned reported values.",
        "latest_value": latest_value,
        "prior_value": prior_value,
        "change": round(delta, 2),
        "percent_change": round(percent, 2) if percent is not None else None,
    }


def _metric_series(metric: str, facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    candidates = _candidate_facts(facts, metric, plan)
    by_period: Dict[str, Dict[str, Any]] = {}
    for fact in candidates:
        period = str(fact.get("period")) if fact.get("period") else None
        if period and period not in by_period:
            by_period[period] = fact
    return [by_period[p] for p in sorted(by_period, key=_period_year, reverse=True)] if by_period else None


def _ebitda(facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    explicit = _reported("ebitda", facts, plan)
    if explicit:
        # Guard against answering an adjusted EBITDA question with a generic label.
        source_label = str(explicit["inputs"][0].get("name", "")).lower()
        if "adjusted" not in source_label or "adjusted" in str(plan.get("definition", "")).lower():
            explicit["definition"] = "Reported EBITDA as presented in the filing."
            return explicit

    ebit = _pair("ebit", facts, plan)
    dep = _pair("depreciation", facts, plan)
    if not ebit or not dep:
        # Fall back to the single requested period for value questions.
        ebit_fact = _best_fact(facts, "ebit", _latest_period(_candidate_facts(facts, "ebit", plan), plan.get("target_period")), plan)
        dep_fact = _best_fact(facts, "depreciation", _latest_period(_candidate_facts(facts, "depreciation", plan), plan.get("target_period")), plan)
        if not ebit_fact or not dep_fact or str(ebit_fact.get("period")) != str(dep_fact.get("period")):
            return None
        value = float(ebit_fact.get("value")) + abs(float(dep_fact.get("value")))
        return {
            "metric": "ebitda", "status": "reconstructed", "answer": round(value, 2), "period": ebit_fact.get("period"),
            "formula": "EBIT + depreciation & amortisation", "inputs": [_input("EBIT", ebit_fact), _input("Depreciation & amortisation", dep_fact)],
            "source": {"items": [ebit_fact, dep_fact]}, "confidence": "medium", "scope": ebit_fact.get("scope"),
            "definition": "Reconstructed EBITDA from reported EBIT plus depreciation & amortisation.",
        }

    e, d = ebit[0], dep[0]
    if str(e.get("period")) != str(d.get("period")):
        return None
    value = float(e.get("value")) + abs(float(d.get("value")))
    return {
        "metric": "ebitda", "status": "reconstructed", "answer": round(value, 2), "period": e.get("period"),
        "formula": "EBIT + depreciation & amortisation", "inputs": [_input("EBIT", e), _input("Depreciation & amortisation", d)],
        "source": {"items": [e, d]}, "confidence": "medium", "scope": e.get("scope"),
        "definition": "Reconstructed EBITDA from reported EBIT plus depreciation & amortisation.",
    }


def _enterprise_value(facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    period = plan.get("target_period") or _latest_period(facts)
    market = _best_fact(facts, "market_capitalization", period, plan)
    cash = _best_fact(facts, "cash_and_equivalents", period, plan)
    debt_result = _total_debt([f for f in facts if str(f.get("period")) == str(period)] if period else list(facts))
    if not market or not cash or not debt_result:
        return None
    answer = round(float(market.get("value")) + float(debt_result.get("answer")) - float(cash.get("value")), 2)
    debt_items = _evidence_items(debt_result.get("source"))
    return {
        "metric": "enterprise_value", "status": "derived", "answer": answer, "period": period,
        "formula": "market capitalization + total debt − cash", "inputs": [_input("Market capitalization", market), *debt_result.get("inputs", []), _input("Cash", cash)],
        "source": {"items": [market, *debt_items, cash]}, "confidence": "medium" if debt_result.get("status") != "reported" else "high",
        "scope": market.get("scope") or cash.get("scope"),
        "definition": "Enterprise value reconstructed as market capitalization plus gross debt less cash and cash equivalents.",
    }


def _arithmetic(metrics: List[str], operation: str, facts: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selected = []
    for metric in metrics:
        result = _reported(metric, facts, {**plan, "target_period": plan.get("target_period")})
        if not result:
            return None
        selected.append(result)
    values = [float(r["answer"]) for r in selected]
    if operation == "sum":
        answer = sum(values)
        formula = " + ".join(metrics)
    else:
        if len(values) < 2:
            return None
        answer = values[0] - values[1]
        formula = " − ".join(metrics[:2])
    items = [item for result in selected for item in _evidence_items(result.get("source"))]
    return {
        "metric": "arithmetic_result", "status": "derived", "answer": round(answer, 2),
        "period": selected[0].get("period"), "formula": formula,
        "inputs": [result["inputs"][0] for result in selected], "source": {"items": items},
        "confidence": "high" if all(result.get("confidence") == "high" for result in selected) else "medium",
        "scope": selected[0].get("scope"), "definition": "Deterministic arithmetic over aligned reported financial facts.",
    }


def _deterministic_computation(question: str, data: Dict[str, Any], selected_facts: List[Dict[str, Any]], plan: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    plan = plan or {}
    store = build_fact_store(data)
    facts = store.get("facts", [])
    # Use retrieval-selected facts first, but replenish from the authoritative store when a
    # multi-component calculation needs an input that ranking omitted.
    working = list(selected_facts)
    selected_ids = {f.get("fact_id") for f in working}
    for fact in facts:
        if fact.get("fact_id") not in selected_ids:
            working.append(fact)

    metrics = list(plan.get("metrics") or [])
    operation = plan.get("operation", "value")
    lowered = question.lower()
    if not metrics:
        if "total debt" in lowered or lowered.strip() == "debt":
            metrics = ["total_debt"]
        elif "enterprise value" in lowered or re.search(r"\bev\b", lowered):
            metrics = ["enterprise_value"]
        elif "ebitda" in lowered:
            metrics = ["ebitda"]

    if "total_debt" in metrics:
        debt = _total_debt(working)
        if debt and operation == "value":
            return debt
        if debt and operation in {"yoy_change", "yoy_percent"}:
            # Debt is a balance-sheet stock; compare the complete reconstructed debt at two periods.
            periods = sorted({str(f.get("period")) for f in working if f.get("period") and f.get("statement") == "balance_sheet"}, key=_period_year, reverse=True)
            target = plan.get("target_period") or (periods[0] if periods else None)
            prior = plan.get("comparison_period") or (periods[1] if len(periods) > 1 else None)
            if target and prior:
                a = _total_debt([f for f in working if str(f.get("period")) == str(target)])
                b = _total_debt([f for f in working if str(f.get("period")) == str(prior)])
                if a and b:
                    delta = float(a["answer"]) - float(b["answer"])
                    pct = None if float(b["answer"]) == 0 else delta / float(b["answer"]) * 100
                    return {
                        "metric": "total_debt", "status": "derived",
                        "answer": round(pct, 2) if operation == "yoy_percent" and pct is not None else round(delta, 2),
                        "period": f"{target} vs {prior}",
                        "formula": "(latest − prior) / prior × 100" if operation == "yoy_percent" else "latest − prior",
                        "inputs": [{"name": f"Total debt {target}", "value": a["answer"], "page": next((i.get("page") for i in a.get("inputs", []) if i.get("page") is not None), None)}, {"name": f"Total debt {prior}", "value": b["answer"], "page": next((i.get("page") for i in b.get("inputs", []) if i.get("page") is not None), None)}],
                        "source": {"items": _evidence_items(a.get("source")) + _evidence_items(b.get("source"))},
                        "confidence": "high" if a.get("confidence") == b.get("confidence") == "high" else "medium",
                        "scope": a.get("scope") or b.get("scope"),
                        "definition": "Period-over-period movement in reconstructed gross debt.",
                    }

    if metrics == ["ebitda"]:
        if operation in {"yoy_change", "yoy_percent"}:
            candidates = _metric_series("ebitda", working, plan) or []
            if len(candidates) >= 2:
                return _change("ebitda", working, plan)
            # Build EBITDA for the two periods from underlying inputs.
            period_list = sorted({str(f.get("period")) for f in working if f.get("period")}, key=_period_year, reverse=True)
            target = plan.get("target_period") or (period_list[0] if period_list else None)
            prior = plan.get("comparison_period") or (period_list[1] if len(period_list) > 1 else None)
            if target and prior:
                first = _ebitda(working, {**plan, "operation": "value", "target_period": target})
                second = _ebitda(working, {**plan, "operation": "value", "target_period": prior})
                if first and second:
                    delta = float(first["answer"]) - float(second["answer"])
                    pct = None if float(second["answer"]) == 0 else delta / float(second["answer"]) * 100
                    return {
                        "metric": "ebitda", "status": "derived", "answer": round(pct, 2) if operation == "yoy_percent" and pct is not None else round(delta, 2),
                        "period": f"{target} vs {prior}", "formula": "(latest − prior) / prior × 100" if operation == "yoy_percent" else "latest − prior",
                        "inputs": [{"name": f"EBITDA {target}", "value": first["answer"], "page": first["inputs"][0].get("page") if first.get("inputs") else None}, {"name": f"EBITDA {prior}", "value": second["answer"], "page": second["inputs"][0].get("page") if second.get("inputs") else None}],
                        "source": {"items": _evidence_items(first.get("source")) + _evidence_items(second.get("source"))}, "confidence": "medium",
                        "scope": first.get("scope") or second.get("scope"), "definition": "Period-over-period movement in reconstructed or reported EBITDA.",
                    }
        result = _ebitda(working, plan)
        if result:
            return result

    if metrics == ["enterprise_value"]:
        result = _enterprise_value(working, plan)
        if result:
            return result

    if operation in {"sum", "difference"} and len(metrics) >= 2:
        return _arithmetic(metrics, operation, working, plan)

    if metrics:
        metric = metrics[0]
        if operation in {"yoy_change", "yoy_percent"}:
            return _change(metric, working, plan)
        result = _reported(metric, working, plan)
        if result:
            return result

    return None


# -----------------------------
# Answer assembly + verification loop
# -----------------------------


def _from_computation(computation: Dict[str, Any], data: Dict[str, Any], verification: Optional[Dict[str, Any]] = None) -> FinancialAnswer:
    metadata = _metadata(data)
    items = _evidence_items(computation.get("source"))
    return FinancialAnswer(
        metric=computation.get("metric") or "unknown",
        answer=computation.get("answer"),
        period=computation.get("period"),
        currency=metadata.get("currency"),
        unit=metadata.get("unit") or metadata.get("currency_unit"),
        status=computation.get("status", "derived"),
        confidence=computation.get("confidence", "medium"),
        formula=computation.get("formula"),
        inputs=computation.get("inputs") or [],
        sources=_evidence_refs(items),
        explanation=None,
        scope=computation.get("scope") or metadata.get("standalone_or_consolidated"),
        definition=computation.get("definition"),
        warnings=[],
        verification=verification,
    )


def _should_analyze(plan: Dict[str, Any], computation: Optional[Dict[str, Any]]) -> bool:
    return computation is None or bool(plan.get("needs_narrative")) or plan.get("operation") == "explain"


def answer_question(question: str, data: Dict[str, Any]) -> Dict[str, Any]:
    normalized_question = normalize_question(question)
    plan, planner_used, planner_raw = plan_question(normalized_question)
    retrieval = retrieve(normalized_question, data, plan=plan)
    computation = _deterministic_computation(normalized_question, data, retrieval.get("selected_facts", []), plan)

    analysis: Optional[Dict[str, Any]] = None
    controller: Optional[Dict[str, Any]] = None
    llm_used = bool(planner_used)
    warnings = list(retrieval.get("warnings") or [])

    if _should_analyze(plan, computation):
        try:
            analysis = analyze(normalized_question, retrieval, computation)
            llm_used = True
        except RuntimeError as exc:
            warnings.append(f"Analyst unavailable: {exc}")

    # Always audit the final factual claim when an LLM is available. For pure deterministic
    # answers, the verifier protects against wrong period/scope/component assembly.
    draft_for_verifier = analysis or (computation or {})
    try:
        if draft_for_verifier:
            controller = critique(normalized_question, retrieval, computation, draft_for_verifier)
            llm_used = True
    except RuntimeError as exc:
        warnings.append(f"Verifier unavailable: {exc}")

    # One correction pass, followed by a second audit. Do not let a rejected answer silently pass.
    if controller and not controller.get("approved", False):
        issues = list(controller.get("issues") or []) + list(controller.get("required_changes") or [])
        correction_retrieval = {**retrieval, "warnings": warnings + [f"Verifier: {issue}" for issue in issues]}
        try:
            analysis = analyze(normalized_question, correction_retrieval, computation)
            llm_used = True
            second = critique(normalized_question, correction_retrieval, computation, analysis)
            controller = second
            llm_used = True
        except RuntimeError as exc:
            warnings.append(f"Correction pass unavailable: {exc}")

    if computation is not None:
        answer = _from_computation(computation, data, controller)
        if analysis:
            answer.explanation = analysis.get("answer_text") or analysis.get("explanation")
        elif computation.get("status") in {"reported", "derived", "reconstructed"}:
            answer.explanation = (
                f"Reported from the cited financial statement." if computation.get("status") == "reported"
                else f"Calculated deterministically using {computation.get('formula')}."
            )
    elif analysis:
        answer = FinancialAnswer(
            metric=analysis.get("metric") or (plan.get("metrics") or ["unknown"])[0],
            answer=analysis.get("answer"),
            period=analysis.get("period"),
            currency=analysis.get("currency") or _metadata(data).get("currency"),
            unit=analysis.get("unit") or _metadata(data).get("unit") or _metadata(data).get("currency_unit"),
            status=analysis.get("status", "ambiguous"),
            confidence=analysis.get("confidence", "low"),
            formula=analysis.get("formula"),
            inputs=analysis.get("inputs") or [],
            sources=_evidence_refs(retrieval.get("selected_facts", [])),
            explanation=analysis.get("answer_text") or analysis.get("explanation"),
            scope=analysis.get("scope") or plan.get("scope"),
            definition=plan.get("definition"),
            warnings=warnings,
            verification=controller,
        )
    else:
        answer = FinancialAnswer(
            metric=(plan.get("metrics") or ["unknown"])[0],
            answer=None,
            period=plan.get("target_period"),
            currency=_metadata(data).get("currency"),
            unit=_metadata(data).get("unit") or _metadata(data).get("currency_unit"),
            status="not_available" if retrieval.get("selected_facts") else "ambiguous",
            confidence="low",
            formula=None,
            inputs=[],
            sources=_evidence_refs(retrieval.get("selected_facts", [])),
            explanation="The available evidence was insufficient to produce a defensible answer.",
            scope=plan.get("scope"),
            definition=plan.get("definition"),
            warnings=warnings,
            verification=controller,
        )

    answer.warnings.extend(warnings)
    if controller and not controller.get("approved", False):
        answer.confidence = "low"
        answer.warnings.append("Final model verification did not approve the answer; treat it as provisional.")

    return {
        **answer.as_dict(),
        "llm_used": llm_used,
        "models": {
            "embedding": retrieval.get("embedding_model"),
            "planner": PLANNER_MODEL,
            "analyst": ANALYST_MODEL,
            "verifier": VERIFIER_MODEL,
        },
        "normalized_question": normalized_question,
        "plan": plan,
        "retrieval": retrieval,
        "deterministic_computation": computation,
        "analysis": analysis,
        "controller": controller,
        "planner_raw_output": planner_raw,
    }
