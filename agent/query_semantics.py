from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from extractor.financial_resolver import resolve_metric, resolve_raw_text


def normalize_question(question: str) -> str:
    text = str(question or "").strip()
    replacements = {
        "operational income": "operating income",
        "operational profit": "operating profit",
        "income from operations": "operating income",
        "operating earnings": "operating income",
        "total operating expenses": "total expenses",
        "total operating expense": "total expenses",
        "total operating expenditure": "total expenses",
        "total operating expenditures": "total expenses",
        "expenses": "total expenses",
        "expense": "total expenses",
    }
    for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
        text = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, text, flags=re.I)
    return text


def _select_pair(candidate: Dict[str, Any]) -> Optional[Tuple[float, float, Optional[str], Optional[str]]]:
    values = candidate.get("values") or []
    periods = candidate.get("periods") or []
    if len(values) < 2:
        return None
    return (
        float(values[0]), float(values[1]),
        periods[0] if len(periods) > 0 else None,
        periods[1] if len(periods) > 1 else None,
    )


def _pair_for_metric(metric: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = resolve_metric(metric, data) or resolve_raw_text(metric, data)
    for candidate in candidates:
        pair = _select_pair(candidate)
        if pair:
            return {"candidate": candidate, "pair": pair}
    return None


def _derive_ebitda_pair(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    explicit = _pair_for_metric("ebitda", data)
    if explicit:
        latest, prior, latest_period, prior_period = explicit["pair"]
        return {"latest": latest, "prior": prior, "latest_period": latest_period, "prior_period": prior_period, "source": explicit["candidate"], "inputs": [], "formula": None}

    ebit = _pair_for_metric("ebit", data)
    dep = _pair_for_metric("depreciation", data)
    if ebit and dep:
        e_latest, e_prior, e_lp, e_pp = ebit["pair"]
        d_latest, d_prior, d_lp, d_pp = dep["pair"]
        if (e_lp and d_lp and e_lp != d_lp) or (e_pp and d_pp and e_pp != d_pp):
            return None
        return {
            "latest": e_latest + d_latest, "prior": e_prior + d_prior,
            "latest_period": e_lp or d_lp, "prior_period": e_pp or d_pp,
            "source": {"items": [ebit["candidate"], dep["candidate"]]},
            "inputs": [
                {"name": "EBIT", "value": e_latest, "prior_value": e_prior, "page": ebit["candidate"].get("page")},
                {"name": "Depreciation & amortisation", "value": d_latest, "prior_value": d_prior, "page": dep["candidate"].get("page")},
            ],
            "formula": "EBIT + depreciation & amortisation",
        }

    pbt = _pair_for_metric("pbt", data)
    finance = _pair_for_metric("finance_costs", data)
    if pbt and finance and dep:
        p_latest, p_prior, p_lp, p_pp = pbt["pair"]
        f_latest, f_prior, f_lp, f_pp = finance["pair"]
        d_latest, d_prior, d_lp, d_pp = dep["pair"]
        if len({p for p in (p_lp, f_lp, d_lp) if p}) > 1 or len({p for p in (p_pp, f_pp, d_pp) if p}) > 1:
            return None
        return {
            "latest": p_latest + f_latest + d_latest, "prior": p_prior + f_prior + d_prior,
            "latest_period": p_lp or f_lp or d_lp, "prior_period": p_pp or f_pp or d_pp,
            "source": {"items": [pbt["candidate"], finance["candidate"], dep["candidate"]]},
            "inputs": [
                {"name": "PBT", "value": p_latest, "prior_value": p_prior, "page": pbt["candidate"].get("page")},
                {"name": "Finance costs", "value": f_latest, "prior_value": f_prior, "page": finance["candidate"].get("page")},
                {"name": "Depreciation & amortisation", "value": d_latest, "prior_value": d_prior, "page": dep["candidate"].get("page")},
            ],
            "formula": "PBT + finance costs + depreciation & amortisation",
        }
    return None


def compute_ebitda_change(question: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pair = _derive_ebitda_pair(data)
    if not pair or pair["prior"] == 0:
        return None
    delta = pair["latest"] - pair["prior"]
    percent = delta / pair["prior"] * 100.0
    lowered = question.lower()
    percentage_request = any(term in lowered for term in ("% increase", "% decrease", "percentage", "percent", "growth rate", "% growth"))
    return {
        "metric": "ebitda", "status": "derived",
        "answer": round(percent, 2) if percentage_request else round(delta, 2),
        "latest_value": round(pair["latest"], 2), "prior_value": round(pair["prior"], 2),
        "change": round(delta, 2), "percent_change": round(percent, 2),
        "latest_period": pair["latest_period"], "prior_period": pair["prior_period"],
        "period": f"{pair['latest_period']} vs {pair['prior_period']}" if pair["latest_period"] and pair["prior_period"] else None,
        "formula": "(latest − prior) / prior × 100" if percentage_request else "latest − prior",
        "source": pair["source"], "inputs": pair["inputs"] or [], "confidence": "high",
    }


def expense_candidates(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    aliases = (
        "total expenses", "total expense", "total expenditure", "total expenditures",
        "total costs", "total operating expenses", "total operating expenditure",
    )
    results: List[Dict[str, Any]] = []
    year_re = re.compile(r"(?:FY\s*)?20\d{2}(?:[-/–]\d{2})?", re.I)
    for statement_type, bucket in (data.get("statement_tables") or {}).items():
        if statement_type != "income_statement":
            continue
        for table in bucket.get("tables", []):
            rows = table.get("table") or []
            periods: List[str] = []
            for row in rows[:8]:
                periods = year_re.findall(" ".join(map(str, row)))
                if periods:
                    break
            for row in rows:
                if not isinstance(row, list) or not row:
                    continue
                label = " ".join(str(cell or "").strip().lower() for cell in row if cell is not None).strip()
                matched = next((alias for alias in aliases if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", label)), None)
                if not matched:
                    continue
                values: List[float] = []
                for cell in row[1:]:
                    text = str(cell or "").strip()
                    negative = "(" in text and ")" in text
                    cleaned = re.sub(r"[^0-9.\-]", "", text.replace("−", "-"))
                    if not cleaned or cleaned in {"-", "."}:
                        continue
                    try:
                        value = float(cleaned)
                    except ValueError:
                        continue
                    values.append(-abs(value) if negative else value)
                if values:
                    results.append({
                        "metric": "total_expenses", "matched_alias": matched, "values": values,
                        "periods": periods, "page": table.get("page_number_human"),
                        "statement": statement_type, "table_title": table.get("table_title"),
                        "source": table.get("source"), "score": float(table.get("score", 0) or 0),
                        "validated": bool(table.get("validated")), "assignment": table.get("statement_assignment"),
                    })
    return results
