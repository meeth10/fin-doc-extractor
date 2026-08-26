"""Package-level compatibility shims for the financial extraction engine."""

# Keep the resolver implementation stable while enforcing the public shape of
# EBITDA growth results expected by the calculation/UI layers.
from . import financial_resolver as _financial_resolver

_original_compute_ebitda_change = _financial_resolver.compute_ebitda_change


def _compute_ebitda_change_with_schema(question, data):
    result = _original_compute_ebitda_change(question, data)
    if result is None:
        return None

    series = result.get("series") or {}
    periods = series.get("periods") or []
    values = series.get("values") or []

    if len(values) >= 2:
        latest = float(values[0])
        prior = float(values[1])
        delta = latest - prior
        pct = (delta / prior * 100.0) if prior != 0 else None

        result.setdefault("latest_value", latest)
        result.setdefault("prior_value", prior)
        result.setdefault("change", round(delta, 2))
        if pct is not None:
            result.setdefault("percent_change", round(pct, 2))
        if periods:
            result.setdefault("latest_period", periods[0])
        if len(periods) > 1:
            result.setdefault("prior_period", periods[1])

    return result


_financial_resolver.compute_ebitda_change = _compute_ebitda_change_with_schema
