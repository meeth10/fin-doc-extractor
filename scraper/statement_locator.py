from __future__ import annotations

from itertools import product

from .models import StatementRegion

TYPES = ("balance_sheet", "income_statement", "cash_flow")


def _peaks(pages: list[dict], threshold: float = 0.45) -> dict[str, list[tuple[int, float]]]:
    result = {kind: [] for kind in TYPES}
    for page in pages:
        scores = page.get("statement_scores", {})
        number = int(page["number"])
        for kind in TYPES:
            score = float(scores.get(kind, 0.0) or 0.0)
            if score >= threshold:
                result[kind].append((number, score))
    return result


def locate_statements(pages: list[dict], max_span: int = 8) -> list[StatementRegion]:
    """Find the strongest nearby BS/IS/CF cluster, then absorb likely continuation pages."""
    peaks = _peaks(pages)
    combos: list[tuple[float, dict[str, tuple[int, float]]]] = []
    for bs, inc, cf in product(peaks["balance_sheet"], peaks["income_statement"], peaks["cash_flow"]):
        nums = [bs[0], inc[0], cf[0]]
        if len(set(nums)) < 3:
            continue
        span = max(nums) - min(nums)
        if span > max_span:
            continue
        ordered = sorted((bs, inc, cf), key=lambda x: x[0])
        proximity = sum(1 / (1 + abs(a[0] - b[0])) for a, b in ((ordered[0], ordered[1]), (ordered[1], ordered[2])))
        score = bs[1] + inc[1] + cf[1] + proximity - 0.05 * span
        combos.append((score, {"balance_sheet": bs, "income_statement": inc, "cash_flow": cf}))

    if combos:
        _, chosen = max(combos, key=lambda x: x[0])
        starts = {kind: value[0] for kind, value in chosen.items()}
        regions: list[StatementRegion] = []
        ordered = sorted(starts.items(), key=lambda x: x[1])
        for idx, (kind, start) in enumerate(ordered):
            next_start = ordered[idx + 1][1] if idx + 1 < len(ordered) else start + 3
            end = max(start, next_start - 1)
            regions.append(StatementRegion(kind, start, end, round(chosen[kind][1], 3), list(range(start, end + 1))))
        return sorted(regions, key=lambda x: x.start_page)

    # Fallback: return isolated high-confidence candidates without pretending the block is certain.
    regions = []
    for kind in TYPES:
        if peaks[kind]:
            page, score = max(peaks[kind], key=lambda x: (x[1], -x[0]))
            regions.append(StatementRegion(kind, page, page, round(score, 3), [page]))
    return sorted(regions, key=lambda x: x.start_page)
