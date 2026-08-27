from __future__ import annotations

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


def _best_anchors(peaks: dict[str, list[tuple[int, float]]], max_span: int) -> dict[str, tuple[int, float]] | None:
    """Choose one anchor per statement from the tightest high-confidence neighborhood."""
    if sum(bool(values) for values in peaks.values()) < 3:
        return None

    candidates: list[tuple[float, int, dict[str, tuple[int, float]]]] = []
    for bs in peaks["balance_sheet"]:
        for inc in peaks["income_statement"]:
            for cf in peaks["cash_flow"]:
                anchors = {
                    "balance_sheet": bs,
                    "income_statement": inc,
                    "cash_flow": cf,
                }
                pages = [page for page, _ in anchors.values()]
                if len(set(pages)) != 3:
                    continue

                span = max(pages) - min(pages)
                if span > max_span:
                    continue

                ordered_pages = sorted(pages)
                adjacency = sum(
                    1.0 / (1.0 + abs(ordered_pages[i] - ordered_pages[i - 1]))
                    for i in range(1, len(ordered_pages))
                )
                score = sum(score for _, score in anchors.values())
                score += adjacency
                score -= 0.05 * span
                candidates.append((score, span, anchors))

    if not candidates:
        return None

    _, _, chosen = max(candidates, key=lambda item: (item[0], -item[1]))
    return chosen


def locate_statements(pages: list[dict], max_span: int = 8) -> list[StatementRegion]:
    """Locate the balance-sheet/income/cash-flow neighborhood from page structure.

    This layer deliberately does not normalize accounting terminology. It uses
    structural signals and proximity; semantic mapping belongs downstream.
    """
    peaks = _peaks(pages)
    chosen = _best_anchors(peaks, max_span)

    if chosen:
        starts = {kind: anchor[0] for kind, anchor in chosen.items()}
        ordered = sorted(starts.items(), key=lambda item: item[1])
        regions: list[StatementRegion] = []

        for index, (kind, start) in enumerate(ordered):
            if index + 1 < len(ordered):
                end = ordered[index + 1][1] - 1
            else:
                # Give the final statement a small continuation allowance.
                end = start + 2

            regions.append(
                StatementRegion(
                    statement_type=kind,
                    start_page=start,
                    end_page=end,
                    confidence=round(chosen[kind][1], 3),
                    evidence_pages=list(range(start, end + 1)),
                )
            )

        return regions

    # No complete neighborhood: return the strongest isolated candidate per type.
    regions: list[StatementRegion] = []
    for kind in TYPES:
        if not peaks[kind]:
            continue
        page, score = max(peaks[kind], key=lambda item: (item[1], -item[0]))
        regions.append(
            StatementRegion(
                statement_type=kind,
                start_page=page,
                end_page=page,
                confidence=round(score, 3),
                evidence_pages=[page],
            )
        )

    return sorted(regions, key=lambda region: region.start_page)
