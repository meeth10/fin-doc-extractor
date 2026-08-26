"""Financial PDF evidence extraction pipeline."""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional
import pymupdf as fitz
import pdfplumber
from .classify import classify_page
from .ocr import ocr_page
from .locator import locate_financial_statements
from .table_extract import extract_table_candidates
from .metadata import detect_metadata

RENDER_DPI = 120
ProgressCallback = Optional[Callable[[int, str], None]]
TARGET_STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")
SEQUENCE_MAX_SPAN = 5
MAX_CONTINUATION_PAGES = 2
TITLE_PATTERNS = {
    "balance_sheet": [
        r"^(?:consolidated\s+)?balance\s+sheets?(?:\s+(?:as\s+(?:at|on|of)\b|for\s+the\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+financial\s+position(?:\s+(?:as\s+(?:at|on|of)\b|for\s+the\b).*)?$",
    ],
    "income_statement": [
        r"^(?:consolidated\s+)?profit\s+and\s+loss\s+statements?(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+profit\s+and\s+loss(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?income\s+statements?(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+operations?(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+income(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
    ],
    "cash_flow": [
        r"^(?:consolidated\s+)?cash\s+flow\s+statements?(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
        r"^(?:consolidated\s+)?statements?\s+of\s+cash\s+flows?(?:\s+(?:for\s+the\b|as\s+at\b).*)?$",
    ],
}
_TITLE_RE = {k: [re.compile(p, re.I) for p in v] for k, v in TITLE_PATTERNS.items()}
YEAR_RE = re.compile(r"\b(?:FY\s*)?20\d{2}(?:[-/–]\d{2})?\b", re.I)
INDEX_MARKERS = (
    "index to consolidated financial statements",
    "index to financial statements",
    "financial statements and supplementary data",
    "table of contents",
)
RELATED_STATEMENT_MARKERS = (
    "statements of comprehensive income",
    "statement of comprehensive income",
    "other comprehensive income",
)


def _progress(callback: ProgressCallback, percent: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, percent)), message)


def get_all_page_text(doc: "fitz.Document", progress_callback: ProgressCallback = None) -> List[Dict]:
    results = []
    total = max(1, doc.page_count)
    for i, page in enumerate(doc):
        method = classify_page(page)
        text = page.get_text("text") if method == "digital" else ocr_page(page)
        results.append({"page_number": i, "method": method, "text": text})
        if (i + 1) == total or (i + 1) % max(1, total // 20) == 0:
            _progress(progress_callback, 5 + int(((i + 1) / total) * 35), f"Reading pages ({i + 1}/{total})")
    return results


def _flagged_pages_from_sections(sections: List[Dict]) -> set[int]:
    flagged = set()
    for section in sections:
        flagged.update(range(section["start_page"], section["end_page"] + 1))
    return flagged


def _render_pages(doc: "fitz.Document", page_numbers: set[int], image_dir: Path, progress_callback: ProgressCallback = None) -> List[Dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    page_list = sorted(page_numbers)
    total = max(1, len(page_list))
    for i, pn in enumerate(page_list, start=1):
        path = image_dir / f"page_{pn + 1:04d}.png"
        doc[pn].get_pixmap(matrix=mat, alpha=False).save(str(path))
        rendered.append({"page_number_human": pn + 1, "path": str(path), "dpi": RENDER_DPI, "format": "png"})
        _progress(progress_callback, 80 + int((i / total) * 15), f"Rendering page images ({i}/{total})")
    return rendered


def _nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _looks_title_like(normalized: str) -> bool:
    if len(normalized.split()) > 24:
        return False
    markers = ("the amount", "the company", "is disclosed", "is recognised", "are disclosed", "recognised in", "refer to", "see note", "as per note", "during the year", "following", "pursuant", "thereof", "below", "above")
    return not any(marker in normalized.lower() for marker in markers)


def _page_is_excluded_from_primary(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return any(marker in normalized for marker in INDEX_MARKERS + RELATED_STATEMENT_MARKERS)


def _primary_title(page_text: str) -> Optional[Dict]:
    lines = _nonempty_lines(page_text)
    for idx, line in enumerate(lines[:18]):
        normalized = re.sub(r"\s+", " ", line).strip()
        lower = normalized.lower()
        if not normalized or len(normalized) > 140 or not _looks_title_like(normalized):
            continue
        if any(marker in lower for marker in INDEX_MARKERS + RELATED_STATEMENT_MARKERS):
            return None
        for statement_type, patterns in _TITLE_RE.items():
            if any(p.fullmatch(normalized) for p in patterns):
                context = " ".join(lines[idx:idx + 5])
                return {"statement_type": statement_type, "title": normalized, "line_index": idx, "has_year_header": bool(YEAR_RE.search(context)), "is_primary_title": idx <= 8}
    return None


def _annotate_page_titles(page_records: List[Dict]) -> None:
    for record in page_records:
        record["statement_title"] = _primary_title(record["text"])
        record["statement_excluded"] = _page_is_excluded_from_primary(record["text"])


def _statement_sequence_candidates(page_records: List[Dict], table_records: List[Dict]) -> Dict[str, List[int]]:
    """Find the strongest BS/IS/CF title cluster within five pages, regardless of order."""
    titles: Dict[str, List[Dict]] = {t: [] for t in TARGET_STATEMENTS}
    for record in page_records:
        title = record.get("statement_title")
        if title and title["statement_type"] in TARGET_STATEMENTS and title["is_primary_title"]:
            titles[title["statement_type"]].append({"page": record["page_number"], "year": title["has_year_header"]})
    table_score: Dict[int, float] = {}
    table_validated: Dict[int, bool] = {}
    for table in table_records:
        if table["validated"]:
            table_score[table["page_number"]] = max(table_score.get(table["page_number"], 0.0), float(table["score"]))
            table_validated[table["page_number"]] = True

    combos = []
    for bs in titles["balance_sheet"]:
        for inc in titles["income_statement"]:
            for cf in titles["cash_flow"]:
                pages = [bs["page"], inc["page"], cf["page"]]
                span = max(pages) - min(pages)
                if len(set(pages)) != 3 or span > SEQUENCE_MAX_SPAN:
                    continue
                proximity = sum(1.0 / (1.0 + abs(pages[i] - pages[j])) for i, j in ((0, 1), (0, 2), (1, 2)))
                year_support = 0.75 * sum(int(x["year"]) for x in (bs, inc, cf))
                table_support = sum(table_score.get(p, 0.0) for p in pages)
                validation_support = sum(int(table_validated.get(p, False)) for p in pages)
                score = 10.0 + proximity + year_support + table_support + 1.5 * validation_support - 0.75 * span
                combos.append((score, bs["page"], inc["page"], cf["page"]))
    if combos:
        _, bs, inc, cf = max(combos, key=lambda x: x[0])
        return {"balance_sheet": [bs], "income_statement": [inc], "cash_flow": [cf]}
    return {t: ([max(pages, key=lambda x: (x["year"], -x["page"]))["page"]] if pages else []) for t, pages in titles.items()}


def _attach_statement_types(page_records: List[Dict], table_records: List[Dict]) -> Dict[str, List[int]]:
    """Assign statement types using page titles, cluster proximity and explicit exclusions."""
    _annotate_page_titles(page_records)
    selected = _statement_sequence_candidates(page_records, table_records)
    title_by_page = {r["page_number"]: r.get("statement_title") for r in page_records}
    excluded_pages = {r["page_number"] for r in page_records if r.get("statement_excluded")}
    selected_pages = [selected[t][0] for t in TARGET_STATEMENTS if selected.get(t)]
    complete_cluster = len(selected_pages) == 3 and (max(selected_pages) - min(selected_pages) <= SEQUENCE_MAX_SPAN)

    for table in table_records:
        table["statement_type"] = None
        table["statement_assignment"] = "unassigned"
        table["statement_confidence"] = 0.0
        title = title_by_page.get(table["page_number"])
        table["table_title"] = title["title"] if title else None

    for statement_type, pages in selected.items():
        for table in table_records:
            if table["validated"] and table["page_number"] in pages and table["page_number"] not in excluded_pages:
                table["statement_type"] = statement_type
                table["statement_assignment"] = "title" if complete_cluster else "title_candidate"
                table["statement_confidence"] = 1.0 if complete_cluster else 0.82

    if complete_cluster:
        ordered = sorted((pages[0], kind) for kind, pages in selected.items())
        cluster_end = ordered[-1][0]
        for table in table_records:
            if table["statement_type"] is not None or not table["validated"]:
                continue
            pn = table["page_number"]
            if pn in excluded_pages:
                continue
            for idx, (start, kind) in enumerate(ordered):
                next_start = ordered[idx + 1][0] if idx + 1 < len(ordered) else cluster_end + MAX_CONTINUATION_PAGES + 1
                if start < pn < next_start and pn <= start + MAX_CONTINUATION_PAGES:
                    table["statement_type"] = kind
                    table["statement_assignment"] = "continuation"
                    table["statement_confidence"] = 0.92
                    break
    else:
        for table in table_records:
            if table["statement_type"] is None and table["validated"] and table["page_number"] not in excluded_pages:
                title = title_by_page.get(table["page_number"])
                if title and title["statement_type"] in TARGET_STATEMENTS:
                    table["statement_type"] = title["statement_type"]
                    table["statement_assignment"] = "title_candidate"
                    table["statement_confidence"] = 0.82
    return selected


def _isolated_statement_outputs(tables: List[Dict]) -> Dict[str, Dict]:
    outputs: Dict[str, Dict] = {}
    for statement_type in TARGET_STATEMENTS:
        statement_tables = [t for t in tables if t.get("statement_type") == statement_type]
        statement_tables.sort(key=lambda t: (0 if t.get("statement_assignment") in {"title", "title_candidate"} else 1, t["page_number"], -t["score"]))
        outputs[statement_type] = {"statement_type": statement_type, "table_count": len(statement_tables), "pages": sorted({t["page_number_human"] for t in statement_tables}), "status": "validated" if any(t["validated"] for t in statement_tables) else ("provisional" if statement_tables else "empty"), "tables": statement_tables}
    return outputs


def extract(pdf_path: str, out_dir: str = None, include_notes: bool = False, debug: bool = False, render_images: bool = True, progress_callback: ProgressCallback = None) -> Dict:
    pdf_path = str(pdf_path)
    t0 = time.time()
    _progress(progress_callback, 1, "Opening PDF")
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    page_records = get_all_page_text(doc, progress_callback)
    pages_text = [r["text"] for r in page_records]
    _progress(progress_callback, 42, "Detecting document metadata")
    doc_metadata = detect_metadata(pages_text)
    _progress(progress_callback, 46, "Locating financial statements")
    sections, ambiguous_pages, page_scores = locate_financial_statements(pages_text, include_notes=include_notes)
    flagged_pages = _flagged_pages_from_sections(sections)
    table_pages = {s.page_number for s in page_scores if s.status in {"confident", "ambiguous"}}
    _progress(progress_callback, 50, "Extracting table candidates")
    table_records: List[Dict] = []
    with pdfplumber.open(pdf_path) as pl_doc:
        table_page_list = sorted(table_pages)
        total_table_pages = max(1, len(table_page_list))
        for i, pn in enumerate(table_page_list, start=1):
            candidates = extract_table_candidates(pl_doc.pages[pn], doc[pn])
            for rank, candidate in enumerate(candidates):
                table_records.append({"page_number": pn, "page_number_human": pn + 1, "rank": rank + 1, "scope": "financial_locator", **candidate})
            _progress(progress_callback, 50 + int((i / total_table_pages) * 25), f"Extracting tables ({i}/{total_table_pages})")
        if not any(t["validated"] for t in table_records):
            _progress(progress_callback, 75, "Trying document-wide table fallback")
            for pn in range(total_pages):
                if pn in table_pages:
                    continue
                candidates = extract_table_candidates(pl_doc.pages[pn], doc[pn])
                for rank, candidate in enumerate(candidates[:3]):
                    if candidate["score"] >= 0.45:
                        table_records.append({"page_number": pn, "page_number_human": pn + 1, "rank": rank + 1, "scope": "global_fallback", **candidate})
    _attach_statement_types(page_records, table_records)
    validated = [t for t in table_records if t["validated"]]
    statement_candidates = [t for t in table_records if t.get("statement_type") in TARGET_STATEMENTS]
    isolated_statements = _isolated_statement_outputs(statement_candidates)
    best_by_page: Dict[int, Dict] = {}
    for t in validated:
        current = best_by_page.get(t["page_number"])
        if current is None or t["score"] > current["score"]:
            best_by_page[t["page_number"]] = t
    output_pages = []
    for r in page_records:
        pn = r["page_number"]
        if pn not in flagged_pages:
            continue
        page_tables = [t for t in table_records if t["page_number"] == pn]
        output_pages.append({"page_number": pn, "page_number_human": pn + 1, "extraction_method": r["method"], "raw_text": r["text"], "table_candidate_count": len(page_tables), "best_table": best_by_page.get(pn), "tables": [t["table"] for t in page_tables if t["validated"]]})
    result = {"schema_version": "2.3", "source_file": pdf_path, "total_pages": total_pages, "document_metadata": doc_metadata.as_dict(), "sections_found": sections, "ambiguous_pages": [p + 1 for p in ambiguous_pages], "flagged_page_count": len(flagged_pages), "pages": output_pages, "statement_tables": isolated_statements, "table_summary": {"candidate_count": len(table_records), "validated_count": len(validated), "pages_with_validated_tables": len(best_by_page), "best_document_table": max(validated, key=lambda t: t["score"], default=None)}, "elapsed_seconds": round(time.time() - t0, 2)}
    if debug:
        result["_debug"] = {"page_scores": [{"page_number_human": s.page_number + 1, "status": s.status, "best_category": s.best_category, "confidence": s.confidence, "category_scores": s.category_scores, "signals": s.signals} for s in page_scores if s.status != "none"], "table_candidates": [{k: v for k, v in t.items() if k not in {"table"}} for t in table_records]}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        document_path = out / (Path(pdf_path).stem + "_document.json")
        tables_path = out / (Path(pdf_path).stem + "_tables.json")
        visuals_path = out / (Path(pdf_path).stem + "_visuals.json")
        text_payload = dict(result)
        text_payload.pop("_debug", None); text_payload.pop("table_summary", None); text_payload.pop("pages", None)
        text_payload["pages"] = [{"page_number": r["page_number"], "page_number_human": r["page_number"] + 1, "extraction_method": r["method"], "raw_text": r["text"]} for r in page_records]
        with document_path.open("w", encoding="utf-8") as f: json.dump(text_payload, f, indent=2, ensure_ascii=False)
        with tables_path.open("w", encoding="utf-8") as f: json.dump({"schema_version": "2.3", "source_file": pdf_path, "tables": table_records, "statement_tables": isolated_statements, "best_document_table": result["table_summary"]["best_document_table"]}, f, indent=2, ensure_ascii=False)
        statement_files = {}
        for statement_type, payload in isolated_statements.items():
            path = out / f"{statement_type}.json"
            path.write_text(json.dumps({"schema_version": "2.3", "source_file": pdf_path, **payload}, indent=2, ensure_ascii=False), encoding="utf-8")
            statement_files[statement_type] = str(path)
        if render_images:
            image_dir = out / "pages"; rendered = _render_pages(doc, set(range(total_pages)), image_dir, progress_callback)
        else: rendered = []
        with visuals_path.open("w", encoding="utf-8") as f: json.dump({"schema_version": "2.3", "source_file": pdf_path, "render_dpi": RENDER_DPI, "pages": rendered}, f, indent=2)
        result["artifacts"] = {"document_json": str(document_path), "tables_json": str(tables_path), "visuals_json": str(visuals_path), "page_image_dir": str(out / "pages") if render_images else None, "statement_files": statement_files}
    _progress(progress_callback, 100, "Extraction complete")
    doc.close()
    return result