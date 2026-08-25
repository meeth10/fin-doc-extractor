"""Financial PDF evidence extraction pipeline."""

from __future__ import annotations

import json
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
            pct = 5 + int(((i + 1) / total) * 35)
            _progress(progress_callback, pct, f"Reading pages ({i + 1}/{total})")
    return results


def _flagged_pages_from_sections(sections: List[Dict]) -> set[int]:
    flagged = set()
    for section in sections:
        flagged.update(range(section["start_page"], section["end_page"] + 1))
    return flagged


def _render_pages(
    doc: "fitz.Document",
    page_numbers: set[int],
    image_dir: Path,
    progress_callback: ProgressCallback = None,
) -> List[Dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    zoom = RENDER_DPI / 72
    mat = fitz.Matrix(zoom, zoom)
    page_list = sorted(page_numbers)
    total = max(1, len(page_list))
    for i, pn in enumerate(page_list, start=1):
        pix = doc[pn].get_pixmap(matrix=mat, alpha=False)
        filename = f"page_{pn + 1:04d}.png"
        path = image_dir / filename
        pix.save(str(path))
        rendered.append({
            "page_number_human": pn + 1,
            "path": str(path),
            "dpi": RENDER_DPI,
            "format": "png",
        })
        pct = 80 + int((i / total) * 15)
        _progress(progress_callback, pct, f"Rendering page images ({i}/{total})")
    return rendered


def _attach_statement_types(table_records: List[Dict], page_scores: List) -> None:
    """Assign statement type title-first, then fall back to page classification.

    Explicit financial-statement titles are the strongest semantic evidence:
    "Balance Sheet", "Statement of Profit and Loss", "Cash Flow Statement", etc.
    Only when no explicit title was found do we use a confident page-level score.
    """
    score_by_page = {s.page_number: s for s in page_scores}
    for table in table_records:
        statement_type = table.get("statement_type_from_title")
        statement_confidence = 1.0 if statement_type in TARGET_STATEMENTS else 0.0
        assignment = "title" if statement_type in TARGET_STATEMENTS else None

        if statement_type not in TARGET_STATEMENTS:
            score = score_by_page.get(table["page_number"])
            if score and score.status == "confident" and score.best_category in TARGET_STATEMENTS:
                statement_type = score.best_category
                statement_confidence = score.confidence
                assignment = "page_classifier"

        table["statement_type"] = statement_type if statement_type in TARGET_STATEMENTS else None
        table["statement_confidence"] = statement_confidence
        table["statement_assignment"] = assignment


def _isolated_statement_outputs(tables: List[Dict]) -> Dict[str, Dict]:
    outputs: Dict[str, Dict] = {}
    for statement_type in TARGET_STATEMENTS:
        statement_tables = [
            t for t in tables
            if t.get("statement_type") == statement_type
        ]
        statement_tables.sort(
            key=lambda t: (
                0 if t.get("statement_assignment") == "title" else 1,
                t["page_number"],
                -t["score"],
            )
        )
        outputs[statement_type] = {
            "statement_type": statement_type,
            "table_count": len(statement_tables),
            "pages": sorted({t["page_number_human"] for t in statement_tables}),
            "status": "validated" if any(t["validated"] for t in statement_tables) else ("provisional" if statement_tables else "empty"),
            "tables": statement_tables,
        }
    return outputs


def extract(
    pdf_path: str,
    out_dir: str = None,
    include_notes: bool = False,
    debug: bool = False,
    render_images: bool = True,
    progress_callback: ProgressCallback = None,
) -> Dict:
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
    sections, ambiguous_pages, page_scores = locate_financial_statements(
        pages_text, include_notes=include_notes
    )
    flagged_pages = _flagged_pages_from_sections(sections)
    table_pages = {
        s.page_number
        for s in page_scores
        if s.status in {"confident", "ambiguous"}
    }

    _progress(progress_callback, 50, "Extracting table candidates")
    table_records: List[Dict] = []
    with pdfplumber.open(pdf_path) as pl_doc:
        table_page_list = sorted(table_pages)
        total_table_pages = max(1, len(table_page_list))
        for i, pn in enumerate(table_page_list, start=1):
            candidates = extract_table_candidates(pl_doc.pages[pn], doc[pn])
            for rank, candidate in enumerate(candidates):
                table_records.append({
                    "page_number": pn,
                    "page_number_human": pn + 1,
                    "rank": rank + 1,
                    "scope": "financial_locator",
                    **candidate,
                })
            pct = 50 + int((i / total_table_pages) * 25)
            _progress(progress_callback, pct, f"Extracting tables ({i}/{total_table_pages})")

        if not any(t["validated"] for t in table_records):
            _progress(progress_callback, 75, "Trying document-wide table fallback")
            for pn in range(total_pages):
                if pn in table_pages:
                    continue
                candidates = extract_table_candidates(pl_doc.pages[pn], doc[pn])
                for rank, candidate in enumerate(candidates[:3]):
                    if candidate["score"] >= 0.45:
                        table_records.append({
                            "page_number": pn,
                            "page_number_human": pn + 1,
                            "rank": rank + 1,
                            "scope": "global_fallback",
                            **candidate,
                        })

    _attach_statement_types(table_records, page_scores)

    # Keep all title-assigned tables for statement isolation. A title is a
    # semantic assignment even when table-shape validation is imperfect; such
    # tables are marked provisional rather than discarded.
    statement_candidates = [
        t for t in table_records
        if t.get("statement_type") in TARGET_STATEMENTS
    ]
    for t in statement_candidates:
        if t.get("statement_assignment") == "title" and not t.get("validated"):
            t["statement_assignment"] = "provisional"

    validated = [t for t in table_records if t["validated"]]
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
        output_pages.append({
            "page_number": pn,
            "page_number_human": pn + 1,
            "extraction_method": r["method"],
            "raw_text": r["text"],
            "table_candidate_count": len(page_tables),
            "best_table": best_by_page.get(pn),
            "tables": [t["table"] for t in page_tables if t["validated"]],
        })

    result = {
        "schema_version": "2.2",
        "source_file": pdf_path,
        "total_pages": total_pages,
        "document_metadata": doc_metadata.as_dict(),
        "sections_found": sections,
        "ambiguous_pages": [p + 1 for p in ambiguous_pages],
        "flagged_page_count": len(flagged_pages),
        "pages": output_pages,
        "statement_tables": isolated_statements,
        "table_summary": {
            "candidate_count": len(table_records),
            "validated_count": len(validated),
            "pages_with_validated_tables": len(best_by_page),
            "best_document_table": max(validated, key=lambda t: t["score"], default=None),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    if debug:
        result["_debug"] = {
            "page_scores": [
                {
                    "page_number_human": s.page_number + 1,
                    "status": s.status,
                    "best_category": s.best_category,
                    "confidence": s.confidence,
                    "category_scores": s.category_scores,
                    "signals": s.signals,
                }
                for s in page_scores if s.status != "none"
            ],
            "table_candidates": [
                {k: v for k, v in t.items() if k not in {"table"}}
                for t in table_records
            ],
        }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        document_path = out / (Path(pdf_path).stem + "_document.json")
        tables_path = out / (Path(pdf_path).stem + "_tables.json")
        visuals_path = out / (Path(pdf_path).stem + "_visuals.json")

        text_payload = dict(result)
        text_payload.pop("_debug", None)
        text_payload.pop("table_summary", None)
        text_payload.pop("pages", None)
        text_payload["pages"] = [
            {
                "page_number": r["page_number"],
                "page_number_human": r["page_number"] + 1,
                "extraction_method": r["method"],
                "raw_text": r["text"],
            }
            for r in page_records
        ]
        with document_path.open("w", encoding="utf-8") as f:
            json.dump(text_payload, f, indent=2, ensure_ascii=False)

        with tables_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": "2.2",
                    "source_file": pdf_path,
                    "tables": table_records,
                    "statement_tables": isolated_statements,
                    "best_document_table": result["table_summary"]["best_document_table"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        statement_files = {}
        for statement_type, payload in isolated_statements.items():
            path = out / f"{statement_type}.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "2.2",
                        "source_file": pdf_path,
                        **payload,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            statement_files[statement_type] = str(path)

        if render_images:
            image_dir = out / "pages"
            rendered = _render_pages(doc, set(range(total_pages)), image_dir, progress_callback)
        else:
            rendered = []
        with visuals_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": "2.2",
                    "source_file": pdf_path,
                    "render_dpi": RENDER_DPI,
                    "pages": rendered,
                },
                f,
                indent=2,
            )

        result["artifacts"] = {
            "document_json": str(document_path),
            "tables_json": str(tables_path),
            "visuals_json": str(visuals_path),
            "page_image_dir": str(out / "pages") if render_images else None,
            "statement_files": statement_files,
        }

    _progress(progress_callback, 100, "Extraction complete")
    doc.close()
    return result
