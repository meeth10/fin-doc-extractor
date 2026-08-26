from __future__ import annotations

import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from extractor.pipeline import extract
from extractor.financial_facts import build_fact_store
from extractor.trusted_financials import trusted_answer
from agent.financial_agent import answer_question

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
STATIC = Path(__file__).resolve().parent / "static"
RUNS.mkdir(exist_ok=True)
executor = ThreadPoolExecutor(max_workers=2)

STATEMENT_TYPES = ("balance_sheet", "income_statement", "cash_flow")
STATEMENT_LABELS = {
    "balance_sheet": "Balance Sheet",
    "income_statement": "Income Statement",
    "cash_flow": "Cash Flow",
}
CORE_ASSIGNMENTS = {"title", "continuation"}

app = FastAPI(title="Fin Doc Extractor")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/runs", StaticFiles(directory=RUNS), name="runs")


class AskRequest(BaseModel):
    question: str
    run_id: str | None = None


def _status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def _write_status(run_dir: Path, **payload) -> None:
    data = {"status": "queued", "progress": 0, "message": "Queued", **payload}
    _status_path(run_dir).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _page_scores(result: dict) -> dict[int, dict]:
    return {int(item["page_number_human"]) - 1: item for item in result.get("_debug", {}).get("page_scores", [])}


def _table_statement_type(table: dict, page_scores: dict[int, dict]) -> tuple[str | None, str]:
    if not table.get("validated"):
        return None, "rejected"
    page = int(table.get("page_number", -1))
    ps = page_scores.get(page)
    if not ps:
        return None, "unassigned"
    category = ps.get("best_category")
    status = ps.get("status")
    if category not in STATEMENT_TYPES:
        return None, "unassigned"
    if status == "confident":
        return category, "validated"
    if status == "ambiguous" and float(table.get("score", 0) or 0) >= 0.85:
        return category, "provisional"
    return None, "unassigned"


def _build_statement_tables(result: dict, tables: dict) -> dict:
    page_scores = _page_scores(result)
    grouped = {key: {"label": STATEMENT_LABELS[key], "tables": [], "pages": [], "status": "empty"} for key in STATEMENT_TYPES}
    for table in tables.get("tables", []):
        statement_type, assignment = _table_statement_type(table, page_scores)
        enriched = dict(table)
        enriched["statement_type"] = statement_type
        enriched["statement_assignment"] = assignment
        enriched["statement_confidence"] = page_scores.get(int(table.get("page_number", -1)), {}).get("confidence") if statement_type else None
        if statement_type:
            grouped[statement_type]["tables"].append(enriched)
    for key, bucket in grouped.items():
        bucket["tables"].sort(key=lambda t: (t.get("statement_assignment") != "validated", -(float(t.get("score", 0) or 0)), int(t.get("page_number", 10**9))))
        bucket["pages"] = sorted({int(t["page_number_human"]) for t in bucket["tables"]})
        if any(t["statement_assignment"] == "validated" for t in bucket["tables"]):
            bucket["status"] = "validated"
        elif bucket["tables"]:
            bucket["status"] = "provisional"
    return grouped


def _core_statement_outputs(statement_tables: dict) -> dict:
    core = {}
    for key in STATEMENT_TYPES:
        bucket = statement_tables.get(key, {})
        all_tables = bucket.get("tables", [])
        kept = [t for t in all_tables if t.get("statement_assignment") in CORE_ASSIGNMENTS]
        core[key] = {
            **bucket,
            "tables": kept,
            "pages": sorted({int(t["page_number_human"]) for t in kept}),
            "candidate_count": len(all_tables),
            "status": "validated" if any(t.get("statement_assignment") == "title" for t in kept) else ("provisional" if kept else "empty"),
        }
    return core


def _run_extraction(run_id: str, source_name: str) -> None:
    run_dir = RUNS / run_id
    pdf_path = run_dir / "source.pdf"
    try:
        _write_status(run_dir, status="running", progress=5, message="Reading PDF and extracting evidence", source_name=source_name)
        result = extract(str(pdf_path), out_dir=str(run_dir), debug=True, render_images=True)
        _write_status(run_dir, status="running", progress=90, message="Preparing statement and fact outputs", source_name=source_name)

        document = json.loads(Path(result["artifacts"]["document_json"]).read_text(encoding="utf-8"))
        tables = json.loads(Path(result["artifacts"]["tables_json"]).read_text(encoding="utf-8"))
        visuals = json.loads(Path(result["artifacts"]["visuals_json"]).read_text(encoding="utf-8"))
        for page in visuals.get("pages", []):
            filename = Path(page["path"]).name
            page["url"] = f"/runs/{run_id}/pages/{filename}"

        extracted_statement_tables = result.get("statement_tables") or _build_statement_tables(result, tables)
        statement_tables = _core_statement_outputs(extracted_statement_tables)
        statement_counts = {key: len(bucket.get("tables", [])) for key, bucket in statement_tables.items()}
        candidate_counts = {key: bucket.get("candidate_count", 0) for key, bucket in statement_tables.items()}

        fact_input = {
            "summary": {"source_name": source_name, "metadata": result["document_metadata"]},
            "document": document,
            "statement_tables": statement_tables,
        }
        fact_store = build_fact_store(fact_input)
        (run_dir / "financial_facts.json").write_text(json.dumps(fact_store, indent=2, ensure_ascii=False), encoding="utf-8")

        for key in STATEMENT_TYPES:
            payload = {"schema_version": "1.1", "statement_type": key, "statement_label": STATEMENT_LABELS[key], "source_file": source_name, "status": statement_tables[key]["status"], "pages": statement_tables[key]["pages"], "candidate_count": candidate_counts[key], "tables": statement_tables[key]["tables"]}
            (run_dir / f"{key}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        payload = {
            "run_id": run_id,
            "summary": {"source_name": source_name, "total_pages": result["total_pages"], "metadata": result["document_metadata"], "sections": result["sections_found"], "table_summary": result["table_summary"], "statement_counts": statement_counts, "statement_candidate_counts": candidate_counts, "fact_count": fact_store["fact_count"], "elapsed_seconds": result["elapsed_seconds"]},
            "statement_tables": statement_tables,
            "financial_facts": fact_store,
            "document": document,
            "tables": tables,
            "visuals": visuals,
        }
        (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        _write_status(run_dir, status="complete", progress=100, message="Extraction complete", source_name=source_name)
    except Exception as exc:
        _write_status(run_dir, status="failed", progress=100, message=str(exc), source_name=source_name)


@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/extract")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    with (run_dir / "source.pdf").open("wb") as out:
        shutil.copyfileobj(file.file, out)
    _write_status(run_dir, status="queued", progress=0, message="Queued", source_name=file.filename)
    executor.submit(_run_extraction, run_id, file.filename)
    return {"run_id": run_id, "status": "queued"}


@app.post("/api/ask")
def ask_financials(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    run_dir = RUNS / request.run_id if request.run_id else None
    if run_dir is None or not (run_dir / "result.json").exists():
        candidates = [p for p in RUNS.iterdir() if p.is_dir() and (p / "result.json").exists()]
        if not candidates:
            raise HTTPException(status_code=404, detail="No completed document is available. Upload a PDF first.")
        run_dir = max(candidates, key=lambda p: (p / "result.json").stat().st_mtime)
    try:
        data = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        result = answer_question(question, data)
        trusted = trusted_answer(question, data)
        if trusted:
            result.update({
                "metric": trusted["metric"],
                "answer": trusted["answer"],
                "period": trusted["period"],
                "status": trusted["status"],
                "confidence": trusted["confidence"],
                "formula": trusted["formula"],
                "inputs": trusted["inputs"],
                "source_page": trusted["source_page"],
                "explanation": trusted["explanation"],
                "trusted_numeric_grounding": True,
            })
        result["run_id"] = run_dir.name
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str):
    path = _status_path(RUNS / run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/result")
def run_result(run_id: str):
    run_dir = RUNS / run_id
    result_path = run_dir / "result.json"
    if not result_path.exists():
        status_path = _status_path(run_dir)
        if not status_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")
        return {"ready": False, "status": json.loads(status_path.read_text(encoding="utf-8"))}
    return {"ready": True, **json.loads(result_path.read_text(encoding="utf-8"))}


@app.get("/api/runs/{run_id}/source")
def source_pdf(run_id: str):
    path = RUNS / run_id / "source.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return FileResponse(path, media_type="application/pdf", filename="source.pdf")
