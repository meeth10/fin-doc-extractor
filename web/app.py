from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from extractor.pipeline import extract

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
STATIC = Path(__file__).resolve().parent / "static"
RUNS.mkdir(exist_ok=True)

app = FastAPI(title="Fin Doc Extractor")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/runs", StaticFiles(directory=RUNS), name="runs")


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
    pdf_path = run_dir / "source.pdf"

    with pdf_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        result = extract(str(pdf_path), out_dir=str(run_dir), debug=True, render_images=True)
    except Exception as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    document = json.loads(Path(result["artifacts"]["document_json"]).read_text(encoding="utf-8"))
    tables = json.loads(Path(result["artifacts"]["tables_json"]).read_text(encoding="utf-8"))
    visuals = json.loads(Path(result["artifacts"]["visuals_json"]).read_text(encoding="utf-8"))

    for page in visuals.get("pages", []):
        filename = Path(page["path"]).name
        page["url"] = f"/runs/{run_id}/pages/{filename}"

    return {
        "run_id": run_id,
        "summary": {
            "source_name": file.filename,
            "total_pages": result["total_pages"],
            "metadata": result["document_metadata"],
            "sections": result["sections_found"],
            "table_summary": result["table_summary"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
        "document": document,
        "tables": tables,
        "visuals": visuals,
    }


@app.get("/api/runs/{run_id}/source")
def source_pdf(run_id: str):
    path = RUNS / run_id / "source.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return FileResponse(path, media_type="application/pdf", filename="source.pdf")
