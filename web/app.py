from __future__ import annotations

import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
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
executor = ThreadPoolExecutor(max_workers=2)


def _status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def _write_status(run_dir: Path, **payload) -> None:
    data = {"status": "queued", "progress": 0, "message": "Queued", **payload}
    _status_path(run_dir).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_extraction(run_id: str, source_name: str) -> None:
    run_dir = RUNS / run_id
    pdf_path = run_dir / "source.pdf"

    def progress(percent: int, message: str) -> None:
        _write_status(run_dir, status="running", progress=percent, message=message, source_name=source_name)

    try:
        progress(1, "Starting extraction")
        result = extract(
            str(pdf_path),
            out_dir=str(run_dir),
            debug=True,
            render_images=True,
            progress_callback=progress,
        )

        document = json.loads(Path(result["artifacts"]["document_json"]).read_text(encoding="utf-8"))
        tables = json.loads(Path(result["artifacts"]["tables_json"]).read_text(encoding="utf-8"))
        visuals = json.loads(Path(result["artifacts"]["visuals_json"]).read_text(encoding="utf-8"))

        for page in visuals.get("pages", []):
            filename = Path(page["path"]).name
            page["url"] = f"/runs/{run_id}/pages/{filename}"

        payload = {
            "run_id": run_id,
            "summary": {
                "source_name": source_name,
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
    pdf_path = run_dir / "source.pdf"

    with pdf_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    _write_status(run_dir, status="queued", progress=0, message="Queued", source_name=file.filename)
    executor.submit(_run_extraction, run_id, file.filename)
    return {"run_id": run_id, "status": "queued"}


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
