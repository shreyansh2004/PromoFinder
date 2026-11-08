"""Web app: FastAPI backend + static single-page UI.

Run:  .venv\\Scripts\\python -m uvicorn app:app --port 8000     then open http://localhost:8000
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from promofinder.llm import LLM, model_options  # noqa: E402
from promofinder.pipeline import run  # noqa: E402
from promofinder.ranker import ART, SuccessModel  # noqa: E402

UPLOADS, OUTPUTS, FEEDBACK = ROOT / "uploads", ROOT / "outputs", ROOT / "feedback"
for p in (UPLOADS, OUTPUTS, FEEDBACK):
    p.mkdir(exist_ok=True)

app = FastAPI(title="Promo Finder")
JOBS: dict[str, dict] = {}

import os  # noqa: E402

# Hosted safety: set APP_ACCESS_CODE so only people with the code can spend the AI key (rule-based mode
# stays open), and ALLOW_LOCAL_PATH=0 so the server never reads its own folders.
ACCESS_CODE = os.environ.get("APP_ACCESS_CODE", "")
ALLOW_LOCAL = os.environ.get("ALLOW_LOCAL_PATH", "1") == "1"


def _ai_allowed(use_llm: bool, code: str) -> bool:
    if use_llm and ACCESS_CODE and code.strip() != ACCESS_CODE:
        raise HTTPException(403, "Wrong access code for the AI model. Leave AI off to run the rule-based mode.")
    return use_llm


def _start(job_id: str, folder: Path, use_llm: bool, top_n: int, name: str, model: str = "", episodes: str = ""):
    job = JOBS[job_id]

    def progress(msg):
        job["log"].append({"t": round(time.time() - job["started"], 1), "msg": msg})

    def work():
        try:
            load_dotenv(ROOT / ".env", override=True)  # pick up keys added while the server runs
            res = run(folder, OUTPUTS / job_id, llm=LLM(model=model or None), top_n=top_n, use_llm=use_llm,
                      progress=progress, episodes=episodes or None)
            res["show_name"] = name
            job.update(status="done", result=res)
        except Exception as e:  # surface the reason in the UI
            job.update(status="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, daemon=True).start()


def _new_job(name: str) -> str:
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    JOBS[job_id] = {"id": job_id, "status": "running", "log": [], "started": time.time(), "name": name}
    return job_id


@app.get("/api/config")
def config():
    load_dotenv(ROOT / ".env", override=True)
    llm, model = LLM(), SuccessModel()
    return {"llm_available": llm.available, "llm": llm.describe(), "models": model_options(),
            "ai_requires_code": bool(ACCESS_CODE), "allow_local_path": ALLOW_LOCAL,
            "success_model": model.meta.get("summary") if model.available else None}


@app.post("/api/runs")
async def create_run(files: list[UploadFile] = File(...), use_llm: bool = Form(True), top_n: int = Form(15),
                     show_name: str = Form(""), model: str = Form(""), episodes: str = Form(""),
                     access_code: str = Form("")):
    use_llm = _ai_allowed(use_llm, access_code)
    top_n = max(5, min(int(top_n), 40))
    job_id = _new_job(show_name or "Uploaded show")
    folder = UPLOADS / job_id
    folder.mkdir(parents=True)
    n = 0
    for f in files:
        data = await f.read()
        fname = Path(f.filename or "file").name
        if fname.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for m in z.infolist():
                    if m.is_dir() or not m.filename.lower().endswith(".txt"):
                        continue
                    (folder / Path(m.filename).name).write_bytes(z.read(m))  # flatten: no path traversal
                    n += 1
        elif fname.lower().endswith(".txt"):
            (folder / fname).write_bytes(data)
            n += 1
    if n == 0:
        shutil.rmtree(folder, ignore_errors=True)
        JOBS.pop(job_id, None)
        raise HTTPException(400, "No .txt screenplay files found. Upload episode .txt files or a .zip of them.")
    _start(job_id, folder, use_llm, top_n, show_name or f"{n} episodes", model, episodes)
    return {"id": job_id, "files": n}


class LocalRun(BaseModel):
    path: str
    use_llm: bool = True
    top_n: int = 15
    show_name: str = ""
    model: str = ""
    episodes: str = ""
    access_code: str = ""


@app.post("/api/runs/local")
def create_local_run(body: LocalRun):
    if not ALLOW_LOCAL:
        raise HTTPException(403, "Folder paths are disabled on the hosted app. Upload the .txt files or a .zip instead.")
    body.use_llm = _ai_allowed(body.use_llm, body.access_code)
    folder = Path(body.path.strip().strip('"'))
    if not folder.is_dir() or not list(folder.glob("*.txt")):
        raise HTTPException(400, f"Folder not found or has no .txt files: {body.path}")
    name = body.show_name or folder.name
    job_id = _new_job(name)
    _start(job_id, folder, body.use_llm, body.top_n, name, body.model, body.episodes)
    return {"id": job_id}


@app.get("/api/runs")
def list_runs():
    return [{"id": j["id"], "name": j["name"], "status": j["status"]} for j in JOBS.values()]


@app.get("/api/runs/{job_id}")
def get_run(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Run not found")
    out = {k: v for k, v in job.items() if k != "result"}
    if job["status"] == "done":
        out["result"] = job["result"]
    return out


class Feedback(BaseModel):
    promo_id: str
    decision: str  # approved | rejected
    note: str = ""


@app.post("/api/runs/{job_id}/feedback")
def feedback(job_id: str, fb: Feedback):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Run not found or not finished")
    promo = next((p for p in job["result"]["promos"] if p["id"] == fb.promo_id), None)
    if not promo:
        raise HTTPException(404, "Promo not found")
    f = FEEDBACK / "feedback.csv"
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "run", "show", "promo_id", "decision", "note", "segments", "final_score", "model_prob"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), job_id, job["name"], fb.promo_id, fb.decision, fb.note,
                    json.dumps(promo["segments"]), promo["scores"].get("final"), promo["scores"].get("model_prob")])
    promo["decision"] = fb.decision
    return {"ok": True}


@app.get("/api/runs/{job_id}/download/{name}")
def download(job_id: str, name: str):
    if name not in {"promos.csv", "promos.json", "report.md"}:
        raise HTTPException(404, "Unknown file")
    f = OUTPUTS / job_id / name
    if not f.exists():
        raise HTTPException(404, "File not ready")
    return FileResponse(f, filename=f"{job_id}-{name}")


@app.get("/api/insights")
def insights():
    f = ART / "insights.json"
    if not f.exists():
        return JSONResponse({"available": False})
    out = {"available": True, **json.loads(f.read_text(encoding="utf-8"))}
    v = ART / "signal_validation.json"
    if v.exists():
        out["validation"] = json.loads(v.read_text(encoding="utf-8"))
    return out


@app.get("/health")
def health():
    return PlainTextResponse("ok")


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
