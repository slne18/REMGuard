"""Upload page for a new night.

The container is stateless. The fitted model and the healthy reference are
read from disk and are not changed by a request.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import mne
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from remguard.data.edf_header import read_edf_header
from remguard.expert_scores import expert_comparison, parse_expert_scores
from remguard.night_score import model_path, reference_path, repo_root, score_compact_folder, score_recording
from remguard.preprocessing import preprocess_recording

mne.set_log_level("ERROR")

app = FastAPI(title="REMGuard")
STATIC = Path(__file__).resolve().parent / "static"
_model = None
_reference = None
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_score_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        if not model_path().exists():
            raise HTTPException(status_code=503, detail="The REM model has not been fitted yet.")
        _model = joblib.load(model_path())["model"]
    return _model


def get_reference() -> dict:
    global _reference
    if _reference is None:
        if not reference_path().exists():
            raise HTTPException(status_code=503, detail="The healthy reference has not been written yet.")
        _reference = json.loads(reference_path().read_text(encoding="utf-8"))
    return _reference


@app.get("/health")
def health() -> dict:
    ready = model_path().exists() and reference_path().exists()
    return {"ready": ready}


@app.get("/reference")
def reference() -> dict:
    payload = get_reference()
    example = repo_root() / "data" / "preprocess" / "n6"
    return {
        "elevated_percent": payload["elevated_percent"],
        "median": payload["median"],
        "p90": payload["p90"],
        "rule": payload["rule"],
        "example_available": (example / "preprocessing.json").is_file(),
    }


@app.post("/score")
async def score(
    recording: UploadFile = File(...),
    scores: UploadFile | None = File(None),
) -> dict:
    name = recording.filename or "night.edf"
    if not name.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Upload one EDF file that contains EEG, EOG, and chin EMG.")
    data = await recording.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    score_text = await _score_text(scores)
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running"}
    threading.Thread(target=_score_job, args=(job_id, name, data, score_text), daemon=True).start()
    return {"status": "running", "job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="That score is no longer available.")
    if job["status"] == "error":
        raise HTTPException(status_code=400, detail=job["detail"])
    if job["status"] == "running":
        return {"status": "running", "job_id": job_id}
    return job["result"]


@app.post("/score/example")
def score_example() -> dict:
    folder = repo_root() / "data" / "preprocess" / "n6"
    if not (folder / "preprocessing.json").is_file():
        raise HTTPException(status_code=404, detail="The example night is not on this machine.")
    scored = score_compact_folder(folder, get_model())
    return _payload("n6", scored, None)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def _score_job(job_id: str, name: str, data: bytes, score_text: str | None) -> None:
    try:
        with _score_lock:
            with TemporaryDirectory() as temp:
                path = Path(temp) / "night.edf"
                path.write_bytes(data)
                expert = None
                if score_text is not None:
                    header = read_edf_header(path)
                    expert = parse_expert_scores(score_text, header.start_time)
                prepared = preprocess_recording(path)
                scored = score_recording(prepared, get_model())
                result = _payload(name, scored, expert)
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "result": result}
    except ValueError as exc:
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "detail": str(exc)}
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "detail": f"The EDF could not be scored: {exc}"}


async def _score_text(scores: UploadFile | None) -> str | None:
    if scores is None or not scores.filename:
        return None
    if not scores.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="The expert score file should be a .txt file.")
    data = await scores.read()
    if not data.strip():
        raise HTTPException(status_code=400, detail="The expert score file is empty.")
    return data.decode("utf-8-sig", errors="replace")


def _payload(name: str, scored, expert) -> dict:
    reference = get_reference()
    values = reference["elevated_percent"]
    p90 = float(reference["p90"])
    percent = None if scored.elevated_percent is None else round(float(scored.elevated_percent), 2)
    flagged = percent is not None and percent >= p90
    rank = None
    if percent is not None and values:
        rank = round(100.0 * sum(value < percent for value in values) / len(values), 1)
    return {
        "name": name,
        "elevated_percent": percent,
        "rem_minutes": round(scored.rem_minutes, 1),
        "quiet_level_uv": None if scored.quiet_level_uv is None else round(scored.quiet_level_uv, 2),
        "threshold_uv": None if scored.threshold_uv is None else round(scored.threshold_uv, 2),
        "duration_sec": round(scored.duration_sec, 1),
        "rem_blocks": scored.rem_blocks,
        "preview_times": scored.preview_times,
        "preview_uv": [round(value, 2) for value in scored.preview_uv],
        "n_rem_epochs": scored.n_rem_epochs,
        "n_epochs": scored.n_epochs,
        "flagged": flagged,
        "rank": rank,
        "healthy_elevated_percent": values,
        "healthy_median": reference["median"],
        "healthy_p90": p90,
        "expert": None
        if expert is None
        else expert_comparison(expert, scored.rem_blocks, scored.n_epochs, scored.duration_sec),
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")
