"""Song-to-dance API.

Upload a song, get a generated SMPL dance back as viewer JSON or BVH.

Engines (MODE env var, picked automatically if unset):
  stitch  beat-matched retrieval and stitching of real AIST++ moves (CPU, default
          when LIBRARY points at a built library.npz)
  edge    pretrained EDGE diffusion model (GPU, when EDGE_DIR has checkpoint.pt)
  mock    procedural motion, for UI work only
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import motion

EDGE_DIR = Path(os.environ.get("EDGE_DIR", "/opt/EDGE"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/choreo-jobs"))
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
LIBRARY = Path(os.environ.get("LIBRARY", Path(__file__).resolve().parent.parent / "data" / "library.npz"))
MODE = os.environ.get("MODE") or (
    "stitch" if LIBRARY.exists() else "edge" if (EDGE_DIR / "checkpoint.pt").exists() else "mock")
MAX_SECONDS = 30 if MODE == "edge" else 180
_choreo = None


def choreographer():
    global _choreo
    if _choreo is None:
        import choreographer as C
        _choreo = C.Choreographer(C.Library.load(LIBRARY))
    return _choreo
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Choreography API")
jobs: dict[str, dict] = {}
gpu_lock = threading.Lock()  # one EDGE run at a time per GPU


def _ffmpeg_clip(src: Path, dst: Path, start: float, seconds: float):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds),
         "-i", str(src), "-ac", "1", "-ar", "44100", str(dst)],
        check=True,
    )


def _run_edge(job_dir: Path, clip: Path, seconds: int) -> dict:
    # EDGE slices the song into 5 s windows (2.5 s stride) and picks a random
    # contiguous run of out_length seconds. The clip is trimmed to exactly
    # out_length, so the only possible run starts at 0 and stays in sync.
    music = job_dir / "music"
    music.mkdir(exist_ok=True)
    shutil.copy(clip, music / "song.wav")
    out = job_dir / "motions"
    cmd = [
        sys.executable, "test.py",
        "--feature_type", "jukebox",
        "--out_length", str(seconds),
        "--music_dir", str(music),
        "--checkpoint", str(EDGE_DIR / "checkpoint.pt"),
        "--save_motions", "--motion_save_dir", str(out),
        "--render_dir", str(job_dir / "renders"),
        "--no_render",
    ]
    with gpu_lock:
        proc = subprocess.run(cmd, cwd=EDGE_DIR, capture_output=True, text=True)
    (job_dir / "edge.log").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"EDGE failed: {proc.stderr[-800:]}")
    pkls = glob.glob(str(out / "**" / "*.pkl"), recursive=True)
    if not pkls:
        raise RuntimeError("EDGE produced no motion file")
    with open(pkls[0], "rb") as f:
        return pickle.load(f)


def _work(job_id: str, src: Path, start: float, seconds: int, genre: str, seed: int, energy: str):
    job = jobs[job_id]
    job_dir = DATA_DIR / job_id
    try:
        clip = job_dir / "clip.wav"
        job["status"] = "preparing audio"
        _ffmpeg_clip(src, clip, start, seconds)
        job["status"] = "generating dance"
        if MODE == "stitch":
            sample, info = choreographer().generate(clip, seconds, genre=genre, seed=seed, energy=energy)
            job.update(info)
        elif MODE == "edge":
            sample = _run_edge(job_dir, clip, seconds)
        else:
            sample = motion.mock_sample(seconds, seed=int(uuid.UUID(job_id)) % 2**32)
        (job_dir / "motion.json").write_text(json.dumps(motion.to_viewer_json(sample)))
        (job_dir / "motion.bvh").write_text(motion.to_bvh(sample))
        with open(job_dir / "motion.pkl", "wb") as f:
            pickle.dump(sample, f)
        job["status"] = "done"
    except Exception as e:  # surfaced to the client
        job["status"] = "error"
        job["error"] = str(e)


@app.get("/api/health")
def health():
    import choreographer as C
    import intent
    return {"mode": MODE, "max_seconds": MAX_SECONDS, "genres": C.GENRES if MODE == "stitch" else {},
            "assistant": intent.llm_status()}


@app.post("/api/generate")
async def generate(file: UploadFile = File(...), start: float = Form(0.0), seconds: int = Form(30),
                   genre: str = Form("auto"), seed: int | None = Form(None), energy: str = Form("match")):
    # EDGE's window maths needs a multiple of 5 s, at least 5 s.
    seconds = max(5, min(MAX_SECONDS, int(round(seconds / 5)) * 5))
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), "little")
    job_id = str(uuid.uuid4())
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True)
    src = job_dir / ("upload" + Path(file.filename or "song").suffix)
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jobs[job_id] = {"id": job_id, "status": "queued", "seconds": seconds, "start": max(0.0, start)}
    threading.Thread(target=_work, args=(job_id, src, max(0.0, start), seconds, genre, seed, energy if energy in ("calm", "match", "intense") else "match"), daemon=True).start()
    return jobs[job_id]


class Prompt(BaseModel):
    text: str


@app.post("/api/interpret")
def interpret(p: Prompt):
    """Chat box: turn a typed request into dance settings with the on-device LLM."""
    import intent
    return intent.interpret(p.text)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    return jobs[job_id]


FILES = {"motion.json": "application/json", "motion.bvh": "text/plain", "clip.wav": "audio/wav", "motion.pkl": "application/octet-stream"}


@app.get("/api/jobs/{job_id}/{name}")
def job_file(job_id: str, name: str):
    if job_id not in jobs or name not in FILES:
        raise HTTPException(404)
    path = DATA_DIR / job_id / name
    if not path.exists():
        raise HTTPException(409, "not ready")
    return FileResponse(path, media_type=FILES[name], filename=f"dance-{job_id[:8]}{Path(name).suffix}")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
