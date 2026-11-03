"""CodeSage FastAPI app: mining + bug-risk prediction + code review."""
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from . import github_client, mining, model, reviewer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

STATE = {"source": None, "commits": [], "hotspots": [], "cochange": [], "clusters": [], "bundle": None, "error": None}


def analyze(commits, source):
    STATE.update(source=source, commits=commits, error=None)
    STATE["hotspots"] = mining.bug_hotspots(commits, top=50)
    STATE["cochange"] = mining.co_change(commits)
    STATE["clusters"] = mining.bug_fix_clusters(commits)
    try:
        STATE["bundle"] = model.train(commits)
    except ValueError as e:
        STATE["bundle"], STATE["error"] = None, str(e)


app = FastAPI(title="CodeSage", version="1.0.0")


class MineReq(BaseModel):
    repo: str = Field(..., min_length=1)
    max_commits: int = Field(100, gt=0)


class FileChange(BaseModel):
    filename: str
    additions: int = 0
    deletions: int = 0


class PredictReq(BaseModel):
    files: list[FileChange]
    message: str = ""


class ReviewReq(BaseModel):
    code: str = Field(..., max_length=20000)
    filename: str = "snippet.py"
    review_focus: str = Field("", max_length=4000)
    review_mode: Literal["snippet", "diff"] = "snippet"
    review_strategy: Literal["static", "dynamic", "adaptive"] = "dynamic"


@app.post("/api/mine")
def mine(req: MineReq, github_token: str | None = Header(default=None, alias="X-GitHub-Token")):
    try:
        repo = github_client.parse_repo(req.repo)
        commits = github_client.fetch_commits(repo, req.max_commits, token=github_token)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"GitHub request failed: {type(e).__name__}")
    analyze(commits, repo)
    return summary()


@app.get("/api/summary")
def summary():
    commits = STATE["commits"]
    fixes = sum(mining.is_bug_fix(c["message"]) for c in commits)
    return {
        "source": STATE["source"], "commits": len(commits), "bug_fix_commits": fixes,
        "files": len({f["filename"] for c in commits for f in c["files"]}),
        "hotspots": STATE["hotspots"][:15], "cochange": STATE["cochange"], "clusters": STATE["clusters"],
        "model": STATE["bundle"]["metrics"] if STATE["bundle"] else None, "model_error": STATE["error"],
    }


@app.post("/api/predict")
def predict(req: PredictReq):
    if not STATE["bundle"]:
        raise HTTPException(409, STATE["error"] or "No trained model. Mine a repository first.")
    if not req.files:
        raise HTTPException(400, "Provide at least one changed file.")
    return model.predict(STATE["bundle"], [f.model_dump() for f in req.files], req.message)


@app.post("/api/review")
def review(req: ReviewReq, gemini_api_key: str | None = Header(default=None, alias="X-Gemini-Api-Key")):
    if not req.code.strip():
        raise HTTPException(400, "Paste some code to review.")
    if req.review_mode == "diff" and not reviewer.is_unified_diff(req.code):
        raise HTTPException(400, "Paste a unified diff that includes at least one @@ hunk header.")
    history = next((h for h in STATE["hotspots"] if h["file"] == req.filename or h["file"].endswith("/" + req.filename)), None)
    return reviewer.review(
        req.code, req.filename, history, req.review_focus, gemini_api_key,
        req.review_mode, req.review_strategy,
    )


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
