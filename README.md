# GEN_AI-CodeSage
CodeSage ,Understand every change. Ship with confidence.  CodeSage mines a Git repository's history to learn where bugs come from, predicts how risky a new change is, and reviews code with a GenAI assistant t mined history. GenAI Gemini reviews code using static findings, mined file history and returns issues, a patch and a test case
# CodeSage — Understand every change. Ship with confidence.

CodeSage mines a Git repository's history to learn where bugs come from, predicts how risky a new
change is, and reviews code with a GenAI assistant that is grounded in that mined history.

## What it does

| Layer | Technique | File |
|---|---|---|
| Data collection | GitHub REST API (commits + per-file changes) | `app/github_client.py` |
| Data mining | Bug-fix detection, per-file hotspots, co-change pairs (frequent pattern mining), TF-IDF + KMeans clustering of bug-fix messages | `app/mining.py` |
| Prediction | RandomForest "bug-prone commit" classifier, SZZ-lite labelling, leak-free historical features, chronological evaluation | `app/model.py` |
| GenAI | Gemini reviews code using static findings + mined file history and returns issues, a patch and a test case | `app/reviewer.py` |
| API + UI | FastAPI backend, single-page dashboard | `app/main.py`, `app/static/index.html` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # Windows PowerShell: Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 and enter a public GitHub repository as `owner/name` or paste its
GitHub URL in the Data Mining tab. CodeSage analyzes that repository's real commit history;
it does not load synthetic data on startup.

For larger history scans, create a GitHub fine-grained personal access token with read-only
access to public repositories. You can paste it into the optional GitHub token field in the
Data Mining page; that token is sent for that request only and is not stored. Alternatively,
set `GITHUB_TOKEN=your-token` in the local `.env` file and restart the app. Mining makes roughly
one GitHub API request per commit; authenticating raises the typical rate limit. `.env` is
ignored by Git; never commit or share your token.

To enable GenAI reviews, create a Gemini API key in
[Google AI Studio](https://aistudio.google.com/apikey). Paste it into the optional password field
on the GenAI Review page to use it for that review only; the app clears it after the request and
does not store it. Alternatively, add it to the local `.env` file as `GEMINI_API_KEY=your-key` and
restart the app. The Gemini API has free-tier quota limits and eligibility may vary by model and
account. `.env` is ignored by Git; never commit or share your API key.

The GenAI Review page supports full-snippet and unified-diff reviews, including change-focused
findings with new-file line numbers. Select an advanced review profile (for example security,
reliability, concurrency, or production readiness), then filter findings by severity and the
model-reported confidence (an estimate, not a calibrated probability). Suggested patches and
generated tests can be downloaded from the review results. Treat generated patches and tests as
suggestions and inspect and run them before using them.

Choose a review strategy independently of snippet/diff input:
- **Static** runs local rules only and never sends code to Gemini.
- **Dynamic** combines the static checks with a Gemini review and requires a Gemini API key.
- **Adaptive** selects a review focus and focused/deep review depth from the code, filename,
  static findings, and available mined history. It falls back to local checks if no key is set.

“Dynamic” refers to AI-assisted analysis; CodeSage does not execute submitted code or tests.

Run tests: `python -m pytest -q`

### Environment variables (`.env`)
- `GITHUB_TOKEN` (optional): raises the typical GitHub limit from 60 to 5000 requests/hour. Mining needs about
  1 request per commit, so use it for anything beyond ~50 commits.
- `GEMINI_API_KEY` (optional): enables the Gemini reviewer by default. A key entered in the review
  page is used for that request and takes precedence. Without either key, static analysis runs.
- `GEMINI_MODEL` (optional): defaults to `gemini-2.5-flash`.

## API

| Method | Path | Body | Purpose |
|---|---|---|---|
| POST | `/api/mine` | `{"repo": "owner/name", "max_commits": 100}` (any positive commit count; no fixed upper cap) | Mine + train |
| GET | `/api/summary` | – | Hotspots, co-change, clusters, model metrics |
| POST | `/api/predict` | `{"files":[{"filename":"a.py","additions":50,"deletions":5}],"message":""}` | Risk of a change |
| POST | `/api/review` | `{"code":"...","filename":"checkout.py","review_mode":"snippet","review_strategy":"dynamic"}` | Snippet review; input mode is `"snippet"` or `"diff"`, and strategy is `"static"`, `"dynamic"`, or `"adaptive"` |

Interactive docs: http://127.0.0.1:8000/docs

## How the model avoids cheating
- **Labels** look forward: a non-fix commit is "bug-introducing" if a later bug-fix commit (within 40 commits) touches one of its files.
- **Features** look backward only: file history is updated *after* each commit's features are computed.
- **Evaluation** is a chronological 80/20 split, not a random shuffle.

## Limitations (be upfront about these in interviews)
- Bug-fix detection uses commit-message keywords, so it is noisy; real SZZ would use blame on the fix diff.
- Recent commits are slightly under-labelled because their fixes may not exist yet.
- Automated tests use a synthetic fixture; the application itself only analyzes repositories you provide.
- Static rules are intentionally small; the LLM layer covers deeper issues.

## Ideas to extend
Add GitHub Issues linking, SHAP explanations, a PR webhook that comments risk on pull requests, Docker, and a React frontend.

## Resume bullets
- Built a code-quality platform that mines GitHub commit history (hotspots, co-change patterns, bug-theme clustering) and trains a RandomForest model to flag bug-prone changes using leak-free, chronologically evaluated features.
- Developed a FastAPI service and dashboard with an LLM-powered reviewer that combines static analysis with project-specific bug history to explain risks and generate patches and test cases; covered by an automated pytest suite.
