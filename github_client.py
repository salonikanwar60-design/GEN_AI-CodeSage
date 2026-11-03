"""Fetch commit history (with per-file changes) from the GitHub REST API."""
import os
import re
from urllib.parse import urlsplit

import requests

API = "https://api.github.com"


def _headers(token: str | None = None):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "codesage"}
    token = (token or os.getenv("GITHUB_TOKEN", "")).strip()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def parse_repo(repo: str) -> str:
    """Accept 'owner/name' or a full GitHub URL and return 'owner/name'."""
    repo = repo.strip()
    if "://" in repo:
        parsed = urlsplit(repo)
        if parsed.scheme not in {"https", "http"} or parsed.hostname != "github.com":
            raise ValueError("Enter a GitHub URL or repository as 'owner/name'")
        repo = parsed.path.strip("/")
    else:
        repo = repo.rstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+", repo):
        raise ValueError("Enter a GitHub URL or repository as 'owner/name'")
    return repo


def fetch_commits(repo: str, max_commits: int = 100, token: str | None = None):
    """Return commits as: {sha, message, author, date, files:[{filename, additions, deletions}]}.

    Costs 1 request per commit for file details, so keep max_commits modest
    unless GITHUB_TOKEN is set.
    """
    repo = parse_repo(repo)
    listing = []
    page = 1
    while len(listing) < max_commits:
        r = requests.get(
            f"{API}/repos/{repo}/commits",
            params={"per_page": min(100, max_commits - len(listing)), "page": page},
            headers=_headers(token),
            timeout=20,
        )
        if r.status_code == 403:
            raise RuntimeError("GitHub rate limit hit. Set GITHUB_TOKEN in .env and retry.")
        if r.status_code == 404:
            raise RuntimeError("Repository not found (or private).")
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        listing.extend(batch)
        page += 1

    commits = []
    for item in listing[:max_commits]:
        d = requests.get(f"{API}/repos/{repo}/commits/{item['sha']}", headers=_headers(token), timeout=20)
        if d.status_code == 403:
            raise RuntimeError("GitHub rate limit hit. Set GITHUB_TOKEN in .env and retry.")
        d.raise_for_status()
        data = d.json()
        commit = data["commit"]
        commits.append({
            "sha": data["sha"][:10],
            "message": commit["message"].split("\n")[0][:200],
            "author": (commit.get("author") or {}).get("name", "unknown"),
            "date": (commit.get("author") or {}).get("date", ""),
            "files": [
                {"filename": f["filename"], "additions": f.get("additions", 0), "deletions": f.get("deletions", 0)}
                for f in data.get("files", [])
            ],
        })
    return commits
