"""Data-mining routines over commit history."""
import math
import re
from collections import Counter, defaultdict
from itertools import combinations

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

BUG_RE = re.compile(
    r"\b(fix(es|ed)?|bug(s)?|hotfix|patch|crash(es)?|error|regression|resolve[sd]?|issue\s*#?\d+)\b", re.I
)


def is_bug_fix(message: str) -> bool:
    return bool(BUG_RE.search(message or ""))


def bug_hotspots(commits, top=15):
    """Per-file bug statistics. risk_score = bug_ratio * log(1 + commits)."""
    stats = defaultdict(lambda: {"commits": 0, "bug_fixes": 0, "churn": 0, "authors": set()})
    for c in commits:
        fix = is_bug_fix(c["message"])
        for f in c["files"]:
            s = stats[f["filename"]]
            s["commits"] += 1
            s["bug_fixes"] += int(fix)
            s["churn"] += f["additions"] + f["deletions"]
            s["authors"].add(c["author"])
    rows = []
    for name, s in stats.items():
        ratio = s["bug_fixes"] / s["commits"]
        rows.append({
            "file": name,
            "commits": s["commits"],
            "bug_fixes": s["bug_fixes"],
            "bug_ratio": round(ratio, 3),
            "churn": s["churn"],
            "authors": len(s["authors"]),
            "risk_score": round(ratio * math.log1p(s["commits"]), 3),
        })
    rows.sort(key=lambda r: (r["risk_score"], r["bug_fixes"]), reverse=True)
    return rows[:top]


def co_change(commits, min_support=3, max_files=25, top=15):
    """Frequent file pairs that change together (support + confidence A->B)."""
    single, pair = Counter(), Counter()
    for c in commits:
        names = sorted({f["filename"] for f in c["files"]})
        if len(names) > max_files:  # skip mass-refactor commits
            continue
        single.update(names)
        pair.update(combinations(names, 2))
    rows = []
    for (a, b), n in pair.items():
        if n >= min_support:
            rows.append({
                "file_a": a, "file_b": b, "support": n,
                "confidence_a_to_b": round(n / single[a], 2),
                "confidence_b_to_a": round(n / single[b], 2),
            })
    rows.sort(key=lambda r: r["support"], reverse=True)
    return rows[:top]


def bug_fix_clusters(commits, k=5):
    """Cluster bug-fix commit messages (TF-IDF + KMeans) to reveal recurring bug themes."""
    msgs = [re.sub(r"\d+", " ", c["message"].lower()) for c in commits if is_bug_fix(c["message"])]
    if len(msgs) < 6:
        return []
    k = max(2, min(k, len(msgs) // 3))
    vec = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2))
    X = vec.fit_transform(msgs)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    terms = vec.get_feature_names_out()
    out = []
    for i in range(k):
        top_idx = km.cluster_centers_[i].argsort()[::-1][:5]
        members = [m for m, lab in zip(msgs, km.labels_) if lab == i]
        out.append({
            "cluster": i,
            "size": len(members),
            "top_terms": [terms[j] for j in top_idx],
            "example": members[0].strip(),
        })
    out.sort(key=lambda r: r["size"], reverse=True)
    return out
