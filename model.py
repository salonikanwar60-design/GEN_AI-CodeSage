"""Bug-prone commit predictor.

Labels use an SZZ-lite heuristic: a non-fix commit is labelled "bug-introducing"
if one of its files is touched by a later bug-fix commit within WINDOW commits.
Features only use history available *before* each commit (no future leakage).
"""
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .mining import is_bug_fix

WINDOW = 40
FEATURES = [
    "n_files", "additions", "deletions", "churn", "max_file_churn", "n_extensions",
    "hour", "weekday", "msg_len", "prior_bug_ratio", "prior_bug_max", "prior_commits_mean",
]


def _parse(d):
    try:
        return datetime.fromisoformat(d.replace("Z", "+00:00"))
    except Exception:
        return datetime.now()


def _ext(name):
    return name.rsplit(".", 1)[-1] if "." in name else ""


def _features(files, message, when, stats):
    adds = sum(f["additions"] for f in files)
    dels = sum(f["deletions"] for f in files)
    churns = [f["additions"] + f["deletions"] for f in files] or [0]
    seen = [stats[f["filename"]] for f in files if f["filename"] in stats]
    ratios = [b / c for c, b in seen if c]
    return [
        len(files), adds, dels, adds + dels, max(churns), len({_ext(f["filename"]) for f in files}),
        when.hour, when.weekday(), len(message or ""),
        float(np.mean(ratios)) if ratios else 0.0,
        max([b for _, b in seen], default=0),
        float(np.mean([c for c, _ in seen])) if seen else 0.0,
    ]


def build_dataset(commits):
    commits = sorted(commits, key=lambda c: c["date"])
    fix = [is_bug_fix(c["message"]) for c in commits]
    label = [0] * len(commits)
    for i, c in enumerate(commits):
        if not fix[i]:
            continue
        for f in c["files"]:
            for j in range(i - 1, max(-1, i - 1 - WINDOW), -1):
                if not fix[j] and any(g["filename"] == f["filename"] for g in commits[j]["files"]):
                    label[j] = 1
                    break
    stats, X, y = {}, [], []
    for i, c in enumerate(commits):
        if not fix[i]:
            X.append(_features(c["files"], c["message"], _parse(c["date"]), stats))
            y.append(label[i])
        for f in c["files"]:  # update history AFTER computing features
            s = stats.setdefault(f["filename"], [0, 0])
            s[0] += 1
            s[1] += int(fix[i])
    return np.array(X, dtype=float), np.array(y), stats


def train(commits):
    X, y, stats = build_dataset(commits)
    if len(y) < 30 or len(set(y)) < 2:
        raise ValueError("Not enough labelled history to train (need >=30 non-fix commits with both classes).")
    split = int(len(y) * 0.8)  # chronological split
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    params = dict(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    metrics = {"samples": int(len(y)), "positive_rate": round(float(y.mean()), 3), "test_samples": int(len(yte))}
    if len(set(ytr)) == 2 and len(yte):
        clf = RandomForestClassifier(**params).fit(Xtr, ytr)
        pred, proba = clf.predict(Xte), clf.predict_proba(Xte)[:, 1]
        metrics.update(
            precision=round(float(precision_score(yte, pred, zero_division=0)), 3),
            recall=round(float(recall_score(yte, pred, zero_division=0)), 3),
            f1=round(float(f1_score(yte, pred, zero_division=0)), 3),
            roc_auc=round(float(roc_auc_score(yte, proba)), 3) if len(set(yte)) == 2 else None,
        )
    final = RandomForestClassifier(**params).fit(X, y)  # refit on all history for serving
    importances = sorted(zip(FEATURES, final.feature_importances_), key=lambda t: t[1], reverse=True)
    metrics["feature_importance"] = [{"feature": k, "importance": round(float(v), 3)} for k, v in importances[:6]]
    return {"model": final, "stats": stats, "metrics": metrics}


def predict(bundle, files, message=""):
    stats = bundle["stats"]
    x = np.array([_features(files, message, datetime.now(), stats)], dtype=float)
    p = float(bundle["model"].predict_proba(x)[0][1])
    level = "low" if p < 0.3 else "medium" if p < 0.6 else "high"
    risky = []
    for f in files:
        c, b = stats.get(f["filename"], (0, 0))
        if c and b / c >= 0.25:
            risky.append({"file": f["filename"], "past_bug_ratio": round(b / c, 2), "past_commits": c})
    reasons = []
    if risky:
        reasons.append(f"{len(risky)} file(s) have a high historical bug-fix ratio")
    churn = sum(f["additions"] + f["deletions"] for f in files)
    if churn > 300:
        reasons.append(f"Large change ({churn} changed lines)")
    if len(files) > 8:
        reasons.append(f"Touches many files ({len(files)})")
    return {"risk_probability": round(p, 3), "risk_level": level, "risky_files": risky, "reasons": reasons}
