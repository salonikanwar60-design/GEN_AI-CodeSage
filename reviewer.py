"""Code reviewer: static checks + optional Gemini review grounded in mined bug history."""
import ast
import json
import os
import re

import requests

SEV = {"high": 0, "medium": 1, "low": 2}

PY_RULES = [
    (r"^\s*except\s*:", "high", "Bare `except:` swallows every error, including KeyboardInterrupt.", "Catch specific exceptions, e.g. `except ValueError as e:`."),
    (r"\b(eval|exec)\s*\(", "high", "`eval`/`exec` on dynamic input allows code injection.", "Use `ast.literal_eval` or an explicit parser."),
    (r"(password|passwd|secret|api[_-]?key|token)\s*=\s*['\"][^'\"]{3,}['\"]", "high", "Hard-coded credential.", "Load it from an environment variable or a secrets manager."),
    (r"\.execute\(\s*(f['\"]|['\"][^'\"]*['\"]\s*(%|\+))", "high", "SQL built with string formatting (SQL injection risk).", "Use parameterised queries: `cur.execute(sql, params)`."),
    (r"subprocess\.\w+\(.*shell\s*=\s*True", "high", "`shell=True` enables command injection.", "Pass a list of args and drop `shell=True`."),
    (r"pickle\.loads?\(", "medium", "Unpickling untrusted data can execute code.", "Use JSON or a safe serialisation format."),
    (r"requests\.(get|post|put|delete|patch)\((?![^)]*timeout)", "medium", "HTTP call without a timeout can hang forever.", "Add `timeout=10` (or similar)."),
    (r"^[^#]*=\s*open\(", "medium", "File opened without a context manager may leak the handle.", "Use `with open(...) as f:`."),
    (r"[=!]=\s*None", "low", "Comparing to None with ==/!=.", "Use `is None` / `is not None`."),
    (r"^\s*print\(", "low", "Debug `print` left in code.", "Use the `logging` module."),
]
JS_RULES = [
    (r"\beval\s*\(", "high", "`eval` allows code injection.", "Parse data with JSON.parse or a dedicated parser."),
    (r"\.innerHTML\s*=", "high", "Assigning `innerHTML` can cause XSS.", "Use `textContent` or sanitise the input."),
    (r"(password|secret|api[_-]?key|token)\s*[:=]\s*['\"][^'\"]{3,}['\"]", "high", "Hard-coded credential.", "Read it from environment/config on the server."),
    (r"catch\s*\([^)]*\)\s*\{\s*\}", "medium", "Empty catch block hides failures.", "Handle or at least log the error."),
    (r"(?<![=!<>])==(?!=)|!=(?!=)", "low", "Loose equality performs type coercion.", "Use `===` / `!==`."),
    (r"\bvar\s+\w+", "low", "`var` is function-scoped and error-prone.", "Use `const` / `let`."),
    (r"console\.log\(", "low", "Debug `console.log` left in code.", "Remove it or use a logger."),
]


def detect_language(filename: str, code: str = "") -> str:
    f = (filename or "").lower()
    if f.endswith(".py"):
        return "python"
    if f.endswith((".js", ".jsx", ".ts", ".tsx")):
        return "javascript"
    return "python" if re.search(r"^\s*(def |import |from \w+ import)", code, re.M) else "javascript"


def static_review(code: str, language: str, line_numbers: list[int] | None = None):
    rules = PY_RULES if language == "python" else JS_RULES
    issues = []
    for index, line in enumerate(code.splitlines()):
        n = line_numbers[index] if line_numbers else index + 1
        for pat, sev, problem, fix in rules:
            if re.search(pat, line, re.I):
                issues.append({"line": n, "severity": sev, "problem": problem, "fix": fix,
                               "source": "static", "confidence": 0.9})
    if language == "python" and line_numbers is None:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for d in node.args.defaults + [x for x in node.args.kw_defaults if x]:
                        if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                            issues.append({"line": d.lineno, "severity": "medium", "source": "static",
                                           "confidence": 0.9,
                                           "problem": "Mutable default argument is shared between calls.",
                                           "fix": "Default to `None` and create the object inside the function."})
        except SyntaxError as e:
            issues.append({"line": e.lineno or 1, "severity": "high", "source": "static",
                           "confidence": 0.9,
                           "problem": f"Syntax error: {e.msg}", "fix": "Fix the syntax before review."})
    issues.sort(key=lambda i: (SEV[i["severity"]], i["line"]))
    return issues


SYSTEM = (
    "You are a senior code reviewer. You receive code or a unified diff, findings from a static analyser, and "
    "bug-history statistics mined from the project's own git history. For diffs, focus on concrete regressions "
    "introduced by added or changed lines and use new-file line numbers. Respond with ONLY valid JSON: "
    '{"summary": str, "issues": [{"line": int, "severity": "high|medium|low", "confidence": number from 0 to 1, '
    '"problem": str, "fix": str}], '
    '"suggested_patch": str, "test_case": str}. Be specific, do not invent problems, and mention when '
    "the mined history makes a finding more important."
)

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _diff_added_lines(diff: str) -> tuple[str, list[int]]:
    added = []
    line_numbers = []
    new_line = 0
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("+++"):
            continue
        match = HUNK_HEADER.match(line)
        if match:
            new_line = int(match.group(1))
            in_hunk = True
        elif not in_hunk or line.startswith("\\"):
            continue
        elif line.startswith("+"):
            added.append(line[1:])
            line_numbers.append(new_line)
            new_line += 1
        elif line.startswith("-"):
            continue
        elif line.startswith(" "):
            new_line += 1
    return "\n".join(added), line_numbers


def is_unified_diff(diff: str) -> bool:
    return any(HUNK_HEADER.match(line) for line in diff.splitlines())


ADAPTIVE_PROFILES = {
    "security": "Focus on security vulnerabilities, validation, authorization, secrets, injection, and trust boundaries.",
    "performance": "Focus on algorithmic complexity, memory use, unnecessary work, and resource lifecycle.",
    "database": "Focus on query safety, transaction boundaries, data consistency, and connection lifecycle.",
    "frontend": "Focus on accessibility, browser safety, user-visible errors, and responsive behavior.",
    "concurrency": "Focus on race conditions, shared state, locking, async cancellation, and deadlocks.",
    "reliability": "Focus on failure modes, error handling, retries, timeouts, and resource cleanup.",
    "general": "Focus on correctness, maintainability, and edge cases.",
}


def _adaptive_profile(
    code: str,
    filename: str,
    issues: list[dict],
    history: dict | None,
) -> tuple[str, str]:
    lower_code = code.lower()
    lower_filename = filename.lower()
    if re.search(r"\b(eval|exec)\s*\(|shell\s*=\s*true|innerhtml|execute\s*\(.*(?:select|insert|update|delete)", lower_code):
        profile = "security"
    elif re.search(r"\b(select|insert|update|delete|commit|rollback)\b|\.execute\s*\(", lower_code) or "model" in lower_filename:
        profile = "database"
    elif re.search(r"\b(async|await|thread|mutex|lock|atomic|parallel)\b", lower_code):
        profile = "concurrency"
    elif lower_filename.endswith((".html", ".css", ".jsx", ".tsx")):
        profile = "frontend"
    elif len(code.splitlines()) >= 80 or re.search(r"\b(for|while)\b.*\b(for|while)\b", lower_code):
        profile = "performance"
    elif issues or (history and history.get("bug_ratio", 0) >= 0.3 and history.get("commits", 0) >= 5):
        profile = "reliability"
    else:
        profile = "general"

    depth = "deep" if len(code.splitlines()) >= 100 or len(issues) >= 3 else "focused"
    if history and history.get("bug_ratio", 0) >= 0.3 and history.get("commits", 0) >= 5:
        depth = "deep"
    return profile, depth


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e + 1])


def review(
    code: str,
    filename: str = "snippet.py",
    history: dict | None = None,
    review_focus: str = "",
    api_key: str | None = None,
    review_mode: str = "snippet",
    review_strategy: str = "dynamic",
):
    language = detect_language(filename, code)
    reviewed_code, line_numbers = (code, None)
    if review_mode == "diff":
        reviewed_code, line_numbers = _diff_added_lines(code)
    static = static_review(reviewed_code, language, line_numbers)
    adaptive_profile = None
    review_depth = None
    if review_strategy == "adaptive":
        adaptive_profile, review_depth = _adaptive_profile(reviewed_code, filename, static, history)
    result = {
        "mode": "static",
        "review_mode": review_mode,
        "review_strategy": review_strategy,
        "language": language,
        "summary": "",
        "issues": static,
        "suggested_patch": "",
        "test_case": "",
        "history_context": history,
    }
    if adaptive_profile:
        result.update(adaptive_profile=adaptive_profile, review_depth=review_depth)

    if history:
        result["summary"] = (f"{filename} has a bug-fix ratio of {history['bug_ratio']:.0%} over "
                             f"{history['commits']} commits in this repo's history; review changes here carefully. ")
    if static:
        result["summary"] += f"Static analysis found {len(static)} issue(s)."

    if review_strategy == "static":
        return result

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        strategy_name = "Adaptive" if review_strategy == "adaptive" else "Dynamic"
        result["note"] = (
            f"{strategy_name} review needs a Gemini API key. Showing static checks only; "
            "enter a key for this review or configure GEMINI_API_KEY in .env."
        )
        return result
    try:
        numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(code.splitlines(), 1))
        focus_parts = []
        if adaptive_profile:
            focus_parts.append(
                f"Adaptive profile: {adaptive_profile}. Review depth: {review_depth}. "
                f"{ADAPTIVE_PROFILES[adaptive_profile]}"
            )
        if review_focus.strip():
            focus_parts.append(f"Additional review guidance from the user:\n{review_focus.strip()}")
        focus = "\n\n".join(focus_parts)
        focus = f"{focus}\n\n" if focus else ""
        prompt = (f"Review mode: {review_mode}\nReview strategy: {review_strategy}\nFile: {filename} ({language})\n\n{focus}"
                  f"Mined history for this file: {json.dumps(history)}\n\n"
                  f"Static findings: {json.dumps(static)}\n\nCode (with line numbers):\n{numbered}")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "maxOutputTokens": 2000,
                    "temperature": 0.1,
                },
            },
            timeout=45,
        )
        response.raise_for_status()
        response_data = response.json()
        response_text = "".join(
            part.get("text", "")
            for part in response_data["candidates"][0]["content"]["parts"]
        )
        data = _extract_json(response_text)
        ai_issues = data.get("issues", static)
        if not isinstance(ai_issues, list):
            ai_issues = static
        else:
            normalized_issues = []
            for issue in ai_issues:
                if not isinstance(issue, dict) or issue.get("severity") not in SEV:
                    continue
                try:
                    confidence = min(1.0, max(0.0, float(issue.get("confidence", 0.75))))
                    line = max(1, int(issue.get("line", 1)))
                except (TypeError, ValueError):
                    confidence, line = 0.75, 1
                normalized_issues.append({
                    "line": line,
                    "severity": issue["severity"],
                    "confidence": confidence,
                    "problem": str(issue.get("problem", "")),
                    "fix": str(issue.get("fix", "")),
                    "source": "genai",
                })
            ai_issues = normalized_issues
        result.update(mode="genai", summary=data.get("summary", result["summary"]),
                      issues=ai_issues, suggested_patch=data.get("suggested_patch", ""),
                      test_case=data.get("test_case", ""))
    except Exception as exc:  # network/auth/parse errors -> graceful fallback
        result["note"] = f"GenAI review unavailable ({type(exc).__name__}); showing static analysis only."
    return result
