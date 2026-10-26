# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pyyaml>=6.0"]
# ///
"""Run the evaluation questions against the EU-RAG API and write evals.md.

    uv run evals/run_evals.py run --label baseline     # ask all questions, save run, rebuild report
    uv run evals/run_evals.py run --category GDPR      # subset (ids / categories)
    uv run evals/run_evals.py report                   # rescore all saved runs, rebuild report (no API calls)

Each run is stored verbatim (answers, citations, latency) in evals/runs/. Scores are always
recomputed from those raw results with the current questions.yaml and metric definitions, so
adding a metric or fixing an expectation applies to old runs too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import httpx
import yaml

EVALS_DIR = Path(__file__).parent
QUESTIONS = EVALS_DIR / "questions.yaml"
RUNS_DIR = EVALS_DIR / "runs"
REPORT = EVALS_DIR / "evals.md"

# Heuristic: phrases the model uses when it declines or says the sources don't cover a question.
_DECLINE = re.compile(
    r"(?:sources?|context|information|documents?)(?: provided| given)?\s+(?:do(?:es)?\s+not|don't)\s+"
    r"(?:appear to\s+)?(?:contain|provide|include|cover|mention|address|specify|reference)"
    r"|not (?:\w+ )?(?:referenced|covered|mentioned|addressed|included|found) in the (?:provided |given )?"
    r"(?:sources|context|information)"
    r"|(?:cannot|can't|unable to) (?:answer|find|provide|determine)"
    r"|does not exist|insufficient information"
    r"|outside (?:the )?scope of the (?:provided |given )?(?:sources|context|information)",
    re.I,
)


# --- Scoring ------------------------------------------------------------------


def unit_key(act: str, unit: str, number: str) -> tuple[str, str, str]:
    return (act, unit, str(number).upper())


def expected_units(q: dict, required_only: bool = False) -> set[tuple]:
    """Expected units; `optional: true` entries count as correct citations but not towards recall."""
    return {
        unit_key(e["act"], e["unit"], e["number"])
        for e in q.get("expected", [])
        if not (required_only and e.get("optional"))
    }


def cited_units(r: dict) -> set[tuple]:
    return {unit_key(c["act"], c["unit_type"], c["number"]) for c in r.get("citations", [])}


def is_decline(r: dict) -> bool:
    # Runs before the `declined` flag existed only have the answer text to go by.
    return r.get("status") == 404 or bool(r.get("declined")) or bool(_DECLINE.search(r.get("answer", "")))


def _mention_pattern(unit: str, number: str) -> re.Pattern:
    prefix = r"Art(?:icle|\.)?s?" if unit == "article" else r"Annex(?:es)?"
    return re.compile(rf"\b{prefix}\s*{re.escape(number)}(?![\dA-Za-z])", re.I)


def answerable(q: dict) -> bool:
    return not q.get("expect_decline")


def m_hit(q, r):
    return float(bool(expected_units(q) & cited_units(r)))


def m_recall(q, r):
    exp = expected_units(q, required_only=True)
    return len(exp & cited_units(r)) / len(exp)


def m_precision(q, r):
    # Recitals are excluded: citing the recital that explains an expected article is not wrong.
    cited = {u for u in cited_units(r) if u[1] != "recital"}
    return len(cited & expected_units(q)) / len(cited) if cited else None


def m_act_coverage(q, r):
    return float({u[0] for u in expected_units(q)} <= {u[0] for u in cited_units(r)})


def m_mentioned(q, r):
    exp = [e for e in q["expected"] if not e.get("optional")]
    return sum(bool(_mention_pattern(e["unit"], str(e["number"])).search(r.get("answer", ""))) for e in exp) / len(exp)


def m_false_decline(q, r):
    return float(is_decline(r))


def m_declined(q, r):
    return float(is_decline(r))


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    applies: Callable[[dict], bool]
    fn: Callable[[dict, dict], float | None]
    higher_is_better: bool = True
    description: str = ""


# Add new metrics here; they show up in every table and are applied to all saved runs.
METRICS = [
    Metric("hit", "Hit", answerable, m_hit, description="at least one expected unit is cited"),
    Metric("recall", "Recall", answerable, m_recall, description="share of required (non-optional) expected units cited"),
    Metric("precision", "Precision", answerable, m_precision,
           description="share of cited articles/annexes that are expected (recitals ignored)"),
    Metric("act_coverage", "Act cov.", answerable, m_act_coverage,
           description="every expected act appears among the citations"),
    Metric("mentioned", "Mentioned", answerable, m_mentioned,
           description="share of required expected units named in the answer text (e.g. 'Article 6')"),
    Metric("false_decline", "False decl.", answerable, m_false_decline, higher_is_better=False,
           description="answerable question but the answer reads as a refusal (heuristic)"),
    Metric("declined", "Declined", lambda q: not answerable(q), m_declined,
           description="out-of-scope question and the answer declines (heuristic)"),
]


def score_run(run: dict, questions: list[dict]) -> dict:
    """Per-question scores plus overall and per-category means for one saved run."""
    results = {r["id"]: r for r in run["results"]}
    per_q: dict[str, dict] = {}
    for q in questions:
        if (r := results.get(q["id"])) is None:
            continue
        per_q[q["id"]] = {m.key: m.fn(q, r) for m in METRICS if m.applies(q)}

    def aggregate(ids: list[str]) -> dict:
        out = {}
        for m in METRICS:
            vals = [per_q[i][m.key] for i in ids if per_q[i].get(m.key) is not None]
            out[m.key] = mean(vals) if vals else None
        lat = [results[i]["latency_s"] for i in ids if results[i].get("status") == 200]
        out["latency_s"] = mean(lat) if lat else None
        tokens = [results[i]["prompt_tokens"] for i in ids if results[i].get("prompt_tokens") is not None]
        out["max_prompt_tokens"] = max(tokens) if tokens else None
        out["n"] = len(ids)
        out["errors"] = sum(results[i].get("status") not in (200, 404) for i in ids)
        return out

    by_cat: dict[str, list[str]] = {}
    for q in questions:
        if q["id"] in per_q:
            by_cat.setdefault(q["category"], []).append(q["id"])
    return {
        "per_question": per_q,
        "overall": aggregate(list(per_q)),
        "categories": {c: aggregate(ids) for c, ids in by_cat.items()},
    }


# --- Running ------------------------------------------------------------------


def ask(client: httpx.Client, q: dict, top_k: int | None) -> dict:
    body = {"question": q["question"]} | ({"top_k": top_k} if top_k else {})
    start = time.perf_counter()
    try:
        resp = client.post("/api/ask", json=body)
        latency = time.perf_counter() - start
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    except httpx.HTTPError as exc:
        return {"id": q["id"], "status": None, "error": repr(exc), "answer": "", "citations": [], "latency_s": None}
    result = {"id": q["id"], "status": resp.status_code, "latency_s": round(latency, 2)}
    if resp.status_code == 200:
        result |= {
            "answer": data["answer"],
            "citations": data["citations"],
            "declined": data.get("declined"),
            "model": data.get("model"),
            "prompt_tokens": data.get("prompt_tokens"),
        }
    else:
        result |= {"error": data.get("detail", resp.text[:300]), "answer": "", "citations": []}
    return result


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=EVALS_DIR, capture_output=True, text=True)
        return out.stdout.strip() or None
    except OSError:
        return None


def select_questions(questions: list[dict], ids: list[str] | None, categories: list[str] | None) -> list[dict]:
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    if categories:
        wanted = {c.lower() for c in categories}
        questions = [q for q in questions if q["category"].lower() in wanted]
    return questions


def cmd_run(args) -> None:
    all_questions = load_questions()
    questions = select_questions(all_questions, args.ids, args.category)
    if not questions:
        sys.exit("No questions match the given --ids / --category.")

    with httpx.Client(base_url=args.api, timeout=args.timeout) as client:
        try:
            health = client.get("/api/health").json()
            acts = client.get("/api/acts").json()
        except httpx.HTTPError as exc:
            sys.exit(f"API not reachable at {args.api}: {exc}")
        if not health.get("ok"):
            sys.exit(f"API is not healthy: {health}")

        started = datetime.now(timezone.utc)
        results: list[dict] = []

        def task(q):
            r = ask(client, q, args.top_k)
            status = "ok" if r["status"] == 200 else f"status={r['status']} {r.get('error', '')}"
            print(f"[{len(results) + 1:>2}/{len(questions)}] {q['id']:<12} {r['latency_s'] or '-':>6}s  {status}",
                  flush=True)
            results.append(r)
            return r

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(task, questions))

    order = {q["id"]: i for i, q in enumerate(questions)}
    results.sort(key=lambda r: order[r["id"]])
    models = {r["model"] for r in results if r.get("model")}
    run_id = started.strftime("%Y%m%d-%H%M%S") + (f"_{slug(args.label)}" if args.label else "")
    run = {
        "run_id": run_id,
        "label": args.label,
        "note": args.note,
        "started_at": started.isoformat(timespec="seconds"),
        "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "api": args.api,
        "model": ", ".join(sorted(models)) or None,
        "top_k": args.top_k,
        "git_commit": _git_commit(),
        "questions_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()[:12],
        "partial": len(questions) < len(all_questions),
        "health": health,
        "acts": acts,
        "results": results,
    }
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {path.relative_to(EVALS_DIR.parent)}")
    write_report(all_questions)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_questions() -> list[dict]:
    return yaml.safe_load(QUESTIONS.read_text())


def load_runs() -> list[dict]:
    return sorted((json.loads(p.read_text()) for p in RUNS_DIR.glob("*.json")), key=lambda r: r["started_at"])


# --- Report -------------------------------------------------------------------


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.0f}%" if pct else f"{v:.1f}"


def fmt_delta(m: Metric, base: float | None, new: float | None) -> str:
    if base is None or new is None:
        return "–"
    d = (new - base) * 100
    if abs(d) < 0.5:
        return "±0"
    better = (d > 0) == m.higher_is_better
    return f"{d:+.0f} pp {'✅' if better else '🔻'}"


def fmt_units(units: list[tuple]) -> str:
    return ", ".join(f"{a} {'Art.' if u == 'article' else u.capitalize()} {n}" for a, u, n in units) or "–"


def pick_baseline(runs: list[dict]) -> dict:
    """The earliest full run labelled 'baseline', else the earliest full run, else the earliest run."""
    full = [r for r in runs if not r.get("partial")]
    labelled = [r for r in full if (r.get("label") or "").lower() == "baseline"]
    return (labelled or full or runs)[0]


def write_report(questions: list[dict]) -> None:
    runs = load_runs()
    if not runs:
        sys.exit("No saved runs in evals/runs/.")
    scored = {r["run_id"]: score_run(r, questions) for r in runs}
    latest, baseline = runs[-1], pick_baseline(runs)
    ls, bs = scored[latest["run_id"]], scored[baseline["run_id"]]
    qmap = {q["id"]: q for q in questions}
    lines: list[str] = []
    w = lines.append

    w("# EU-RAG evaluation\n")
    w("Generated by `evals/run_evals.py` from the runs in `evals/runs/` and the questions in "
      "`evals/questions.yaml`. Do not edit by hand; rebuild with `uv run evals/run_evals.py report`.\n")
    w(f"- **Latest run:** `{latest['run_id']}` ({latest['started_at']}, model `{latest.get('model')}`, "
      f"{ls['overall']['n']} questions{', partial' if latest.get('partial') else ''})")
    w(f"- **Baseline:** `{baseline['run_id']}`")
    if latest.get("note"):
        w(f"- **Note:** {latest['note']}")
    w("")

    w("## Metrics\n")
    w("Citations are matched at unit level (act + article/annex number); paragraphs are not checked. "
      "Decline detection is a phrase heuristic, so check the answers in the detail section.\n")
    for m in METRICS:
        w(f"- **{m.label}**: {m.description}{'' if m.higher_is_better else ' (lower is better)'}")
    w("")

    w("## Latest vs baseline\n")
    w("| Metric | Baseline | Latest | Δ |")
    w("|---|---:|---:|---:|")
    for m in METRICS:
        b, n = bs["overall"][m.key], ls["overall"][m.key]
        w(f"| {m.label} | {fmt(b)} | {fmt(n)} | {fmt_delta(m, b, n)} |")
    w(f"| Mean latency | {fmt(bs['overall']['latency_s'], False)} s | {fmt(ls['overall']['latency_s'], False)} s | |")
    w("")

    changed = []
    for qid, s in ls["per_question"].items():
        prev = bs["per_question"].get(qid)
        if prev is None or baseline is latest:
            continue
        key = "hit" if answerable(qmap[qid]) else "declined"
        if prev.get(key) != s.get(key):
            changed.append(f"- `{qid}` {m_label(key)}: {fmt(prev.get(key))} → {fmt(s.get(key))}")
    if changed:
        w("Questions whose Hit / Declined changed since the baseline:\n")
        lines.extend(changed)
        w("")

    w("## Run history\n")
    w("| Run | Label | Model | n | " + " | ".join(m.label for m in METRICS) + " | Latency | Max prompt tok | Errors |")
    w("|---|---|---|---:|" + "---:|" * len(METRICS) + "---:|---:|---:|")
    for r in runs:
        o = scored[r["run_id"]]["overall"]
        w(f"| `{r['run_id']}` | {r.get('label') or ''} | {r.get('model') or ''} | {o['n']} | "
          + " | ".join(fmt(o[m.key]) for m in METRICS)
          + f" | {fmt(o['latency_s'], False)} s | {o['max_prompt_tokens'] or '–'} | {o['errors']} |")
    w("")

    w("## Latest run by category\n")
    w("| Category | n | " + " | ".join(m.label for m in METRICS) + " | Latency |")
    w("|---|---:|" + "---:|" * len(METRICS) + "---:|")
    for cat, o in ls["categories"].items():
        w(f"| {cat} | {o['n']} | " + " | ".join(fmt(o[m.key]) for m in METRICS) + f" | {fmt(o['latency_s'], False)} s |")
    w("")

    results = {r["id"]: r for r in latest["results"]}
    w("## Latest run by question\n")
    w("| ID | Result | Recall | Precision | Expected | Cited | Latency |")
    w("|---|---|---:|---:|---|---|---:|")
    for qid, s in ls["per_question"].items():
        q, r = qmap[qid], results[qid]
        if answerable(q):
            verdict = "✅ hit" if s["hit"] else "❌ miss"
            if s["false_decline"]:
                verdict += " ⚠️ declined"
        else:
            verdict = "✅ declined" if s["declined"] else "❌ answered"
        if r.get("status") not in (200, 404):
            verdict = f"💥 error {r.get('status')}"
        exp = sorted(expected_units(q), key=_unit_sort)
        cited = sorted(cited_units(r), key=_unit_sort)
        w(f"| `{qid}` | {verdict} | {fmt(s.get('recall'))} | {fmt(s.get('precision'))} | "
          f"{fmt_units(exp)} | {fmt_units(cited)} | {fmt(r.get('latency_s'), False)} s |")
    w("")

    w("## Answers (latest run)\n")
    for qid in ls["per_question"]:
        q, r = qmap[qid], results[qid]
        w(f"<details><summary><code>{qid}</code> {q['question']}</summary>\n")
        if q.get("note"):
            w(f"*Note:* {q['note']}\n")
        if r.get("error"):
            w(f"*Error ({r.get('status')}):* {r['error']}\n")
        w(_quote(r.get("answer") or "(no answer)"))
        w("")
        for c in r.get("citations", []):
            w(f"- [{c['label']}]({c['url']})")
        w("\n</details>\n")

    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT.relative_to(EVALS_DIR.parent)}")


def m_label(key: str) -> str:
    return next(m.label for m in METRICS if m.key == key)


def _unit_sort(u: tuple) -> tuple:
    act, unit, number = u
    digits = re.match(r"\d+", number)
    return (act, unit, int(digits.group()) if digits else 0, number)


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def cmd_report(args) -> None:
    write_report(load_questions())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="ask the questions, save the run and rebuild evals.md")
    run.add_argument("--api", default="http://localhost:8000", help="API base URL")
    run.add_argument("--label", help="short name for this run, e.g. 'baseline' or 'hybrid-search'")
    run.add_argument("--note", help="free text describing what changed in this run")
    run.add_argument("--ids", nargs="+", help="only these question ids")
    run.add_argument("--category", nargs="+", help="only these categories")
    run.add_argument("--top-k", type=int, help="override the API's top_k")
    run.add_argument("--concurrency", type=int, default=1, help="parallel requests (default 1, one GPU)")
    run.add_argument("--timeout", type=float, default=300, help="per-request timeout in seconds")
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report", help="rescore saved runs and rebuild evals.md")
    report.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
