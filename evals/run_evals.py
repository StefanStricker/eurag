# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pyyaml>=6.0"]
# ///
"""Evaluate EU-RAG against evals/questions.yaml. Everything is stored in one JSON file per run.

    uv run evals/run_evals.py check                       # validate questions.yaml against the index
    uv run evals/run_evals.py run --label baseline        # ask all questions, save evals/runs/<id>.json, score it
    uv run evals/run_evals.py run --ids gdpr-01 --label x # subset (--ids / --category)
    uv run evals/run_evals.py run --retrieval-only --label e-qwen3  # retrieval metrics only, no LLM (seconds)
    uv run evals/run_evals.py judge baseline              # local LLM grades the answers (resumable)
    uv run evals/run_evals.py score hybrid --vs baseline  # rescore and compare two runs

`run` stores the raw answers, citations and the retrieval trace (`debug` of /api/ask). `judge` adds a
`judge` field per result, `score` (re)writes the `scores` section; both edit the run file in place.
Scores are recomputed from the raw results with the current questions.yaml, so fixing an expectation
or adding a metric applies to old runs. Runs are referenced by id or any part of it (latest match).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, quantiles

import httpx
import yaml

EVALS_DIR = Path(__file__).parent
ROOT = EVALS_DIR.parent
QUESTIONS = EVALS_DIR / "questions.yaml"
RUNS_DIR = EVALS_DIR / "runs"

# Heuristic for runs before the API's `declined` flag: phrases the model uses when the sources don't cover a question.
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


def retrieved_units(r: dict, in_context_only: bool = False) -> list[tuple] | None:
    """Retrieved units in rank order (first occurrence); None for runs without a retrieval trace."""
    if not (debug := r.get("debug")):
        return None
    hits = [h for h in debug["retrieved"] if h["in_context"] or not in_context_only]
    return list(dict.fromkeys(unit_key(h["act"], h["unit_type"], h["number"]) for h in hits))


def is_decline(r: dict) -> bool:
    if r.get("status") == 404:
        return True
    if "declined" in r:  # the API's flag; the heuristic would also catch "say what is not covered" caveats
        return bool(r["declined"])
    return bool(_DECLINE.search(r.get("answer", "")))


def covers_paragraph(paragraphs: str, paragraph: str) -> bool:
    """Same as eurag.retrieval.covers_paragraph: whether a chunk's range ("1-3", "4", "" = whole unit) includes it."""
    if not paragraphs:
        return True
    lo, _, hi = paragraphs.partition("-")
    return int(lo) <= int(paragraph) <= int(hi or lo)


def answerable(q: dict) -> bool:
    return not q.get("expect_decline")


def has_paragraphs(q: dict) -> bool:
    return answerable(q) and any(e.get("paragraph") and not e.get("optional") for e in q["expected"])


def _recall(q: dict, units) -> float | None:
    if units is None:
        return None
    exp = expected_units(q, required_only=True)
    return len(exp & set(units)) / len(exp)


def m_hit(q, r):
    return float(bool(expected_units(q) & cited_units(r)))


def m_recall(q, r):
    return _recall(q, cited_units(r))


def m_precision(q, r):
    # Recitals are excluded: citing the recital that explains an expected article is not wrong.
    cited = {u for u in cited_units(r) if u[1] != "recital"}
    return len(cited & expected_units(q)) / len(cited) if cited else None


def m_paragraph_recall(q, r):
    exp = [e for e in q["expected"] if e.get("paragraph") and not e.get("optional")]
    covered = 0
    for e in exp:
        key = unit_key(e["act"], e["unit"], e["number"])
        covered += any(
            unit_key(c["act"], c["unit_type"], c["number"]) == key and covers_paragraph(c["paragraphs"], e["paragraph"])
            for c in r.get("citations", [])
        )
    return covered / len(exp)


def m_retrieved_recall(q, r):
    return _recall(q, retrieved_units(r))


def m_context_recall(q, r):
    return _recall(q, retrieved_units(r, in_context_only=True))


def m_mrr(q, r):
    if (units := retrieved_units(r)) is None:
        return None
    exp = expected_units(q)
    return next((1 / rank for rank, u in enumerate(units, start=1) if u in exp), 0.0)


def m_decline(q, r):
    return float(is_decline(r))


_CORRECTNESS = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def m_correct(q, r):
    return _CORRECTNESS.get((r.get("judge") or {}).get("correctness"))


def m_faithful(q, r):
    faithful = (r.get("judge") or {}).get("faithful")
    return None if faithful is None else float(faithful)


@dataclass(frozen=True)
class Metric:
    key: str
    applies: Callable[[dict], bool]
    fn: Callable[[dict, dict], float | None]
    description: str
    higher_is_better: bool = True


# Add new metrics here; `score` applies them to every run.
METRICS = [
    Metric("retrieved_recall", answerable, m_retrieved_recall, "required units among all retrieved chunks"),
    Metric("context_recall", answerable, m_context_recall, "required units among the chunks that fit the prompt"),
    Metric("mrr", answerable, m_mrr, "1 / rank of the first expected unit in the retrieval order"),
    Metric("hit", answerable, m_hit, "at least one expected unit is cited"),
    Metric("recall", answerable, m_recall, "share of required expected units cited"),
    Metric("precision", answerable, m_precision, "share of cited articles/annexes that are expected (recitals ignored)"),
    Metric("paragraph_recall", has_paragraphs, m_paragraph_recall, "expected paragraphs covered by a citation"),
    Metric("false_decline", answerable, m_decline, "answerable question but the answer declines", False),
    Metric("declined", lambda q: not answerable(q), m_decline, "out-of-scope question and the answer declines"),
    Metric("correct", answerable, m_correct, "judge: correct = 1, partial = 0.5, wrong = 0"),
    Metric("faithful", answerable, m_faithful, "judge: every claim is supported by the cited sources"),
]
RETRIEVAL_METRICS = {"retrieved_recall", "context_recall", "mrr"}  # the only ones a retrieval-only run has


def scored(m: Metric, q: dict, r: dict) -> bool:
    return m.applies(q) and (m.key in RETRIEVAL_METRICS or not r.get("retrieval_only"))


def outcome(q: dict, r: dict) -> str:
    """ok, or where the answer went wrong: error / declined / answered / retrieval_miss / budget_cut / generation_miss.

    Retrieval-only results stop at the prompt: ok / retrieval_miss / budget_cut, n/a for out-of-scope questions."""
    if r.get("status") not in (200, 404):
        return "error"
    if r.get("retrieval_only"):
        if not answerable(q):
            return "n/a"
        missing = expected_units(q, required_only=True)
        if missing - set(retrieved_units(r) or []):
            return "retrieval_miss"
        if missing - set(retrieved_units(r, in_context_only=True) or []):
            return "budget_cut"
        return "ok"
    if not answerable(q):
        return "ok" if is_decline(r) else "answered"
    missing = expected_units(q, required_only=True) - cited_units(r)
    if not missing:
        return "ok"
    if is_decline(r):
        return "declined"
    if (retrieved := retrieved_units(r)) is None:
        return "miss"  # no retrieval trace (old runs)
    if missing - set(retrieved):
        return "retrieval_miss"
    if missing - set(retrieved_units(r, in_context_only=True)):
        return "budget_cut"
    return "generation_miss"


def score_run(run: dict, questions: list[dict]) -> dict:
    """Per-question scores, overall and per-category means for one run."""
    results = {r["id"]: r for r in run["results"]}
    asked = [q for q in questions if q["id"] in results]
    per_q = {
        q["id"]: {m.key: m.fn(q, results[q["id"]]) for m in METRICS if scored(m, q, results[q["id"]])}
        | {"outcome": outcome(q, results[q["id"]])}
        for q in asked
    }

    def aggregate(ids: list[str]) -> dict:
        out: dict = {"n": len(ids)}
        for m in METRICS:
            vals = [per_q[i][m.key] for i in ids if per_q[i].get(m.key) is not None]
            out[m.key] = round(mean(vals), 4) if vals else None
        lat = [results[i]["latency_s"] for i in ids if results[i].get("status") == 200]
        out["latency_mean_s"] = round(mean(lat), 2) if lat else None
        out["latency_p95_s"] = round(quantiles(lat, n=20, method="inclusive")[-1], 2) if len(lat) >= 2 else None
        out["outcomes"] = dict(Counter(per_q[i]["outcome"] for i in ids).most_common())
        return out

    by_cat: dict[str, list[str]] = {}
    for q in asked:
        by_cat.setdefault(q["category"], []).append(q["id"])
    # Cross-cutting slice: annexes are rarely retrieved and spread over all categories.
    by_cat["annex questions"] = [q["id"] for q in asked if answerable(q) and any(e["unit"] == "annex"
                                                                                 for e in q["expected"])]
    return {
        "scored_at": now(),
        "questions_sha256": questions_sha(),
        "overall": aggregate(list(per_q)),
        "by_category": {c: aggregate(ids) for c, ids in by_cat.items()},
        "per_question": per_q,
    }


def fmt(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def print_scores(run: dict, s: dict, base: dict | None = None, b: dict | None = None) -> None:
    """Print scores `s` of `run`; with a base run, `s` and `b` must be scored on the same questions."""
    print(f"\n{run['run_id']}  ({s['overall']['n']} questions)" + (f"  vs  {base['run_id']}" if base else ""))
    print(f"{'metric':<18}{'base':>8}{'run':>8}{'Δ pp':>8}" if b else f"{'metric':<18}{'run':>8}")
    for m in METRICS:
        new = s["overall"][m.key]
        if not b:
            print(f"{m.key:<18}{fmt(new):>8}")
            continue
        old = b["overall"][m.key]
        delta = f"{(new - old) * 100:+.0f}" if None not in (new, old) else "–"
        print(f"{m.key:<18}{fmt(old):>8}{fmt(new):>8}{delta:>8}")
    print(f"{'latency mean/p95':<18}{s['overall']['latency_mean_s']} s / {s['overall']['latency_p95_s']} s")
    print("outcomes: " + ", ".join(f"{k} {v}" for k, v in s["overall"]["outcomes"].items()))
    print("\n" + f"{'category':<22}{'n':>4}{'ctx rec':>9}{'hit':>6}{'recall':>8}{'correct':>9}")
    for cat, o in s["by_category"].items():
        print(f"{cat:<22}{o['n']:>4}{fmt(o['context_recall']):>9}{fmt(o['hit']):>6}{fmt(o['recall']):>8}"
              f"{fmt(o['correct']):>9}")
    if b:
        changed = [(qid, b["per_question"][qid]["outcome"], p["outcome"]) for qid, p in s["per_question"].items()
                   if qid in b["per_question"] and b["per_question"][qid]["outcome"] != p["outcome"]]
        fixed = [c for c in changed if c[2] == "ok"]
        broken = [c for c in changed if c[1] == "ok"]
        print(f"\nfixed {len(fixed)}, broken {len(broken)}, other changes {len(changed) - len(fixed) - len(broken)}")
        for qid, old, new in changed:
            print(f"  {qid:<14}{old} → {new}")


# --- Runs ---------------------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def questions_sha() -> str:
    return hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()[:12]


def load_questions() -> list[dict]:
    return yaml.safe_load(QUESTIONS.read_text())


def find_run(ref: str) -> Path:
    """The run file whose id equals `ref`, else the latest one containing it."""
    paths = sorted(RUNS_DIR.glob("*.json"))
    exact = [p for p in paths if p.stem == ref]
    matches = exact or [p for p in paths if ref in p.stem]
    if not matches:
        sys.exit(f"No run matches {ref!r} in {RUNS_DIR.relative_to(ROOT)}.")
    return matches[-1]


def save_run(path: Path, run: dict) -> None:
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")


def env_port(name: str, default: int) -> int:
    """A host port from the repo's .env (the compose port mappings), else the default."""
    env = ROOT / ".env"
    text = env.read_text() if env.exists() else ""
    m = re.search(rf"^{name}=(\d+)", text, re.M)
    return int(m.group(1)) if m else default


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def ask(client: httpx.Client, q: dict, top_k: int | None, retrieval_only: bool = False) -> dict:
    body = {"question": q["question"], "debug": True} | ({"top_k": top_k} if top_k else {})
    start = time.perf_counter()
    try:
        resp = client.post("/api/retrieve" if retrieval_only else "/api/ask", json=body)
        latency = time.perf_counter() - start
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    except httpx.HTTPError as exc:
        return {"id": q["id"], "status": None, "error": repr(exc), "answer": "", "citations": [], "latency_s": None}
    result = {"id": q["id"], "status": resp.status_code, "latency_s": round(latency, 2)}
    if retrieval_only:
        result["retrieval_only"] = True
    if resp.status_code == 200 and retrieval_only:
        result["debug"] = data.get("debug")
    elif resp.status_code == 200:
        result |= {k: data.get(k) for k in ("answer", "citations", "declined", "prompt_tokens", "debug")}
    else:
        result |= {"error": data.get("detail", resp.text[:300]), "answer": "", "citations": []}
    return result


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
            config = client.get("/api/config").json()
        except httpx.HTTPError as exc:
            sys.exit(f"API not reachable at {args.api}: {exc}")
        if not health.get("ok"):
            sys.exit(f"API is not healthy: {health}")

        started = datetime.now(timezone.utc)
        results: list[dict] = []

        def task(q):
            r = ask(client, q, args.top_k, args.retrieval_only)
            results.append(r)
            status = "ok" if r["status"] == 200 else f"status={r['status']} {r.get('error', '')}"
            print(f"[{len(results):>3}/{len(questions)}] {q['id']:<14} {r['latency_s'] or '-':>6}s  {status}", flush=True)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(task, questions))

    order = {q["id"]: i for i, q in enumerate(questions)}
    results.sort(key=lambda r: order[r["id"]])
    run_id = started.strftime("%Y%m%d-%H%M%S") + (f"_{re.sub(r'[^a-z0-9]+', '-', args.label.lower()).strip('-')}"
                                                  if args.label else "")
    run = {
        "run_id": run_id,
        "label": args.label,
        "note": args.note,
        "started_at": started.isoformat(timespec="seconds"),
        "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "api": args.api,
        "top_k": args.top_k,
        "retrieval_only": args.retrieval_only,
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "app", "data")),
        "questions_sha256": questions_sha(),
        "partial": len(questions) < len(all_questions),
        "health": health,
        "config": config,
        "results": results,
    }
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{run_id}.json"
    run["scores"] = score_run(run, all_questions)
    save_run(path, run)
    print(f"Saved {path.relative_to(ROOT)}")
    print_scores(run, run["scores"])


def cmd_score(args) -> None:
    questions = load_questions()
    path = find_run(args.run)
    run = json.loads(path.read_text())
    run["scores"] = score_run(run, questions)
    save_run(path, run)
    if not args.vs:
        print_scores(run, run["scores"])
        return
    base_path = find_run(args.vs)
    base = json.loads(base_path.read_text())
    base["scores"] = score_run(base, questions)
    save_run(base_path, base)
    # Paired: both runs scored on the questions they both answered.
    ids = {r["id"] for r in run["results"]} & {r["id"] for r in base["results"]}
    common = [q for q in questions if q["id"] in ids]
    print_scores(run, score_run(run, common), base, score_run(base, common))


# --- Judge --------------------------------------------------------------------

JUDGE_PROMPT = """You grade answers of a legal research assistant for EU data law.
You get a question, the provisions a correct answer should rely on, the assistant's answer and the full \
text of the sources the answer cites. Judge only against those source texts, not outside knowledge.

- correctness: "correct" if the answer addresses the question accurately and covers its main points; \
"partial" if it is right but incomplete or partly off; "wrong" if it is incorrect, off-topic or relies on \
the wrong provisions.
- faithful: true if every claim in the answer is supported by the cited source texts, false otherwise.

Reply with JSON only: {"correctness": "correct" | "partial" | "wrong", "faithful": true | false, \
"reason": "<one sentence>"}"""


def judge_messages(q: dict, r: dict) -> list[dict]:
    expected = ", ".join(f"{e['act']} {e['unit']} {e['number']}" + (f"({e['paragraph']})" if e.get("paragraph") else "")
                         for e in q["expected"] if not e.get("optional"))
    sources = "\n\n".join(f"[{s['id']}] {s['text']}" for s in r["debug"]["sources"]) or "(no sources cited)"
    user = (f"Question: {q['question']}\n\nExpected provisions: {expected}\n\nAnswer:\n{r['answer']}\n\n"
            f"Cited sources:\n{sources}")
    return [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}]


def judge_one(client: httpx.Client, model: str, q: dict, r: dict) -> dict:
    if is_decline(r):
        return {"correctness": "wrong", "faithful": None, "reason": "declined an answerable question"}
    body = {"model": model, "messages": judge_messages(q, r), "stream": False, "format": "json",
            "options": {"temperature": 0, "num_ctx": 16384}}
    resp = client.post("/api/chat", json=body)
    resp.raise_for_status()
    verdict = json.loads(resp.json()["message"]["content"])
    if verdict.get("correctness") not in _CORRECTNESS or not isinstance(verdict.get("faithful"), bool):
        raise ValueError(f"unexpected judge reply: {verdict}")
    return {k: verdict[k] for k in ("correctness", "faithful")} | {"reason": str(verdict.get("reason", ""))}


def cmd_judge(args) -> None:
    questions = load_questions()
    qmap = {q["id"]: q for q in questions}
    path = find_run(args.run)
    run = json.loads(path.read_text())
    if run.get("retrieval_only"):
        sys.exit(f"{path.stem} is a retrieval-only run, there are no answers to judge.")
    todo = [r for r in run["results"]
            if r["id"] in qmap and answerable(qmap[r["id"]]) and r.get("status") == 200 and "judge" not in r]
    if any(not r.get("debug") for r in todo):
        sys.exit(f"{path.stem} has no retrieval trace (cited source texts); re-run it to judge it.")
    print(f"Judging {len(todo)} answers of {path.stem} with {args.model}")
    with httpx.Client(base_url=args.ollama, timeout=args.timeout) as client:
        names = {m["name"] for m in client.get("/api/tags").json()["models"]}
        if args.model not in names and f"{args.model}:latest" not in names:
            sys.exit(f"Judge model {args.model} not pulled: docker compose exec ollama ollama pull {args.model}")
        try:
            for i, r in enumerate(todo, start=1):
                try:
                    r["judge"] = judge_one(client, args.model, qmap[r["id"]], r)
                    print(f"[{i:>3}/{len(todo)}] {r['id']:<14} {r['judge']['correctness']:<8} "
                          f"faithful={r['judge']['faithful']}", flush=True)
                except (httpx.HTTPError, ValueError) as exc:  # left unjudged; the next `judge` retries it
                    print(f"[{i:>3}/{len(todo)}] {r['id']:<14} failed: {exc}", flush=True)
        finally:  # keep finished verdicts on Ctrl-C
            run["judge_model"] = args.model
            run["scores"] = score_run(run, questions)
            save_run(path, run)
    print_scores(run, run["scores"])


# --- Check --------------------------------------------------------------------


def index_units(qdrant: str, collection: str) -> set[tuple]:
    units: set[tuple] = set()
    offset = None
    with httpx.Client(base_url=qdrant, timeout=60) as client:
        while True:
            body = {"limit": 1000, "with_payload": ["act", "unit_type", "number"], "with_vector": False}
            resp = client.post(f"/collections/{collection}/points/scroll", json=body | ({"offset": offset} if offset else {}))
            resp.raise_for_status()
            page = resp.json()["result"]
            units |= {unit_key(p["payload"]["act"], p["payload"]["unit_type"], p["payload"]["number"])
                      for p in page["points"]}
            if (offset := page.get("next_page_offset")) is None:
                return units


def cmd_check(args) -> None:
    questions = load_questions()
    try:
        units = index_units(args.qdrant, args.collection)
    except httpx.HTTPError as exc:
        sys.exit(f"Qdrant not reachable at {args.qdrant}: {exc}")
    acts = {u[0] for u in units}
    problems: list[str] = []
    for qid, n in Counter(q.get("id") for q in questions).items():
        if n > 1:
            problems.append(f"{qid}: duplicate id ({n}x)")
    for q in questions:
        qid = q.get("id", "?")
        problems += [f"{qid}: missing {f!r}" for f in ("id", "category", "question") if not q.get(f)]
        if answerable(q) and not q.get("expected"):
            problems.append(f"{qid}: no expected units and not expect_decline")
        for e in q.get("expected", []):
            if e.get("act") not in acts:
                problems.append(f"{qid}: unknown act {e.get('act')!r}")
            elif unit_key(e["act"], e["unit"], e["number"]) not in units:
                problems.append(f"{qid}: {e['act']} {e['unit']} {e['number']} not in the index")
            if e.get("paragraph") and not str(e["paragraph"]).isdigit():
                problems.append(f"{qid}: paragraph {e['paragraph']!r} is not a number")
    print(f"{len(questions)} questions ({sum(not answerable(q) for q in questions)} expect a decline), "
          f"{len(units)} units in the index")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate questions.yaml against the index")
    check.add_argument("--qdrant", default=f"http://localhost:{env_port('QDRANT_PORT', 6333)}", help="Qdrant URL")
    check.add_argument("--collection", default="eu_acts", help="collection or alias")
    check.set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="ask the questions, save and score the run")
    run.add_argument("--api", default=f"http://localhost:{env_port('API_PORT', 8000)}", help="API base URL")
    run.add_argument("--label", help="short name for this run, e.g. 'baseline' or 'hybrid'")
    run.add_argument("--note", help="free text describing what changed in this run")
    run.add_argument("--ids", nargs="+", help="only these question ids")
    run.add_argument("--category", nargs="+", help="only these categories")
    run.add_argument("--top-k", type=int, help="override the API's top_k")
    run.add_argument("--retrieval-only", action="store_true", help="only retrieve (/api/retrieve), no LLM answers")
    run.add_argument("--concurrency", type=int, default=1, help="parallel requests (default 1, one GPU)")
    run.add_argument("--timeout", type=float, default=300, help="per-request timeout in seconds")
    run.set_defaults(func=cmd_run)

    judge = sub.add_parser("judge", help="grade a run's answers with a local LLM")
    judge.add_argument("run", help="run id or part of it")
    judge.add_argument("--model", default="mistral-nemo:12b", help="Ollama judge model")
    judge.add_argument("--ollama", default=f"http://localhost:{env_port('OLLAMA_PORT', 11434)}", help="Ollama URL")
    judge.add_argument("--timeout", type=float, default=300, help="per-request timeout in seconds")
    judge.set_defaults(func=cmd_judge)

    score = sub.add_parser("score", help="rescore a run, optionally compare it with another")
    score.add_argument("run", help="run id or part of it")
    score.add_argument("--vs", help="run to compare against, e.g. baseline")
    score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
