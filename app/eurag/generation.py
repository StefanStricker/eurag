"""Prompt construction and citation handling."""

from __future__ import annotations

import re

from eurag.config import Settings
from eurag.retrieval import Hit

SYSTEM_PROMPT = """You are a legal research assistant for European Union data law \
(GDPR, AI Act, Data Act, Data Governance Act, ePrivacy Directive).

Rules:
- Answer ONLY using the numbered sources provided. Do not use outside knowledge.
- Cite every statement with the source number in square brackets at the end of the sentence, \
e.g. "... [1]." or "... [2][3]." Cite each source at most once per sentence.
- Articles are binding law; recitals only explain them. Base your answer on articles where \
possible and use recitals for context.
- Name the act and article (e.g. "Article 6(1)(a) GDPR") when you rely on it.
- If the sources do not contain the answer, say so clearly instead of guessing.
- Be precise and concise; use short paragraphs or bullet points.
- Answer in the language of the question."""

_CITATION = re.compile(r"\[(\d+)\]")
_CITATION_RUN = re.compile(r"\[\d+\](?:\s*\[\d+\])*")
_CITATION_ONLY_LINE = re.compile(r"^\s*(?:\[\d+\]\s*)+$\n?", re.M)


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] {h.payload['text']}")
    return "\n\n".join(blocks)


_TEMPLATE_OVERHEAD_CHARS = 200  # chat template, "Sources:" / "Question:" labels
_SOURCE_OVERHEAD_CHARS = 8  # "[n] " marker and blank line per source


def fit_context(question: str, hits: list[Hit], settings: Settings) -> list[Hit]:
    """Keep hits, in priority order, while the prompt fits the context window minus the answer reserve.

    A hit that doesn't fit is skipped rather than ending the walk, so smaller later chunks can still be
    added. The first hit is always kept.
    """
    budget = (settings.num_ctx - settings.answer_tokens) * settings.chars_per_token
    budget -= len(SYSTEM_PROMPT) + len(question) + _TEMPLATE_OVERHEAD_CHARS
    kept: list[Hit] = []
    for h in hits:
        size = len(h.payload["text"]) + _SOURCE_OVERHEAD_CHARS
        if kept and size > budget:
            continue
        kept.append(h)
        budget -= size
    return kept


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    user = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _dedupe_run(m: re.Match) -> str:
    """Drop repeated markers within a citation run, e.g. [2][1][2] -> [2][1]."""
    return "".join(f"[{n}]" for n in dict.fromkeys(_CITATION.findall(m.group(0))))


def clean_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """Remove citation markers pointing at non-existent sources.

    Returns the cleaned answer and the sorted list of valid cited source numbers.
    """
    cited: set[int] = set()

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            cited.add(n)
            return m.group(0)
        return ""

    answer = _CITATION.sub(repl, answer)
    answer = _CITATION_RUN.sub(_dedupe_run, answer)
    answer = _CITATION_ONLY_LINE.sub("", answer).strip()
    return answer, sorted(cited)


def citation_payload(n: int, hit: Hit) -> dict:
    p = hit.payload
    body = p["text"].split("\n", 1)[-1]
    return {
        "id": n,
        "label": p["label"],
        "act": p["act"],
        "act_title": p["act_title"],
        "unit_type": p["unit_type"],
        "number": p["number"],
        "title": p["title"],
        "chapter": p["chapter"],
        "paragraphs": p["paragraphs"],
        "page": p["page_start"],
        "url": p["url"],
        "score": round(hit.score, 4),
        "snippet": body[:400] + ("…" if len(body) > 400 else ""),
    }


def build_citations(hits: list[Hit], cited: list[int]) -> list[dict]:
    """Citations for the sources the answer actually used (all retrieved if none were cited)."""
    ids = cited or list(range(1, len(hits) + 1))
    return [citation_payload(n, hits[n - 1]) for n in ids]
