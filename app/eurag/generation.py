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
- If the sources do not answer the question, start your reply with NOT_COVERED, then say in one or \
two sentences what is missing. Cite nothing in that case. If they answer only part of it, answer that \
part normally and say what is not covered.
- Be precise and concise; use short paragraphs or bullet points.
- Answer in the language of the question."""

DECLINE_MARKER = "NOT_COVERED"
DECLINE_FALLBACK = "The indexed acts do not cover this question."

_DECLINE = re.compile(rf"^[\s*_]*{DECLINE_MARKER}[\s*_]*[:.\-–]?\s*", re.I)
_CITATION = re.compile(r"\[(\d+)\]")
_CITATION_RUN = re.compile(r"\[\d+\](?:\s*\[\d+\])*")
_CITATION_ONLY_LINE = re.compile(r"^\s*(?:\[\d+\]\s*)+$\n?", re.M)
_DOUBLE_SPACE = re.compile(r"(?<=\S)[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+(?=[.,;:!?])")


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


def clean_citations(answer: str, n_sources: int) -> tuple[str, list[int], bool]:
    """Remove the decline marker and citation markers pointing at non-existent sources.

    Returns the cleaned answer, the sorted list of valid cited source numbers and whether the model
    declined. A declined answer keeps no citations.
    """
    answer, n = _DECLINE.subn("", answer, count=1)
    declined = bool(n)
    cited: set[int] = set()

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_sources and not declined:
            cited.add(n)
            return m.group(0)
        return ""

    answer = _CITATION.sub(repl, answer)
    answer = _CITATION_RUN.sub(_dedupe_run, answer)
    answer = _CITATION_ONLY_LINE.sub("", answer)
    answer = _SPACE_BEFORE_PUNCT.sub("", _DOUBLE_SPACE.sub(" ", answer)).strip()
    if declined and not answer:
        answer = DECLINE_FALLBACK
    return answer, sorted(cited), declined


class DeclineFilter:
    """Holds back streamed tokens until it is clear whether the reply starts with the decline marker,
    so the marker itself is never streamed to the client."""

    def __init__(self) -> None:
        self._buffer = ""
        self._decided = False
        self._strip_next = False
        self.declined = False

    def feed(self, token: str) -> str:
        if self._decided:
            if self._strip_next:
                token = token.lstrip(" \t\n*_:.-–")
                self._strip_next = not token
            return token
        self._buffer += token
        lead = self._buffer.lstrip(" \t\n*_").upper()
        if len(lead) < len(DECLINE_MARKER) and DECLINE_MARKER.startswith(lead):
            return ""
        self._decided = True
        if m := _DECLINE.match(self._buffer):
            self.declined = True
            rest = self._buffer[m.end() :]
            self._strip_next = not rest
            return rest
        return self._buffer

    def flush(self) -> str:
        """Text still held back when the stream ends (a reply shorter than the marker)."""
        return "" if self._decided else self._buffer


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
    """Citations for the sources the answer actually used."""
    return [citation_payload(n, hits[n - 1]) for n in cited]
