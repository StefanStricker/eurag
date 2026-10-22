"""Vector search over the indexed acts, plus direct lookup of explicitly cited articles."""

from __future__ import annotations

import re
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from eurag.config import Settings
from eurag.ollama_client import OllamaClient

ARTICLE_BONUS = 0.02
EXPLICIT_SCORE = 1.0
EXPLICIT_MAX_PARTS = 3  # parts of an explicitly named article kept when no paragraph is given
# Recitals explain the binding articles but rarely share their wording. For the top-ranked
# recitals, the articles of the same act closest to the recital's vector are added as well.
RECITAL_EXPANSION_DEPTH = 4  # consider recitals among the first N hits
RECITAL_EXPANSION_ARTICLES = 2  # articles added per recital
RECITAL_EXPANSION_MAX = 4  # max extra chunks beyond top_k

# Phrases users write for each act (short name as stored in the payload -> regex).
ACT_ALIASES = {
    "GDPR": r"\bgdpr\b|general data protection regulation|2016/679",
    "AI Act": r"\bai act\b|artificial intelligence act|2024/1689",
    "Data Act": r"(?<!governance )\bdata act\b|2023/2854",
    "Data Governance Act": r"\bdga\b|data governance act|2022/868",
    "ePrivacy Directive": r"e-?privacy|2002/58",
}
_ARTICLE_REF = re.compile(r"\bart(?:icle|\.)?\s*(\d+[a-z]?)\b(?:\s*\((\d+)\))?", re.I)


@dataclass
class Hit:
    score: float
    payload: dict
    id: str | None = None

    @property
    def key(self) -> tuple:
        p = self.payload
        return (p["celex"], p["unit_type"], p["number"], p["part"])


def mentioned_acts(question: str) -> list[str]:
    q = question.lower()
    return [act for act, pattern in ACT_ALIASES.items() if re.search(pattern, q)]


def explicit_article_refs(question: str, acts: list[str] | None) -> list[tuple[str, str, str | None]]:
    """(act, article number, paragraph or None) triples the question references unambiguously."""
    refs = _ARTICLE_REF.findall(question)
    if not refs:
        return []
    named = mentioned_acts(question) or (acts or [])
    if len(named) != 1:
        return []
    return [(named[0], number, para or None) for number, para in dict.fromkeys(refs)]


def covers_paragraph(paragraphs: str, paragraph: str) -> bool:
    """Whether a chunk's paragraph range ("1-3", "4", or "" for a whole article) includes `paragraph`."""
    if not paragraphs:
        return True
    lo, _, hi = paragraphs.partition("-")
    return int(lo) <= int(paragraph) <= int(hi or lo)


def _act_filter(acts: list[str] | None) -> models.Filter | None:
    if not acts:
        return None
    return models.Filter(must=[models.FieldCondition(key="act", match=models.MatchAny(any=acts))])


class Retriever:
    def __init__(self, settings: Settings, qdrant: AsyncQdrantClient, ollama: OllamaClient):
        self.settings = settings
        self.qdrant = qdrant
        self.ollama = ollama

    async def _lookup_article(self, act: str, number: str, paragraph: str | None, vector: list[float]) -> list[Hit]:
        """Parts of an explicitly named article: those covering `paragraph`, else the ones closest to the question."""
        article = models.Filter(
            must=[
                models.FieldCondition(key="act", match=models.MatchValue(value=act)),
                models.FieldCondition(key="unit_type", match=models.MatchValue(value="article")),
                models.FieldCondition(key="number", match=models.MatchValue(value=number)),
            ]
        )
        points = []
        if paragraph:
            points, _ = await self.qdrant.scroll(self.settings.collection, scroll_filter=article, limit=50)
            points = [p for p in points if covers_paragraph(p.payload["paragraphs"], paragraph)]
        if not points:
            result = await self.qdrant.query_points(
                self.settings.collection, query=vector, query_filter=article, limit=EXPLICIT_MAX_PARTS, with_payload=True
            )
            points = result.points
        points.sort(key=lambda p: p.payload["part"])
        return [Hit(EXPLICIT_SCORE, p.payload) for p in points]

    async def _articles_for_recital(self, recital: Hit) -> list[Hit]:
        result = await self.qdrant.query_points(
            self.settings.collection,
            query=recital.id,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="act", match=models.MatchValue(value=recital.payload["act"])),
                    models.FieldCondition(key="unit_type", match=models.MatchValue(value="article")),
                ]
            ),
            limit=RECITAL_EXPANSION_ARTICLES,
            with_payload=True,
        )
        return [Hit(recital.score * p.score, p.payload, str(p.id)) for p in result.points]

    async def search(self, question: str, acts: list[str] | None = None, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.settings.top_k
        [vector] = await self.ollama.embed([question])
        result = await self.qdrant.query_points(
            self.settings.collection,
            query=vector,
            query_filter=_act_filter(acts),
            limit=top_k * 2,
            with_payload=True,
        )
        hits = [
            Hit(p.score + (ARTICLE_BONUS if p.payload["unit_type"] == "article" else 0.0), p.payload, str(p.id))
            for p in result.points
        ]
        hits.sort(key=lambda h: h.score, reverse=True)

        explicit: list[Hit] = []
        for act, number, paragraph in explicit_article_refs(question, acts):
            explicit += await self._lookup_article(act, number, paragraph, vector)

        # Articles found via a recital are placed right after it, so the model reads them together.
        expansions: dict[tuple, list[Hit]] = {}
        budget = RECITAL_EXPANSION_MAX
        for recital in [h for h in hits[:RECITAL_EXPANSION_DEPTH] if h.payload["unit_type"] == "recital"]:
            related = await self._articles_for_recital(recital)
            expansions[recital.key] = related[: max(budget, 0)]
            budget -= len(related)

        merged: dict[tuple, Hit] = {}
        for h in explicit + hits[:top_k]:
            merged.setdefault(h.key, h)
            for related in expansions.get(h.key, []):
                merged.setdefault(related.key, related)
        return list(merged.values())
