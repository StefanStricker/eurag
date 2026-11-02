"""Dense or hybrid (dense + BM25, RRF) search over the indexed acts, plus direct lookup of explicitly cited articles."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from eurag.config import Settings
from eurag.ollama_client import OllamaClient
from eurag.sparse import DENSE, SPARSE, SparseEncoder

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
# "Article 6(1)", "Articles 13 and 14", "Art. 13, 14 and 15", "Articles 13-15"; "Annex III", "Annexes I to III".
_SEP = r"\s*(?:,|and|or|to|-|–)\s*"
_ARTICLE_ITEM = r"\d+[a-z]?\b(?:\s*\(\d+\))?"
_ANNEX_ITEM = r"[IVXLC]+\b"
_ARTICLE_REF = re.compile(rf"\b(?:articles?|arts?\.?)\s*({_ARTICLE_ITEM}(?:{_SEP}{_ARTICLE_ITEM})*)", re.I)
_ANNEX_REF = re.compile(rf"\bannex(?:es)?\s+({_ANNEX_ITEM}(?:{_SEP}{_ANNEX_ITEM})*)", re.I)
_ITEM = re.compile(r"(\d+[a-z]?|[IVXLC]+\b)(?:\s*\((\d+)\))?|(to|-|–)", re.I)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
MAX_RANGE = 10  # longer ranges ("Articles 1-99") keep only their two ends
MAX_EXPLICIT_REFS = 6  # units fetched directly per question


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


def _roman_to_int(numeral: str) -> int:
    values = [_ROMAN[c] for c in numeral]
    return sum(-v if v < nxt else v for v, nxt in zip(values, values[1:] + [0]))


def _int_to_roman(n: int) -> str:
    out = ""
    numerals = ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    for value, numeral in numerals:
        count, n = divmod(n, value)
        out += numeral * count
    return out


def _expand_items(items: str, unit_type: str) -> list[tuple[str, str | None]]:
    """(number, paragraph or None) pairs of a reference list; ranges between plain numbers are expanded."""
    out: list[tuple[str, str | None]] = []
    is_range = False
    for number, para, range_marker in _ITEM.findall(items):
        if range_marker:
            is_range = True
            continue
        number = number.upper() if unit_type == "annex" else number.lower()
        if is_range and out and out[-1][1] is None and not para:
            to_int = _roman_to_int if unit_type == "annex" else (lambda s: int(s) if s.isdigit() else None)
            lo, hi = to_int(out[-1][0]), to_int(number)
            if lo is not None and hi is not None and 0 < hi - lo <= MAX_RANGE:
                fmt = _int_to_roman if unit_type == "annex" else str
                out += [(fmt(n), None) for n in range(lo + 1, hi)]
        out.append((number, para or None))
        is_range = False
    return out


def explicit_refs(question: str, acts: list[str] | None) -> list[tuple[str, str, str, str | None]]:
    """(act, unit type, number, paragraph or None) tuples the question references unambiguously."""
    matches = [(m.start(), "article", m.group(1)) for m in _ARTICLE_REF.finditer(question)]
    matches += [(m.start(), "annex", m.group(1)) for m in _ANNEX_REF.finditer(question)]
    if not matches:
        return []
    named = mentioned_acts(question) or (acts or [])
    if len(named) != 1:
        return []
    refs = [
        (named[0], unit_type, number, para)
        for _, unit_type, items in sorted(matches)
        for number, para in _expand_items(items, unit_type)
    ]
    return list(dict.fromkeys(refs))[:MAX_EXPLICIT_REFS]


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
        self.sparse = SparseEncoder(settings.sparse_model) if settings.retrieval_mode == "hybrid" else None

    async def _lookup_unit(
        self, act: str, unit_type: str, number: str, paragraph: str | None, vector: list[float]
    ) -> list[Hit]:
        """Parts of a named article or annex: those covering `paragraph`, else the ones closest to the question."""
        unit = models.Filter(
            must=[
                models.FieldCondition(key="act", match=models.MatchValue(value=act)),
                models.FieldCondition(key="unit_type", match=models.MatchValue(value=unit_type)),
                models.FieldCondition(key="number", match=models.MatchValue(value=number)),
            ]
        )
        points = []
        if paragraph:
            points, _ = await self.qdrant.scroll(self.settings.collection, scroll_filter=unit, limit=50)
            points = [p for p in points if covers_paragraph(p.payload["paragraphs"], paragraph)]
        if not points:
            result = await self.qdrant.query_points(
                self.settings.collection,
                query=vector,
                using=DENSE,
                query_filter=unit,
                limit=EXPLICIT_MAX_PARTS,
                with_payload=True,
            )
            points = result.points
        points.sort(key=lambda p: p.payload["part"])
        return [Hit(EXPLICIT_SCORE, p.payload) for p in points]

    async def _articles_for_recital(self, recital: Hit) -> list[Hit]:
        result = await self.qdrant.query_points(
            self.settings.collection,
            query=recital.id,
            using=DENSE,
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

    async def _candidates(self, question: str, vector: list[float], acts: list[str] | None, limit: int) -> list[Hit]:
        """Ranked search hits. Dense: cosine plus the article bonus. Hybrid: RRF of dense and BM25, no bonus
        (RRF scores are rank-based, ~1/(rrf_k + rank), so a cosine-scale bonus would override the fusion)."""
        flt = _act_filter(acts)
        if self.sparse is None:
            result = await self.qdrant.query_points(
                self.settings.collection, query=vector, using=DENSE, query_filter=flt, limit=limit, with_payload=True
            )
            bonus = self.settings.article_bonus
            hits = [
                Hit(p.score + (bonus if p.payload["unit_type"] == "article" else 0.0), p.payload, str(p.id))
                for p in result.points
            ]
            hits.sort(key=lambda h: h.score, reverse=True)
            return hits

        n = max(self.settings.hybrid_prefetch, limit)
        dense, sparse = await self.qdrant.query_batch_points(
            self.settings.collection,
            requests=[
                models.QueryRequest(query=vector, using=DENSE, filter=flt, limit=n, with_payload=True),
                models.QueryRequest(
                    query=self.sparse.query(question), using=SPARSE, filter=flt, limit=n, with_payload=True
                ),
            ],
        )
        # Weighted RRF, fused here: Qdrant's RRF weights rescale ranks rather than contributions, so a
        # large dense weight flattens the dense ranking and leaves the order to BM25.
        k = self.settings.rrf_k
        fused: dict[str, Hit] = {}
        for weight, result in ((self.settings.hybrid_dense_weight, dense), (1.0, sparse)):
            for rank, p in enumerate(result.points):  # rank from 0, as Qdrant's RRF
                hit = fused.setdefault(str(p.id), Hit(0.0, p.payload, str(p.id)))
                hit.score += weight / (k + rank)
        return sorted(fused.values(), key=lambda h: h.score, reverse=True)[:limit]

    async def search(self, question: str, acts: list[str] | None = None, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.settings.top_k
        [vector] = await self.ollama.embed([question], query=True)
        hits = await self._candidates(question, vector, acts, top_k * 2)

        explicit: list[Hit] = []
        for act, unit_type, number, paragraph in explicit_refs(question, acts):
            explicit += await self._lookup_unit(act, unit_type, number, paragraph, vector)

        # Articles found via a recital are placed right after it, so the model reads them together.
        recitals = [h for h in hits[:RECITAL_EXPANSION_DEPTH] if h.payload["unit_type"] == "recital"]
        found = await asyncio.gather(*(self._articles_for_recital(r) for r in recitals))
        expansions: dict[tuple, list[Hit]] = {}
        budget = RECITAL_EXPANSION_MAX
        for recital, related in zip(recitals, found):  # budget in rank order
            expansions[recital.key] = related[: max(budget, 0)]
            budget -= len(related)

        merged: dict[tuple, Hit] = {}
        for h in explicit + hits[:top_k]:
            merged.setdefault(h.key, h)
            for related in expansions.get(h.key, []):
                merged.setdefault(related.key, related)
        return list(merged.values())
