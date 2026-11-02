from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from eurag import retrieval
from eurag.config import get_settings
from eurag.generation import (
    SYSTEM_PROMPT,
    DeclineFilter,
    build_citations,
    build_messages,
    clean_citations,
    debug_payload,
    fit_context,
)
from eurag.ollama_client import OllamaClient, query_prefix
from eurag.retrieval import Hit, Retriever

STATIC_DIR = Path(__file__).parent / "static"
_CONFIG_SETTINGS = ("llm_model", "embed_model", "embed_dim", "sparse_model", "retrieval_mode", "hybrid_prefetch",
                    "rrf_k", "hybrid_dense_weight", "rerank", "rerank_model", "rerank_candidates",
                    "rerank_article_bonus", "rerank_search_weight", "rerank_gpu_mem_gb", "query_embed_num_gpu",
                    "llm_num_gpu", "top_k", "article_bonus", "num_ctx", "answer_tokens", "temperature",
                    "chars_per_token")
_RETRIEVAL_CONSTANTS = ("EXPLICIT_SCORE", "EXPLICIT_MAX_PARTS", "RECITAL_EXPANSION_DEPTH",
                        "RECITAL_EXPANSION_ARTICLES", "RECITAL_EXPANSION_MAX", "MAX_RANGE", "MAX_EXPLICIT_REFS")
log = logging.getLogger("eurag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    app.state.settings = settings
    app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    app.state.ollama = OllamaClient(settings)
    app.state.retriever = Retriever(settings, app.state.qdrant, app.state.ollama)
    app.state.acts_cache = {}  # physical collection -> /api/acts result
    yield
    await app.state.ollama.aclose()
    await app.state.qdrant.close()


app = FastAPI(title="EU-RAG", description="Question answering over EU data law with citations", lifespan=lifespan)


@app.exception_handler(httpx.HTTPError)
async def llm_unavailable(request: Request, exc: httpx.HTTPError) -> JSONResponse:
    """Only Ollama calls raise raw httpx errors (qdrant-client wraps its own), so this means the LLM failed."""
    log.warning("Ollama request failed: %r", exc)
    return JSONResponse(status_code=502, content={"detail": f"LLM unavailable: {exc!r}"})


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    acts: list[str] | None = Field(default=None, description="Restrict search to these acts (short names)")
    top_k: int | None = Field(default=None, ge=1, le=20)
    debug: bool = Field(default=False, description="Include the retrieval trace and cited source texts (evals)")


class Citation(BaseModel):
    id: int
    label: str
    act: str
    act_title: str
    unit_type: str
    number: str
    title: str
    chapter: str
    paragraphs: str
    page: int
    url: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    declined: bool = Field(description="The model found no answer in the sources; citations are empty")
    model: str
    prompt_tokens: int | None = None
    debug: dict | None = None


async def _retrieve(request: Request, body: AskRequest) -> tuple[list[Hit], list[Hit]]:
    """All retrieved hits and the ones that fit the context budget."""
    hits = await request.app.state.retriever.search(body.question, body.acts, body.top_k)
    if not hits:
        raise HTTPException(404, "No relevant passages found. Has the ingest step run?")
    kept = fit_context(body.question, hits, request.app.state.settings)
    if len(kept) < len(hits):
        log.info("context budget: kept %d of %d retrieved chunks", len(kept), len(hits))
    return hits, kept


@app.post("/api/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    retrieved, hits = await _retrieve(request, body)
    raw, prompt_tokens = await request.app.state.ollama.chat(build_messages(body.question, hits))
    answer, cited, declined = clean_citations(raw, len(hits))
    return AskResponse(
        answer=answer,
        citations=[Citation(**c) for c in build_citations(hits, cited)],
        declined=declined,
        model=request.app.state.settings.llm_model,
        prompt_tokens=prompt_tokens,
        debug=debug_payload(retrieved, hits, cited) if body.debug else None,
    )


@app.post("/api/retrieve")
async def retrieve(body: AskRequest, request: Request) -> dict:
    """Retrieval only, no LLM call: the same trace as `debug` of /api/ask, without cited sources (evals)."""
    retrieved, hits = await _retrieve(request, body)
    return {"debug": debug_payload(retrieved, hits, [])}


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Server-sent events: `token` events, then one `done` event with the cleaned answer, citations and `declined`.

    Tokens are held back until the decline marker is ruled out, so it never reaches the client.
    """
    _, hits = await _retrieve(request, body)
    ollama: OllamaClient = request.app.state.ollama

    async def events():
        parts: list[str] = []
        stats: dict = {}
        decline_filter = DeclineFilter()
        try:
            async for token in ollama.chat_stream(build_messages(body.question, hits), stats):
                parts.append(token)
                if text := decline_filter.feed(token):
                    yield _sse("token", text)
            if text := decline_filter.flush():
                yield _sse("token", text)
        except httpx.HTTPError as exc:  # the response has started, so report it as an event instead of a 502
            yield _sse("error", f"LLM unavailable: {exc!r}")
            return
        except Exception as exc:  # surface other errors to the client instead of dropping the stream
            yield _sse("error", str(exc))
            return
        answer, cited, declined = clean_citations("".join(parts), len(hits))
        yield _sse(
            "done",
            {
                "answer": answer,
                "citations": build_citations(hits, cited),
                "declined": declined,
                "prompt_tokens": stats.get("prompt_tokens"),
            },
        )

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/acts")
async def acts(request: Request) -> list[dict]:
    """Acts in the index with unit counts; cached per physical collection, so a re-index invalidates it."""
    settings = request.app.state.settings
    qdrant: AsyncQdrantClient = request.app.state.qdrant
    collection = await _live_collection(qdrant, settings.collection)
    cache: dict[str, list[dict]] = request.app.state.acts_cache
    if collection not in cache:
        cache.clear()
        cache[collection] = await _count_acts(qdrant, collection)
    return cache[collection]


async def _live_collection(qdrant: AsyncQdrantClient, alias: str) -> str:
    """The physical collection behind the alias (the alias name itself for a pre-alias index)."""
    aliases = {a.alias_name: a.collection_name for a in (await qdrant.get_aliases()).aliases}
    return aliases.get(alias, alias)


async def _count_acts(qdrant: AsyncQdrantClient, collection: str) -> list[dict]:
    counts: Counter[tuple[str, str]] = Counter()
    titles: dict[str, str] = {}
    offset = None
    while True:
        points, offset = await qdrant.scroll(
            collection, limit=1000, offset=offset, with_payload=["act", "act_title", "unit_type", "part"]
        )
        for p in points:
            titles[p.payload["act"]] = p.payload["act_title"]
            if p.payload["part"] == 0:
                counts[(p.payload["act"], p.payload["unit_type"])] += 1
        if offset is None:
            break
    return [
        {
            "act": act,
            "title": title,
            "articles": counts[(act, "article")],
            "recitals": counts[(act, "recital")],
            "annexes": counts[(act, "annex")],
        }
        for act, title in titles.items()
    ]


@app.get("/api/config")
async def config(request: Request) -> dict:
    """Effective settings that shape answers, so an eval run can be tied to the exact configuration."""
    settings = request.app.state.settings
    qdrant: AsyncQdrantClient = request.app.state.qdrant
    collection = await _live_collection(qdrant, settings.collection)
    metadata = (await qdrant.get_collection(collection)).config.metadata or {}
    return {
        "settings": {name: getattr(settings, name) for name in _CONFIG_SETTINGS},
        "embed_query_prefix": query_prefix(settings.embed_model),
        "retrieval": {name.lower(): getattr(retrieval, name) for name in _RETRIEVAL_CONSTANTS},
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "index": {
            "alias": settings.collection,
            "collection": collection,
            "fingerprint": (metadata.get("fingerprint") or "")[:12] or None,
            "chunker_version": metadata.get("chunker_version"),
            "embed_model": metadata.get("embed_model"),
            "sparse_model": metadata.get("sparse_model"),
        },
    }


@app.get("/api/health")
async def health(request: Request) -> dict:
    settings = request.app.state.settings
    status: dict[str, object] = {}
    try:
        models = await request.app.state.ollama.models()
        status["ollama"] = "ok"
        status["models"] = {
            m: any(name == m or name == f"{m}:latest" for name in models)
            for m in (settings.llm_model, settings.embed_model)
        }
    except Exception as exc:
        status["ollama"] = f"error: {exc}"
    try:
        count = await request.app.state.qdrant.count(settings.collection)
        status["qdrant"] = "ok"
        status["chunks"] = count.count
    except Exception as exc:
        status["qdrant"] = f"error: {exc}"
    status["ok"] = status.get("ollama") == "ok" and status.get("qdrant") == "ok"
    return status


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
