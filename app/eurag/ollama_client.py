"""Minimal async client for the Ollama HTTP API."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from eurag.config import Settings

log = logging.getLogger("eurag.ollama")

# Query prefixes the embedding models were trained with (model name without tag -> prefix). Documents get none.
QUERY_PREFIXES = {
    "qwen3-embedding": (
        "Instruct: Given a question about EU data law, retrieve the legal provisions that answer it\nQuery:"
    ),
    "snowflake-arctic-embed2": "query: ",
}


def query_prefix(model: str) -> str:
    return QUERY_PREFIXES.get(model.split(":")[0], "")


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._http = httpx.AsyncClient(base_url=settings.ollama_url, timeout=settings.request_timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def embed(self, texts: list[str], query: bool = False) -> list[list[float]]:
        """Embed chunk texts, or search queries with the model's query prefix."""
        if query:
            texts = [query_prefix(self.settings.embed_model) + t for t in texts]
        r = await self._http.post("/api/embed", json={"model": self.settings.embed_model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]

    def _chat_body(self, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.settings.llm_model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.settings.temperature,
                "num_ctx": self.settings.num_ctx,
                "num_predict": self.settings.answer_tokens,
            },
        }

    def _prompt_tokens(self, data: dict) -> int | None:
        """Prompt size reported by Ollama; warns when it is close to the context window (likely truncated)."""
        n = data.get("prompt_eval_count")
        if n is not None and n >= 0.9 * self.settings.num_ctx:
            log.warning("prompt uses %d of %d context tokens, it may have been truncated", n, self.settings.num_ctx)
        return n

    async def chat(self, messages: list[dict]) -> tuple[str, int | None]:
        """Return the answer and the number of prompt tokens."""
        r = await self._http.post("/api/chat", json=self._chat_body(messages, stream=False))
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"], self._prompt_tokens(data)

    async def chat_stream(self, messages: list[dict], stats: dict | None = None) -> AsyncIterator[str]:
        """Yield answer tokens; `stats["prompt_tokens"]` is filled from the final message."""
        async with self._http.stream("POST", "/api/chat", json=self._chat_body(messages, stream=True)) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if token := data.get("message", {}).get("content"):
                    yield token
                if data.get("done"):
                    if stats is not None:
                        stats["prompt_tokens"] = self._prompt_tokens(data)
                    break

    async def models(self) -> list[str]:
        r = await self._http.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
