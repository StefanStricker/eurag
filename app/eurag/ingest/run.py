"""Index the PDFs in DATA_DIR into Qdrant.

Usage: python -m eurag.ingest.run [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from collections import Counter

import yaml
from qdrant_client import AsyncQdrantClient, models

from eurag.config import get_settings
from eurag.ingest.chunker import ActInfo, Chunk, chunk_document
from eurag.ingest.pdf_extract import extract_lines
from eurag.ollama_client import OllamaClient

log = logging.getLogger("eurag.ingest")
BATCH = 32
_NS = uuid.UUID("6f1c1a52-6a2e-4e0a-9a0e-2f5b1b0c7e11")


def load_chunks() -> list[Chunk]:
    settings = get_settings()
    registry = yaml.safe_load((settings.data_dir / "acts.yaml").read_text())
    chunks: list[Chunk] = []
    for filename, meta in registry.items():
        path = settings.data_dir / filename
        if not path.exists():
            log.warning("missing %s, skipping", path)
            continue
        doc_chunks = chunk_document(extract_lines(path), ActInfo(**meta))
        units = Counter(c.unit_type for c in doc_chunks if c.part == 0)
        log.info("%-20s %s -> %d chunks", meta["act"], dict(units), len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks


def point_id(c: Chunk) -> str:
    return str(uuid.uuid5(_NS, f"{c.celex}/{c.unit_type}/{c.number}/{c.part}"))


async def ingest(force: bool) -> None:
    settings = get_settings()
    chunks = load_chunks()
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    ollama = OllamaClient(settings)
    try:
        if await qdrant.collection_exists(settings.collection):
            count = (await qdrant.count(settings.collection, exact=True)).count
            if count == len(chunks) and not force:
                log.info("collection %s already holds %d chunks, nothing to do", settings.collection, count)
                return
            await qdrant.delete_collection(settings.collection)

        await qdrant.create_collection(
            settings.collection,
            vectors_config=models.VectorParams(size=settings.embed_dim, distance=models.Distance.COSINE),
        )
        for field in ("act", "unit_type", "number"):
            await qdrant.create_payload_index(settings.collection, field, models.PayloadSchemaType.KEYWORD)

        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            vectors = await ollama.embed([c.text for c in batch])
            await qdrant.upsert(
                settings.collection,
                points=[
                    models.PointStruct(id=point_id(c), vector=v, payload=c.to_payload())
                    for c, v in zip(batch, vectors, strict=True)
                ],
            )
            log.info("embedded %d/%d", min(i + BATCH, len(chunks)), len(chunks))
        log.info("done: %d chunks in %s", len(chunks), settings.collection)
    finally:
        await ollama.aclose()
        await qdrant.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-index even if the collection is up to date")
    asyncio.run(ingest(parser.parse_args().force))


if __name__ == "__main__":
    main()
