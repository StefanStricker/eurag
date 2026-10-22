"""Index the PDFs in DATA_DIR into Qdrant.

Usage: python -m eurag.ingest.run [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from collections import Counter
from pathlib import Path

import yaml
from qdrant_client import AsyncQdrantClient, models

from eurag.config import get_settings
from eurag.ingest.chunker import ActInfo, Chunk, Unit, chunk_units, merge_recitals, parse_units
from eurag.ingest.pdf_extract import extract_lines
from eurag.ollama_client import OllamaClient

log = logging.getLogger("eurag.ingest")
BATCH = 32
_NS = uuid.UUID("6f1c1a52-6a2e-4e0a-9a0e-2f5b1b0c7e11")


def load_act(data_dir: Path, filename: str, meta: dict) -> tuple[ActInfo, list[Unit]]:
    """Parse one acts.yaml entry; `recitals` names the OJ original when the main text lacks them."""
    meta = dict(meta)
    recitals_file = meta.pop("recitals", None)
    units = parse_units(extract_lines(data_dir / filename))
    if recitals_file:
        units = merge_recitals(units, parse_units(extract_lines(data_dir / recitals_file)))
    return ActInfo(**meta), units


def load_chunks() -> list[Chunk]:
    settings = get_settings()
    registry = yaml.safe_load((settings.data_dir / "acts.yaml").read_text())
    chunks: list[Chunk] = []
    for filename, meta in registry.items():
        missing = [f for f in (filename, meta.get("recitals")) if f and not (settings.data_dir / f).exists()]
        if missing:
            log.warning("missing %s, skipping", ", ".join(missing))
            continue
        info, units = load_act(settings.data_dir, filename, meta)
        doc_chunks = chunk_units(units, info)
        counts = Counter(c.unit_type for c in doc_chunks if c.part == 0)
        log.info("%-20s %s -> %d chunks", meta["act"], dict(counts), len(doc_chunks))
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
