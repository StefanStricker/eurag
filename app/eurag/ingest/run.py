"""Index the PDFs in DATA_DIR into Qdrant.

The index is a physical collection `<collection>_<fingerprint>_<build time>` behind the alias
`settings.collection`. The fingerprint covers the embedding model, chunker version, acts.yaml and
the PDFs; when it changes, a new collection is built and the alias is switched atomically, so the
API keeps serving the old index until the new one is complete.

Usage: python -m eurag.ingest.run [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import time
import uuid
from collections import Counter
from pathlib import Path

import yaml
from qdrant_client import AsyncQdrantClient, models

from eurag.config import Settings, get_settings
from eurag.ingest.chunker import CHUNKER_VERSION, ActInfo, Chunk, Unit, chunk_units, merge_recitals, parse_units
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


def fingerprint(settings: Settings) -> str:
    """Hash of everything that determines the index contents."""
    h = hashlib.sha256()
    h.update(f"{settings.embed_model}\0{settings.embed_dim}\0{CHUNKER_VERSION}\0".encode())
    registry_file = settings.data_dir / "acts.yaml"
    h.update(registry_file.read_bytes())
    registry = yaml.safe_load(registry_file.read_text())
    files = sorted({f for name, meta in registry.items() for f in (name, meta.get("recitals")) if f})
    for f in files:
        path = settings.data_dir / f
        h.update(f"\0{f}\0".encode())
        h.update(path.read_bytes() if path.exists() else b"<missing>")
    return h.hexdigest()


def point_id(c: Chunk) -> str:
    return str(uuid.uuid5(_NS, f"{c.celex}/{c.unit_type}/{c.number}/{c.part}"))


async def _build(qdrant: AsyncQdrantClient, ollama: OllamaClient, name: str, fp: str, chunks: list[Chunk]) -> None:
    settings = get_settings()
    await qdrant.create_collection(
        name,
        vectors_config=models.VectorParams(size=settings.embed_dim, distance=models.Distance.COSINE),
        metadata={"fingerprint": fp, "chunker_version": CHUNKER_VERSION, "embed_model": settings.embed_model},
    )
    for field in ("act", "unit_type", "number"):
        await qdrant.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vectors = await ollama.embed([c.text for c in batch])
        await qdrant.upsert(
            name,
            points=[
                models.PointStruct(id=point_id(c), vector=v, payload=c.to_payload())
                for c, v in zip(batch, vectors, strict=True)
            ],
        )
        log.info("embedded %d/%d", min(i + BATCH, len(chunks)), len(chunks))


async def ingest(force: bool) -> None:
    settings = get_settings()
    alias = settings.collection
    fp = fingerprint(settings)
    prefix = f"{alias}_{fp[:12]}_"
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    ollama = OllamaClient(settings)
    try:
        aliases = {a.alias_name: a.collection_name for a in (await qdrant.get_aliases()).aliases}
        live = aliases.get(alias)
        if live and live.startswith(prefix) and not force:
            log.info("alias %s -> %s matches fingerprint %s, nothing to do", alias, live, fp[:12])
            return

        chunks = load_chunks()
        target = prefix + time.strftime("%Y%m%d%H%M%S")
        log.info("building %s (%d chunks), alias %s currently -> %s", target, len(chunks), alias, live)
        await _build(qdrant, ollama, target, fp, chunks)

        collections = {c.name for c in (await qdrant.get_collections()).collections}
        if alias in collections:
            # Pre-alias layout: a real collection holds the alias name. Unavoidably brief downtime.
            log.info("migrating: deleting collection %s to free the alias name", alias)
            await qdrant.delete_collection(alias)
        switch = [models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=target, alias_name=alias))]
        if live:
            switch.insert(0, models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
        await qdrant.update_collection_aliases(change_aliases_operations=switch)
        log.info("alias %s -> %s", alias, target)

        # Drop the previous index and leftovers of interrupted builds (not collections behind other aliases).
        in_use = {c for a, c in aliases.items() if a != alias} | {target}
        for name in sorted(collections - in_use):
            if name.startswith(f"{alias}_"):
                await qdrant.delete_collection(name)
                log.info("deleted old collection %s", name)
        log.info("done: %d chunks in %s", len(chunks), target)
    finally:
        await ollama.aclose()
        await qdrant.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-index even if the fingerprint matches")
    asyncio.run(ingest(parser.parse_args().force))


if __name__ == "__main__":
    main()
