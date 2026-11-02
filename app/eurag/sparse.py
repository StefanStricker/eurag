"""BM25 sparse vectors (fastembed, CPU) for the keyword half of hybrid search.

The collection stores term weights only; Qdrant applies IDF itself (`Modifier.IDF`), so it stays
correct for the whole collection. Queries get unweighted terms (`query_embed`).
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding
from qdrant_client import models

DENSE = "dense"  # named vectors in the collection
SPARSE = "bm25"


class SparseEncoder:
    def __init__(self, model: str):
        self._model = SparseTextEmbedding(model)

    def documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [_vector(e) for e in self._model.embed(texts)]

    def query(self, text: str) -> models.SparseVector:
        return _vector(next(iter(self._model.query_embed(text))))


def _vector(e) -> models.SparseVector:
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
