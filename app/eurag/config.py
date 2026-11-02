from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    llm_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "snowflake-arctic-embed2"
    embed_dim: int = 1024
    sparse_model: str = "Qdrant/bm25"  # fastembed sparse model for the keyword half of hybrid search
    collection: str = "eu_acts"
    data_dir: Path = Path("/data")
    top_k: int = 8
    retrieval_mode: Literal["dense", "hybrid"] = "dense"  # hybrid: dense + BM25 prefetch, fused with RRF
    hybrid_prefetch: int = 40  # candidates per prefetch (dense and BM25) before fusion
    rrf_k: int = 60
    hybrid_dense_weight: float = 1.0  # RRF weight of the dense ranking; BM25 has weight 1
    article_bonus: float = 0.02  # added to the cosine score of article chunks (dense mode only)
    num_ctx: int = 8192
    answer_tokens: int = 1024  # reserved for the answer (num_predict)
    chars_per_token: float = 3.5  # conservative estimate for English legal text
    temperature: float = 0.0  # greedy: reproducible answers and eval runs
    request_timeout: float = 300.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
