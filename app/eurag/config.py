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
    # Questions naming 2+ acts: one search per act, each act gets top_k // n slots (not combined with rerank)
    act_quota: bool = True
    rerank: bool = False  # re-sort the search candidates with a cross-encoder before the top_k cut
    rerank_model: str = "onnx-community/bge-reranker-v2-m3-ONNX"  # fp16 ONNX, on the GPU (eurag.rerank.CUSTOM_MODELS)
    rerank_candidates: int = 30  # search hits scored by the reranker
    rerank_article_bonus: float = 0.0  # added to the rerank score (0-1) of article chunks
    rerank_search_weight: float = 0.0  # > 0: RRF of rerank and search ranking with this weight on the search side
    rerank_gpu_mem_gb: float = 1.4  # cap of onnxruntime's GPU memory pool (weights 1.1 GB; batch 1 of the longest chunks fits 1.3)
    # Ollama `num_gpu` (GPU layers), -1 = Ollama decides. With the reranker on the 8 GB GPU, Ollama's own estimate
    # offloads LLM layers to the CPU although they fit: query embeddings go to the CPU (0) and the LLM is forced (99).
    query_embed_num_gpu: int = -1  # search queries only; ingest keeps the GPU
    llm_num_gpu: int = -1
    num_ctx: int = 8192
    answer_tokens: int = 1024  # reserved for the answer (num_predict)
    chars_per_token: float = 3.5  # conservative estimate for English legal text
    temperature: float = 0.0  # greedy: reproducible answers and eval runs
    request_timeout: float = 300.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
