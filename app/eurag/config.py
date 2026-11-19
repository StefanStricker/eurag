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
    sparse_model: str = "Qdrant/bm25"
    collection: str = "eu_acts"
    data_dir: Path = Path("/data")
    top_k: int = 8
    retrieval_mode: Literal["dense", "hybrid"] = "dense" 
    hybrid_prefetch: int = 40 
    rrf_k: int = 60
    hybrid_dense_weight: float = 1.0  
    article_bonus: float = 0.02
    act_quota: bool = True
    rerank: bool = False 
    rerank_model: str = "onnx-community/bge-reranker-v2-m3-ONNX"  
    rerank_candidates: int = 30 
    rerank_article_bonus: float = 0.0 
    rerank_search_weight: float = 0.0 
    rerank_gpu_mem_gb: float = 1.4
    query_embed_num_gpu: int = -1  
    llm_num_gpu: int = -1
    num_ctx: int = 8192
    answer_tokens: int = 1024  
    chars_per_token: float = 3.5 
    temperature: float = 0.0  
    request_timeout: float = 300.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
