from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    llm_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "snowflake-arctic-embed2"
    embed_dim: int = 1024
    collection: str = "eu_acts"
    data_dir: Path = Path("/data")
    top_k: int = 8
    article_bonus: float = 0.02  # added to the cosine score of article chunks
    num_ctx: int = 8192
    answer_tokens: int = 1024  # reserved for the answer (num_predict)
    chars_per_token: float = 3.5  # conservative estimate for English legal text
    temperature: float = 0.0  # greedy: reproducible answers and eval runs
    request_timeout: float = 300.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
