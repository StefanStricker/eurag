FROM python:3.12.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/models
WORKDIR /app

# Dependencies first (cached layer), then the project itself, both from uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra dev --no-install-project
# BM25 model for hybrid search (tokenizer + stopwords, no weights), baked in so containers start offline
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"
COPY eurag ./eurag
RUN uv sync --frozen --extra dev --no-editable

COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "eurag.api:app", "--host", "0.0.0.0", "--port", "8000"]
