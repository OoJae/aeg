# Aeg gateway (FastAPI) — serves the API + dashboard at / on Railway's $PORT.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# onnxruntime (via fastembed) needs libgomp at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first (cached layer): install everything except the project itself
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# then the source + build/install the aeg package
COPY . .
RUN uv sync --no-dev --frozen

# cognee writes its embedded stores here (always-writable, ephemeral)
ENV AEG_SCRATCH_DIR=/tmp/aeg_cognee

EXPOSE 8080
CMD ["sh", "-c", "uv run uvicorn aeg.gateway:app --host 0.0.0.0 --port ${PORT:-8080}"]
