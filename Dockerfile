ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/eleven-am/bragi"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libopus0 \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash bragi

WORKDIR /home/bragi/app

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /bin/uv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --compile-bytecode --no-install-project --all-extras --python 3.12

COPY --chown=bragi . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --compile-bytecode --all-extras --python 3.12

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python --reinstall --pre torch torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

RUN mkdir -p /data/models /data/voices /data/keys /data/huggingface && \
    chown -R bragi:bragi /home/bragi /data

USER bragi

ENV PATH="/home/bragi/app/.venv/bin:$PATH" \
    BRAGI_HOST=0.0.0.0 \
    BRAGI_PORT=8000 \
    BRAGI_DEVICE=auto \
    BRAGI_MODEL_CACHE_DIR=/data/models \
    BRAGI_VOICE_STORE_DIR=/data/voices \
    BRAGI_KEY_STORE_DIR=/data/keys \
    HF_HOME=/data/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_CPP_LOG_LEVEL=ERROR

EXPOSE 8000

CMD ["python", "-m", "bragi.main"]
