# syntax=docker/dockerfile:1
# One Dockerfile for both images; EXTRAS picks what gets installed:
#   gateway:  --extra api
#   worker:   --extra worker (default)
ARG EXTRAS="--extra worker"

FROM python:3.11-slim-bookworm AS builder
ARG EXTRAS
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

# git: the anonymizer is a git dependency; build-essential: it compiles a Cython extension
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# dependencies first, so code changes don't invalidate the (large) torch layer
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project ${EXTRAS}

COPY .project-root ./
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen ${EXTRAS}


FROM python:3.11-slim-bookworm
# ffmpeg: torchcodec decodes audio through it
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# /app must be writable: Hydra writes its run logs to outputs/ in the working directory
# the cache dir exists up front, so the volume mounted there (Whisper weights) is writable by app
RUN useradd --create-home --uid 1000 app && mkdir /app && chown app:app /app \
    && install -d -o app -g app /home/app/.cache
WORKDIR /app
# the project is installed editable, so configs/ and .project-root are found relative to src/
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER app

CMD ["globi-worker"]
