# syntax=docker/dockerfile:1
#
# Both build and runtime stages start from the SAME python base image and
# borrow the uv binary, so the interpreter that builds the virtualenv is
# byte-identical to the one that runs it. Installing from uv.lock makes the
# container match `uv sync` on a teammate's laptop exactly.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.15

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uvbin

# --------------------------------------------------------------- builder ----
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
COPY --from=uvbin /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, as their own layer: editing src/ must not re-download
# ~1GB of wheels. Only a change to pyproject.toml or uv.lock busts this.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ------------------------------------------------------------ base runtime --
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime-base

# libgomp1: the OpenMP runtime XGBoost links against. Without it, `import
# xgboost` fails at load time with a dlopen error.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# UID 1000 matches the first human user on most Linux hosts, so files written
# into the bind-mounted data/ and artifacts/ stay editable outside the container.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash app

WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HOME=/home/app \
    TABPFN_MODEL_CACHE_DIR=/home/app/.cache/tabpfn

COPY --chown=app:app config.yaml README.md ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app app/ ./app/
COPY --chown=app:app tests/ ./tests/

RUN mkdir -p /app/data/raw /app/data/processed \
             /app/artifacts/models /app/artifacts/results /app/artifacts/figures \
             /home/app/.cache/tabpfn \
    && chown -R app:app /app/data /app/artifacts /home/app/.cache

# ----------------------------------------------------------------- runtime --
# Default target: production-ish image, no test tooling.
FROM runtime-base AS runtime
COPY --from=builder --chown=app:app /app/.venv /app/.venv
USER app
EXPOSE 8501

# TabPFN downloads its pretrained checkpoint on first use; the named volume in
# docker-compose.yml keeps that a one-off rather than once per container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/Home.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# --------------------------------------------------------------------- dev --
# Same image plus dev dependencies, so `make docker-test` runs the suite under
# exactly the environment the app ships with. Used by CI and by the test service.
FROM runtime-base AS dev
COPY --from=uvbin /uv /uvx /bin/
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app pyproject.toml uv.lock ./
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked \
    && chown -R app:app /app/.venv
USER app
CMD ["pytest"]
