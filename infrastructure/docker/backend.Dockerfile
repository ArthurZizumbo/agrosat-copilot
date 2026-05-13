# syntax=docker/dockerfile:1.7
# AgroSatCopilot — Backend FastAPI multi-stage image
# Stages: builder (poetry export) → dev (uvicorn --reload) → runtime (slim production)

# ----------------------------------------------------------------------------
# Stage 1: builder — exporta requirements.txt y construye wheels para grupos
# main + test (no se incluye grupo ml: vive en inference-worker image aparte).
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV POETRY_VERSION=2.2.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential libpq-dev libgeos-dev libproj-dev gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /build
COPY pyproject.toml poetry.lock* ./

# Export to requirements.txt (main + test groups; no ml/geo/paper/dagster)
RUN poetry self add poetry-plugin-export \
    && poetry export --with test --without-hashes --format requirements.txt -o requirements.txt

# Build wheels
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ----------------------------------------------------------------------------
# Stage 2: dev — incluye poetry + hot reload + bind-mount friendly
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential libpq-dev libgeos-dev libproj-dev gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --with dev,test --no-root

# Source mounted via docker-compose volumes at runtime
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ----------------------------------------------------------------------------
# Stage 3: runtime — slim production image (<2GB), no poetry, non-root user
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 libgeos-c1v5 libproj25 libgdal36 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 agrosat \
    && useradd --system --uid 1001 --gid agrosat --home-dir /app --shell /bin/bash agrosat

WORKDIR /app

# Install wheels exported from builder stage
COPY --from=builder /wheels /wheels
COPY --from=builder /build/requirements.txt /app/requirements.txt
RUN pip install --no-index --find-links=/wheels -r /app/requirements.txt \
    && rm -rf /wheels /app/requirements.txt

# Copy application code (backend + shared ml utils used by API layer only)
COPY --chown=agrosat:agrosat backend/ /app/backend/
COPY --chown=agrosat:agrosat ml/ /app/ml/

USER agrosat
EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
