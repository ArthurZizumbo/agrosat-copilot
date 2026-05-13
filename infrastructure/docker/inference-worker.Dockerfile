# syntax=docker/dockerfile:1.7
# AgroSatCopilot - Inference Worker GPU L4 image
# Stages: builder (poetry export con grupos ml + ml-cuda + geo) -> runtime (CUDA 13.0 base + wheels)
#
# Diferencia clave vs backend.Dockerfile:
# - INCLUYE grupos `ml` + `ml-cuda` (torch+cu130, transformers, peft, vllm, flash-attn, etc.)
#   requeridos por los Pub/Sub workers que reciben jobs de inferencia pesada.
# - Base image NVIDIA CUDA 13.0 (alineada a torch 2.11.0+cu130) para GPU L4 / H100.
#
# Tamano objetivo: < 10 GB compressed (torch CUDA + flash-attn + vllm son pesados).

# ----------------------------------------------------------------------------
# Stage 1: builder - exporta requirements con grupos `ml,ml-cuda,geo` y construye wheels
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

# Exportar con grupos:
# - `ml`           (transformers, peft, segmentation_models, monai, xgboost, lightgbm)
# - `ml-gpu`       (torch+cu130, bitsandbytes — cross-platform GPU)
# - `ml-gpu-linux` (flash-attn, vllm — Linux only, sirve para Cloud Run/Azure H100)
# - `geo`          (rasterio, shapely, geopandas)
RUN poetry self add poetry-plugin-export \
    && poetry export --with ml,ml-gpu,ml-gpu-linux,geo --without-hashes \
       --format requirements.txt -o requirements.txt

# ----------------------------------------------------------------------------
# Stage 2: runtime - CUDA 13.0 base + wheels
# ----------------------------------------------------------------------------
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        libpq5 libgeos-c1v5 libproj22 libgdal30 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 agrosat \
    && useradd --system --uid 1001 --gid agrosat --home-dir /app --shell /bin/bash agrosat

WORKDIR /app

COPY --from=builder /build/requirements.txt /app/requirements.txt
RUN python3.12 -m pip install --upgrade pip \
    && python3.12 -m pip install -r /app/requirements.txt \
    && rm -f /app/requirements.txt

# Codigo: backend (workers/) + ml (inferencia + agente)
COPY --chown=agrosat:agrosat backend/ /app/backend/
COPY --chown=agrosat:agrosat ml/ /app/ml/

USER agrosat
EXPOSE 8000

# Entry point: arranca el worker de Pub/Sub que escucha `inference-jobs`.
CMD ["python3.12", "-m", "backend.app.workers.inference_worker"]
