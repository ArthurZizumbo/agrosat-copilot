# syntax=docker/dockerfile:1.7
# AgroSatCopilot - ML Training image (US-022b-A, ADR-006)
#
# Stages: builder (poetry export con grupos `ml` + `ml-gpu` + `ml-gpu-linux` + `geo`)
#         runtime (CUDA 13.0 base + wheels + cliente MLflow + DVC + breizhcrops)
#
# Diferencia clave vs inference-worker.Dockerfile:
# - Optimizada para JOBS de entrenamiento (Vertex AI custom-jobs en L4 spot 24 GB).
#   - Incluye `breizhcrops` (US-022b-C TempCNN/InceptionTime).
#   - Incluye `dvc[gs]` + `mlflow` ya activos para que el job escriba runs y
#     publique datasets versionados a `gs://agrosat-artifacts-dev/`.
#   - NO incluye `backend/` ni Pub/Sub workers — es imagen de batch, no de serving.
# - Base image NVIDIA CUDA 13.0 alineada a torch 2.11.0+cu130 (mismo runtime
#   que `inference-worker` para compatibilidad de checkpoints).
# - ENTRYPOINT por defecto: `poetry run python` para que `containerSpec.args`
#   de Vertex AI invoque scripts arbitrarios sin reescribir el CMD.
#
# A-1 smoke (validacion local sin GPU):
#   docker build -f infrastructure/docker/ml-train.Dockerfile -t ml-train:dev .
#   docker run --rm ml-train:dev \
#     poetry run python -c "import torch, mlflow, breizhcrops; print(torch.__version__)"
#
# Tamano objetivo: < 11 GB compressed (torch CUDA + flash-attn + vllm son pesados).
# Tamano efectivo se valida en Cloud Build (timeout 1800s + machineType E2_HIGHCPU_8).

# ----------------------------------------------------------------------------
# Stage 1: builder - exporta requirements con grupos `ml,ml-gpu,ml-gpu-linux,geo`
#                    y descarga wheels (sin instalar) para acelerar runtime stage.
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

# Grupos:
# - `ml`           transformers, peft, mlflow, dvc[gs], breizhcrops, sklearn, xgboost, sentence-transformers
# - `ml-gpu`       torch 2.11.0+cu130 + bitsandbytes (cross-platform)
# - `geo`          rasterio, shapely, geopandas (FE temporal + spatial CV requiere)
#
# US-022-c P1 fix #5 (2026-05-23): `ml-gpu-linux` (flash-attn + vllm) EXCLUIDO
# del builder. Esos paquetes son SOLO para V3-V5 H100 LoRA Gemma 4/Qwen3.5;
# FarSLIP (train.py) usa CLIP standard y NO los necesita. flash-attn requiere
# `nvcc` (no incluido en cuda:13.0.0-runtime). Cuando llegue V3 se construira
# imagen separada `ml-train-llm.Dockerfile` con base devel + flash-attn/vllm.
RUN poetry self add poetry-plugin-export \
    && poetry export --with ml,ml-gpu,geo --without-hashes \
       --format requirements.txt -o requirements.txt

# ----------------------------------------------------------------------------
# Stage 2: runtime - CUDA 13.0 base + wheels
# ----------------------------------------------------------------------------
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=2.2.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# US-022-c P1 etapa 2 fix (2026-05-23): Ubuntu 24.04 (noble) trae python3.12 nativo
# en los repos default. Elimina software-properties-common + add-apt-repository
# (bug Cloud Build con resolv.conf busy) y el PPA deadsnakes. Libs C++ con sufijos
# Noble: libgeos-c1t64, libproj25, libgdal34. `python3.12-distutils` removido
# (no existe en Noble; setuptools lo provee). PEP 668 obliga `--break-system-packages`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        python3.12 python3.12-venv python3-pip \
        libpq5 libgeos-c1t64 libproj25 libgdal34 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 agrosat \
    && useradd --system --uid 1001 --gid agrosat --home-dir /app --shell /bin/bash agrosat \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python

# Poetry necesario en runtime: la imagen se invoca como
#   `bash -c "poetry run python ..."` desde l4_spot.yaml (paridad con dev local).
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Instala dependencias desde el requirements exportado en builder.
# US-022-c P1 fix #4+5 (2026-05-23):
#  (a) pip de debian no se desinstala -> --ignore-installed.
#  (b) torch 2.11.0+cu130 viene del index custom `pytorch-cu130` que poetry source
#      con priority=explicit NO exporta como --extra-index-url -> agregamos el flag.
#  (c) flash-attn/vllm/xformers EXCLUIDOS del builder (no necesarios para FarSLIP,
#      requieren nvcc que esta solo en cuda:devel). Sin esos paquetes el install
#      es un solo comando, sin --no-build-isolation.
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
COPY --from=builder /build/requirements.txt /app/requirements.txt
RUN python3.12 -m pip install --ignore-installed --break-system-packages --upgrade \
        pip setuptools wheel \
    && python3.12 -m pip install --break-system-packages \
        --extra-index-url ${PYTORCH_INDEX_URL} -r /app/requirements.txt \
    && rm -f /app/requirements.txt

# Copia el pyproject.toml y poetry.lock para que `poetry run` resuelva el
# entorno (poetry usa el python del sistema porque VIRTUALENVS_CREATE=false).
COPY --chown=agrosat:agrosat pyproject.toml poetry.lock* /app/

# Codigo necesario para training: SOLO `ml/` (NO backend, NO frontend).
# Los notebooks NO se copian — el job lee/escribe datasets desde GCS/DVC.
COPY --chown=agrosat:agrosat ml/ /app/ml/
COPY --chown=agrosat:agrosat scripts/ /app/scripts/

USER agrosat

# Healthcheck: verifica que las 3 deps criticas importen.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch, mlflow, breizhcrops" || exit 1

# CMD por defecto: smoke trivial. Vertex AI lo sobreescribe via containerSpec.args.
CMD ["python", "-c", "import torch, mlflow, breizhcrops; print('ml-train ready: torch', torch.__version__, 'mlflow', mlflow.__version__, 'breizhcrops', breizhcrops.__version__)"]
