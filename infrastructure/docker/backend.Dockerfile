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
# Stage 2b: dvc-data — pull the Voting-3 v2 champion OOF artifacts (US-081 AC5).
#
# The copilot's ``classify_new_parcel`` tool serves the EPIC 12 Voting-3 v2
# champion by default; it reads the per-parcel fold-5 OOF parquets that live in
# DVC (gs://agrosat-dvc-remote), NOT in git. This stage materializes the MINIMAL
# artifact set the vote needs into ``/dvc/ml/eval`` so the runtime image can copy
# them in. PINNED weights (0.902 / 0.0 / 0.098) mean the vote needs NO PASTIS-R
# GT/geometry at load time, so we pull ONLY:
#   - ml/eval/oof_new32                              (tsvit-pheno-v2, 32 timesteps)
#   - ml/eval/oof/oof_parcel_utae_fold5.parquet      (U-TAE member)
#   - ml/eval/oof/oof_parcel_xgb-alphaearth_fold5.parquet  (xgb member + fallback)
#
# Credentials: a Cloud Build run mounts the GCS key as the build secret
# ``gcs_dvc_key`` (see infrastructure/cloudbuild.yaml). When the secret is absent
# (a local ``docker build`` with no creds) the pull is SKIPPED with a loud notice
# and the runtime falls back to the git-tracked ``ml/`` (no parquets) -> the agent
# degrades to xgb-alphaearth at request time and logs ``classify_voting3_unavailable``
# (honest degradation, never a fabricated posterior). It never fails the build.
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS dvc-data

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "dvc[gs]>=3.67.1,<4"

WORKDIR /dvc

# Copy only the DVC metadata needed to resolve + pull the champion's OOF outputs
# (the .dvc pointer files + the repo DVC config), not the whole tree.
COPY .dvc/config /dvc/.dvc/config
COPY ml/eval/oof_new32.dvc /dvc/ml/eval/oof_new32.dvc
COPY ml/eval/oof/oof_parcel_utae_fold5.parquet.dvc /dvc/ml/eval/oof/oof_parcel_utae_fold5.parquet.dvc
COPY ml/eval/oof/oof_parcel_xgb-alphaearth_fold5.parquet.dvc /dvc/ml/eval/oof/oof_parcel_xgb-alphaearth_fold5.parquet.dvc

# DVC requires a git repo to resolve relative .dvc paths; init a throwaway one.
RUN git init -q . \
    && dvc config core.no_scm true 2>/dev/null || true

# Pull with the mounted GCS key when present; skip cleanly when it is not.
RUN --mount=type=secret,id=gcs_dvc_key,required=false \
    set -eu; \
    if [ -f /run/secrets/gcs_dvc_key ]; then \
        export GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcs_dvc_key; \
        echo "US-081 AC5: pulling Voting-3 v2 OOF artifacts from DVC remote..."; \
        dvc pull --no-run-cache \
            ml/eval/oof_new32.dvc \
            ml/eval/oof/oof_parcel_utae_fold5.parquet.dvc \
            ml/eval/oof/oof_parcel_xgb-alphaearth_fold5.parquet.dvc; \
        echo "US-081 AC5: DVC pull complete."; \
    else \
        echo "US-081 AC5 WARNING: no gcs_dvc_key build secret; skipping DVC pull."; \
        echo "  The runtime image will lack the OOF parquets and the copilot will"; \
        echo "  degrade voting3 -> xgb-alphaearth at request time (logged)."; \
        mkdir -p ml/eval/oof_new32 ml/eval/oof; \
    fi

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

# Overlay the Voting-3 v2 champion OOF parquets pulled from DVC (US-081 AC5) on
# top of the git-tracked ``ml/eval`` tree (which carries only the .dvc pointers).
# When the dvc-data stage skipped the pull (no creds), these dirs are empty and
# the runtime degrades voting3 -> xgb-alphaearth at request time (logged).
COPY --from=dvc-data --chown=agrosat:agrosat /dvc/ml/eval/oof_new32/ /app/ml/eval/oof_new32/
COPY --from=dvc-data --chown=agrosat:agrosat /dvc/ml/eval/oof/ /app/ml/eval/oof/

USER agrosat
EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
