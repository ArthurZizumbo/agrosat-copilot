# syntax=docker/dockerfile:1.7
# AgroSatCopilot — Dagster orchestrator image
# Base python:3.12-slim + poetry + grupo `dagster` del pyproject.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    DAGSTER_HOME=/app/dagster_home

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Install dagster + integrations via poetry group `dagster`
# (dagster-postgres ya está pinneado en el grupo `dagster` del pyproject — no
# instalar via pip en paralelo, para evitar mismatch con el lockfile).
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main,dagster --no-root

# Copy dagster project + dagster.yaml
COPY dagster_project/ /app/dagster_project/
COPY dagster.yaml /app/dagster_home/dagster.yaml

# Non-root user
RUN groupadd --system --gid 1001 dagster \
    && useradd --system --uid 1001 --gid dagster --home-dir /app --shell /bin/bash dagster \
    && mkdir -p /app/dagster_home \
    && chown -R dagster:dagster /app

USER dagster
EXPOSE 3000

CMD ["dagster", "dev", "-m", "dagster_project.definitions", "-h", "0.0.0.0", "-p", "3000"]
