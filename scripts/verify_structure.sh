#!/usr/bin/env bash
# verify_structure.sh — chequea AC-4 de US-001
# Falla si falta cualquier directorio canónico del monorepo.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REQUIRED_DIRS=(
  "backend"
  "frontend"
  "frontend/pages"
  "frontend/components"
  "frontend/composables"
  "frontend/stores"
  "frontend/layouts"
  "frontend/middleware"
  "frontend/plugins"
  "frontend/server"
  "frontend/types"
  "frontend/assets"
  "frontend/i18n/locales"
  "frontend/public"
  "ml"
  "ml/configs"
  "ml/agent"
  "dagster_project"
  "infrastructure/docker"
  "infrastructure/terraform/modules/gcp"
  "infrastructure/terraform/modules/azure"
  "infrastructure/terraform/environments/dev"
  "notebooks"
  "data"
  "docs"
  "docs/decisions"
  "docs/orchestration"
  "paper"
  "scripts"
  ".github/workflows"
  "db/migrations"
)

REQUIRED_FILES=(
  "LICENSE"
  "README.md"
  "AGENTS.md"
  "CLAUDE.md"
  "Makefile"
  "pyproject.toml"
  ".env.example"
  ".dbmate.yaml"
  "docker-compose.yml"
  "dagster.yaml"
  ".github/CODEOWNERS"
  ".github/workflows/ci.yml"
  "infrastructure/docker/backend.Dockerfile"
  "infrastructure/docker/frontend.Dockerfile"
  "infrastructure/docker/dagster.Dockerfile"
  "infrastructure/docker/inference-worker.Dockerfile"
  "infrastructure/cloudbuild.yaml"
  "infrastructure/terraform/environments/dev/main.tf"
  "ml/configs/l4_spot.yaml"
  "scripts/seed.py"
  "db/migrations/20260511213942_initial_schema.sql"
  "docs/decisions/ADR-001-no-cookiecutter-externo.md"
  "docs/decisions/ADR-002-single-env-dev.md"
  "docs/decisions/ADR-003-upstash-redis.md"
  "docs/decisions/ADR-004-poetry-optional-groups-and-aiplatform-ml.md"
)

missing=0

for d in "${REQUIRED_DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "MISSING DIR : $d"
    missing=1
  fi
done

for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING FILE: $f"
    missing=1
  fi
done

if [[ "$missing" -eq 0 ]]; then
  echo "structure OK (AC-4 verified)"
  exit 0
else
  echo ""
  echo "structure check FAILED — completa los archivos/directorios listados arriba."
  exit 1
fi
