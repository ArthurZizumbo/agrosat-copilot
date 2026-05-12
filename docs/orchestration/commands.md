# Comandos Make — AgroSatCopilot

> Lista completa de targets `make`. Resumen ejecutivo en [`AGENTS.md`](../../AGENTS.md).

## Desarrollo local

```bash
make dev                    # docker-compose: 8 servicios (api, frontend, postgres, redis, titiler, mlflow, dagster, ollama)
make stop
```

## Quality gates (reemplaza pre-commit)

```bash
make check                  # lint + secrets-scan + notebooks-strip + i18n-check (obligatorio antes de PR)
make lint                   # ruff check + ruff format --check + mypy (backend + ml) + pnpm lint
make format                 # ruff format
make secrets-scan           # gitleaks detect --no-banner --redact
make notebooks-strip        # nbstripout sobre notebooks/*.ipynb
make i18n-check             # valida claves it/es/en sincronizadas
```

## Base de datos (dbmate)

```bash
make db-migrate             # dbmate up
make db-rollback            # dbmate down
make db-new name=xxx        # dbmate new create_xxx_table
make db-status
make db-seed
make db-shell
```

## ML / Training

```bash
make train-l4 epic=E4 us=US-020          # spot L4 24GB (baselines, dev)
make train-h100 window=V3 script=train_gemma4_lora.py
make azure-h100-start       # enciende VM Azure H100 spot con auto-shutdown 12h
make azure-h100-stop
make azure-h100-status
make serve-qwen35           # vLLM Qwen3.5-35B-A3B en H100
make dvc-push
make dvc-pull
make mlflow-ui              # MLflow UI :5000
make dagster-ui             # Dagster UI :3001
```

## Eval

```bash
make eval-agromind variant=gemini
make eval-agromind variant=qwen35
make eval-geoanalyst variant=gemini
```

## Tests

```bash
make test                   # pytest backend con coverage ≥70 %
make test-unit
make test-integration
make test-e2e               # Playwright
make test-frontend          # vitest
```

## Terraform

```bash
make tf-init env=dev
make tf-plan env=dev
make tf-apply env=dev
make tf-fmt
make tf-validate env=dev
```

## Deploy

```bash
make deploy-staging         # Cloud Build → staging
make deploy-prod            # solo desde main
```

## FinOps & Seguridad

```bash
make security-audit
make cost-audit
make scale-to-zero-check
```
