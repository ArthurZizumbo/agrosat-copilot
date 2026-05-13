.PHONY: help bootstrap bootstrap-gpu bootstrap-gpu-linux verify-structure dev stop test lint format check secrets-scan notebooks-strip i18n-check db-migrate db-rollback db-new db-status db-seed train-l4 train-h100 azure-h100-start azure-h100-stop azure-h100-status mlflow-ui dagster-ui dvc-push dvc-pull eda-sentinel2 eval-agromind eval-geoanalyst serve-qwen35 cost-audit deploy-staging deploy-prod tf-init tf-plan tf-apply tf-fmt tf-validate

help:
	@echo "AgroSatCopilot — comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

# === Bootstrap (reemplaza el hook post_gen_project.py de cookiecutter) ===
bootstrap:  ## Instala deps Python + Node (poetry + pnpm) — sin GPU
	poetry install --with dev,test,ml,geo,dagster,paper
	cd frontend && pnpm install

bootstrap-gpu:  ## Como bootstrap + torch CUDA 13.0 + bitsandbytes (Win/Linux con GPU NVIDIA)
	poetry install --with dev,test,ml,ml-gpu,geo,dagster,paper
	cd frontend && pnpm install

bootstrap-gpu-linux:  ## Como bootstrap-gpu + flash-attn + vllm (solo Linux, replica cloud)
	poetry install --with dev,test,ml,ml-gpu,ml-gpu-linux,geo,dagster,paper
	cd frontend && pnpm install

verify-structure:  ## Valida estructura de directorios (AC-4 de US-001)
	@bash scripts/verify_structure.sh

# === Dev ===
dev:  ## Levanta docker-compose con 8 servicios (carga puertos desde .env.local)
	docker compose --env-file .env.local up -d
	@echo "API: http://localhost:$${API_HOST_PORT:-8010}  Frontend: http://localhost:$${FRONTEND_HOST_PORT:-3010}  Dagster: http://localhost:$${DAGSTER_HOST_PORT:-3011}  MLflow: http://localhost:$${MLFLOW_HOST_PORT:-5010}"

stop:  ## Detiene docker-compose
	docker compose --env-file .env.local down

# === Lint & format ===
lint:  ## ruff + ruff format check + mypy
	cd backend && poetry run ruff check .
	cd backend && poetry run ruff format --check .
	cd backend && poetry run mypy app/
	cd ml && poetry run ruff check .
	cd frontend && pnpm lint

format:  ## ruff format
	cd backend && poetry run ruff format .
	cd ml && poetry run ruff format .

secrets-scan:  ## gitleaks secret scanning (reemplazo del hook pre-commit)
	gitleaks detect --no-banner --redact

notebooks-strip:  ## nbstripout sobre todos los notebooks (reemplazo del hook pre-commit)
	poetry run nbstripout notebooks/*.ipynb

i18n-check:  ## valida que las claves i18n existan en it/es/en
	cd frontend && pnpm i18n:check

check: lint secrets-scan notebooks-strip i18n-check  ## suite local previa a PR (reemplaza pre-commit)

# === Tests ===
test:  ## pytest backend con cobertura
	cd backend && poetry run pytest --cov=app --cov-report=term-missing --cov-fail-under=70

test-unit:
	cd backend && poetry run pytest tests/unit -v

test-integration:
	cd backend && poetry run pytest tests/integration -v

test-e2e:  ## Playwright E2E
	cd frontend && pnpm test:e2e

test-frontend:
	cd frontend && pnpm test

# === DB ===
db-migrate:  ## dbmate up
	dbmate up

db-rollback:
	dbmate down

db-new:  ## make db-new name=create_xxx
	dbmate new $(name)

db-status:
	dbmate status

db-seed:
	poetry run python scripts/seed.py

db-shell:
	docker compose exec postgres psql -U agrosat -d agrosat

# === ML / Training ===
train-l4:  ## Spot L4 24GB (baselines, dev)
	@echo "Lanzando job en GCP L4 spot para epic=$(epic) us=$(us)"
	gcloud ai custom-jobs create --region=$(GCP_REGION) \
	  --display-name=train-$(epic)-$(us) \
	  --config=ml/configs/l4_spot.yaml

train-h100:  ## Azure H100 96GB ventana=Vn script=xxx.py
	@echo "Lanzando $(script) en H100 ventana $(window)"
	ssh agrosat@$(shell az vm show -d -g agrosat-rg -n agrosat-h100-prod --query publicIps -o tsv) \
	  "cd ~/agro_sat_copilot && poetry run python ml/train/$(script)"

azure-h100-start:
	bash scripts/azure_h100_start.sh

azure-h100-stop:
	bash scripts/azure_h100_stop.sh

azure-h100-status:
	bash scripts/azure_h100_status.sh

serve-qwen35:  ## Lanza vLLM Qwen3.5-35B-A3B en H100
	bash scripts/serve_qwen35.sh

# === DVC / MLflow / Dagster ===
dvc-push:
	dvc push

dvc-pull:
	dvc pull

mlflow-ui:
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

dagster-ui:
	dagster dev -m dagster_project.definitions

# === EDA / Notebooks ===
eda-sentinel2:  ## Ejecuta el notebook US-010 con papermill (sample_size=100000)
	poetry run papermill notebooks/02a_eda_sentinel2.ipynb /tmp/02a_out.ipynb -p sample_size 100000

# === Eval ===
eval-agromind:  ## make eval-agromind variant=gemini
	poetry run python ml/agent/eval/eval_agromind.py --variant=$(variant)

eval-geoanalyst:
	poetry run python ml/agent/eval/eval_geoanalyst.py --variant=$(variant)

# === FinOps ===
cost-audit:
	bash scripts/cost_audit.sh

scale-to-zero-check:
	gcloud run services list --format='table(metadata.name,spec.template.spec.containers[0].resources.limits.cpu,spec.template.metadata.annotations.\"autoscaling.knative.dev/minScale\")'

# === Terraform ===
tf-init:  ## make tf-init env=dev
	cd infrastructure/terraform/environments/$(env) && terraform init

tf-plan:
	cd infrastructure/terraform/environments/$(env) && terraform plan -out tfplan

tf-apply:
	cd infrastructure/terraform/environments/$(env) && terraform apply tfplan

tf-fmt:
	terraform fmt -recursive infrastructure/terraform/

tf-validate:
	cd infrastructure/terraform/environments/$(env) && terraform validate

# === Deploy ===
deploy-staging:  ## Cloud Build → staging
	gcloud builds submit --config=infrastructure/cloudbuild.yaml --substitutions=_ENV=staging

deploy-prod:
	@[ "$(shell git rev-parse --abbrev-ref HEAD)" = "main" ] || (echo "ERROR: deploy-prod solo desde main"; exit 1)
	gcloud builds submit --config=infrastructure/cloudbuild.yaml --substitutions=_ENV=prod

# === Security ===
security-audit:
	bash scripts/security_audit.sh
