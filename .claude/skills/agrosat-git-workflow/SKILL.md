---
name: agrosat-git-workflow
description: Manage Conventional Commits, branches (feature/E{epic}-US-XXX-{slug}), and PR workflow for AgroSatCopilot. Use when committing, creating branches, opening PRs, or closing User Stories.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Git Workflow Skill

## Branches

- `main` — protegido, deploys a prod desde aquí
- `develop` — integración continua, deploys a staging
- `feature/E{epic}-US-XXX-{slug}` — una rama por user story
- `hotfix/XXX` — solo para fixes urgentes en prod

## Conventional Commits con Scope de Epica

```
feat(E0): add cookiecutter template for monorepo
fix(E7): handle vLLM 503 in agent fallback
docs(E1): document AlphaEarth licensing in DATA_LICENSE.md
test(E5): add TSViT unit tests
chore(E10): bump Evidently to 0.4.40
refactor(E8): extract STAC search to service layer
perf(E6): enable FlashAttention-2 in Gemma 4 training
```

| Tipo | Cuándo |
|------|--------|
| feat | nueva feature |
| fix | bug fix |
| docs | documentación |
| style | formato (no afecta lógica) |
| refactor | refactor sin cambio de comportamiento |
| perf | mejora de performance |
| test | añadir/corregir tests |
| chore | mantenimiento, deps |
| ci | pipeline CI/CD |

## Flujo por User Story

```bash
# Inicio
git checkout develop && git pull
git checkout -b feature/E6-US-029-gemma4-lora

# Trabajo iterativo
git add ml/train/train_gemma4_lora.py configs/gemma4_h100.yaml
git commit -m "feat(E6): scaffold Gemma 4 LoRA training script"

git add ml/utils/mlflow_utils.py
git commit -m "feat(E6): add @track_experiment decorator"

# Push y PR
git push -u origin feature/E6-US-029-gemma4-lora
gh pr create --base develop --title "feat(E6): US-029 fine-tune Gemma 4 26B-MoE LoRA" \
  --body "## US-029\n\n[Refinamiento v5](context/RefinamientoPlaneacionAgroSatCopilot_v5.md)..."

# Tras merge a develop
gh pr merge --squash
git checkout develop && git pull
git branch -d feature/E6-US-029-gemma4-lora
```

## Cierre de US

```bash
# 1. PR mergeado a develop
# 2. Crear docs/us-resolved/us-029.md con resumen
# 3. Verificar MLflow run + DVC commit + atribución de licencias
# 4. Marcar US como completada en tracking (notion/asana/...)
```

## Hotfix Flow

```bash
git checkout main
git checkout -b hotfix/critical-csp-issue
# fix...
git commit -m "fix(security): close CSP wildcard"
git push -u origin hotfix/critical-csp-issue
gh pr create --base main
# tras merge: cherry-pick a develop también
```

## Quality Gates (sin pre-commit)

El proyecto NO usa `pre-commit`. Las garantías que antes vivían como hooks ahora se ejecutan vía Makefile (manual antes de PR) y GitHub Actions (CI):

```bash
make lint              # ruff check + ruff format --check + mypy (backend + ml) + pnpm lint
make secrets-scan      # gitleaks detect --no-banner --redact
make notebooks-strip   # nbstripout sobre notebooks/*.ipynb
make i18n-check        # valida claves it/es/en sincronizadas
make check             # corre los 4 anteriores
```

Todos estos comandos son obligatorios en CI (`.github/workflows/ci.yml`) sobre cada PR a `develop` y `main`.

## QA Checklist

- [ ] Branch sigue convención
- [ ] Conventional Commits con scope
- [ ] Sin emojis
- [ ] Pre-commit hooks pasan
- [ ] PR description referencia US-XXX
- [ ] PR target es develop (no main)
- [ ] Code review aprobado por 1+ team
- [ ] CI verde antes de merge
