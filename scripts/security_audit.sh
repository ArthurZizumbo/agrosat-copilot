#!/usr/bin/env bash
set -e
echo "=== Secret scanning (gitleaks) ==="
gitleaks detect --no-banner --redact || true

echo "=== Backend deps (pip-audit) ==="
cd backend && poetry run pip-audit || true; cd ..

echo "=== Frontend deps (pnpm audit) ==="
cd frontend && pnpm audit --prod || true; cd ..

echo "=== Hardcoded secrets check ==="
grep -rE "(api[_-]?key|password|secret).*=.*['\"][a-zA-Z0-9]{20,}" backend/app/ frontend/ 2>/dev/null && echo "FAIL: possible secrets" || echo "OK"

echo "=== ruff security (S rules) ==="
cd backend && poetry run ruff check --select=S . || true; cd ..
