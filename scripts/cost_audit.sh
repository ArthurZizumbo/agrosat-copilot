#!/usr/bin/env bash
set -euo pipefail
echo "=== GCP costs (last 30 days) ==="
gcloud billing budgets list || echo "(billing API not configured)"

echo ""
echo "=== Azure costs (last 30 days) ==="
az consumption usage list --top 100 \
  --query "[].{date:usageStart, service:meterCategory, cost:pretaxCost}" \
  --output table 2>/dev/null || echo "(az CLI not authenticated)"

echo ""
echo "=== Cloud Run scale-to-zero check ==="
gcloud run services list --format='table(metadata.name,status.url,metadata.annotations.\"run.googleapis.com/minScale\")' || true

echo ""
echo "=== Azure H100 VM state ==="
bash scripts/azure_h100_status.sh || echo "(VM not provisioned yet)"
