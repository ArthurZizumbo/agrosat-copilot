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
gcloud run services list --format='table(metadata.name,status.url,metadata.annotations."run.googleapis.com/minScale")' || true

echo ""
echo "=== Vertex AI custom-jobs (US-022b-A — debe ser vacio o JOB_STATE_SUCCEEDED/FAILED/CANCELLED) ==="
# A-6 FinOps gate: cualquier job en JOB_STATE_RUNNING o JOB_STATE_PENDING indica
# que hay una L4 colgada quemando spot. El script imprime pero NO falla (audit-only);
# si quieres gate duro en CI, agrega `| grep -v RUNNING || exit 1` aguas abajo.
gcloud ai custom-jobs list \
  --region="${GCP_REGION:-us-central1}" \
  --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING" \
  --format='table(displayName,state,createTime)' 2>/dev/null \
  || echo "(gcloud ai not available or no jobs)"

echo ""
echo "=== Azure H100 VM state ==="
bash scripts/azure_h100_status.sh || echo "(VM not provisioned yet)"
