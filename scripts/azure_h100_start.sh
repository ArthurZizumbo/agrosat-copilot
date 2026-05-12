#!/usr/bin/env bash
# Inicia VM Azure H100 spot con auto-shutdown 12h
set -euo pipefail
RG="${AZURE_RESOURCE_GROUP:-agrosat-rg}"
VM="${AZURE_H100_VM_NAME:-agrosat-h100-prod}"

echo "Starting Azure H100 spot VM..."
az vm start --resource-group "$RG" --name "$VM"

SHUTDOWN_TIME=$(date -u -d '+12 hours' +'%Y-%m-%dT%H:%M:%SZ')
az vm auto-shutdown -g "$RG" -n "$VM" --time "$SHUTDOWN_TIME"

IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
echo "H100 ready at $IP — auto-shutdown $SHUTDOWN_TIME (12 h)"
