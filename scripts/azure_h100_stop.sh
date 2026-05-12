#!/usr/bin/env bash
set -euo pipefail
RG="${AZURE_RESOURCE_GROUP:-agrosat-rg}"
VM="${AZURE_H100_VM_NAME:-agrosat-h100-prod}"

az vm deallocate --resource-group "$RG" --name "$VM"
echo "VM deallocated. Spot price no longer accumulating."
