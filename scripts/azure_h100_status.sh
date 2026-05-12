#!/usr/bin/env bash
set -euo pipefail
RG="${AZURE_RESOURCE_GROUP:-agrosat-rg}"
VM="${AZURE_H100_VM_NAME:-agrosat-h100-prod}"

az vm show -g "$RG" -n "$VM" --show-details --query \
  "{PowerState:powerState, IP:publicIps, AutoShutdown:tags.autoShutdown}"
