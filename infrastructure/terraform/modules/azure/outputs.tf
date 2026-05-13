output "vm_id" {
  description = "Resource id of the H100 VM."
  value       = azurerm_linux_virtual_machine.h100.id
}

output "vm_name" {
  description = "Name of the H100 VM."
  value       = azurerm_linux_virtual_machine.h100.name
}

output "vm_public_ip" {
  description = "Public IP of the H100 VM (for SSH from whitelisted CIDRs)."
  value       = azurerm_public_ip.h100.ip_address
}

output "vm_principal_id" {
  description = "System-assigned managed identity principal id of the VM."
  value       = azurerm_linux_virtual_machine.h100.identity[0].principal_id
}

output "resource_group_name" {
  description = "Resource group name."
  value       = azurerm_resource_group.rg.name
}

output "blob_endpoint" {
  description = "Primary blob endpoint URL."
  value       = azurerm_storage_account.checkpoints.primary_blob_endpoint
}

output "blob_container_name" {
  description = "Container name for LoRA checkpoints."
  value       = azurerm_storage_container.checkpoints.name
}

output "key_vault_id" {
  description = "Key Vault resource id."
  value       = azurerm_key_vault.kv.id
}

output "key_vault_uri" {
  description = "Key Vault URI."
  value       = azurerm_key_vault.kv.vault_uri
}
