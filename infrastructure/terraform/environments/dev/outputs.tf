output "api_url" {
  description = "Cloud Run API public URL."
  value       = module.gcp.api_url
}

output "frontend_url" {
  description = "Cloud Run frontend public URL."
  value       = module.gcp.frontend_url
}

output "tiling_url" {
  description = "Cloud Run TiTiler public URL."
  value       = module.gcp.tiling_url
}

output "db_connection_name" {
  description = "Cloud SQL connection name for Auth Proxy."
  value       = module.gcp.db_connection_name
}

output "gcs_data_bucket" {
  description = "Data bucket."
  value       = module.gcp.gcs_data_bucket
}

output "gcs_artifacts_bucket" {
  description = "MLflow artifacts bucket."
  value       = module.gcp.gcs_artifacts_bucket
}

output "gcs_dvc_bucket" {
  description = "DVC remote bucket."
  value       = module.gcp.gcs_dvc_bucket
}

output "gcs_tfstate_bucket" {
  description = "Terraform state bucket."
  value       = module.gcp.gcs_tfstate_bucket
}

output "artifact_registry_url" {
  description = "Artifact Registry Docker prefix."
  value       = module.gcp.artifact_registry_url
}

output "pubsub_inference_topic" {
  description = "Pub/Sub inference jobs topic."
  value       = module.gcp.pubsub_inference_topic
}

output "service_account_emails" {
  description = "GCP service account emails by role."
  value       = module.gcp.service_account_emails
}

output "azure_h100_vm_name" {
  description = "Azure H100 VM name."
  value       = module.azure.vm_name
}

output "azure_h100_public_ip" {
  description = "Azure H100 public IP (SSH from whitelisted CIDRs only)."
  value       = module.azure.vm_public_ip
}

output "azure_blob_endpoint" {
  description = "Azure Blob endpoint for LoRA checkpoints."
  value       = module.azure.blob_endpoint
}

output "azure_key_vault_uri" {
  description = "Azure Key Vault URI."
  value       = module.azure.key_vault_uri
}
