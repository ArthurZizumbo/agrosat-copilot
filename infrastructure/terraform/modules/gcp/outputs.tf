output "api_url" {
  description = "Public URL of the Cloud Run api service."
  value       = google_cloud_run_v2_service.service["api"].uri
}

output "frontend_url" {
  description = "Public URL of the Cloud Run frontend service."
  value       = google_cloud_run_v2_service.service["frontend"].uri
}

output "tiling_url" {
  description = "Public URL of the Cloud Run tiling service."
  value       = google_cloud_run_v2_service.service["tiling"].uri
}

output "inference_worker_url" {
  description = "Internal URL of the inference worker Cloud Run service."
  value       = google_cloud_run_v2_service.service["inference-worker"].uri
}

output "db_connection_name" {
  description = "Cloud SQL instance connection name (project:region:instance) for Auth Proxy."
  value       = google_sql_database_instance.postgres.connection_name
}

output "db_instance_name" {
  description = "Cloud SQL instance short name."
  value       = google_sql_database_instance.postgres.name
}

output "gcs_data_bucket" {
  description = "GCS bucket for raw + processed data."
  value       = google_storage_bucket.data.name
}

output "gcs_artifacts_bucket" {
  description = "GCS bucket for MLflow artifacts."
  value       = google_storage_bucket.artifacts.name
}

output "gcs_dvc_bucket" {
  description = "GCS bucket configured as DVC remote."
  value       = google_storage_bucket.dvc_remote.name
}

output "gcs_tfstate_bucket" {
  description = "GCS bucket used as Terraform remote state backend."
  value       = google_storage_bucket.tfstate.name
}

output "artifact_registry_url" {
  description = "Artifact Registry Docker repo URL prefix."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "pubsub_inference_topic" {
  description = "Pub/Sub topic for inference jobs."
  value       = google_pubsub_topic.inference_jobs.name
}

output "pubsub_inference_results_topic" {
  description = "Pub/Sub topic for inference results."
  value       = google_pubsub_topic.inference_results.name
}

output "service_account_emails" {
  description = "Map of role -> service account email."
  value       = { for k, sa in google_service_account.sa : k => sa.email }
}

output "db_password_secret_id" {
  description = "Secret Manager secret id holding the Cloud SQL password."
  value       = google_secret_manager_secret.db_password.secret_id
}
