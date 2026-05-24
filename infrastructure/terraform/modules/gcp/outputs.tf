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

output "gcs_vertex_artifacts_bucket" {
  description = "GCS bucket dedicado a outputs de jobs Vertex AI L4 (US-022b-A, sufijado por environment)."
  value       = google_storage_bucket.vertex_artifacts.name
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

# US-022b-A — MLflow + ml-train runner outputs
output "mlflow_tracking_uri" {
  description = "MLflow tracking server URI (Cloud Run v2 scale-to-zero). Pasar como MLFLOW_TRACKING_URI a los jobs Vertex AI L4."
  value       = google_cloud_run_v2_service.mlflow.uri
}

output "mlflow_service_name" {
  description = "MLflow Cloud Run service name."
  value       = google_cloud_run_v2_service.mlflow.name
}

output "ml_train_runner_sa_email" {
  description = "Service account email del runner Vertex AI L4 (US-022b-A). Referenciado por ml/configs/l4_spot.yaml."
  value       = google_service_account.sa["ml_train_run"].email
}

# US-022-c P1 etapa 5 fix (2026-05-24): outputs de la VM FarSLIP Pub/Sub.
output "farslip_vm_name" {
  description = "Compute Engine instance name de la VM FarSLIP L4 (event-driven via Pub/Sub)."
  value       = google_compute_instance.farslip_trainer.name
}

output "farslip_vm_zone" {
  description = "Zona GCP donde corre la VM FarSLIP."
  value       = google_compute_instance.farslip_trainer.zone
}

output "farslip_jobs_topic" {
  description = "Nombre del topic Pub/Sub para encolar trabajos FarSLIP. Publish: gcloud pubsub topics publish."
  value       = google_pubsub_topic.farslip_jobs.name
}

output "farslip_vm_sa_email" {
  description = "Email de la SA dedicada a la VM FarSLIP (least privilege)."
  value       = google_service_account.farslip_vm_sa.email
}

output "farslip_vm_start_command" {
  description = "Comando exacto para arrancar la VM (Compute Engine la deja TERMINATED por default)."
  value       = "gcloud compute instances start ${google_compute_instance.farslip_trainer.name} --zone=${google_compute_instance.farslip_trainer.zone} --project=${var.project_id}"
}

output "farslip_publish_example" {
  description = "Comando exemplar para encolar un trabajo via Pub/Sub."
  value       = "gcloud pubsub topics publish ${google_pubsub_topic.farslip_jobs.name} --message='{\"command\":\"make farslip-train-smoke\",\"label\":\"smoke-2026-05-24\",\"timeout_seconds\":3600}' --project=${var.project_id}"
}
