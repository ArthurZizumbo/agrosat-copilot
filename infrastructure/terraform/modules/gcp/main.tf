# AgroSatCopilot — Module: GCP
# Provisions Cloud Run x4, Cloud SQL (PostGIS-ready), GCS buckets, Pub/Sub,
# Secret Manager, Artifact Registry, IAM (least privilege).
#
# Decisions enforced (ADR-002 + irrevocables):
# - region: europe-west1
# - Cloud Run min_instances = 0 (scale-to-zero)
# - Cloud SQL with backups + PITR
# - NO roles/owner on runtime SAs
# - All secrets in Secret Manager (no plaintext .tf)

locals {
  name_suffix = var.environment
  common_labels = merge(var.labels, {
    environment = var.environment
    managed_by  = "terraform"
  })

  cloud_run_services = {
    api = {
      image  = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo_id}/api:latest"
      cpu    = "1"
      memory = "512Mi"
      port   = 8000
      public = true
    }
    frontend = {
      image  = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo_id}/frontend:latest"
      cpu    = "1"
      memory = "256Mi"
      port   = 3000
      public = true
    }
    tiling = {
      image  = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo_id}/tiling:latest"
      cpu    = "1"
      memory = "512Mi"
      port   = 8000
      public = true
    }
    inference-worker = {
      image  = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo_id}/inference-worker:latest"
      cpu    = "4"
      memory = "16Gi"
      port   = 8000
      public = false
    }
  }

  service_accounts = {
    api      = "agrosat-api-sa"
    frontend = "agrosat-frontend-sa"
    tiling   = "agrosat-tiling-sa"
    worker   = "agrosat-worker-sa"
    dagster  = "agrosat-dagster-sa"
    ci       = "agrosat-ci-sa"
  }

  secret_ids = [
    "agrosat-hf-token",
    "agrosat-clerk-secret",
    "agrosat-cdse-password",
    "agrosat-gee-sa-json",
    "agrosat-vllm-api-key",
    "agrosat-db-password",
    "agrosat-upstash-rest-url",
    "agrosat-upstash-rest-token",
  ]
}

# ----------------------------------------------------------------------------
# IAM — service accounts (least privilege)
# ----------------------------------------------------------------------------
resource "google_service_account" "sa" {
  for_each     = local.service_accounts
  project      = var.project_id
  account_id   = each.value
  display_name = "AgroSatCopilot ${each.key} (${var.environment})"
}

# Minimal role bindings per SA (NO roles/owner anywhere)
resource "google_project_iam_member" "api_roles" {
  for_each = toset([
    "roles/cloudsql.client",
    "roles/pubsub.publisher",
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectViewer",
    "roles/aiplatform.user",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.sa["api"].email}"
}

resource "google_project_iam_member" "frontend_roles" {
  for_each = toset([
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.sa["frontend"].email}"
}

resource "google_project_iam_member" "tiling_roles" {
  for_each = toset([
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.sa["tiling"].email}"
}

resource "google_project_iam_member" "worker_roles" {
  for_each = toset([
    "roles/cloudsql.client",
    "roles/pubsub.subscriber",
    "roles/pubsub.publisher",
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectAdmin",
    "roles/aiplatform.user",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.sa["worker"].email}"
}

resource "google_project_iam_member" "dagster_roles" {
  for_each = toset([
    "roles/cloudsql.client",
    "roles/storage.objectAdmin",
    "roles/pubsub.editor",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.sa["dagster"].email}"
}

resource "google_project_iam_member" "ci_roles" {
  for_each = toset([
    "roles/run.developer",
    "roles/artifactregistry.writer",
    "roles/cloudbuild.builds.editor",
    "roles/iam.serviceAccountUser",
    "roles/storage.objectAdmin",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.sa["ci"].email}"
}

# ----------------------------------------------------------------------------
# Artifact Registry — Docker repo
# ----------------------------------------------------------------------------
resource "google_artifact_registry_repository" "docker" {
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_registry_repo_id
  description   = "AgroSatCopilot container images (api, frontend, tiling, worker)."
  format        = "DOCKER"
  labels        = local.common_labels
}

# ----------------------------------------------------------------------------
# Cloud SQL — PostgreSQL 15 (PostGIS + pgvector via flags / extensions)
# ----------------------------------------------------------------------------
resource "random_password" "db_password" {
  length  = 24
  special = true
}

resource "google_secret_manager_secret" "db_password" {
  project   = var.project_id
  secret_id = "agrosat-db-password"
  replication {
    auto {}
  }
  labels = local.common_labels
}

resource "google_secret_manager_secret_version" "db_password_v" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = random_password.db_password.result
}

resource "google_sql_database_instance" "postgres" {
  project          = var.project_id
  name             = "agrosat-pg-${local.name_suffix}"
  region           = var.region
  database_version = "POSTGRES_15"

  deletion_protection = var.environment == "prod"

  settings {
    tier              = var.db_tier
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"
    disk_size         = var.db_disk_size_gb
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 7
        retention_unit   = "COUNT"
      }
    }

    ip_configuration {
      ipv4_enabled    = true
      ssl_mode        = "ENCRYPTED_ONLY"
      private_network = null
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = false
    }

    user_labels = local.common_labels
  }
}

resource "google_sql_database" "agrosat" {
  project  = var.project_id
  name     = var.db_name
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "agrosat" {
  project  = var.project_id
  name     = var.db_user
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}

# ----------------------------------------------------------------------------
# GCS buckets
# ----------------------------------------------------------------------------
locals {
  buckets = {
    data       = "agrosat-data-${local.name_suffix}"
    artifacts  = "agrosat-artifacts-${local.name_suffix}"
    dvc_remote = "agrosat-dvc-remote-${local.name_suffix}"
    tfstate    = "agrosat-tfstate"
  }
}

resource "google_storage_bucket" "data" {
  project                     = var.project_id
  name                        = local.buckets.data
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = var.environment != "prod"

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = local.common_labels
}

resource "google_storage_bucket" "artifacts" {
  project                     = var.project_id
  name                        = local.buckets.artifacts
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = var.environment != "prod"

  versioning {
    enabled = true
  }

  labels = local.common_labels
}

resource "google_storage_bucket" "dvc_remote" {
  project                     = var.project_id
  name                        = local.buckets.dvc_remote
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = var.environment != "prod"

  versioning {
    enabled = true
  }

  labels = local.common_labels
}

resource "google_storage_bucket" "tfstate" {
  project                     = var.project_id
  name                        = local.buckets.tfstate
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 30
    }
    action {
      type = "Delete"
    }
  }

  labels = local.common_labels
}

# ----------------------------------------------------------------------------
# Pub/Sub — inference jobs + results
# ----------------------------------------------------------------------------
resource "google_pubsub_topic" "inference_jobs" {
  project = var.project_id
  name    = "inference-jobs-${local.name_suffix}"
  labels  = local.common_labels

  message_retention_duration = "86400s"
}

resource "google_pubsub_topic" "inference_results" {
  project = var.project_id
  name    = "inference-results-${local.name_suffix}"
  labels  = local.common_labels

  message_retention_duration = "86400s"
}

resource "google_pubsub_subscription" "inference_jobs_worker" {
  project = var.project_id
  name    = "inference-jobs-worker-${local.name_suffix}"
  topic   = google_pubsub_topic.inference_jobs.name

  ack_deadline_seconds       = 600
  message_retention_duration = "604800s"
  retain_acked_messages      = false

  expiration_policy {
    ttl = ""
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  labels = local.common_labels
}

resource "google_pubsub_subscription" "inference_results_api" {
  project = var.project_id
  name    = "inference-results-api-${local.name_suffix}"
  topic   = google_pubsub_topic.inference_results.name

  ack_deadline_seconds       = 60
  message_retention_duration = "604800s"

  expiration_policy {
    ttl = ""
  }

  labels = local.common_labels
}

# ----------------------------------------------------------------------------
# Secret Manager — non-DB secrets (5 of 6; db_password defined above)
# ----------------------------------------------------------------------------
resource "google_secret_manager_secret" "other" {
  for_each  = toset([for s in local.secret_ids : s if s != "agrosat-db-password"])
  project   = var.project_id
  secret_id = each.value

  replication {
    auto {}
  }

  labels = local.common_labels
}

# ----------------------------------------------------------------------------
# Cloud Run v2 — 4 services (api, frontend, tiling, inference-worker)
# ----------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "service" {
  for_each = local.cloud_run_services

  project  = var.project_id
  name     = "agrosat-${each.key}-${local.name_suffix}"
  location = var.region
  ingress  = each.value.public ? "INGRESS_TRAFFIC_ALL" : "INGRESS_TRAFFIC_INTERNAL_ONLY"

  labels = local.common_labels

  template {
    service_account = google_service_account.sa[
      each.key == "inference-worker" ? "worker" : each.key
    ].email

    scaling {
      min_instance_count = var.cloudrun_min_instances
      max_instance_count = var.cloudrun_max_instances
    }

    containers {
      image = each.value.image

      resources {
        limits = {
          cpu    = each.value.cpu
          memory = each.value.memory
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      ports {
        container_port = each.value.port
      }

      env {
        name  = "ENV"
        value = var.environment
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
    }

    timeout = "300s"

    labels = local.common_labels
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_artifact_registry_repository.docker,
    google_project_iam_member.api_roles,
    google_project_iam_member.frontend_roles,
    google_project_iam_member.tiling_roles,
    google_project_iam_member.worker_roles,
  ]
}

# Allow public unauth invocation only for services flagged public
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  for_each = { for k, v in local.cloud_run_services : k => v if v.public }

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.service[each.key].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
