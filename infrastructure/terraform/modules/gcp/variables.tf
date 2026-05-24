variable "project_id" {
  description = "GCP project id (e.g. agrosat-prod)."
  type        = string
}

variable "region" {
  description = "Primary GCP region for all regional resources."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment label (dev|staging|prod). Drives naming suffix."
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of dev, staging, prod."
  }
}

variable "db_name" {
  description = "Cloud SQL database name."
  type        = string
  default     = "agrosat"
}

variable "db_user" {
  description = "Cloud SQL primary user."
  type        = string
  default     = "agrosat"
}

variable "db_tier" {
  description = "Cloud SQL machine tier."
  type        = string
  default     = "db-f1-micro"
}

variable "db_disk_size_gb" {
  description = "Cloud SQL disk size in GB."
  type        = number
  default     = 20
}

variable "cloudrun_min_instances" {
  description = "Minimum Cloud Run instances (0 = scale-to-zero, mandatory in dev)."
  type        = number
  default     = 0
  validation {
    condition     = var.cloudrun_min_instances >= 0
    error_message = "cloudrun_min_instances cannot be negative."
  }
}

variable "cloudrun_max_instances" {
  description = "Maximum Cloud Run instances per service."
  type        = number
  default     = 10
}

variable "artifact_registry_repo_id" {
  description = "Artifact Registry Docker repository id."
  type        = string
  default     = "agrosat"
}

variable "labels" {
  description = "Common resource labels applied where supported."
  type        = map(string)
  default = {
    project = "agrosat"
    owner   = "mlops"
  }
}

variable "farslip_vm_zone" {
  description = "GCP zone donde provisionar la VM FarSLIP L4 y su disco persistente. us-west4-a (Las Vegas) confirmado con capacidad 2026-05-24 — us-central1 entero STOCKOUT G2+L4. Quota Compute Engine NVIDIA_L4_GPUS ya aprobada (=1.0) en us-west4."
  type        = string
  default     = "us-west4-a"
}
