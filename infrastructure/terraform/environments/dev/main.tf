# AgroSatCopilot — environment: dev
# Single active environment per ADR-002. Invokes GCP + Azure modules with real values.

provider "google" {
  project = var.project_id
  region  = var.gcp_region
}

provider "google-beta" {
  project = var.project_id
  region  = var.gcp_region
}

provider "azurerm" {
  subscription_id = var.azure_subscription_id
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

module "gcp" {
  source = "../../modules/gcp"

  project_id             = var.project_id
  region                 = var.gcp_region
  environment            = "dev"
  db_name                = "agrosat"
  db_user                = "agrosat"
  db_tier                = "db-f1-micro"
  cloudrun_min_instances = 0
  cloudrun_max_instances = 10

  labels = {
    project     = "agrosat"
    environment = "dev"
    owner       = "mlops"
  }
}

module "azure" {
  source = "../../modules/azure"

  location                    = var.azure_location
  resource_group_name         = "agrosat-rg"
  vm_name                     = "agrosat-h100-prod"
  vm_size                     = "Standard_NC40ads_H100_v5"
  use_spot                    = var.use_spot
  max_bid_price               = var.max_bid_price
  allowed_ssh_cidrs           = var.allowed_ssh_cidrs
  admin_username              = "agrosat"
  admin_ssh_public_key        = var.admin_ssh_public_key
  shutdown_time_utc           = "2300"
  shutdown_notification_email = var.shutdown_notification_email
  blob_container_name         = "agrosat-checkpoints"

  tags = {
    project     = "agrosat"
    environment = "dev"
    owner       = "mlops"
  }
}
