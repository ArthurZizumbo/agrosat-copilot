variable "project_id" {
  description = "GCP project id."
  type        = string
  default     = "agrosat-prod"
}

variable "gcp_region" {
  description = "Primary GCP region."
  type        = string
  default     = "europe-west1"
}

variable "azure_subscription_id" {
  description = "Azure subscription id."
  type        = string
}

variable "azure_location" {
  description = "Azure region for H100 VM."
  type        = string
  default     = "westeurope"
}

variable "allowed_ssh_cidrs" {
  description = "Whitelisted CIDRs allowed to SSH into the H100 VM (max 5)."
  type        = list(string)
}

variable "admin_ssh_public_key" {
  description = "SSH public key for the Azure VM admin user."
  type        = string
}

variable "use_spot" {
  description = "Use Azure Spot pricing for H100 (default true)."
  type        = bool
  default     = true
}

variable "max_bid_price" {
  description = "Maximum spot bid price USD/hour. -1 = market."
  type        = number
  default     = -1
}

variable "shutdown_notification_email" {
  description = "Email for auto-shutdown notifications."
  type        = string
  default     = ""
}
