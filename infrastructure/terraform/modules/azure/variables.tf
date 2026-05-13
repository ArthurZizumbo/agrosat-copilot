variable "location" {
  description = "Azure region (westeurope = Italy proximity, H100 availability)."
  type        = string
  default     = "westeurope"
}

variable "resource_group_name" {
  description = "Resource group name."
  type        = string
  default     = "agrosat-rg"
}

variable "vm_name" {
  description = "Name of the H100 virtual machine."
  type        = string
  default     = "agrosat-h100-prod"
}

variable "vm_size" {
  description = "Azure VM size (H100 NVL 96GB)."
  type        = string
  default     = "Standard_NC40ads_H100_v5"
}

variable "use_spot" {
  description = "Use spot pricing (true) or on-demand (false). Default true to honor budget."
  type        = bool
  default     = true
}

variable "max_bid_price" {
  description = "Maximum spot bid price USD/hour. -1 means up to current market price."
  type        = number
  default     = -1
}

variable "allowed_ssh_cidrs" {
  description = "List of CIDR ranges allowed to SSH (whitelist of dev IPs, max 3)."
  type        = list(string)
  validation {
    condition     = length(var.allowed_ssh_cidrs) > 0 && length(var.allowed_ssh_cidrs) <= 5
    error_message = "Provide between 1 and 5 CIDRs (whitelist dev IPs)."
  }
}

variable "admin_username" {
  description = "Linux admin username."
  type        = string
  default     = "agrosat"
}

variable "admin_ssh_public_key" {
  description = "SSH public key for admin user."
  type        = string
}

variable "shutdown_time_utc" {
  description = "Daily auto-shutdown time in HHMM UTC (e.g. 2300 for 23:00 UTC)."
  type        = string
  default     = "2300"
}

variable "shutdown_notification_email" {
  description = "Email to notify before auto-shutdown."
  type        = string
  default     = ""
}

variable "blob_container_name" {
  description = "Blob container for LoRA checkpoints."
  type        = string
  default     = "agrosat-checkpoints"
}

variable "tags" {
  description = "Common Azure tags."
  type        = map(string)
  default = {
    project    = "agrosat"
    owner      = "mlops"
    managed_by = "terraform"
  }
}

variable "purge_protection_enabled" {
  description = "Habilita purge protection en Key Vault (CIS Azure). Default false en dev por ADR-002; true en staging/prod."
  type        = bool
  default     = false
}
