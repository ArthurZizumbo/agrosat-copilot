# AgroSatCopilot — Module: Azure (H100 single VM + Blob + Key Vault)
# Enforces irrevocables:
# - Size: Standard_NC40ads_H100_v5 (H100 NVL 96GB)
# - Spot priority by default (eviction_policy=Deallocate), toggleable
# - Auto-shutdown daily (mandatory)
# - SSH NSG whitelist (no 0.0.0.0/0)
# - Secrets in Key Vault

locals {
  storage_account_name = lower(replace("agrosat${var.location}sa", "-", ""))
}

resource "random_string" "sa_suffix" {
  length  = 5
  upper   = false
  special = false
  numeric = true
}

# ----------------------------------------------------------------------------
# Resource group
# ----------------------------------------------------------------------------
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# ----------------------------------------------------------------------------
# Networking: VNet + Subnet + NSG (SSH whitelist)
# ----------------------------------------------------------------------------
resource "azurerm_virtual_network" "vnet" {
  name                = "agrosat-vnet"
  address_space       = ["10.20.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags
}

resource "azurerm_subnet" "h100" {
  name                 = "agrosat-h100-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.20.1.0/24"]
}

resource "azurerm_network_security_group" "h100" {
  name                = "agrosat-h100-nsg"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags

  security_rule {
    name                       = "AllowSSHFromDevs"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefixes    = var.allowed_ssh_cidrs
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "DenyAllInbound"
    priority                   = 4096
    direction                  = "Inbound"
    access                     = "Deny"
    protocol                   = "*"
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_subnet_network_security_group_association" "h100" {
  subnet_id                 = azurerm_subnet.h100.id
  network_security_group_id = azurerm_network_security_group.h100.id
}

resource "azurerm_public_ip" "h100" {
  name                = "${var.vm_name}-pip"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  allocation_method   = "Static"
  sku                 = "Standard"
  tags                = var.tags
}

resource "azurerm_network_interface" "h100" {
  name                = "${var.vm_name}-nic"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags

  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.h100.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.h100.id
  }
}

# ----------------------------------------------------------------------------
# H100 VM — spot by default, on-demand via use_spot=false
# ----------------------------------------------------------------------------
resource "azurerm_linux_virtual_machine" "h100" {
  name                = var.vm_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  size                = var.vm_size
  admin_username      = var.admin_username

  # Spot toggle
  priority        = var.use_spot ? "Spot" : "Regular"
  eviction_policy = var.use_spot ? "Deallocate" : null
  max_bid_price   = var.use_spot ? var.max_bid_price : null

  network_interface_ids = [azurerm_network_interface.h100.id]

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.admin_ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 256
  }

  source_image_reference {
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-hpc"
    sku       = "2204"
    version   = "latest"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = merge(var.tags, {
    autoshutdown = "enabled"
    priority     = var.use_spot ? "spot" : "ondemand"
  })
}

# ----------------------------------------------------------------------------
# Auto-shutdown (mandatory) — daily at shutdown_time_utc
# ----------------------------------------------------------------------------
resource "azurerm_dev_test_global_vm_shutdown_schedule" "h100" {
  virtual_machine_id    = azurerm_linux_virtual_machine.h100.id
  location              = azurerm_resource_group.rg.location
  enabled               = true
  daily_recurrence_time = var.shutdown_time_utc
  timezone              = "UTC"

  notification_settings {
    enabled         = var.shutdown_notification_email != ""
    time_in_minutes = 30
    email           = var.shutdown_notification_email != "" ? var.shutdown_notification_email : null
  }

  tags = var.tags
}

# ----------------------------------------------------------------------------
# Blob storage for LoRA checkpoints
# ----------------------------------------------------------------------------
resource "azurerm_storage_account" "checkpoints" {
  name                            = substr("${local.storage_account_name}${random_string.sa_suffix.result}", 0, 24)
  resource_group_name             = azurerm_resource_group.rg.name
  location                        = azurerm_resource_group.rg.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  account_kind                    = "StorageV2"
  access_tier                     = "Hot"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  https_traffic_only_enabled      = true

  blob_properties {
    versioning_enabled       = true
    change_feed_enabled      = false
    last_access_time_enabled = false

    delete_retention_policy {
      days = 14
    }
  }

  tags = var.tags
}

resource "azurerm_storage_container" "checkpoints" {
  name                  = var.blob_container_name
  storage_account_id    = azurerm_storage_account.checkpoints.id
  container_access_type = "private"
}

# ----------------------------------------------------------------------------
# Key Vault — secrets for HF token, vLLM key, GCP SA JSON when running on H100
# ----------------------------------------------------------------------------
data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "kv" {
  name                       = substr("agrosat-kv-${random_string.sa_suffix.result}", 0, 24)
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  # CIS Azure Benchmark recomienda purge_protection_enabled=true.
  # En dev permitimos opt-out para poder destruir el KV sin esperar 7-90 dias.
  purge_protection_enabled = var.purge_protection_enabled

  rbac_authorization_enabled = true

  tags = var.tags
}

# Grant VM managed identity read access to Key Vault
resource "azurerm_role_assignment" "vm_kv_reader" {
  scope                = azurerm_key_vault.kv.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_linux_virtual_machine.h100.identity[0].principal_id
}

# Grant deployer (current Terraform principal) acceso a secretos.
# CIS Azure: usar "Key Vault Secrets Officer" en lugar de "Administrator"
# para que el deployer pueda gestionar secretos pero no policies/keys/certs.
resource "azurerm_role_assignment" "deployer_kv_secrets" {
  scope                = azurerm_key_vault.kv.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}
