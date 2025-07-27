variable "resource_group_name" {
  description = "Name of the Resource Group"
  type        = string
  default     = "deepthoughtTerraformRG"
}

variable "environment" {
  description = "Deployment environment tag"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure location"
  type        = string
  default     = "eastus"
}

variable "vnet_name" {
  description = "Name of the Virtual Network"
  type        = string
  default     = "demo-vnet"
}

variable "vnet_address_space" {
  description = "Address space for the VNet"
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_name" {
  description = "Name of the subnet"
  type        = string
  default     = "default"
}

variable "subnet_prefix" {
  description = "Address prefix for the subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "vm_name" {
  description = "Name of the Virtual Machine"
  type        = string
  default     = "demo-vm"
}

variable "vm_size" {
  description = "Size of the Virtual Machine"
  type        = string
  default     = "Standard_B1s"
}

variable "admin_username" {
  description = "Admin username for the VM"
  type        = string
  default     = "azureuser"
}

variable "ssh_public_key_path" {
  description = "Path to the SSH public key"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "ssh_allowed_cidr" {
  description = "CIDR range allowed to SSH"
  type        = string
  default     = "69.92.191.166/32"
}

variable "image_publisher" {
  description = "VM image publisher"
  type        = string
  default     = "Canonical"
}

variable "image_offer" {
  description = "VM image offer"
  type        = string
  default     = "UbuntuServer"
}

variable "image_sku" {
  description = "VM image SKU"
  type        = string
  default     = "18.04-LTS"
}

variable "image_version" {
  description = "VM image version"
  type        = string
  default     = "latest"
}

variable "budget_amount" {
  description = "Monthly budget amount in USD"
  type        = number
  default     = 100
}
variable "budget_threshold_percentage" {
  description = "Threshold percent to notify"
  type        = number
  default     = 80
}
variable "alert_emails" {
  description = "Emails notified when threshold is met"
  type        = list(string)
  default     = ["you@example.com"]
}