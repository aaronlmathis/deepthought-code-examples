variable "nodes" {
  type = map(object({
    memory    = number
    vcpu      = number
    disk_size = number
    ip        = string
  }))

  default     = null
  description = "A map of objects describing the nodes to create. If null, a default cluster is created."
}

variable "ssh_public_key_path" {
  type        = string
  description = "Path to your SSH public key file (e.g., ~/.ssh/id_rsa.pub)."
}

variable "network_gateway" {
  type        = string
  description = "The gateway for the libvirt network."
  default     = "192.168.122.1"
}

variable "network_cidr" {
  type        = string
  description = "The CIDR block for the libvirt network."
  default     = "192.168.122.0/24"
}

variable "network_nameserver" {
  type        = string
  description = "The nameserver for the VMs."
  default     = "8.8.8.8"
}

variable "network_name" {
  type        = string
  description = "The name of the libvirt network to attach VMs to."
  default     = "default"
}

variable "storage_pool_name" {
  type        = string
  description = "The name of the libvirt storage pool to use for VM disks."
  default     = "default"
}

variable "network_interface_name" {
  type        = string
  description = "The name of the primary network interface inside the VM (e.g., ens3, enp1s0)."
  default     = "ens3" # A common default for recent Ubuntu cloud images
}

variable "base_image_name" {
  type        = string
  description = "The name for the base OS image volume."
  default     = "ubuntu-noble-base"
}

variable "base_image_source" {
  type        = string
  description = "The URL from which to download the base OS image."
  default     = "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img"
}