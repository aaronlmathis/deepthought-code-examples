locals {
  # Define common disk sizes for readability
  gb = 1024 * 1024 * 1024

  # The final map of nodes to create. This uses the user-provided `var.nodes` if it's not null,
  # otherwise it falls back to a default configuration.
  nodes = var.nodes != null ? var.nodes : {
    "master-1" = { memory = 4096, vcpu = 2, disk_size = 20 * local.gb, ip = "192.168.122.100" }
    "worker-1" = { memory = 4096, vcpu = 2, disk_size = 20 * local.gb, ip = "192.168.122.101" }
    "worker-2" = { memory = 4096, vcpu = 2, disk_size = 20 * local.gb, ip = "192.168.122.102" }
    "worker-3" = { memory = 4096, vcpu = 2, disk_size = 20 * local.gb, ip = "192.168.122.103" }
    "worker-4" = { memory = 4096, vcpu = 2, disk_size = 20 * local.gb, ip = "192.168.122.104" }
  }
}