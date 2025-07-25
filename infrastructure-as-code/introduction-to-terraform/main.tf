terraform {
  required_providers {
    libvirt = {
      source  = "dmacvicar/libvirt"
      version = "0.7.1"
    }
  }
}

provider "libvirt" {
  uri = "qemu+ssh://amathis@192.168.0.40/system"
}
#Test

# Create cloud-init ISO per VM
resource "libvirt_cloudinit_disk" "cloudinit" {
  for_each = local.nodes

  name = "${each.key}-cloudinit.iso"
  network_config = templatefile("${path.module}/cloud-init/network-config.tpl", {
    ip         = each.value.ip
    gateway    = var.network_gateway
    interface  = var.network_interface_name
    prefix     = tonumber(split("/", var.network_cidr)[1])
    nameserver = var.network_nameserver
  })
  user_data = templatefile("${path.module}/cloud-init/user-data.tpl", {
    hostname = each.key
    ssh_key  = file(pathexpand(var.ssh_public_key_path))
  })
}

# Create a libvirt volume for the base image.
# This will download the image from the source URL on the first run
# and place it in the 'default' storage pool with the correct permissions.
resource "libvirt_volume" "base_image" {
  name   = var.base_image_name
  pool   = var.storage_pool_name
  source = var.base_image_source
  format = "qcow2"
}

# Create OS disk per VM (backed by base image)
resource "libvirt_volume" "disk" {
  for_each       = local.nodes
  name           = "${each.key}.qcow2"
  pool           = var.storage_pool_name
  base_volume_id = libvirt_volume.base_image.id
  size           = each.value.disk_size
  format         = "qcow2"
}

# Define the VM
resource "libvirt_domain" "vm" {
  for_each = local.nodes

  name   = each.key
  memory = each.value.memory
  vcpu   = each.value.vcpu

  disk {
    volume_id = libvirt_volume.disk[each.key].id
  }

  cloudinit = libvirt_cloudinit_disk.cloudinit[each.key].id

  network_interface {
    network_name = var.network_name
  }

  console {
    type        = "pty"
    target_type = "serial"
    target_port = "0"
  }

  graphics {
    type        = "vnc"
    listen_type = "address"
    autoport    = true
  }
}