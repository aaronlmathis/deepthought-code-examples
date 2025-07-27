output "public_ip_address" {
  description = "Public IP address of the Virtual Machine"
  value       = azurerm_public_ip.public_ip.ip_address
}

output "vm_name" {
  description = "Name of the deployed Virtual Machine"
  value       = azurerm_linux_virtual_machine.vm.name
}

output "resource_group_name" {
  description = "Name of the Resource Group"
  value       = azurerm_resource_group.rg.name
}

output "ssh_connection_command" {
  description = "SSH command to connect to the VM"
  value       = "ssh ${var.admin_username}@${azurerm_public_ip.public_ip.ip_address}"
}