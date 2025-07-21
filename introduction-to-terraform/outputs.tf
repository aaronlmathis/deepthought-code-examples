output "vm_ips" {
  description = "A map of VM names to their configured IP addresses."
  value = {
    for name, node in local.nodes : name => node.ip
  }
}

output "ssh_commands" {
  description = "A map of VM names to the SSH command needed to connect to them."
  value = {
    for name, node in local.nodes : name => "ssh ubuntu@${node.ip}"
  }
}