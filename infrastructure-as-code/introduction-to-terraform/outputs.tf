output "master_ips" {
  description = "A map of master VM names to their configured IP addresses."
  value = {
    for name, node in local.nodes : name => node.ip
    if startswith(name, "master-")
  }
}

output "worker_ips" {
  description = "A map of worker VM names to their configured IP addresses."
  value = {
    for name, node in local.nodes : name => node.ip
    if startswith(name, "worker-")
  }
}

output "ssh_commands" {
  description = "A map of VM names to the SSH command needed to connect to them."
  value = {
    for name, node in local.nodes : name => "ssh ubuntu@${node.ip}"
  }
}

