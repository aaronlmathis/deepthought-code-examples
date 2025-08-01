# outputs.tf

output "aks_cluster_name" {
  description = "The name of the created AKS cluster."
  value       = azurerm_kubernetes_cluster.aks.name
}

output "resource_group_name" {
  description = "The name of the resource group where the cluster is deployed."
  value       = data.azurerm_resource_group.rg.name
}

output "kube_config" {
  description = "The Kubernetes config for the created cluster. Treat this like gold."
  value       = azurerm_kubernetes_cluster.aks.kube_config_raw
  sensitive   = true
}