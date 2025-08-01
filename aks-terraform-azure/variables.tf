variable "location" {
  default = "eastus"
}
variable "resource_group_name" {
  default = "rg-cheap-aks-demo"
}

variable "node_vm_size" {
  default = "Standard_B4pls_v2"
}
variable "aks_cluster_name" {
  default = "aks-cheap-demo-cluster"
}

variable "node_count" {
  default = 1
}