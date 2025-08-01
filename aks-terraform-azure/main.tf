# main.tf

# Configure the Azure Provider
provider "azurerm" {
  features {}
}

# Look up the details of the existing Resource Group
data "azurerm_resource_group" "rg" {
  name = var.resource_group_name
}

# Create a Virtual Network
resource "azurerm_virtual_network" "vnet" {
  name                = "vnet-cheap-aks-demo"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  address_space       = ["172.16.0.0/16"]
}

# Create a Subnet inside the Virtual Network
resource "azurerm_subnet" "subnet" {
  name                 = "snet-cheap-aks-demo"
  resource_group_name  = data.azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["172.16.1.0/24"]
}

# Create the AKS cluster
resource "azurerm_kubernetes_cluster" "aks" {
  name                = var.aks_cluster_name
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  dns_prefix          = var.aks_cluster_name

  default_node_pool {
    name       = "nodepool1"
    node_count = var.node_count
    vm_size    = var.node_vm_size
    vnet_subnet_id = azurerm_subnet.subnet.id
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin     = "kubenet"
    load_balancer_sku  = "basic"
  }
}