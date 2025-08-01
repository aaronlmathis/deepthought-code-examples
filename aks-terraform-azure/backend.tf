terraform {
  required_providers {
    azurerm = { source = "hashicorp/azurerm" }
  }
  backend "azurerm" {
    resource_group_name  = "rg-cheap-aks-demo"
    storage_account_name = "tfstatecheapaksdemo"
    container_name       = "tfstate"
    key                  = "aks.tfstate"
  }
} 