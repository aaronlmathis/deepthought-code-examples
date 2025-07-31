terraform {
  backend "azurerm" {
    resource_group_name  = "rg-terraform-state-dev"
    storage_account_name = "tfstatedev1753803491"
    container_name       = "tfstate"
    key                  = "dev.terraform.tfstate"
  }
}
