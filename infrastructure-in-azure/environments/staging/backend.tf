terraform {
  backend "azurerm" {
    resource_group_name  = "rg-terraform-state-staging"
    storage_account_name = "tfstatestaging1753803715"
    container_name       = "tfstate"
    key                  = "staging.terraform.tfstate"
  }
}
