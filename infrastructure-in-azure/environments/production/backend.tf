terraform {
  backend "azurerm" {
    resource_group_name  = "rg-terraform-state-production"
    storage_account_name = "tfstateprod1753803789"
    container_name       = "tfstate"
    key                  = "production.terraform.tfstate"
  }
}
