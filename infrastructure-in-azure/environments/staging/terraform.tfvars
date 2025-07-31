# Staging Environment Configuration
resource_group_name = "rg-terraform-state-staging"
environment = "staging"
location = "East US"
vm_name = "staging-vm"
vm_size = "Standard_B1s"
admin_username = "azureuser"
ssh_allowed_cidr = "0.0.0.0/0"  # More restrictive 
vnet_name = "staging-vnet"
vnet_address_space = "10.1.0.0/16"
subnet_name = "staging-subnet"
subnet_prefix = "10.1.1.0/24"
budget_amount = 100
budget_threshold_percentage = 70
alert_emails = ["staging-team@example.com"]
ssh_public_key_path = "keys/id_rsa.pub"