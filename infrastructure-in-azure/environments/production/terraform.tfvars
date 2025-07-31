# Production Environment Configuration
resource_group_name = "rg-terraform-state-production"
environment = "production"
location = "East US" 
vm_name = "prod-vm"
vm_size = "Standard_B1s"
admin_username = "azureuser"
ssh_allowed_cidr = "0.0.0.0/0" 
vnet_name = "prod-vnet"
vnet_address_space = "10.2.0.0/16"
subnet_name = "prod-subnet"
subnet_prefix = "10.2.1.0/24"
budget_amount = 200
budget_threshold_percentage = 60
alert_emails = ["ops-team@example.com", "alerts@example.com"]
ssh_public_key_path = "keys/id_rsa.pub"