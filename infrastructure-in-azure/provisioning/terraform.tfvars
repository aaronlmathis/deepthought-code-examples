# Development Environment Configuration
resource_group_name = "dt-series-dev-rg"
environment = "dev"
location = "East US"
vm_name = "dt-series-dev-vm"
vm_size = "Standard_B1s"
admin_username = "azureuser"
ssh_allowed_cidr =  "0.0.0.0/0"  # More permissive for dev
vnet_name = "dt-series-dev-vnet"
vnet_address_space = "10.0.0.0/16"
subnet_name = "dt-series-dev-subnet"
subnet_prefix = "10.0.1.0/24"
budget_amount = 50
budget_threshold_percentage = 80
alert_emails = ["aaron@deepthought.sh"]
ssh_public_key_path = "~/.ssh/azure_vm_key.pub"