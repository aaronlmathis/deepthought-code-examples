Part 1: Terraforming Azure with Cost Guardrails

This scaffold will:
- Configure the AzureRM provider
- Create a Resource Group
- Define a Virtual Network + Subnet
- Provision a Public IP and Network Interface
- Deploy a single B-series Linux VM
- Create a monthly budget with alerts
- Output the VM's public IP

Files:
  • README.md
  • main.tf
  • variables.tf
  • outputs.tf
  • budgets.tf
  • .github/workflows/auto_destroy.yml
*/

// README.md
# Part 1: Terraforming Azure with Cost Guardrails

## Prerequisites
- An existing Azure Resource Group named `deepthoughtTerraformRG` (created during SPN setup)
- Azure CLI installed and authenticated (`az login`)
- **Ensure your default subscription is set**:
  ```bash
  az account show                     # verify current subscription
  az account set -s <SUBSCRIPTION_ID> # switch if needed
  ```
- Optionally, set the subscription in env var:
  ```bash
  export AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
  ```
- Terraform installed (>= v1.0)
- GitHub repo configured with Azure Service Principal in `AZURE_CREDENTIALS`

_Before applying_, import the existing RG to Terraform state so it can manage tags:

```bash
terraform init
terraform import azurerm_resource_group.rg /subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/deepthoughtTerraformRG
```

## Usage
```bash
git clone <repo-url> && cd part1-terraform-azure
terraform init
# Optional: override budget defaults via -var flags
terraform plan
terraform apply    # confirm with "yes"
```

### Budget & Alerts
- A monthly budget (`budgets.tf`) is created at the subscription level.
- It triggers an email when spending reaches your threshold (default 80% of $100).
- Customize via `budget_amount`, `budget_threshold_percentage`, and `alert_emails` in `variables.tf`.

### Auto‑Destroy Dev Resources
Resources tagged `environment = "dev"` are cleaned up daily at 02:00 UTC by a GitHub Actions workflow (`.github/workflows/auto_destroy.yml`).

## Cleanup
To manually destroy:
```bash
terraform destroy # removes all Terraform-managed resources (including budget)
```  
To auto‑delete dev groups, see the scheduled workflow above.