# Makefile for Azure VM provisioning and configuration
# This integrates Terraform (infrastructure) with Ansible (configuration)

.PHONY: help plan apply destroy configure deploy clean status ssh validate logs

# Default variables
TERRAFORM_DIR := provisioning
ANSIBLE_DIR := configuration-management
# Use full path or shell expansion for SSH key
SSH_KEY := $(HOME)/.ssh/id_rsa

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Force bash usage for advanced features
SHELL := /bin/bash

help: ## Show this help message
	@echo "Azure VM Infrastructure & Configuration Management"
	@echo "=================================================="
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Typical workflow:"
	@echo "  1. make plan     - Review infrastructure changes"
	@echo "  2. make deploy   - Deploy infrastructure + configuration"
	@echo "  3. make ssh      - Connect to the VM"
	@echo "  4. make destroy  - Clean up resources"

check-prereqs: ## Check if required tools are installed
	@echo -e "$(GREEN)Checking prerequisites...$(NC)"
	@command -v terraform >/dev/null 2>&1 || { echo -e "$(RED)Error: terraform is not installed$(NC)" >&2; exit 1; }
	@command -v ansible >/dev/null 2>&1 || { echo -e "$(RED)Error: ansible is not installed$(NC)" >&2; exit 1; }
	@if [ -f "$(SSH_KEY)" ]; then \
		echo -e "$(GREEN)✓ Found SSH key: $(SSH_KEY)$(NC)"; \
	else \
		echo -e "$(RED)Error: SSH key not found at $(SSH_KEY)$(NC)"; \
		echo -e "$(YELLOW)Please ensure your SSH key exists at $(SSH_KEY)$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(GREEN)✓ All prerequisites satisfied$(NC)"

plan: check-prereqs ## Plan Terraform infrastructure changes
	@echo -e "$(GREEN)Planning infrastructure changes...$(NC)"
	cd $(TERRAFORM_DIR) && source .env && terraform init && terraform plan
	@echo -e "$(GREEN)✓ Plan completed$(NC)"

apply: check-prereqs ## Apply Terraform infrastructure changes
	@echo -e "$(GREEN)Applying infrastructure changes...$(NC)"
	cd $(TERRAFORM_DIR) && source .env && terraform apply -auto-approve
	@echo -e "$(GREEN)✓ Infrastructure deployed$(NC)"

configure: ## Run Ansible configuration on existing VM
	@echo -e "$(GREEN)Configuring VM with Ansible...$(NC)"
	cd $(ANSIBLE_DIR) && ./deploy.sh
	@echo -e "$(GREEN)✓ VM configuration completed$(NC)"

deploy: plan apply configure ## Full deployment: provision infrastructure + configure VM
	@echo -e "$(GREEN)Full deployment completed!$(NC)"
	@echo ""
	@echo -e "$(YELLOW)Deployment Summary:$(NC)"
	@echo "=================="
	@cd $(TERRAFORM_DIR) && echo "VM Name: $$(terraform output -raw vm_name)"
	@cd $(TERRAFORM_DIR) && echo "Public IP: $$(terraform output -raw public_ip_address)"
	@cd $(TERRAFORM_DIR) && echo "SSH Command: $$(terraform output -raw ssh_connection_command)"
	@cd $(TERRAFORM_DIR) && echo "Web App: http://$$(terraform output -raw public_ip_address)"
	@echo ""
	@echo -e "$(GREEN)Your VM is ready for use!$(NC)"

status: ## Show current infrastructure status
	@echo -e "$(GREEN)Infrastructure Status:$(NC)"
	@echo "====================="
	@if [ -f "$(TERRAFORM_DIR)/terraform.tfstate" ]; then \
		cd $(TERRAFORM_DIR) && terraform show -json | jq -r '.values.root_module.resources[] | select(.type=="azurerm_linux_virtual_machine") | "VM Status: " + .values.name + " (" + .values.location + ")"' 2>/dev/null || echo "VM: Deployed"; \
		echo "Public IP: $$(cd $(TERRAFORM_DIR) && terraform output -raw public_ip_address 2>/dev/null || echo 'Not available')"; \
		echo "Resource Group: $$(cd $(TERRAFORM_DIR) && terraform output -raw resource_group_name 2>/dev/null || echo 'Not available')"; \
	else \
		echo -e "$(YELLOW)No infrastructure deployed$(NC)"; \
	fi

ssh: ## SSH into the VM
	@echo -e "$(GREEN)Connecting to VM...$(NC)"
	@cd $(TERRAFORM_DIR) && $$(terraform output -raw ssh_connection_command 2>/dev/null) || { echo -e "$(RED)Cannot get SSH command. Is the VM deployed?$(NC)" >&2; exit 1; }

test: ## Test the deployed web application
	@echo -e "$(GREEN)Testing web application...$(NC)"
	@VM_IP=$$(cd $(TERRAFORM_DIR) && terraform output -raw public_ip_address 2>/dev/null) && \
	if [ -n "$$VM_IP" ]; then \
		echo "Testing HTTP connection to $$VM_IP..."; \
		curl -v --connect-timeout 10 "http://$$VM_IP" || echo -e "$(YELLOW)Connection failed$(NC)"; \
	else \
		echo -e "$(RED)VM IP not available$(NC)"; \
	fi

logs: ## Show Ansible deployment logs
	@echo -e "$(GREEN)Recent deployment logs:$(NC)"
	@if [ -f "$(ANSIBLE_DIR)/ansible.log" ]; then \
		tail -50 "$(ANSIBLE_DIR)/ansible.log"; \
	else \
	echo -e "$(YELLOW)No deployment logs found$(NC)"; \
	fi

clean: ## Clean Terraform cache and temporary files
	@echo -e "$(GREEN)Cleaning temporary files...$(NC)"
	rm -rf $(TERRAFORM_DIR)/.terraform
	rm -f $(TERRAFORM_DIR)/.terraform.lock.hcl
	rm -f $(TERRAFORM_DIR)/terraform.tfstate.backup
	rm -f $(ANSIBLE_DIR)/ansible.log
	@echo -e "$(GREEN)✓ Cleanup completed$(NC)"

validate: ## Validate Terraform and Ansible configurations
	@echo -e "$(GREEN)Validating configurations...$(NC)"
	cd $(TERRAFORM_DIR) && terraform fmt
	cd $(TERRAFORM_DIR) && source .env && terraform init -backend=false
	cd $(TERRAFORM_DIR) && source .env && terraform validate
	cd $(ANSIBLE_DIR) && ansible-playbook --syntax-check site.yml -i inventory/hosts.yml
	@echo -e "$(GREEN)✓ All configurations valid$(NC)"

destroy: ## Destroy all infrastructure
	@echo -e "$(RED)WARNING: This will destroy all infrastructure!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo -e "$(GREEN)Destroying infrastructure...$(NC)"; \
		cd $(TERRAFORM_DIR) && source .env && terraform destroy -auto-approve; \
		echo -e "$(GREEN)✓ Infrastructure destroyed$(NC)"; \
	else \
		echo -e "$(YELLOW)Destroy cancelled$(NC)"; \
	fi

# Set default target
.DEFAULT_GOAL := help