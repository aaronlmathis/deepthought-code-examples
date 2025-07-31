#!/bin/bash

env="$1"   # first argument after the script name

case "$env" in
  dev|staging|production)
    echo "Destroying resources in \"$env\"..."
	rm -rf backend.tf .terraform*
	
	ln -s "../environments/$env/backend.tf" backend.tf
	echo "Running Terraform destroy for \"$env\"..."
	terraform init 
	terraform destroy -auto-approve -var-file="../environments/$env/terraform.tfvars"
	echo "Resources in \"$env\" destroyed successfully."
	;;
  *)
    echo "Usage: $(basename "$0") {dev|staging|production}" >&2
    exit 1
    ;;
esac