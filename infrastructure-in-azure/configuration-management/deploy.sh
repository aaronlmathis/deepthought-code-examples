#!/bin/bash
# Ansible deployment script for Azure VM configuration
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TERRAFORM_DIR="../provisioning"
ANSIBLE_DIR="."

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Ansible is installed
check_ansible() {
    if ! command -v ansible &> /dev/null; then
        print_error "Ansible is not installed. Please install Ansible 11.x"
        exit 1
    fi
    
    ANSIBLE_VERSION=$(ansible --version | head -n1 | awk '{print $3}')
    print_status "Found Ansible version: $ANSIBLE_VERSION"
}

# Install Ansible collections
install_collections() {
    print_status "Installing required Ansible collections..."
    ansible-galaxy collection install -r requirements.yml --force
}

# Get VM IP from Terraform output
# Get VM IP from Terraform output and write inventory file
get_vm_ip() {
    if [ -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
        VM_IP=$(cd "$TERRAFORM_DIR" && terraform output -raw public_ip_address 2>/dev/null || echo "")
        if [ -n "$VM_IP" ]; then
            print_status "Found VM IP from Terraform: $VM_IP"
            print_status "Writing dynamic inventory file..."
            
            # Write the inventory file dynamically
            cat > inventory/hosts.yml << EOF
all:
  children:
    azure_vms:
      hosts:
        demo-vm:
          ansible_host: $VM_IP
          ansible_user: azureuser
          ansible_ssh_private_key_file: ~/.ssh/id_rsa
          ansible_python_interpreter: /usr/bin/python3
      vars:
        ansible_ssh_common_args: '-o StrictHostKeyChecking=no'
        environment_name: "{{ env | default('dev') }}"
        vm_size: Standard_B1s
        location: eastus
EOF
            print_status "Inventory file updated with VM IP: $VM_IP"
        else
            print_warning "Could not get VM IP from Terraform output"
            print_warning "Please manually update inventory/hosts.yml with the correct IP"
        fi
    else
        print_warning "Terraform state file not found"
        print_warning "Please manually update inventory/hosts.yml with the VM IP"
    fi
}

# Test connectivity to VM
test_connectivity() {
    print_status "Testing connectivity to VM..."
    if ansible all -m ping -i inventory/hosts.yml &> /dev/null; then
        print_status "Successfully connected to VM"
    else
        print_error "Cannot connect to VM. Please check:"
        echo "  1. VM IP address in inventory/hosts.yml"
        echo "  2. SSH key permissions"
        echo "  3. Network security group rules"
        echo "  4. VM is running"
        exit 1
    fi
}

# Run Ansible playbook
deploy_configuration() {
    print_status "Starting Ansible deployment..."
    
    # Run the playbook with verbose output
    ansible-playbook -i inventory/hosts.yml site.yml \
        --ssh-extra-args='-o StrictHostKeyChecking=no' \
        -v
    
    if [ $? -eq 0 ]; then
        print_status "Deployment completed successfully!"
        print_status "Your web application should be available at: http://$VM_IP"
    else
        print_error "Deployment failed!"
        exit 1
    fi
}

# Main execution
main() {
    print_status "Starting Azure VM configuration deployment..."
    
    cd "$SCRIPT_DIR"
    
    check_ansible
    install_collections
    get_vm_ip
    test_connectivity
    deploy_configuration
    
    print_status "Deployment script completed!"
}

# Run main function
main "$@"
