.PHONY: plan apply inventory ansible deploy destroy clean-ssh

TF_DIR := introduction-to-terraform
ANSIBLE_DIR := configuration-with-ansible
INVENTORY := $(ANSIBLE_DIR)/inventory.ini
TF_OUTPUT := $(TF_DIR)/terraform_output.json

plan:
    cd $(TF_DIR) && terraform init && terraform plan

apply:
    cd $(TF_DIR) && terraform init && terraform apply -auto-approve
    cd $(TF_DIR) && terraform output -json > terraform_output.json

inventory: apply
    $(ANSIBLE_DIR)/generate_inventory.sh $(TF_OUTPUT) $(INVENTORY)

wait-for-ssh: inventory
    $(ANSIBLE_DIR)/wait_for_ssh.sh $(INVENTORY)

ansible: wait-for-ssh
    ANSIBLE_CONFIG=$(ANSIBLE_DIR)/ansible.cfg ansible-playbook -i $(INVENTORY) $(ANSIBLE_DIR)/site.yml

deploy: apply inventory wait-for-ssh ansible

destroy:
    cd $(TF_DIR) && terraform destroy -auto-approve
    $(MAKE) clean-ssh

clean-ssh:
    @echo "Clearing SSH known_hosts entries for libvirt VMs..."
    @bash -c 'for ip in {100..104}; do ssh-keygen -f "$$HOME/.ssh/known_hosts" -R "192.168.122.$$ip" 2>/dev/null || true; done'
    @echo "SSH known_hosts cleaned"