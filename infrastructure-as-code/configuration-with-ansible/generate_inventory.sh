#!/usr/bin/env bash
set -e

TF_OUTPUT_JSON="$1"
INVENTORY_FILE="$2"

if [[ ! -f "$TF_OUTPUT_JSON" ]]; then
    echo "Error: Terraform output file $TF_OUTPUT_JSON not found"
    exit 1
fi

# Extract SSH configuration from Terraform outputs
SSH_USER=$(jq -r '.ssh_user.value // "ubuntu"' "$TF_OUTPUT_JSON")
SSH_KEY=$(jq -r '.ssh_private_key_path.value // "~/.ssh/id_rsa"' "$TF_OUTPUT_JSON")

# Extract IPs and create inventory entries with SSH config
masters=$(jq -r '.master_ips.value // {} | to_entries[] | "\(.key) ansible_host=\(.value)"' "$TF_OUTPUT_JSON")
workers=$(jq -r '.worker_ips.value // {} | to_entries[] | "\(.key) ansible_host=\(.value)"' "$TF_OUTPUT_JSON")

# Create inventory file
{
    echo "[masters]"
    if [[ -n "$masters" ]]; then
        echo "$masters"
    fi
    echo ""
    echo "[workers]"
    if [[ -n "$workers" ]]; then
        echo "$workers"
    fi
    echo ""
    echo "[all:vars]"
    echo "ansible_user=$SSH_USER"
    echo "ansible_ssh_private_key_file=$SSH_KEY"
    echo "ansible_ssh_common_args='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'"
} > "$INVENTORY_FILE"

echo "Generated inventory file: $INVENTORY_FILE"
cat "$INVENTORY_FILE"