#!/usr/bin/env bash
set -e

INVENTORY_FILE="$1"

if [[ ! -f "$INVENTORY_FILE" ]]; then
    echo "Error: Inventory file $INVENTORY_FILE not found"
    exit 1
fi

echo "Waiting for SSH to be available on all VMs..."

# Extract IPs from inventory file
ips=$(grep -E "ansible_host=" "$INVENTORY_FILE" | sed 's/.*ansible_host=\([0-9.]*\).*/\1/')

for ip in $ips; do
    echo "Waiting for SSH on $ip..."
    timeout=120
    while ! nc -z "$ip" 22 2>/dev/null && [ $timeout -gt 0 ]; do
        sleep 2
        timeout=$((timeout-2))
    done
    
    if [ $timeout -le 0 ]; then
        echo "Timeout waiting for SSH on $ip"
        exit 1
    else
        echo "SSH available on $ip"
    fi
done

echo "All VMs are ready for SSH connections"