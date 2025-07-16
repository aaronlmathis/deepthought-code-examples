#!/bin/bash
# Cleanup script for learning-kubernetes VMs

set -e # Exit immediately if a command exits with a non-zero status.

VM_NAMES=("master-1" "worker-1" "worker-2")
KUBE_POOL_NAME="learning-kubernetes"
BASE_IMG_FILENAME="noble-server-cloudimg-amd64.img"

echo "--- Performing aggressive cleanup for all learning-kubernetes VMs and volumes ---"

# 1. Stop and undefine all VMs
echo "Stopping and undefining all VMs..."
for VM in "${VM_NAMES[@]}"; do
  echo "Processing VM: $VM"
  sudo virsh destroy "$VM" || true # Try to destroy if running
  sudo virsh undefine "$VM" || true # Undefine the VM definition
done

# 2. Get the pool path (needed for direct file operations if necessary)
POOL_PATH=$(sudo virsh pool-dumpxml "$KUBE_POOL_NAME" | grep -oP '(?<=<path>).*?(?=</path>)')
echo "Identified pool path: $POOL_PATH"

# 3. Delete related volumes from the pool for all VMs
echo "Deleting VM disks and seed ISOs from pool..."
for VM in "${VM_NAMES[@]}"; do
  echo "  - Deleting ${VM}.img"
  sudo virsh vol-delete --pool "$KUBE_POOL_NAME" "${VM}.img" || true
  echo "  - Deleting seed-${VM}.iso"
  sudo virsh vol-delete --pool "$KUBE_POOL_NAME" "seed-${VM}.iso" || true
done

# Optional: Delete the base image from the pool. Do this if you suspect the base image itself is the issue
# or if you want to force a fresh upload during the next script run.
echo "  - Deleting base image '$BASE_IMG_FILENAME' from pool (optional, but recommended for clean start)..."
sudo virsh vol-delete --pool "$KUBE_POOL_NAME" "$BASE_IMG_FILENAME" || true

# 4. Clean up any leftover local temporary files (important for fresh start)
echo "Cleaning up local /tmp files related to VMs..."
sudo rm -f "/tmp/$BASE_IMG_FILENAME" || true # If it was left over
sudo rm -rf "/tmp/cloud-init-*" || true

# 5. Clean up any leftover NVRAM files (if they exist and are problematic)
# These are per-VM UEFI variables files, often found in /var/lib/libvirt/qemu/nvram/
echo "Cleaning up NVRAM files..."
for VM in "${VM_NAMES[@]}"; do
  sudo rm -f "/var/lib/libvirt/qemu/nvram/${VM}_VARS.fd" || true
done

echo "--- Cleanup complete. You are now ready to re-run the main provisioning script. ---"