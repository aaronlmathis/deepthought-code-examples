#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# ---- Configurable ----
BASE_IMG_FILENAME="noble-server-cloudimg-amd64.img"
UBUNTU_CLOUD_IMG_URL="https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img"
VM_NAMES=("master-1" "worker-1" "worker-2")
VM_IPS=("192.168.122.101/24" "192.168.122.102/24" "192.168.122.103/24")
SSH_PUB_KEY="$(cat /home/<YOUR USERNAME>/.ssh/id_rsa.pub)" # IMPORTANT: REPLACE WITH YOUR ACTUAL PUBLIC SSH KEY
VM_RAM=4096
VM_VCPUS=2
VM_DISK_SIZE=30 # GB
KUBE_POOL_NAME="learning-kubernetes"    # The pool that we created
# ----------------------

echo "--- Checking Prerequisites and Downloading Base Image ---"

# Check for KVM acceleration (user friendly message)
if ! grep -E 'svm|vmx' /proc/cpuinfo &>/dev/null; then
  echo "ERROR: Your CPU does not support hardware virtualization (Intel VT-x or AMD-V)."
  echo "KVM acceleration will not be available. Please check your CPU capabilities."
  exit 1
fi

if ! lsmod | grep -E 'kvm_intel|kvm_amd' &>/dev/null; then
  echo "WARNING: KVM kernel modules are not loaded. Attempting to load them."
  sudo modprobe kvm_intel || sudo modprobe kvm_amd || { echo "Failed to load KVM modules."; exit 1; }
  sudo modprobe kvm || { echo "Failed to load KVM module."; exit 1; }
  sleep 2 # Give modules time to load
fi

# Define local path for temporary download (always download to /tmp or similar)
LOCAL_BASE_IMG_PATH="/tmp/$BASE_IMG_FILENAME"

# Download base image if not present locally
if [ ! -f "$LOCAL_BASE_IMG_PATH" ]; then
  echo "Downloading Ubuntu cloud image to $LOCAL_BASE_IMG_PATH: $UBUNTU_CLOUD_IMG_URL"
  # Use -O to specify the output filename and path directly
  wget -O "$LOCAL_BASE_IMG_PATH" "$UBUNTU_CLOUD_IMG_URL"
else
  echo "Local base image '$LOCAL_BASE_IMG_PATH' already exists. Skipping download."
fi

# Ensure base image is in the libvirt storage pool
echo "Checking base image '$BASE_IMG_FILENAME' in libvirt storage pool '$KUBE_POOL_NAME'..."
if ! virsh vol-list --pool "$KUBE_POOL_NAME" | grep -q "$BASE_IMG_FILENAME"; then
  echo "Importing base image '$BASE_IMG_FILENAME' into pool '$KUBE_POOL_NAME'..."
  
  # 1. Get the size of the local base image
  # Use du -b for bytes for more precision, as vol-create-as can take bytes
  IMG_SIZE_BYTES=$(sudo du -b "$LOCAL_BASE_IMG_PATH" | awk '{print $1}')

  # 2. Create an empty volume in the pool for the base image
  # Assuming qcow2 format for the base image
  sudo virsh vol-create-as "$KUBE_POOL_NAME" "$BASE_IMG_FILENAME" "${IMG_SIZE_BYTES}B" --format qcow2 || { echo "ERROR: Failed to create volume for base image. Exiting."; exit 1; }
  
  # 3. Upload the local base image into the newly created volume in the pool
  sudo virsh vol-upload --pool "$KUBE_POOL_NAME" "$BASE_IMG_FILENAME" "$LOCAL_BASE_IMG_PATH" || { echo "ERROR: Failed to upload base image. Exiting."; exit 1; }
  
  echo "Base image imported successfully into pool '$KUBE_POOL_NAME'."
else
  echo "Base image '$BASE_IMG_FILENAME' already present in pool '$KUBE_POOL_NAME'."
fi

# Clean up the locally downloaded base image after import
if [ -f "$LOCAL_BASE_IMG_PATH" ]; then
  echo "Cleaning up local downloaded base image: $LOCAL_BASE_IMG_PATH"
  rm "$LOCAL_BASE_IMG_PATH"
fi

echo "--- Starting VM Provisioning ---"

# Loop through each VM to create unique cloud-init data and install the VM
for i in "${!VM_NAMES[@]}"; do
  VM="${VM_NAMES[$i]}"
  CURRENT_IP="${VM_IPS[$i]}"
  
  echo "Processing VM: $VM with IP: $CURRENT_IP"

  # Define paths for temporary cloud-init files and the seed ISO
  TEMP_DIR="/tmp/cloud-init-$$" # Use $$ for a unique temporary directory for each run
  mkdir -p "$TEMP_DIR" || { echo "ERROR: Failed to create temporary directory $TEMP_DIR. Exiting."; exit 1; } # Added error handling for mkdir

  LOCAL_USER_DATA="${TEMP_DIR}/user-data"
  LOCAL_META_DATA="${TEMP_DIR}/meta-data"
  LOCAL_SEED_ISO="${TEMP_DIR}/seed-${VM}.iso" # Always create ISO, not .img

  # Define the volume name for libvirt pool (can be same as ISO filename)
  # It's best practice to keep the .iso extension for clarity in libvirt volumes
  SEED_VOL_NAME="seed-${VM}.iso"

  # 1. Create per-VM user-data and meta-data files in the temporary directory
  # This makes hostnames dynamic via cloud-init
  cat <<EOF > "$LOCAL_USER_DATA"
#cloud-config
preserve_hostname: false
hostname: ${VM}
manage_etc_hosts: true

# Completely disable cloud-init network management
network:
  config: disabled

bootcmd:
  # Kill any DHCP processes immediately
  - [ pkill, -f, dhcp ]
  - [ systemctl, stop, systemd-networkd ]
  - [ systemctl, stop, NetworkManager ]
  # Remove all existing network configs
  - [ rm, -rf, /etc/netplan/* ]
  - [ rm, -f, /etc/cloud/cloud.cfg.d/*network* ]
  - [ rm, -f, /var/lib/dhcp/* ]

timezone: UTC
ssh_pwauth: true
chpasswd:
  list: |
    ubuntu:ubuntu123
  expire: false
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: users, admin, sudo
    shell: /bin/bash
    lock_passwd: false
    ssh_authorized_keys:
      - ${SSH_PUB_KEY}
disable_root: true
package_update: true
package_upgrade: false
packages:
  - qemu-guest-agent
  - net-tools
  - curl
  - wget
  - vim
  - htop

write_files:
  - path: /etc/ssh/sshd_config.d/99-cloud-init.conf
    content: |
      PasswordAuthentication yes
      PubkeyAuthentication yes
      PermitRootLogin no
    permissions: '0644'
  - path: /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg
    content: |
      network: {config: disabled}
    permissions: '0644'
  - path: /etc/systemd/network/10-static.network
    permissions: '0644'
    content: |
      [Match]
      Name=en*

      [Network]
      DHCP=no
      Address=${CURRENT_IP}
      Gateway=192.168.122.1
      DNS=8.8.8.8
      DNS=1.1.1.1
  - path: /etc/systemd/network/20-virtio.network
    permissions: '0644'
    content: |
      [Match]
      Driver=virtio_net

      [Network]
      DHCP=no
      Address=${CURRENT_IP}
      Gateway=192.168.122.1
      DNS=8.8.8.8
      DNS=1.1.1.1

runcmd:
  # Disable all network managers except systemd-networkd
  - [ systemctl, disable, NetworkManager ]
  - [ systemctl, mask, NetworkManager ]
  - [ systemctl, stop, NetworkManager ]
  
  # Kill any remaining DHCP clients
  - [ pkill, -f, dhcp ]
  - [ rm, -rf, /var/lib/dhcp/* ]
  
  # Remove any netplan configs that might interfere
  - [ rm, -rf, /etc/netplan/* ]
  
  # Enable and start systemd-networkd
  - [ systemctl, enable, systemd-networkd ]
  - [ systemctl, enable, systemd-resolved ]
  - [ systemctl, start, systemd-networkd ]
  - [ systemctl, start, systemd-resolved ]
  
  # Wait for network to come up
  - [ sleep, 15 ]
  
  # Verify connectivity
  - [ ip, addr, show ]
  - [ ping, -c, 3, 8.8.8.8 ]
  
  # Enable services
  - [ systemctl, enable, qemu-guest-agent ]
  - [ systemctl, start, qemu-guest-agent ]
  - [ systemctl, restart, sshd ]

final_message: "Cloud-init setup complete for ${VM}. Static IP configured: ${CURRENT_IP}"
EOF

  cat <<EOF > "$LOCAL_META_DATA"
instance-id: ${VM}
local-hostname: ${VM}
EOF

  # 2. Create cloud-init seed ISO image for the current VM in the temporary directory
  echo "Creating cloud-init seed ISO for $VM in $TEMP_DIR..."
  genisoimage -output "$LOCAL_SEED_ISO" -volid cidata -joliet -rock "$LOCAL_USER_DATA" "$LOCAL_META_DATA"

  # 3. Import the seed image (ISO) into the libvirt storage pool
  echo "Importing seed image '$SEED_VOL_NAME' into pool '$KUBE_POOL_NAME'..."
  # Remove existing seed volume if it's there for idempotency
  if virsh vol-list --pool "$KUBE_POOL_NAME" | grep -q "$SEED_VOL_NAME"; then
    echo "Removing existing seed volume '$SEED_VOL_NAME' from pool $KUBE_POOL_NAME"
    sudo virsh vol-delete --pool "$KUBE_POOL_NAME" "$SEED_VOL_NAME"
  fi
  
  # Calculate size of the ISO file
  SEED_SIZE_BYTES=$(du -b "$LOCAL_SEED_ISO" | awk '{print $1}')

  # Create an empty volume in pool with the correct size and raw format
  sudo virsh vol-create-as "$KUBE_POOL_NAME" "$SEED_VOL_NAME" "${SEED_SIZE_BYTES}B" --format iso || { echo "ERROR: Failed to create volume for seed image. Exiting."; exit 1; }

  # Upload local seed ISO image to the newly created volume in the pool
  sudo virsh vol-upload --pool "$KUBE_POOL_NAME" --vol "$SEED_VOL_NAME" "$LOCAL_SEED_ISO" || { echo "ERROR: Failed to upload seed image. Exiting."; exit 1; }

  # Clean up temporary local copies after upload
  echo "Cleaning up temporary cloud-init files in $TEMP_DIR..."
  rm -rf "$TEMP_DIR" # Remove the entire temporary directory

  # 4. Remove existing VM disk in pool if it's there (for idempotency)
  if virsh vol-list --pool "$KUBE_POOL_NAME" | grep -q "${VM}.img"; then
    echo "Removing existing VM disk '${VM}.img' from pool $KUBE_POOL_NAME"
    sudo virsh vol-delete --pool "$KUBE_POOL_NAME" "${VM}.img"
  fi

  # 5. Create new VM disk volume in the pool by cloning the base image using qemu-img
  echo "Creating VM volume '${VM}.img' (cloning from '$BASE_IMG_FILENAME') in pool '$KUBE_POOL_NAME' using qemu-img..."
  
  # Get the full path to the base image within the pool
  BASE_IMG_PATH_IN_POOL="$(sudo virsh pool-dumpxml "$KUBE_POOL_NAME" | grep -oP '(?<=<path>).*?(?=</path>)' | head -1)/$BASE_IMG_FILENAME"
  
  # Get the full path where the new VM disk will reside
  VM_DISK_PATH_IN_POOL="$(sudo virsh pool-dumpxml "$KUBE_POOL_NAME" | grep -oP '(?<=<path>).*?(?=</path>)' | head -1)/${VM}.img"

  # Use qemu-img create to create the COW (Copy-On-Write) clone directly
  # This makes a sparse qcow2 image backed by the base image.
  sudo qemu-img create -f qcow2 -o backing_file="$BASE_IMG_PATH_IN_POOL",backing_fmt=qcow2 "$VM_DISK_PATH_IN_POOL" "${VM_DISK_SIZE}G" || { echo "ERROR: Failed to create cloned VM disk using qemu-img. Exiting."; exit 1; }

  # IMPORTANT: Tell libvirt to refresh its pool so it detects the new file
  sudo virsh pool-refresh "$KUBE_POOL_NAME" || { echo "ERROR: Failed to refresh libvirt pool. Exiting."; exit 1; }

  # 6. Install the VM
  echo "Installing VM: $VM"
  # Check if VM already exists before attempting virt-install
  if virsh list --all | grep -w "$VM" &>/dev/null; then
    echo "VM '$VM' already exists. Skipping virt-install."
    # If it exists but is shut off, start it
    if virsh list --all --state shutoff | grep -w "$VM" &>/dev/null; then
      echo "VM '$VM' is shut off. Starting it..."
      sudo virsh start "$VM"
    fi
    continue # Skip to the next VM in the loop
  fi

  sudo virt-install \
    --name "$VM" \
    --ram "$VM_RAM" \
    --vcpus "$VM_VCPUS" \
    --disk "vol=$KUBE_POOL_NAME/${VM}.img,bus=virtio" \
    --disk "vol=$KUBE_POOL_NAME/$SEED_VOL_NAME,device=cdrom,bus=sata" \
    --os-variant ubuntu24.04 \
    --network network=default \
    --graphics none \
    --import \
    --noautoconsole \
    --autostart \
    --boot hd,cdrom

  echo "VM $VM created successfully."
  

done

echo "--- VM Provisioning Complete ---"
echo ""
echo "To check VM status: sudo virsh list --all"
echo "To get VM IP addresses: sudo virsh net-dhcp-leases default"
echo "To check cloud-init status on a VM: ssh ubuntu@<VM_IP> 'sudo cloud-init status'"