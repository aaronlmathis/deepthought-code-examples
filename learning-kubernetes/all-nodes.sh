#!/bin/bash


# Disable swap immediately
sudo swapoff -a

# Permanently disable swap by commenting out swap entries in /etc/fstab
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

# Verify swap is disabled
free -h

# Update system packages
sudo apt update

# Install containerd and related packages
sudo apt install -y containerd apt-transport-https ca-certificates curl gpg

# Create containerd configuration directory
sudo mkdir -p /etc/containerd

# Generate default containerd configuration
sudo containerd config default | sudo tee /etc/containerd/config.toml

# Configure containerd to use systemd cgroup driver (required for kubeadm)
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml

# Restart and enable containerd
sudo systemctl restart containerd
sudo systemctl enable containerd

# Verify containerd is running
sudo systemctl status containerd

# Load required kernel modules
sudo modprobe overlay
sudo modprobe br_netfilter

# Make modules load on boot
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

# Set required sysctl parameters
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

# Apply sysctl parameters immediately
sudo sysctl --system

# Verify the settings
lsmod | grep br_netfilter
lsmod | grep overlay
sysctl net.bridge.bridge-nf-call-iptables net.bridge.bridge-nf-call-ip6tables net.ipv4.ip_forward

# Add Kubernetes repository signing key
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

# Add Kubernetes repository
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list

# Update package index
sudo apt update

# Install Kubernetes components
sudo apt install -y kubelet kubeadm kubectl

# Prevent automatic updates of these packages
sudo apt-mark hold kubelet kubeadm kubectl

# Enable kubelet service
sudo systemctl enable kubelet