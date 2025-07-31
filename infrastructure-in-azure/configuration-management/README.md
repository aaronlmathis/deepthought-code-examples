# Azure VM Configuration Management

This directory contains professional-grade Ansible playbooks for configuring Azure VMs with security hardening, web application deployment, and monitoring capabilities.

## Overview

The configuration management system provides:

### 🔒 Security & Hardening
- **System Updates**: Automated apt updates and security patches
- **SSH Hardening**: Custom port (2222), disabled root login, strong ciphers
- **Fail2ban**: Intrusion prevention with UFW integration
- **Firewall**: UFW configuration allowing only necessary ports
- **Kernel Hardening**: Security-focused kernel parameters

### 🐳 Application Stack
- **Docker & Docker Compose**: Latest stable versions with security configurations
- **Containerized Web App**: Nginx-based demo application with health checks
- **Runtime Environment**: Node.js 18.x and Python 3 with virtual environments
- **Systemd Integration**: Application service with auto-restart capabilities

### 📊 Monitoring & Maintenance
- **Azure Monitor Agent**: Integration with Azure Monitor for metrics and logs
- **System Monitoring**: Custom scripts for disk, memory, and service health
- **Automated Backups**: Daily system backups with retention policies
- **Log Rotation**: Automated cleanup of Docker and system logs
- **Time Synchronization**: Chrony NTP configuration

## Directory Structure

```
configuration-management/
├── ansible.cfg                 # Ansible configuration
├── site.yml                   # Main playbook
├── deploy.sh                  # Deployment script
├── requirements.yml           # Ansible collections
├── inventory/
│   └── hosts.yml             # Inventory configuration
├── group_vars/
│   ├── all.yml              # Common variables
│   └── azure_vms.yml        # Azure-specific variables
└── roles/
    ├── system-hardening/     # OS security and updates
    ├── firewall/            # UFW firewall configuration
    ├── ssh-hardening/       # SSH security configuration
    ├── fail2ban/           # Intrusion prevention
    ├── time-sync/          # NTP/Chrony configuration
    ├── docker/             # Docker installation
    ├── nodejs-runtime/     # Node.js and Python setup
    ├── web-application/    # Containerized web app
    ├── azure-monitor/      # Azure monitoring setup
    └── cron-jobs/          # Automated maintenance
```

## Prerequisites

### Required Software
- **Ansible 11.x**: Configuration management
- **SSH Access**: Key-based authentication to target VM
- **Python 3**: On both control and target machines

### Required Collections
```bash
ansible-galaxy collection install -r requirements.yml
```

### SSH Key Setup
Ensure your SSH key is available at `~/.ssh/id_rsa` and the public key is deployed to the VM.

## Configuration Variables

### Core Settings (`group_vars/all.yml`)
```yaml
ssh_port: 2222                    # Custom SSH port
app_name: "demo-web-app"          # Application service name
app_port: 80                      # Web application port
nodejs_version: "18"              # Node.js runtime version
timezone: "UTC"                   # System timezone
```

### Security Settings
```yaml
fail2ban_maxretry: 3              # Max failed login attempts
fail2ban_bantime: 3600            # Ban duration (seconds)
ssh_allowed_users: ["azureuser"]  # Allowed SSH users
```

### Azure-Specific (`group_vars/azure_vms.yml`)
```yaml
firewall_rules:                   # UFW firewall rules
  - port: 2222                    # SSH
  - port: 80                      # HTTP
  - port: 443                     # HTTPS
```

## Deployment

### Quick Start
```bash
# Full deployment (recommended)
./deploy.sh

# Or using the Makefile from project root
make deploy
```

### Manual Deployment
```bash
# 1. Install collections
ansible-galaxy collection install -r requirements.yml

# 2. Update inventory with VM IP
vim inventory/hosts.yml

# 3. Test connectivity
ansible all -m ping -i inventory/hosts.yml

# 4. Run playbook
ansible-playbook -i inventory/hosts.yml site.yml
```

### Deployment Script Features
The `deploy.sh` script provides:
- ✅ Prerequisite checking
- ✅ Terraform integration (auto-detects VM IP)
- ✅ Connectivity testing
- ✅ Colored output and error handling
- ✅ Post-deployment verification

## Role Details

### system-hardening
- Updates all packages and enables automatic security updates
- Configures kernel security parameters
- Removes unnecessary packages
- Sets up log rotation

### firewall
- Configures UFW with deny-by-default policy
- Opens only required ports (SSH, HTTP, HTTPS)
- Enables logging for security monitoring

### ssh-hardening
- Changes SSH port to 2222
- Disables root login and password authentication
- Configures strong ciphers and key exchange algorithms
- Sets connection limits and timeouts

### fail2ban
- Monitors SSH authentication logs
- Automatically bans IPs after failed attempts
- Integrates with UFW for blocking

### docker
- Installs Docker CE and Docker Compose
- Configures daemon with security settings
- Adds user to docker group
- Sets up log rotation for containers

### web-application
- Deploys Nginx-based containerized web app
- Creates systemd service for auto-start
- Implements health checks
- Provides production-ready configuration

### azure-monitor
- Installs Azure Monitor Agent (when enabled)
- Configures system monitoring tools (sysstat, iotop)
- Sets up performance data collection

### cron-jobs
- Daily system backups with 7-day retention
- Log rotation and cleanup
- System health monitoring every 15 minutes
- Weekly update checks

## Security Considerations

### Network Security
- UFW firewall with minimal open ports
- SSH on non-standard port (2222)
- Fail2ban for intrusion prevention

### Application Security
- Containerized applications with resource limits
- Security headers in Nginx configuration
- Non-root user execution

### System Security
- Disabled IPv6 (if configured)
- Kernel hardening parameters
- Regular security updates
- Comprehensive logging

## Monitoring & Maintenance

### Automated Tasks
- **02:00 Daily**: System backup
- **03:00 Daily**: Log rotation
- **Every 15min**: System health checks
- **04:00 Weekly**: Update availability check

### Log Files
- `/var/log/backup.log`: Backup operations
- `/var/log/maintenance.log`: Maintenance tasks
- `/var/log/system_monitor.log`: Health monitoring
- `/var/log/updates.log`: Available updates

### Health Checks
The system automatically monitors:
- Disk usage (alerts at >85%)
- Memory usage (alerts at >90%)
- Container health status
- Service availability

## Troubleshooting

### Common Issues

#### SSH Connection Failed
```bash
# Check VM status
make status

# Test connectivity
ansible all -m ping -i inventory/hosts.yml

# Check SSH configuration
ssh -p 2222 azureuser@<VM_IP>
```

#### Web Application Not Accessible
```bash
# Check container status
docker ps

# Check service status
systemctl status demo-web-app

# Check firewall
sudo ufw status

# Test local connectivity
curl http://localhost:80
```

#### Deployment Failures
```bash
# Run with verbose output
ansible-playbook -i inventory/hosts.yml site.yml -vvv

# Check specific role
ansible-playbook -i inventory/hosts.yml site.yml --tags docker

# Skip problematic tasks
ansible-playbook -i inventory/hosts.yml site.yml --skip-tags azure-monitor
```

### Validation Commands
```bash
# Validate syntax
ansible-playbook --syntax-check site.yml -i inventory/hosts.yml

# Check variables
ansible-inventory -i inventory/hosts.yml --list

# Test specific modules
ansible all -m setup -i inventory/hosts.yml
```

## Integration with Terraform

This configuration management system is designed to work seamlessly with the Terraform infrastructure in the `../provisioning` directory:

1. **Automatic IP Detection**: The deploy script reads VM IP from Terraform output
2. **Makefile Integration**: Use `make deploy` for full infrastructure + configuration
3. **State Consistency**: Respects Terraform-managed resources

## Best Practices

### Security
- Regular security updates via unattended-upgrades
- Principle of least privilege for all services
- Comprehensive logging and monitoring
- Regular backup verification

### Operations
- Idempotent playbook design
- Tagged roles for selective execution
- Comprehensive error handling
- Automated health checks

### Maintenance
- Regular playbook updates
- Security patch monitoring
- Log analysis and cleanup
- Backup restoration testing

## Support & Maintenance

This configuration management system is designed for production use with:
- **High Availability**: Auto-restart capabilities for all services
- **Security**: Industry-standard hardening practices
- **Monitoring**: Comprehensive logging and alerting
- **Automation**: Minimal manual intervention required
FOFO
Test
Test
ANOTHER TEST
TEST
#test
#TEST
TEST
