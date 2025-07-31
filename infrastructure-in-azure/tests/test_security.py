"""
Security validation tests
"""
import pytest
import testinfra
import yaml
import os
import time

def get_vm_ip(environment):
    """Get VM IP from inventory file"""
    inventory_path = f"configuration-management/inventory/hosts.yml"
    
    if not os.path.exists(inventory_path):
        pytest.skip(f"Inventory file not found: {inventory_path}")
    
    with open(inventory_path, 'r') as f:
        inventory = yaml.safe_load(f)
    
    vm_hosts = inventory['all']['children']['azure_vms']['hosts']
    vm_name = list(vm_hosts.keys())[0]
    vm_ip = vm_hosts[vm_name]['ansible_host']
    
    return vm_ip, vm_name

@pytest.fixture(scope="module")
def vm_connection(request):
    """Create testinfra connection to VM"""
    environment = request.config.getoption("--env", default="dev")
    vm_ip, vm_name = get_vm_ip(environment)
    
    connection_string = f"ssh://azureuser@{vm_ip}"
    host = testinfra.get_host(connection_string)
    
    # Wait for SSH to be available
    max_retries = 30
    for i in range(max_retries):
        try:
            host.run("echo 'SSH test'")
            break
        except Exception as e:
            if i == max_retries - 1:
                pytest.fail(f"Could not connect to VM after {max_retries} attempts: {e}")
            time.sleep(10)
    
    return host, vm_ip

def test_no_default_passwords(vm_connection):
    """Test that no default passwords are set"""
    host, _ = vm_connection
    
    # Check shadow file for password policies
    shadow_file = host.file("/etc/shadow")
    assert shadow_file.exists
    
    # Root should be locked
    root_entry = host.run("grep '^root:' /etc/shadow")
    assert "!" in root_entry.stdout or "*" in root_entry.stdout, "Root account should be locked"

def test_ssh_security_configuration(vm_connection):
    """Test SSH security settings"""
    host, _ = vm_connection
    
    sshd_config = host.file("/etc/ssh/sshd_config")
    config_content = sshd_config.content_string.lower()
    
    security_tests = [
        ("passwordauthentication no", "Password authentication should be disabled"),
        ("permitrootlogin no", "Root login should be disabled"),
        ("protocol 2", "SSH Protocol 2 should be enforced"),
        ("x11forwarding no", "X11 forwarding should be disabled"),
        ("maxauthtries", "Max auth tries should be limited")
    ]
    
    for setting, message in security_tests:
        assert setting in config_content, message

def test_firewall_rules(vm_connection):
    """Test firewall configuration"""
    host, _ = vm_connection
    
    # Check UFW status
    ufw_status = host.run("ufw status numbered")
    assert "Status: active" in ufw_status.stdout, "UFW should be active"
    
    # Should allow SSH
    assert "22/tcp" in ufw_status.stdout or "ssh" in ufw_status.stdout.lower(), \
        "SSH should be allowed through firewall"
    
    # Should allow HTTP/HTTPS if web server is configured
    status_output = ufw_status.stdout.lower()
    
    # Check that default policies are restrictive
    ufw_verbose = host.run("ufw status verbose")
    assert "default: deny (incoming)" in ufw_verbose.stdout.lower(), \
        "Default incoming policy should be deny"

def test_fail2ban_jails(vm_connection):
    """Test fail2ban jail configuration"""
    host, _ = vm_connection
    
    # Check fail2ban status
    fail2ban_status = host.run("fail2ban-client status")
    assert fail2ban_status.exit_status == 0, "fail2ban should be running"
    
    # Check SSH jail specifically
    ssh_jail = host.run("fail2ban-client status sshd")
    assert ssh_jail.exit_status == 0, "SSH jail should be configured"
    
    # Verify jail configuration
    jail_conf = host.file("/etc/fail2ban/jail.local")
    if jail_conf.exists:
        config_content = jail_conf.content_string
        assert "[sshd]" in config_content, "SSH jail should be configured"

def test_system_hardening(vm_connection):
    """Test system hardening measures"""
    host, _ = vm_connection
    
    # Check if unnecessary services are disabled
    unnecessary_services = ['telnet', 'rsh', 'rlogin']
    
    for service in unnecessary_services:
        service_check = host.run(f"systemctl is-enabled {service} 2>/dev/null || echo 'not-found'")
        assert "not-found" in service_check.stdout or "disabled" in service_check.stdout, \
            f"Service {service} should not be enabled"

def test_file_permissions(vm_connection):
    """Test critical file permissions"""
    host, _ = vm_connection
    
    critical_files = [
        ("/etc/passwd", "644"),
        ("/etc/shadow", "640"),
        ("/etc/group", "644"),
        ("/etc/ssh/sshd_config", "600")
    ]
    
    for file_path, expected_perms in critical_files:
        file_obj = host.file(file_path)
        if file_obj.exists:
            actual_perms = oct(file_obj.mode)[-3:]
            assert actual_perms == expected_perms, \
                f"File {file_path} has permissions {actual_perms}, expected {expected_perms}"

def test_network_security(vm_connection):
    """Test network security configuration"""
    host, _ = vm_connection
    
    # Check for open ports
    netstat_result = host.run("ss -tuln")
    
    # Should have SSH open
    assert ":22 " in netstat_result.stdout, "SSH port should be listening"
    
    # Check for unnecessary open ports
    dangerous_ports = [':23 ', ':135 ', ':139 ', ':445 ', ':3389 ']
    for port in dangerous_ports:
        assert port not in netstat_result.stdout, f"Dangerous port {port.strip()} should not be open"

def test_kernel_parameters(vm_connection):
    """Test kernel security parameters"""
    host, _ = vm_connection
    
    security_params = [
        ("net.ipv4.ip_forward", "0"),
        ("net.ipv4.conf.all.send_redirects", "0"),
        ("net.ipv4.conf.default.send_redirects", "0"),
        ("net.ipv4.conf.all.accept_redirects", "0"),
        ("net.ipv4.conf.default.accept_redirects", "0")
    ]
    
    for param, expected_value in security_params:
        sysctl_result = host.run(f"sysctl {param}")
        if sysctl_result.exit_status == 0:
            actual_value = sysctl_result.stdout.split('=')[-1].strip()
            assert actual_value == expected_value, \
                f"Kernel parameter {param} should be {expected_value}, got {actual_value}"

def test_user_accounts(vm_connection):
    """Test user account security"""
    host, _ = vm_connection
    
    # Check for users with UID 0 (should only be root)
    uid_zero_users = host.run("awk -F: '$3==0 {print $1}' /etc/passwd")
    uid_zero_list = uid_zero_users.stdout.strip().split('\n')
    assert uid_zero_list == ['root'], f"Only root should have UID 0, found: {uid_zero_list}"
    
    # Check for users with empty passwords
    empty_passwords = host.run("awk -F: '$2==\"\" {print $1}' /etc/shadow")
    assert empty_passwords.stdout.strip() == "", "No users should have empty passwords"

def test_log_file_permissions(vm_connection):
    """Test log file permissions"""
    host, _ = vm_connection
    
    log_files = [
        "/var/log/auth.log",
        "/var/log/syslog",
        "/var/log/fail2ban.log"
    ]
    
    for log_file in log_files:
        file_obj = host.file(log_file)
        if file_obj.exists:
            # Log files should not be world-readable
            perms = oct(file_obj.mode)
            assert not (file_obj.mode & 0o004), f"Log file {log_file} should not be world-readable"

def test_cron_security(vm_connection):
    """Test cron security configuration"""
    host, _ = vm_connection
    
    # Check cron permissions
    cron_files = ["/etc/crontab", "/etc/cron.deny"]
    
    for cron_file in cron_files:
        file_obj = host.file(cron_file)
        if file_obj.exists:
            # Cron files should have restricted permissions
            assert file_obj.user == "root", f"{cron_file} should be owned by root"
            assert not (file_obj.mode & 0o022), f"{cron_file} should not be group/world writable"


if __name__ == "__main__":
    pytest.main([__file__])
