"""
Production health checks and security verification. 
These tests ensure the production environment is secure, stable, and compliant with best practices.
"""
import os
import re
import subprocess
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_vm_ip() -> str:
    """Return the VM IP from the workflow’s environment; skip tests if absent."""
    vm_ip = os.environ.get("VM_IP")
    if not vm_ip:
        pytest.skip("VM_IP environment variable not set")
    return vm_ip


def ssh_command(host: str, command: str) -> subprocess.CompletedProcess:
    """Run a command on the remote host via SSH and return the CompletedProcess."""
    try:
        return subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                f"azureuser@{host}",
                command,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"SSH command failed: {exc}")  # surfaces stderr in pytest output


# ── Tests ──────────────────────────────────────────────────────────────────────
def test_vm_health_status():
    """Basic VM health (uptime / mem / root disk)."""
    vm_ip = get_vm_ip()
    result = ssh_command(vm_ip, "uptime && free -m && df -h /")
    assert "load average" in result.stdout, "uptime output missing"
    assert "Mem:" in result.stdout, "memory info missing"
    assert "/" in result.stdout, "disk info missing"
    print("[PASS] VM health checks passed")


def test_critical_services_status():
    """Ensure critical services are active (Ubuntu defaults + hardening)."""
    vm_ip = get_vm_ip()
    critical_services = ["ssh", "fail2ban", "ufw", "chronyd", "rsyslog"]
    for svc in critical_services:
        result = ssh_command(vm_ip, f"systemctl is-active {svc}")
        assert "active" in result.stdout, f"service {svc} is not running"
        print(f"[PASS] {svc} is active")


def test_network_connectivity():
    """Verify outbound connectivity and DNS resolution."""
    vm_ip = get_vm_ip()
    # External reachability
    result = ssh_command(vm_ip, "ping -c 2 8.8.8.8")
    assert result.returncode == 0, "ICMP to 8.8.8.8 failed"
    # DNS resolution
    result = ssh_command(vm_ip, "nslookup google.com")
    assert result.returncode == 0, "DNS resolution failed"
    print("[PASS] network connectivity + DNS OK")


def test_security_baseline():
    """Security sanity checks for processes and *unexpected* open ports."""
    vm_ip = get_vm_ip()

    # 1️⃣  no stray netcat variants
    result = ssh_command(vm_ip, r"ps -eo pid,comm | grep -E '(nc$|netcat$|ncat$)' || true")
    assert result.stdout.strip() == "", f"Suspicious netcat‑like processes:\n{result.stdout}"

    # 2️⃣  listening sockets we do NOT expect (allow 22/ssh, 53/dns‑stub, 68/dhcp)
    allowed_ports = ("22", "53", "68")
    output = ssh_command(vm_ip, "ss -tuln").stdout.splitlines()

    unexpected = [
        line for line in output
        if line and not line.startswith("Netid")                        # skip header
           and not re.search(r":(22|53|68)\b", line)                   # allow common ports
           and "127.0.0.1" not in line and "::1" not in line           # ignore loopback
    ]

    assert not unexpected, "Unexpected listening sockets:\n" + "\n".join(unexpected)
    print("[PASS] baseline security checks passed")


def test_log_analysis():
    """Check logs and tolerate known benign noise on Azure Ubuntu images."""
    vm_ip = get_vm_ip()

    # failed SSH logins < 5
    fails = int(
        ssh_command(vm_ip, "grep -c 'Failed password' /var/log/auth.log || true").stdout.strip() or 0
    )
    assert fails < 5, f"{fails} recent failed SSH logins"

    # system logs present
    assert ssh_command(vm_ip, "test -f /var/log/syslog -o -f /var/log/messages").returncode == 0

    # collect recent priority=err messages and filter benign ones
    raw = ssh_command(vm_ip, "journalctl --since '-1h' --priority err -q || true").stdout.splitlines()
    benign_keys = (
        "RETBleed",                         # firmware warning
        "dhclient",                         # DHCP noisy perms / TIME_MAX
        "I/O error, dev sr0",               # harmless optical‑drive reads
        "Buffer I/O error on dev sr0",
        "execve (/bin/true"                 # dhclient workaround
        "kex_protocol_error",               # harmless aborted SSH handshakes
        "kex_exchange_identification",      # same – banner/ID issues
    )
    noisy_errors = [l for l in raw if not any(k in l for k in benign_keys)]

    assert len(noisy_errors) < 5, "recent critical errors in journal:\n" + "\n".join(noisy_errors)
    print("[PASS] log analysis clean")


if __name__ == "__main__":
    pytest.main([__file__])
    