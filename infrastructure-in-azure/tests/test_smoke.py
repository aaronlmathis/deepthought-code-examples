"""
Basic smoke tests for deployment validation
"""

from pathlib import Path
import os
import subprocess
import time
import pytest

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TF_DIR   = Path(os.getenv("TF_DIR", "../provisioning")).resolve()
SSH_USER = os.getenv("SSH_USER", "azureuser")
SSH_KEY  = os.getenv("SSH_KEY", str(Path.home() / ".ssh/id_rsa"))
VM_IP    = None  # cached after first lookup


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Shortcut around subprocess.run with sane defaults."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        **kwargs,
    )


def get_vm_ip() -> str:
    """Return the VM’s public IP (read once from Terraform)."""
    # Prefer the value handed over by the deploy job
    ip_env = os.getenv("PUBLIC_IP_ADDRESS")
    if ip_env:
        return ip_env

    # Fallback to reading terraform output locally
    global VM_IP
    if VM_IP:
        return VM_IP
    result = _run(
        ["terraform", "output", "-raw", "public_ip_address"],
        cwd=TF_DIR,
    )
    if result.returncode != 0:
        pytest.fail(f"Unable to obtain VM IP from Terraform:\n{result.stderr}")

    VM_IP = result.stdout.strip()
    return VM_IP


def ssh_exec(remote_cmd: str) -> subprocess.CompletedProcess:
    """Execute a command on the VM via SSH and return CompletedProcess."""
    return _run(
        [
            "ssh",
            "-i", SSH_KEY,
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            f"{SSH_USER}@{get_vm_ip()}",
            remote_cmd,
        ]
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_vm_connectivity():
    """VM must respond to ping."""
    ip = get_vm_ip()
    print("[TEST] Testing VM connectivity (TCP-port 22)…")
    result = _run(["nc", "-zvw", "5", ip, "22"])
    assert result.returncode == 0, f"TCP port 22 unreachable on {ip}"
    print("[PASS] VM is reachable")


def test_ssh_connectivity():
    """SSH must succeed."""
    print("[TEST] Testing SSH connectivity…")
    for _ in range(6):  # retry ~30 s total
        result = ssh_exec("echo SSH-OK")
        if result.returncode == 0 and "SSH-OK" in result.stdout:
            print("[PASS] SSH connection successful")
            return
        time.sleep(5)
    pytest.fail(f"SSH failed after retries:\n{result.stderr}")


def test_required_services():
    """Expected services must be active."""
    print("[TEST] Testing required services…")
    required = ["sshd", "fail2ban", "chronyd", "ufw"]
    for svc in required:
        result = ssh_exec(f"systemctl is-active {svc}")
        assert result.returncode == 0 and result.stdout.strip() == "active", (
            f"Service {svc} is NOT running"
        )
        print(f"[PASS] Service {svc} is running")


def test_firewall_rules():
    """SSH port open; example port 8080 closed."""
    ip = get_vm_ip()
    print("[TEST] Testing firewall configuration…")

    # Port 22 open
    assert _run(["nc", "-zvw", "5", ip, "22"]).returncode == 0, "Port 22 unreachable"
    print("[PASS] SSH port 22 is open")

    # Port 8080 closed
    assert _run(["nc", "-zvw", "5", ip, "8080"]).returncode != 0, "Port 8080 open"
    print("[PASS] Unnecessary ports are closed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])