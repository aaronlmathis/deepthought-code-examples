"""
Infrastructure validation tests
"""
import pytest
import os
import yaml

def test_environment_configs_exist():
    """Test that all environment configurations exist"""
    environments = ['dev', 'staging', 'production']
    
    for env in environments:
        # Check Terraform configs
        tfvars_path = f"environments/{env}/terraform.tfvars"
        backend_path = f"environments/{env}/backend.tf"
        ansible_vars_path = f"environments/{env}/ansible_vars.yml"
        
        assert os.path.exists(tfvars_path), f"Missing {tfvars_path}"
        assert os.path.exists(backend_path), f"Missing {backend_path}"
        assert os.path.exists(ansible_vars_path), f"Missing {ansible_vars_path}"

def test_terraform_variables_format():
    """Test that Terraform variables are properly formatted"""
    environments = ['dev', 'staging', 'production']
    required_vars = [
        'resource_group_name',
        'environment',
        'location',
        'vm_name',
        'vm_size'
    ]
    
    for env in environments:
        tfvars_path = f"environments/{env}/terraform.tfvars"
        
        with open(tfvars_path, 'r') as f:
            content = f.read()
            
        for var in required_vars:
            assert var in content, f"Missing required variable {var} in {tfvars_path}"

def test_ansible_variables_format():
    """Test that Ansible variables are properly formatted"""
    environments = ['dev', 'staging', 'production']
    required_vars = [
        'environment_name',
        'ssh_port',
        'fail2ban_maxretry',
        'timezone'
    ]
    
    for env in environments:
        ansible_vars_path = f"environments/{env}/ansible_vars.yml"
        
        with open(ansible_vars_path, 'r') as f:
            vars_data = yaml.safe_load(f)
            
        for var in required_vars:
            assert var in vars_data, f"Missing required variable {var} in {ansible_vars_path}"

def test_environment_isolation():
    """Test that environments have different resource names"""
    environments = ['dev', 'staging', 'production']
    resource_groups = []
    vm_names = []
    
    for env in environments:
        tfvars_path = f"environments/{env}/terraform.tfvars"
        
        with open(tfvars_path, 'r') as f:
            content = f.read()
            
        # Extract resource group name
        for line in content.split('\n'):
            if line.startswith('resource_group_name'):
                rg_name = line.split('=')[1].strip().strip('"')
                resource_groups.append(rg_name)
            elif line.startswith('vm_name'):
                vm_name = line.split('=')[1].strip().strip('"')
                vm_names.append(vm_name)
    
    # Ensure all resource groups are unique
    assert len(set(resource_groups)) == len(resource_groups), "Resource groups must be unique across environments"
    assert len(set(vm_names)) == len(vm_names), "VM names must be unique across environments"

def test_production_security_settings():
    """Test that production has stricter security settings"""
    with open('environments/production/ansible_vars.yml', 'r') as f:
        prod_vars = yaml.safe_load(f)
    
    with open('environments/dev/ansible_vars.yml', 'r') as f:
        dev_vars = yaml.safe_load(f)
    
    # Production should have stricter fail2ban settings
    assert prod_vars['fail2ban_maxretry'] <= dev_vars['fail2ban_maxretry'], \
        "Production should have stricter fail2ban retry limit"
    
    assert prod_vars['fail2ban_bantime'] >= dev_vars['fail2ban_bantime'], \
        "Production should have longer ban times"
    
    # Production should not have debug mode enabled
    assert not prod_vars.get('debug_mode', False), \
        "Production should not have debug mode enabled"

if __name__ == "__main__":
    pytest.main([__file__])
