#cloud-config
hostname: ${hostname}
users:
  - name: ubuntu
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    # For debugging console access, you can set a password. Remove for production.
    password: "atlantis"
    lock_passwd: false
    ssh_authorized_keys:
      - ${ssh_key}
