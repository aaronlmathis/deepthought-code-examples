version: 2
ethernets:
  ${interface}:
    dhcp4: false
    addresses: [${ip}/${prefix}]
    gateway4: ${gateway}
    nameservers:
      addresses: [${nameserver}]