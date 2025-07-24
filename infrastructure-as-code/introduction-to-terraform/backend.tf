terraform {
  cloud {
    organization = "Deep-Thought"
    workspaces {
      name = "homelab"
    }
  }
} 
