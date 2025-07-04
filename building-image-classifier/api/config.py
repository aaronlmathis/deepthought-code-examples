import os
from pathlib import Path
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings  # Changed this line
import torch

class Settings(BaseSettings):
    """
    Application configuration using Pydantic BaseSettings.
    
    This approach provides several production benefits:
    - Type validation ensures configuration values are correct
    - Environment variable support for containerized deployments
    - Default values prevent startup failures
    - Automatic documentation of configuration options
    """
    
    # Application metadata
    app_name: str = "PyramidNet Image Classifier API"
    app_version: str = "1.0.0"
    app_description: str = "Production-grade image classification API using PyramidNet CNN"
    
    # Environment configuration
    environment: str = "development"  # development, staging, production
    debug: bool = True
    
    # API configuration
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True  # Auto-reload for development
    
    # Security settings
    cors_origins: list = ["http://localhost:3000", "http://localhost:8080"]
    api_key: Optional[str] = None  # Optional API key authentication
    max_request_size: int = 10 * 1024 * 1024  # 10MB max file size
    
    # Model configuration
    model_path: str = "models/best_pyramidnet_model.pth"
    model_device: str = "auto"  # auto, cpu, cuda
    model_batch_size: int = 1  # For batch inference
    
    # Performance settings
    worker_timeout: int = 300  # 5 minutes for model loading
    prediction_timeout: int = 30  # 30 seconds for single prediction
    max_concurrent_requests: int = 10
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/api.log"
    access_log: bool = True
    
    # Monitoring and health checks
    health_check_timeout: int = 5
    metrics_enabled: bool = True
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Ensure environment is one of the allowed values"""
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of {allowed}')
        return v
    
    @field_validator('model_device')
    @classmethod
    def validate_device(cls, v):
        """Automatically detect device or validate manual setting"""
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif v in ["cpu", "cuda"]:
            if v == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available")
            return v
        else:
            raise ValueError("Device must be 'auto', 'cpu', or 'cuda'")
    
    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v):
        """Ensure model file exists (skip validation in development if not present)"""
        model_path = Path(v)
        if not model_path.exists():
            # In development, we might not have the model yet
            if cls.model_config.env_prefix and os.getenv(f"{cls.model_config.env_prefix}ENVIRONMENT", "development") != "production":
                print(f"Warning: Model file not found: {v} (OK in development)")
                return v
            else:
                # In production or when no environment is set, require the model file
                raise ValueError(f"Model file not found: {v}")
        return v
    
    def is_production(self) -> bool:
        """Helper to check if running in production"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Helper to check if running in development"""
        return self.environment == "development"
    
    model_config = {
        "env_file": ".env",  # Load from .env file if present
        "env_prefix": "CLASSIFIER_",  # Environment variables with this prefix override settings
        "case_sensitive": False
    }

# Create global settings instance
settings = Settings()

# CIFAR-10 class names - these should match your training data
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# API response messages
API_MESSAGES = {
    "model_loaded": "PyramidNet model loaded successfully",
    "prediction_success": "Image classified successfully",
    "invalid_image": "Invalid or corrupted image file",
    "file_too_large": f"File size exceeds maximum allowed ({settings.max_request_size // (1024*1024)}MB)",
    "prediction_error": "Error occurred during prediction",
    "health_check_passed": "API is healthy and ready to serve requests"
}