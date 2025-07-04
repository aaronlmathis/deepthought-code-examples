from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

class ImageFormat(str, Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    TIF = "tif"

class CIFAR10Class(str, Enum):
    """CIFAR-10 classification classes"""
    AIRPLANE = "airplane"
    AUTOMOBILE = "automobile"
    BIRD = "bird"
    CAT = "cat"
    DEER = "deer"
    DOG = "dog"
    FROG = "frog"
    HORSE = "horse"
    SHIP = "ship"
    TRUCK = "truck"

class PredictionConfidence(BaseModel):
    """Individual class confidence score"""
    class_name: CIFAR10Class = Field(
        ..., 
        description="The name of the classification class"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0.0 and 1.0"
    )

class PredictionResult(BaseModel):
    """
    Main prediction result model.
    
    This model defines the structure of successful prediction responses.
    It includes the primary prediction, confidence scores for all classes,
    and metadata about the prediction process.
    """
    predicted_class: CIFAR10Class = Field(
        ...,
        description="The most likely class for the uploaded image"
    )
    
    predicted_class_id: int = Field(
        ...,
        ge=0,
        le=9,
        description="Numeric ID of the predicted class (0-9)"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted class"
    )
    
    all_confidences: Dict[str, float] = Field(
        ...,
        description="Confidence scores for all CIFAR-10 classes"
    )
    
    prediction_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken for prediction in milliseconds"
    )
    
    model_device: str = Field(
        ...,
        description="Device used for prediction (cpu/cuda)"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when prediction was made"
    )
    
    @field_validator('all_confidences')
    @classmethod
    def validate_all_confidences(cls, v):
        """Ensure all_confidences contains all CIFAR-10 classes"""
        expected_classes = [class_item.value for class_item in CIFAR10Class]
        if set(v.keys()) != set(expected_classes):
            raise ValueError("all_confidences must contain all CIFAR-10 classes")
        
        # Validate that all confidence scores are between 0 and 1
        for class_name, confidence in v.items():
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Confidence for {class_name} must be between 0.0 and 1.0")
        
        return v
    
    @model_validator(mode='after')
    def validate_confidence_consistency(self):
        """Ensure the main confidence matches the predicted class confidence"""
        predicted_class = self.predicted_class
        all_confidences = self.all_confidences
        confidence = self.confidence
        
        if predicted_class in all_confidences:
            expected_confidence = all_confidences[predicted_class]
            if abs(confidence - expected_confidence) > 1e-6:
                raise ValueError("Main confidence score must match predicted class confidence")
        
        return self

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions (future enhancement)"""
    images: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of base64-encoded images (max 10)"
    )
    
    include_all_confidences: bool = Field(
        default=True,
        description="Whether to include confidence scores for all classes"
    )

class BatchPredictionResult(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResult] = Field(
        ...,
        description="List of prediction results"
    )
    
    total_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total time for all predictions in milliseconds"
    )
    
    batch_size: int = Field(
        ...,
        ge=1,
        description="Number of images processed"
    )

class ErrorResponse(BaseModel):
    """
    Standardized error response model.
    
    This ensures all API errors have a consistent structure,
    making it easier for clients to handle errors programmatically.
    """
    error: str = Field(
        ...,
        description="Error type or category"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    detail: Optional[str] = Field(
        None,
        description="Additional error details for debugging"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when error occurred"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Unique identifier for the request (for tracking)"
    )

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(
        ...,
        description="Health status (healthy/unhealthy)"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of health check"
    )
    
    version: str = Field(
        ...,
        description="API version"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready"
    )
    
    uptime_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="API uptime in seconds"
    )

class ModelStatsResponse(BaseModel):
    """Model statistics response model"""
    is_loaded: bool = Field(
        ...,
        description="Whether the model is loaded"
    )
    
    device: Optional[str] = Field(
        None,
        description="Device the model is running on"
    )
    
    load_time_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Time taken to load the model"
    )
    
    prediction_count: int = Field(
        ...,
        ge=0,
        description="Total number of predictions made"
    )
    
    average_prediction_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average prediction time in milliseconds"
    )
    
    model_parameters: int = Field(
        ...,
        ge=0,
        description="Number of trainable parameters in the model"
    )

class ImageUploadResponse(BaseModel):
    """Response for successful image upload and preprocessing"""
    message: str = Field(
        ...,
        description="Success message"
    )
    
    filename: str = Field(
        ...,
        description="Original filename"
    )
    
    file_size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of uploaded file in bytes"
    )
    
    preprocessing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken for image preprocessing"
    )
    
    image_dimensions: Dict[str, int] = Field(
        ...,
        description="Original image dimensions"
    )

# Example models for API documentation
class PredictionExample:
    """Example data for API documentation"""
    
    successful_prediction = {
        "predicted_class": "cat",
        "predicted_class_id": 3,
        "confidence": 0.891,
        "all_confidences": {
            "airplane": 0.012,
            "automobile": 0.008,
            "bird": 0.045,
            "cat": 0.891,
            "deer": 0.023,
            "dog": 0.015,
            "frog": 0.003,
            "horse": 0.002,
            "ship": 0.001,
            "truck": 0.000
        },
        "prediction_time_ms": 45.2,
        "model_device": "cuda",
        "timestamp": "2025-06-21T10:30:45.123456Z"
    }
    
    error_response = {
        "error": "ValidationError",
        "message": "Invalid image format",
        "detail": "Supported formats: jpg, jpeg, png, bmp, tiff",
        "timestamp": "2025-06-21T10:30:45.123456Z",
        "request_id": "req_123456789"
    }
    
    health_check = {
        "status": "healthy",
        "timestamp": "2025-06-21T10:30:45.123456Z",
        "version": "1.0.0",
        "model_loaded": True,
        "uptime_seconds": 3600.5
    }