import asyncio
import io
import logging
import time
from typing import Tuple, Optional, Dict, List
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2

from ..config import settings

class ImageService:
    """
    Production-grade image preprocessing service.
    
    Handles image validation, preprocessing, and conversion to model-ready tensors.
    Designed for single-image processing in a web API context.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Create the transformation pipeline
        # This matches the preprocessing from Part 1, adapted for single images
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Direct resize - simpler for API use
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 dataset statistics
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Quality thresholds
        self.min_size = 8  # Minimum dimension in pixels
        self.max_size = 4096  # Maximum dimension to prevent memory issues
        self.blur_threshold = 100.0  # Laplacian variance threshold
        
    async def validate_image_file(self, file_content: bytes, filename: str) -> None:
        """
        Validate uploaded image file before processing.
        
        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename for format detection
            
        Raises:
            ValueError: If the image is invalid or unsupported
        """
        # Check file size
        if len(file_content) > settings.max_request_size:
            raise ValueError(f"File size exceeds maximum allowed ({settings.max_request_size // (1024*1024)}MB)")
        
        if len(file_content) == 0:
            raise ValueError("Empty file uploaded")
        
        # Check file extension
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        if f'.{file_extension}' not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {file_extension}. Supported formats: {', '.join(self.supported_formats)}")
        
        # Try to open the image to validate it's not corrupted
        try:
            image = Image.open(io.BytesIO(file_content))
            image.verify()  # This will raise an exception if the image is corrupted
        except UnidentifiedImageError:
            raise ValueError("Invalid or corrupted image file")
        except Exception as e:
            raise ValueError(f"Error reading image: {str(e)}")
    
    async def load_and_validate_image(self, file_content: bytes) -> Image.Image:
        """
        Load image from bytes and perform quality validation.
        
        Args:
            file_content: Raw bytes of the image file
            
        Returns:
            PIL Image object ready for preprocessing
            
        Raises:
            ValueError: If the image fails quality checks
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check dimensions
            width, height = image.size
            if width < self.min_size or height < self.min_size:
                raise ValueError(f"Image too small: {width}x{height}. Minimum size: {self.min_size}x{self.min_size}")
            
            if width > self.max_size or height > self.max_size:
                raise ValueError(f"Image too large: {width}x{height}. Maximum size: {self.max_size}x{self.max_size}")
            
            # Check if image is too blurry (optional quality check)
            if await self._is_image_too_blurry(image):
                self.logger.warning("Uploaded image appears to be very blurry")
                # Note: We warn but don't reject - users might want to classify blurry images
            
            return image
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise  # Re-raise validation errors
            raise ValueError(f"Error processing image: {str(e)}")
    
    async def _is_image_too_blurry(self, image: Image.Image) -> bool:
        """
        Check if image is too blurry using Laplacian variance.
        
        This is the same technique we used in Part 1 for filtering blurry images.
        """
        try:
            # Convert PIL image to numpy array for OpenCV
            img_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return laplacian_var < self.blur_threshold
            
        except Exception as e:
            self.logger.warning(f"Error checking image blur: {str(e)}")
            return False  # If we can't check, assume it's okay
    
    async def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image into model-ready tensor.
        
        This applies the same preprocessing pipeline we used in training:
        1. Resize to 32x32 pixels
        2. Convert to tensor (0-1 range)
        3. Normalize with CIFAR-10 statistics
        4. Add batch dimension
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed tensor of shape (1, 3, 32, 32)
        """
        start_time = time.time()
        
        try:
            # Apply the transformation pipeline
            tensor = self.transform(image)
            
            # Add batch dimension (model expects batched input)
            tensor = tensor.unsqueeze(0)  # Shape: (1, 3, 32, 32)
            
            processing_time = time.time() - start_time
            self.logger.debug(f"Image preprocessing completed in {processing_time*1000:.2f}ms")
            
            return tensor
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    async def process_uploaded_file(self, file_content: bytes, filename: str) -> torch.Tensor:
        """
        Complete pipeline: validate, load, and preprocess an uploaded image file.
        
        This is the main entry point for the API - it takes raw uploaded bytes
        and returns a tensor ready for model inference.
        
        Args:
            file_content: Raw bytes of uploaded file
            filename: Original filename
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate the file
            await self.validate_image_file(file_content, filename)
            
            # Step 2: Load and validate the image
            image = await self.load_and_validate_image(file_content)
            
            # Step 3: Preprocess for model
            tensor = await self.preprocess_image(image)
            
            total_time = time.time() - start_time
            self.logger.info(f"Complete image processing pipeline completed in {total_time*1000:.2f}ms")
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {str(e)}")
            raise
    
    def get_preprocessing_info(self) -> Dict[str, any]:
        """
        Get information about the preprocessing pipeline for API documentation.
        """
        return {
            "supported_formats": list(self.supported_formats),
            "target_size": "32x32 pixels",
            "normalization": {
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2023, 0.1994, 0.2010]
            },
            "max_file_size_mb": settings.max_request_size // (1024 * 1024),
            "min_dimension": self.min_size,
            "max_dimension": self.max_size
        }

# Global image service instance
image_service = ImageService()