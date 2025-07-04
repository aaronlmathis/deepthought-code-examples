import asyncio
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from contextlib import asynccontextmanager

from ..config import settings, CIFAR10_CLASSES

# Import our PyramidNet architecture from Part 2
# We'll need to copy the model classes here or import them
class IdentityPadding(nn.Module):
    """Identity padding for PyramidNet residual connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(IdentityPadding, self).__init__()
        if stride == 2:
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self.pooling = None
        self.add_channels = out_channels - in_channels
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        if self.pooling is not None:
            out = self.pooling(out)
        return out

class ResidualBlock(nn.Module):
    """Residual block for PyramidNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)      
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                stride=1, padding=1, bias=False)    
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        self.stride = stride

    def forward(self, x):
        shortcut = self.down_sample(x)
        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out += shortcut
        return out

class PyramidNet(nn.Module):
    """PyramidNet architecture - copied from Part 2"""
    def __init__(self, num_layers, alpha, block, num_classes=10):
        super(PyramidNet, self).__init__()   	
        self.in_channels = 16
        self.num_layers = num_layers
        self.addrate = alpha / (3*self.num_layers*1.0)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self.get_layers(block, stride=1)
        self.layer2 = self.get_layers(block, stride=2)
        self.layer3 = self.get_layers(block, stride=2)

        self.out_channels = int(round(self.out_channels))
        self.bn_out= nn.BatchNorm2d(self.out_channels)
        self.relu_out = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(self.out_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, stride):
        layers_list = []
        for _ in range(self.num_layers):
            self.out_channels = self.in_channels + self.addrate
            layers_list.append(block(int(round(self.in_channels)), 
                                     int(round(self.out_channels)), 
                                     stride))
            self.in_channels = self.out_channels
            stride=1
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn_out(x)
        x = self.relu_out(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x

def pyramidnet():
    """Factory function to create PyramidNet model"""
    block = ResidualBlock
    model = PyramidNet(num_layers=18, alpha=48, block=block)
    return model

class ModelService:
    """
    Production-grade model service for PyramidNet inference.
    
    Key design principles:
    - Singleton pattern: One model instance per application
    - Thread-safe: Handles concurrent requests safely
    - Async-friendly: Non-blocking operations where possible
    - Comprehensive error handling and logging
    - Performance monitoring built-in
    """
    
    def __init__(self):
        self.model: Optional[PyramidNet] = None
        self.device: torch.device = None
        self.is_loaded: bool = False
        self.load_time: Optional[float] = None
        self.prediction_count: int = 0
        self.total_prediction_time: float = 0.0
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()  # For thread-safe model loading
        
    async def load_model(self) -> None:
        """
        Load the trained PyramidNet model with comprehensive error handling.
        
        This method is designed to be called once at application startup.
        It includes validation, performance monitoring, and detailed logging.
        """
        async with self._lock:  # Prevent multiple simultaneous loads
            if self.is_loaded:
                self.logger.info("Model already loaded, skipping...")
                return
                
            start_time = time.time()
            self.logger.info(f"Loading PyramidNet model from {settings.model_path}")
            
            try:
                # Validate model file exists
                model_path = Path(settings.model_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Configure device
                self.device = torch.device(settings.model_device)
                self.logger.info(f"Using device: {self.device}")
                
                # Create model architecture
                self.model = pyramidnet()
                
                # Load checkpoint with proper PyTorch 2.6+ compatibility
                try:
                    # First try with weights_only=True (secure mode)
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                except Exception as e:
                    self.logger.warning(f"Secure loading failed: {e}")
                    self.logger.info("Falling back to weights_only=False for older checkpoint format")
                    # Fall back to weights_only=False for older checkpoints
                    # This is safe if you trust the source of your model file
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                    self.logger.info(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.2f}%")
                else:
                    self.model.load_state_dict(checkpoint)
                    
                # Move to device and set to evaluation mode
                self.model.to(self.device)
                self.model.eval()
                
                # Warm up the model with a dummy prediction
                await self._warmup_model()
                
                self.load_time = time.time() - start_time
                self.is_loaded = True
                
                self.logger.info(f"Model loaded successfully in {self.load_time:.2f} seconds")
                self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {str(e)}")
                self.is_loaded = False
                raise RuntimeError(f"Model loading failed: {str(e)}")
    
    async def _warmup_model(self) -> None:
        """
        Warm up the model with a dummy prediction.
        
        This ensures that CUDA memory is allocated and the model is ready
        for fast inference on the first real request.
        """
        self.logger.info("Warming up model...")
        try:
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            self.logger.info("Model warmup completed")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {str(e)}")
    
    async def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Make a prediction on a single image with comprehensive error handling.
        
        Args:
            image_tensor: Preprocessed image tensor of shape (1, 3, 32, 32)
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Validate input tensor
            if image_tensor.dim() != 4 or image_tensor.shape != (1, 3, 32, 32):
                raise ValueError(f"Expected tensor shape (1, 3, 32, 32), got {image_tensor.shape}")
            
            # Move to device if needed
            if image_tensor.device != self.device:
                image_tensor = image_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence_scores = probabilities.cpu().numpy()[0]
                predicted_class_idx = np.argmax(confidence_scores)
                
            # Prepare response
            prediction_time = time.time() - start_time
            
            response = {
                "predicted_class": CIFAR10_CLASSES[predicted_class_idx],
                "predicted_class_id": int(predicted_class_idx),
                "confidence": float(confidence_scores[predicted_class_idx]),
                "all_confidences": {
                    CIFAR10_CLASSES[i]: float(confidence_scores[i]) 
                    for i in range(len(CIFAR10_CLASSES))
                },
                "prediction_time_ms": round(prediction_time * 1000, 2),
                "model_device": str(self.device)
            }
            
            # Update statistics
            self.prediction_count += 1
            self.total_prediction_time += prediction_time
            
            self.logger.debug(f"Prediction completed in {prediction_time*1000:.2f}ms: {response['predicted_class']} ({response['confidence']:.3f})")
            
            return response
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def batch_predict(self, image_tensors: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple images efficiently.
        
        This method batches multiple images together for more efficient GPU utilization.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not image_tensors:
            return []
        
        start_time = time.time()
        
        try:
            # Stack tensors into a batch
            batch_tensor = torch.stack(image_tensors).to(self.device)
            
            # Make batch prediction
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence_scores = probabilities.cpu().numpy()
                
            # Prepare responses
            responses = []
            for i, scores in enumerate(confidence_scores):
                predicted_class_idx = np.argmax(scores)
                responses.append({
                    "predicted_class": CIFAR10_CLASSES[predicted_class_idx],
                    "predicted_class_id": int(predicted_class_idx),
                    "confidence": float(scores[predicted_class_idx]),
                    "all_confidences": {
                        CIFAR10_CLASSES[j]: float(scores[j]) 
                        for j in range(len(CIFAR10_CLASSES))
                    }
                })
            
            prediction_time = time.time() - start_time
            self.logger.info(f"Batch prediction of {len(image_tensors)} images completed in {prediction_time*1000:.2f}ms")
            
            return responses
            
        except Exception as e:
            error_msg = f"Batch prediction failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model service statistics for monitoring"""
        avg_prediction_time = (
            self.total_prediction_time / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            "is_loaded": self.is_loaded,
            "device": str(self.device) if self.device else None,
            "load_time_seconds": self.load_time,
            "prediction_count": self.prediction_count,
            "average_prediction_time_ms": round(avg_prediction_time * 1000, 2),
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def is_healthy(self) -> bool:
        """Health check for monitoring systems"""
        return self.is_loaded and self.model is not None

# Global model service instance
model_service = ModelService()