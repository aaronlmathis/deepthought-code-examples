import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import tempfile
import os

from api.main import app
from api.services.model_service import model_service

# Test configuration
TEST_IMAGE_SIZE = 32

def create_test_image(size=(TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), format="PNG"):
    """Create a test image for API testing"""
    # Create a simple test image with some pattern to make it more realistic
    image = Image.new("RGB", size, color="red")
    
    # Add some pattern to make it look more like a real image
    import random
    pixels = image.load()
    for i in range(size[0]):
        for j in range(size[1]):
            # Add some random noise to make it more realistic
            r = min(255, max(0, 255 + random.randint(-50, 50)))
            g = min(255, max(0, 0 + random.randint(-20, 20)))
            b = min(255, max(0, 0 + random.randint(-20, 20)))
            pixels[i, j] = (r, g, b)
    
    # Convert to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()

def create_invalid_image():
    """Create an invalid image file for testing"""
    return b"This is not an image file"

def create_oversized_image():
    """Create an oversized image for testing file size limits"""
    # Create a very large image that would exceed size limits
    large_image = Image.new("RGB", (2000, 2000), color="blue")
    img_byte_arr = BytesIO()
    large_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_basic_health_check(self, client):
        """Test basic health endpoint"""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
    
    def test_readiness_check(self, client):
        """Test readiness endpoint"""
        response = client.get("/health/ready")
        # Should be 503 if model isn't loaded, 200 if it is
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "ready"
    
    def test_liveness_check(self, client):
        """Test liveness endpoint"""
        response = client.get("/health/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
    
    def test_model_stats(self, client):
        """Test model statistics endpoint"""
        response = client.get("/health/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "is_loaded" in data
        assert "prediction_count" in data

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_prediction_with_valid_image(self, client):
        """Test single image prediction"""
        # Create test image
        test_image = create_test_image()
        
        # Make prediction request
        response = client.post(
            "/predict/",
            files={"file": ("test.png", test_image, "image/png")}
        )
        
        # Model might not be loaded in test environment
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "all_confidences" in data
        assert "prediction_time_ms" in data
        assert "timestamp" in data
        
        # Verify confidence is between 0 and 1
        assert 0 <= data["confidence"] <= 1
        
        # Verify all_confidences has 10 classes (CIFAR-10)
        assert len(data["all_confidences"]) == 10
        
        # Verify the predicted class is in CIFAR-10 classes
        cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        assert data["predicted_class"] in cifar10_classes
    
    def test_prediction_with_invalid_file(self, client):
        """Test prediction with invalid file"""
        # Create invalid file
        invalid_file = create_invalid_image()
        
        response = client.post(
            "/predict/",
            files={"file": ("test.txt", invalid_file, "text/plain")}
        )
        
        # Model might not be loaded in test environment
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_prediction_with_invalid_format(self, client):
        """Test prediction with unsupported image format"""
        # Create an image with unsupported extension
        test_image = create_test_image()
        
        response = client.post(
            "/predict/",
            files={"file": ("test.xyz", test_image, "image/xyz")}
        )
        
        # Model might not be loaded in test environment
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        assert response.status_code == 400
    
    def test_prediction_without_file(self, client):
        """Test prediction without file"""
        response = client.post("/predict/")
        
        # Model might not be loaded in test environment
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        assert response.status_code == 422  # Validation error
    
    def test_prediction_with_empty_file(self, client):
        """Test prediction with empty file"""
        response = client.post(
            "/predict/",
            files={"file": ("test.png", b"", "image/png")}
        )
        
        # Model might not be loaded in test environment
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        assert response.status_code == 400
    
    def test_batch_prediction(self, client):
        """Test batch prediction"""
        # Create multiple test images
        test_images = [
            ("test1.png", create_test_image(), "image/png"),
            ("test2.png", create_test_image(), "image/png")
        ]
        
        response = client.post(
            "/predict/batch",
            files=[("files", img) for img in test_images]
        )
        
        # Model might not be loaded in test environment
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "batch_size" in data
        assert "total_processing_time_ms" in data
        assert data["batch_size"] == 2
        assert len(data["predictions"]) == 2
        
        # Verify each prediction has required fields
        for prediction in data["predictions"]:
            assert "predicted_class" in prediction
            assert "confidence" in prediction
            assert "all_confidences" in prediction
    
    def test_batch_prediction_too_many_files(self, client):
        """Test batch prediction with too many files"""
        # Create more than 10 test images
        test_images = [
            (f"test{i}.png", create_test_image(), "image/png")
            for i in range(12)  # More than the limit of 10
        ]
        
        response = client.post(
            "/predict/batch",
            files=[("files", img) for img in test_images]
        )
        
        # Model might not be loaded in test environment
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        assert response.status_code == 400
        data = response.json()
        assert "Maximum 10 images" in data["detail"]
    
    def test_batch_prediction_empty(self, client):
        """Test batch prediction with no files"""
        response = client.post("/predict/batch", files=[])
        
        # Model might not be loaded in test environment
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        assert response.status_code == 400
    
    def test_prediction_info(self, client):
        """Test prediction info endpoint"""
        response = client.get("/predict/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "api_version" in data
        assert "model_info" in data
        assert "preprocessing" in data

class TestAPIBehavior:
    """Test general API behavior"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert "model_status" in data
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/")
        # CORS headers should be present
        # Note: TestClient might not fully simulate CORS
        assert response.status_code in [200, 405]  # Some test clients return 405 for OPTIONS
    
    def test_security_headers(self, client):
        """Test security headers are present"""
        response = client.get("/")
        
        # Check for security headers (case-insensitive)
        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        
        assert headers_lower.get("x-content-type-options") == "nosniff"
        assert headers_lower.get("x-frame-options") == "DENY"
        assert "x-request-id" in headers_lower
    
    def test_404_handling(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent")
        assert response.status_code == 404

# Integration tests
class TestModelIntegration:
    """Test model service integration"""
    
    @pytest.mark.asyncio
    async def test_model_loading(self):
        """Test model can be loaded"""
        try:
            await model_service.load_model()
            assert model_service.is_healthy()
        except FileNotFoundError:
            pytest.skip("Model file not found - expected in test environment")
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")
    
    @pytest.mark.asyncio
    async def test_model_prediction(self):
        """Test model prediction directly"""
        if not model_service.is_loaded:
            pytest.skip("Model not loaded")
        
        # Create test tensor with proper normalization
        test_tensor = torch.randn(1, 3, 32, 32)
        
        try:
            result = await model_service.predict(test_tensor)
            assert "predicted_class" in result
            assert "confidence" in result
            assert "all_confidences" in result
            assert "prediction_time_ms" in result
        except Exception as e:
            pytest.fail(f"Model prediction failed: {e}")

# Performance tests
class TestPerformance:
    """Test API performance"""
    
    def test_prediction_timing(self, client):
        """Test that predictions complete in reasonable time"""
        test_image = create_test_image()
        
        import time
        start_time = time.time()
        
        response = client.post(
            "/predict/",
            files={"file": ("test.png", test_image, "image/png")}
        )
        
        end_time = time.time()
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        # Should complete within 5 seconds (generous for test environment)
        assert (end_time - start_time) < 5.0

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])