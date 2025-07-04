import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import our components
from .config import settings, API_MESSAGES
from .routers import health, predict
from .services.model_service import model_service
from .middleware.logging import RequestLoggingMiddleware
from .middleware.security import SecurityHeadersMiddleware
from .middleware.rate_limiting import SimpleRateLimitMiddleware

class PerformanceMonitor:
    """Monitor API performance metrics"""
    
    def __init__(self):
        self.request_times = defaultdict(list)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.start_time = time.time()
    
    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record a request for metrics tracking"""
        self.request_times[endpoint].append(duration)
        self.request_counts[endpoint] += 1
        
        if status_code >= 400:
            self.error_counts[endpoint] += 1
    
    def get_stats(self):
        """Get performance statistics"""
        stats = {}
        for endpoint in self.request_times:
            times = self.request_times[endpoint]
            if times:  # Avoid division by zero
                stats[endpoint] = {
                    "count": self.request_counts[endpoint],
                    "avg_time_ms": sum(times) / len(times),
                    "max_time_ms": max(times),
                    "min_time_ms": min(times),
                    "error_rate": self.error_counts[endpoint] / self.request_counts[endpoint] if self.request_counts[endpoint] > 0 else 0.0
                }
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "endpoint_stats": stats,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values())
        }

# Configure logging
def setup_logging():
    """Configure comprehensive logging for production"""
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler()  # Console output
        ]
    )
    
    # Specific logger for our API
    api_logger = logging.getLogger("api")
    api_logger.setLevel(getattr(logging, settings.log_level))
    
    return api_logger

# Setup logging
logger = setup_logging()

# Initialize performance monitor
monitor = PerformanceMonitor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    
    Handles startup and shutdown tasks:
    - Model loading during startup
    - Resource cleanup during shutdown
    """
    # Startup
    logger.info("Starting PyramidNet Image Classifier API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Model device: {settings.model_device}")
    logger.info(f"Docs enabled: {not settings.is_production()}")
    
    try:
        # Load the model
        await model_service.load_model()
        logger.info("Model loaded successfully - API ready to serve requests")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # In production, you might want to fail fast here
        if settings.is_production():
            raise RuntimeError("Model loading failed in production environment")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down PyramidNet Image Classifier API")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs" if not settings.is_production() else None,  # Disable docs in production
    redoc_url="/redoc" if not settings.is_production() else None,
    lifespan=lifespan
)

# Add middleware (order matters!)
# Security headers should be last to ensure they're applied to all responses
app.add_middleware(SecurityHeadersMiddleware)

# Request logging
app.add_middleware(RequestLoggingMiddleware)

# Rate limiting
if settings.environment != "development":
    app.add_middleware(SimpleRateLimitMiddleware, requests_per_minute=60)

# CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Trusted hosts (security)
if settings.is_production():
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.yourdomain.com", "*.yourdomain.com"]  # Configure for your domain
    )

# Include routers
app.include_router(health.router)
app.include_router(predict.router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    
    Ensures that all errors return consistent JSON responses
    and prevents sensitive information leakage.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    if settings.debug:
        # In development, show detailed error information
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(exc),
                "detail": "Check logs for more information",
                "type": exc.__class__.__name__
            }
        )
    else:
        # In production, hide sensitive error details
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "detail": "Please try again or contact support if the problem persists"
            }
        )

# Root endpoint
@app.get(
    "/",
    summary="API Information",
    description="Get basic information about the PyramidNet Image Classifier API"
)
async def root():
    """
    Root endpoint providing API information.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "environment": settings.environment,
        "model_status": "loaded" if model_service.is_healthy() else "not_loaded",
        "docs_url": "/docs" if not settings.is_production() else "disabled",
        "endpoints": {
            "health": "/health",
            "prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "metrics": "/metrics"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics"""
    try:
        return {
            "model_stats": model_service.get_stats(),
            "performance": monitor.get_stats(),
            "system_info": {
                "device": str(model_service.device) if model_service.device else "unknown",
                "model_loaded": model_service.is_healthy(),
                "environment": settings.environment
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve metrics", "detail": str(e)}
        )

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload and settings.is_development(),
        log_level=settings.log_level.lower(),
        access_log=settings.access_log
    )