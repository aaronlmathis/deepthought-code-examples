import time
from fastapi import APIRouter, HTTPException
from ..models.prediction import HealthCheckResponse, ModelStatsResponse
from ..services.model_service import model_service
from ..config import settings

router = APIRouter(prefix="/health", tags=["Health"])

# Track startup time for uptime calculation
startup_time = time.time()

@router.get(
    "/",
    response_model=HealthCheckResponse,
    summary="Basic health check",
    description="Check if the API is running and responsive"
)
async def health_check():
    """
    Basic health check endpoint.
    
    This endpoint performs a quick check to verify the API is running.
    It's designed to be fast and lightweight for load balancer health checks.
    """
    try:
        uptime = time.time() - startup_time
        
        return HealthCheckResponse(
            status="healthy",
            version=settings.app_version,
            model_loaded=model_service.is_healthy(),
            uptime_seconds=uptime
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@router.get(
    "/ready",
    response_model=HealthCheckResponse,
    summary="Readiness check",
    description="Check if the API is ready to serve prediction requests"
)
async def readiness_check():
    """
    Readiness check endpoint.
    
    This endpoint performs a deeper check to verify the API is ready to serve
    prediction requests. It checks if the model is loaded and functional.
    
    Kubernetes uses readiness checks to determine if a pod should receive traffic.
    """
    try:
        if not model_service.is_healthy():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded or unhealthy"
            )
        
        uptime = time.time() - startup_time
        
        return HealthCheckResponse(
            status="ready",
            version=settings.app_version,
            model_loaded=True,
            uptime_seconds=uptime
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Readiness check failed: {str(e)}"
        )

@router.get(
    "/live",
    response_model=HealthCheckResponse,
    summary="Liveness check",
    description="Check if the API process is alive"
)
async def liveness_check():
    """
    Liveness check endpoint.
    
    This endpoint performs a minimal check to verify the API process is alive.
    It should only fail if the process is completely broken.
    
    Kubernetes uses liveness checks to determine if a pod should be restarted.
    """
    try:
        uptime = time.time() - startup_time
        
        return HealthCheckResponse(
            status="alive",
            version=settings.app_version,
            model_loaded=model_service.is_healthy(),
            uptime_seconds=uptime
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Liveness check failed: {str(e)}"
        )

@router.get(
    "/stats",
    response_model=ModelStatsResponse,
    summary="Model statistics",
    description="Get detailed statistics about the model service"
)
async def model_stats():
    """
    Model statistics endpoint.
    
    Provides detailed information about model performance and usage.
    Useful for monitoring and debugging.
    """
    try:
        stats = model_service.get_stats()
        
        return ModelStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model statistics: {str(e)}"
        )