import logging
import uuid
import time
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse

from ..models.prediction import PredictionResult, BatchPredictionResult, ErrorResponse, PredictionExample
from ..services.model_service import model_service
from ..services.image_service import image_service
from ..config import settings, API_MESSAGES

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)

async def check_model_ready():
    """Dependency to ensure model is loaded before prediction requests"""
    if not model_service.is_healthy():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again in a few moments."
        )

@router.post(
    "/",
    response_model=PredictionResult,
    summary="Classify an uploaded image",
    description="Upload an image file and get a CIFAR-10 classification prediction",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": PredictionExample.successful_prediction
                }
            }
        },
        400: {
            "description": "Bad request - invalid image or format",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": PredictionExample.error_response
                }
            }
        },
        413: {"description": "File too large"},
        503: {"description": "Model not ready"},
    }
)
async def predict_image(
    file: UploadFile = File(
        ...,
        description="Image file to classify (JPEG, PNG, BMP, TIFF)",
        media_type="image/*"
    ),
    request_id: Optional[str] = Form(
        None,
        description="Optional request ID for tracking"
    ),
    model_ready: None = Depends(check_model_ready)
):
    """
    Classify an uploaded image using the trained PyramidNet model.
    
    This endpoint accepts image files in common formats and returns:
    - The predicted CIFAR-10 class
    - Confidence score for the prediction
    - Confidence scores for all classes
    - Processing time and metadata
    
    **Supported formats:** JPEG, PNG, BMP, TIFF
    **Maximum file size:** 10MB
    **Expected accuracy:** ~94% on CIFAR-10 test set
    """
    # Generate request ID if not provided
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info(f"Prediction request {request_id} started for file: {file.filename}")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file provided"
            )
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        logger.debug(f"Request {request_id}: File size: {len(file_content)} bytes")
        
        # Process image through the pipeline
        try:
            processed_tensor = await image_service.process_uploaded_file(
                file_content, 
                file.filename
            )
        except ValueError as e:
            logger.warning(f"Request {request_id}: Image validation failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # Make prediction
        try:
            prediction_result = await model_service.predict(processed_tensor)
        except Exception as e:
            logger.error(f"Request {request_id}: Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Please try again."
            )
        
        logger.info(
            f"Request {request_id}: Successful prediction - {prediction_result['predicted_class']} "
            f"({prediction_result['confidence']:.3f})"
        )
        
        # Add metadata to response
        prediction_result['timestamp'] = datetime.now(timezone.utc)
        
        return PredictionResult(**prediction_result)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction"
        )

@router.post(
    "/batch",
    response_model=BatchPredictionResult,
    summary="Classify multiple images",
    description="Upload multiple images and get predictions for all of them",
    responses={
        400: {"description": "Invalid request or image format", "model": ErrorResponse},
        413: {"description": "Request too large"},
        503: {"description": "Model not ready"},
    }
)
async def batch_predict(
    files: List[UploadFile] = File(
        ...,
        description="List of image files to classify (max 10 files)",
        media_type="image/*"
    ),
    request_id: Optional[str] = Form(
        None,
        description="Optional request ID for tracking"
    ),
    model_ready: None = Depends(check_model_ready)
):
    """
    Classify multiple uploaded images in a single request.
    
    This endpoint is useful for batch processing scenarios where you need
    to classify several images at once. It's more efficient than making
    multiple single-image requests.
    
    **Limitations:**
    - Maximum 10 images per request
    - Same file size limits apply to each image
    - Total request size cannot exceed server limits
    """
    # Generate request ID if not provided
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info(f"Batch prediction request {request_id} started with {len(files)} files")
    
    # Validate batch size
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch request"
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one image file must be provided"
        )
    
    start_time = time.time()
    processed_tensors = []
    
    try:
        # Process each image
        for i, file in enumerate(files):
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {i+1} has no filename"
                )
            
            file_content = await file.read()
            
            # Process image through pipeline
            try:
                tensor = await image_service.process_uploaded_file(
                    file_content, 
                    file.filename
                )
                processed_tensors.append(tensor)
            except ValueError as e:
                logger.warning(f"Request {request_id}: File {i+1} validation failed: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"File {i+1} ({file.filename}): {str(e)}"
                )
        
        # Make batch prediction
        try:
            predictions = await model_service.batch_predict(processed_tensors)
        except Exception as e:
            logger.error(f"Request {request_id}: Batch prediction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Batch prediction failed. Please try again."
            )
        
        # Add metadata to each prediction
        prediction_results = []
        for pred in predictions:
            pred['timestamp'] = datetime.now(timezone.utc)
            prediction_results.append(PredictionResult(**pred))
        
        total_time = time.time() - start_time
        
        result = BatchPredictionResult(
            predictions=prediction_results,
            total_processing_time_ms=round(total_time * 1000, 2),
            batch_size=len(files)
        )
        
        logger.info(
            f"Request {request_id}: Batch prediction completed - {len(files)} images "
            f"processed in {total_time*1000:.2f}ms"
        )
        
        return result
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during batch prediction"
        )

@router.get(
    "/info",
    summary="Get prediction endpoint information",
    description="Get information about supported formats and preprocessing"
)
async def prediction_info():
    """
    Get information about the prediction endpoints.
    
    Returns details about supported formats, preprocessing pipeline,
    and usage limits.
    """
    try:
        preprocessing_info = image_service.get_preprocessing_info()
        
        return {
            "api_version": settings.app_version,
            "model_info": {
                "architecture": "PyramidNet",
                "dataset": "CIFAR-10",
                "num_classes": 10,
                "input_size": "32x32",
                "expected_accuracy": "~94%"
            },
            "preprocessing": preprocessing_info,
            "batch_limits": {
                "max_files_per_batch": 10,
                "max_concurrent_requests": settings.max_concurrent_requests
            },
            "performance": {
                "typical_prediction_time_ms": "20-50ms",
                "batch_processing_advantage": "2-5x faster than individual requests"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get prediction info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve prediction information"
        )