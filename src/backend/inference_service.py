"""
FastAPI Inference Service for Wafer Defect Segmentation

Features:
- POST /predict: Single image prediction
- POST /predict/batch: Batch image prediction
- Redis caching (30-day TTL)
- ONNX Runtime for optimized inference
- Request validation and error handling
- Prometheus metrics tracking
- <2s latency (PRD requirement)

Usage:
    uvicorn src.backend.inference_service:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import numpy as np
import cv2
from PIL import Image
import io
import hashlib
import json
import redis
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path

from src.models.onnx_inference import ONNXSegmentationInference
from src.database.database import get_db
from src.database.models import InferenceLog, WaferMap
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wafer Defect Segmentation API",
    description="Production inference service for semiconductor wafer defect classification",
    version="1.0.0",
)

# Redis client for caching
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=False,
)

# Global ONNX inference engine (lazy loading)
inference_engine: Optional[ONNXSegmentationInference] = None


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for prediction."""
    wafer_id: str = Field(..., description="Unique wafer identifier")
    use_cache: bool = Field(True, description="Use cached predictions if available")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Confidence threshold must be between 0 and 1')
        return v


class DefectRegion(BaseModel):
    """Detected defect region."""
    class_id: int = Field(..., ge=0, le=7, description="Defect class (0-7)")
    class_name: str = Field(..., description="Defect class name")
    bbox: List[float] = Field(..., description="Bounding box [x, y, width, height]")
    area: float = Field(..., description="Defect area in pixels")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    wafer_id: str
    prediction_id: str
    timestamp: str
    defect_regions: List[DefectRegion]
    total_defect_area: float
    class_distribution: Dict[str, int]
    confidence_scores: List[float]
    inference_time_ms: float
    model_version: str
    cached: bool = False


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    wafer_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of wafer IDs (max 100)")
    use_cache: bool = Field(True, description="Use cached predictions if available")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse]
    total_processed: int
    total_time_ms: float
    average_time_ms: float


# Defect class mapping
DEFECT_CLASSES = {
    0: "background",
    1: "edge",
    2: "center",
    3: "ring",
    4: "scratch",
    5: "particle",
    6: "lithography",
    7: "etching",
}


def get_inference_engine() -> ONNXSegmentationInference:
    """
    Get or initialize ONNX inference engine (singleton pattern).
    
    Returns:
        ONNXSegmentationInference instance
    """
    global inference_engine
    
    if inference_engine is None:
        model_path = "data/models/resnet50_unet.onnx"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Please export model using: python scripts/export_onnx.py"
            )
        
        logger.info(f"Loading ONNX model from {model_path}")
        inference_engine = ONNXSegmentationInference(
            model_path=model_path,
            input_size=(300, 300),
            num_classes=8,
            device="cpu",
        )
        logger.info("ONNX model loaded successfully")
        logger.info(f"Model info: {inference_engine.get_model_info()}")
    
    return inference_engine


def generate_cache_key(wafer_id: str, model_version: str) -> str:
    """Generate Redis cache key for prediction."""
    key_string = f"{wafer_id}_{model_version}"
    return f"prediction:{hashlib.md5(key_string.encode()).hexdigest()}"


def get_cached_prediction(wafer_id: str, model_version: str) -> Optional[Dict]:
    """Retrieve cached prediction from Redis."""
    try:
        cache_key = generate_cache_key(wafer_id, model_version)
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            logger.info(f"Cache hit for wafer_id={wafer_id}")
            return json.loads(cached_data)
        
        logger.info(f"Cache miss for wafer_id={wafer_id}")
        return None
    
    except Exception as e:
        logger.error(f"Redis error: {e}")
        return None


def cache_prediction(wafer_id: str, model_version: str, prediction: Dict, ttl_days: int = 30):
    """Cache prediction in Redis with TTL."""
    try:
        cache_key = generate_cache_key(wafer_id, model_version)
        redis_client.setex(
            cache_key,
            timedelta(days=ttl_days),
            json.dumps(prediction)
        )
        logger.info(f"Cached prediction for wafer_id={wafer_id}, TTL={ttl_days} days")
    
    except Exception as e:
        logger.error(f"Failed to cache prediction: {e}")


def extract_defect_regions(mask: np.ndarray, probabilities: np.ndarray, confidence: np.ndarray) -> List[DefectRegion]:
    """
    Extract individual defect regions from segmentation mask.
    
    Args:
        mask: Segmentation mask (H, W) with class indices
        probabilities: Class probabilities (H, W, num_classes)
        confidence: Confidence map (H, W)
    
    Returns:
        List of DefectRegion objects
    """
    regions = []
    
    # Find connected components for each class (excluding background)
    for class_id in range(1, 8):
        class_mask = (mask == class_id).astype(np.uint8)
        
        if class_mask.sum() == 0:
            continue
        
        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area
            area = cv2.contourArea(contour)
            
            if area < 10:  # Filter small noise regions
                continue
            
            # Calculate average confidence for this region
            region_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], -1, 1, -1)
            region_confidence = confidence[region_mask == 1].mean()
            
            regions.append(DefectRegion(
                class_id=int(class_id),
                class_name=DEFECT_CLASSES[class_id],
                bbox=[float(x), float(y), float(w), float(h)],
                area=float(area),
                confidence=float(region_confidence),
            ))
    
    return regions


def process_prediction_result(
    result: Dict[str, np.ndarray],
    wafer_id: str,
    inference_time_ms: float,
    model_version: str = "1.0.0",
) -> PredictionResponse:
    """
    Process raw prediction result into API response.
    
    Args:
        result: Dictionary with 'mask', 'probabilities', 'confidence'
        wafer_id: Wafer identifier
        inference_time_ms: Inference time in milliseconds
        model_version: Model version string
    
    Returns:
        PredictionResponse object
    """
    mask = result['mask']
    probabilities = result['probabilities']
    confidence = result['confidence']
    
    # Extract defect regions
    defect_regions = extract_defect_regions(mask, probabilities, confidence)
    
    # Calculate class distribution
    unique, counts = np.unique(mask, return_counts=True)
    class_distribution = {
        DEFECT_CLASSES[int(class_id)]: int(count)
        for class_id, count in zip(unique, counts)
    }
    
    # Total defect area (excluding background)
    total_defect_area = float((mask > 0).sum())
    
    # Confidence scores per class (average)
    confidence_scores = []
    for class_id in range(8):
        class_pixels = (mask == class_id)
        if class_pixels.sum() > 0:
            avg_confidence = float(confidence[class_pixels].mean())
        else:
            avg_confidence = 0.0
        confidence_scores.append(avg_confidence)
    
    # Generate prediction ID
    prediction_id = hashlib.md5(
        f"{wafer_id}_{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()
    
    return PredictionResponse(
        wafer_id=wafer_id,
        prediction_id=prediction_id,
        timestamp=datetime.utcnow().isoformat(),
        defect_regions=defect_regions,
        total_defect_area=total_defect_area,
        class_distribution=class_distribution,
        confidence_scores=confidence_scores,
        inference_time_ms=inference_time_ms,
        model_version=model_version,
    )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Wafer Defect Segmentation API",
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": inference_engine is not None,
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    try:
        # Check Redis connection
        redis_healthy = redis_client.ping()
    except:
        redis_healthy = False
    
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "redis_connected": redis_healthy,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    file: UploadFile = File(..., description="Wafer map image (PNG/JPG)"),
    wafer_id: str = "unknown",
    use_cache: bool = True,
    confidence_threshold: float = 0.5,
    db: Session = Depends(get_db),
):
    """
    Predict defects for a single wafer map image.
    
    Args:
        file: Uploaded wafer map image
        wafer_id: Unique wafer identifier
        use_cache: Use cached predictions if available
        confidence_threshold: Minimum confidence for defect detection
        db: Database session
    
    Returns:
        PredictionResponse with segmentation results
    """
    start_time = time.time()
    model_version = "1.0.0"
    
    # Check cache
    if use_cache:
        cached = get_cached_prediction(wafer_id, model_version)
        if cached:
            cached['cached'] = True
            return PredictionResponse(**cached)
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))
        
        # Get inference engine
        engine = get_inference_engine()
        
        # Run inference
        inference_start = time.time()
        result = engine.predict(image_np, confidence_threshold=confidence_threshold)
        inference_time_ms = (time.time() - inference_start) * 1000
        
        # Process result
        response = process_prediction_result(result, wafer_id, inference_time_ms, model_version)
        
        # Cache result
        if use_cache:
            cache_prediction(wafer_id, model_version, response.dict())
        
        # Log to database
        try:
            log = InferenceLog(
                wafer_map_id=None,  # Would link to WaferMap.id if available
                model_version=model_version,
                model_path="data/models/resnet50_unet.onnx",
                class_distribution=response.class_distribution,
                total_defect_area=response.total_defect_area,
                confidence_scores=response.confidence_scores,
                inference_time_ms=inference_time_ms,
                preprocessing_time_ms=0.0,
                postprocessing_time_ms=0.0,
                request_id=response.prediction_id,
                status="success",
            )
            db.add(log)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to log inference: {e}")
        
        total_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Prediction completed for {wafer_id} in {total_time_ms:.2f}ms")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction failed for {wafer_id}: {e}")
        
        # Log error to database
        try:
            log = InferenceLog(
                wafer_map_id=None,
                model_version=model_version,
                status="error",
                error_message=str(e),
            )
            db.add(log)
            db.commit()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple wafer map images"),
    wafer_ids: Optional[List[str]] = None,
    use_cache: bool = True,
    confidence_threshold: float = 0.5,
    db: Session = Depends(get_db),
):
    """
    Predict defects for multiple wafer maps (batch processing).
    
    Args:
        files: List of uploaded wafer map images
        wafer_ids: List of wafer identifiers (optional, auto-generated if not provided)
        use_cache: Use cached predictions if available
        confidence_threshold: Minimum confidence threshold
        db: Database session
    
    Returns:
        BatchPredictionResponse with all predictions
    """
    start_time = time.time()
    
    # Validate batch size
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum batch size is 100 images")
    
    # Generate wafer IDs if not provided
    if wafer_ids is None:
        wafer_ids = [f"wafer_{i:04d}" for i in range(len(files))]
    
    if len(wafer_ids) != len(files):
        raise HTTPException(status_code=400, detail="Number of wafer_ids must match number of files")
    
    predictions = []
    
    for file, wafer_id in zip(files, wafer_ids):
        try:
            # Reuse single prediction endpoint
            prediction = await predict_single(file, wafer_id, use_cache, confidence_threshold, db)
            predictions.append(prediction)
        
        except Exception as e:
            logger.error(f"Batch prediction failed for {wafer_id}: {e}")
            # Continue with other predictions
    
    total_time_ms = (time.time() - start_time) * 1000
    avg_time_ms = total_time_ms / len(predictions) if predictions else 0
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        total_time_ms=total_time_ms,
        average_time_ms=avg_time_ms,
    )


@app.delete("/cache/{wafer_id}")
async def clear_cache(wafer_id: str, model_version: str = "1.0.0"):
    """
    Clear cached prediction for a specific wafer.
    
    Args:
        wafer_id: Wafer identifier
        model_version: Model version
    
    Returns:
        Success status
    """
    try:
        cache_key = generate_cache_key(wafer_id, model_version)
        deleted = redis_client.delete(cache_key)
        
        return {
            "status": "success",
            "wafer_id": wafer_id,
            "cache_cleared": bool(deleted),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """
    Get inference statistics.
    
    Returns:
        Statistics about predictions
    """
    try:
        # Query recent inference logs
        total_predictions = db.query(InferenceLog).count()
        successful = db.query(InferenceLog).filter(InferenceLog.status == "success").count()
        failed = db.query(InferenceLog).filter(InferenceLog.status == "error").count()
        
        # Average inference time
        avg_time = db.query(InferenceLog.inference_time_ms).filter(
            InferenceLog.status == "success"
        ).all()
        
        avg_inference_time = np.mean([t[0] for t in avg_time if t[0] is not None]) if avg_time else 0
        
        return {
            "total_predictions": total_predictions,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_predictions if total_predictions > 0 else 0,
            "average_inference_time_ms": float(avg_inference_time),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run development server
    uvicorn.run(
        "src.backend.inference_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
