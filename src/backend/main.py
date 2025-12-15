"""
FastAPI Main Application - ResNet Wafer Map Classifier

Enterprise-grade wafer defect classification platform with:
- Inference API (ONNX optimized, <2s latency)
- Training orchestration (Kubernetes jobs)
- Active learning workflow (85% annotation reduction)
- Annotation management (COCO export)
- Authentication & authorization (JWT + RBAC)

Performance targets:
- Mean IoU: >95%
- Latency: <2s per wafer
- Throughput: 10K wafers/day
- Uptime: 99.9%
- ROI: $2M+ annual savings
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, Any

from src.backend import auth, inference_service, annotation_service
from src.backend.training_service import router as training_router
from src.backend.active_learning_service import router as active_learning_router
from src.database.database import engine, Base
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Starting ResNet Wafer Map Classifier API...")
    
    # Initialize database
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    # Load inference model (lazy loading in inference service)
    logger.info("Inference engine will be loaded on first request")
    
    # Check external dependencies
    try:
        from redis import Redis
        redis_client = Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.warning(f"Redis not available: {e}. Caching disabled.")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ResNet Wafer Map Classifier",
    description="Enterprise ML platform for semiconductor wafer defect classification",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and response times."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {duration:.3f}s with status {response.status_code}"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(duration)
    
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": request.url.path,
        },
    )


# Include routers
app.include_router(auth.router)
app.include_router(inference_service.router)
app.include_router(annotation_service.router)
app.include_router(training_router)
app.include_router(active_learning_router)


# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """API root with basic information."""
    return {
        "name": "ResNet Wafer Map Classifier",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "metrics": "/metrics",
        },
    }


# Health check
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check.
    
    Checks:
    - API service
    - Database connection
    - Redis cache
    - Kubernetes availability
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {},
    }
    
    # Check database
    try:
        from src.database.database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        from redis import Redis
        redis_client = Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unavailable: {str(e)}"
        # Redis is optional, don't mark as degraded
    
    # Check Kubernetes
    try:
        from kubernetes import client, config
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        v1.list_node()
        health_status["checks"]["kubernetes"] = "healthy"
    except Exception as e:
        health_status["checks"]["kubernetes"] = f"unavailable: {str(e)}"
        # K8s may not be available in dev
    
    return health_status


# Metrics endpoint
@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get system metrics.
    
    Returns:
    - Request counts
    - Error rates
    - Response times
    - Resource usage
    """
    from src.database.database import SessionLocal
    from src.database.models import InferenceLog, TrainingJob, Annotation
    
    db = SessionLocal()
    
    try:
        # Inference metrics
        total_inferences = db.query(InferenceLog).count()
        avg_inference_time = db.query(func.avg(InferenceLog.inference_time_ms)).scalar() or 0
        
        # Training metrics
        total_training_jobs = db.query(TrainingJob).count()
        completed_jobs = db.query(TrainingJob).filter(TrainingJob.status == "completed").count()
        
        # Annotation metrics
        total_annotations = db.query(Annotation).count()
        verified_annotations = db.query(Annotation).filter(Annotation.is_verified == True).count()
        
        metrics = {
            "inference": {
                "total_predictions": total_inferences,
                "average_latency_ms": float(avg_inference_time),
                "target_latency_ms": 2000,
                "latency_compliance": avg_inference_time < 2000,
            },
            "training": {
                "total_jobs": total_training_jobs,
                "completed_jobs": completed_jobs,
                "success_rate": completed_jobs / total_training_jobs if total_training_jobs > 0 else 0,
            },
            "annotations": {
                "total_annotations": total_annotations,
                "verified_annotations": verified_annotations,
                "verification_rate": verified_annotations / total_annotations if total_annotations > 0 else 0,
            },
        }
        
        return metrics
    
    finally:
        db.close()


# API documentation
@app.get("/api/info")
async def api_info() -> Dict[str, Any]:
    """
    Get API information and capabilities.
    """
    return {
        "title": "ResNet Wafer Map Classifier API",
        "version": "1.0.0",
        "description": "Enterprise ML platform for semiconductor wafer defect classification",
        "features": {
            "inference": {
                "description": "Real-time wafer defect prediction",
                "endpoints": [
                    "POST /predict - Single wafer prediction",
                    "POST /predict/batch - Batch prediction",
                    "GET /stats - Inference statistics",
                ],
                "performance": {
                    "latency_target": "<2s per wafer",
                    "throughput_target": "10K wafers/day",
                    "optimization": "ONNX Runtime + Redis caching",
                },
            },
            "training": {
                "description": "Distributed model training orchestration",
                "endpoints": [
                    "POST /training/jobs - Create training job",
                    "GET /training/jobs - List jobs",
                    "GET /training/metrics/{id} - Get training metrics",
                ],
                "capabilities": {
                    "modes": ["supervised", "active_learning", "semi_supervised", "fine_tuning"],
                    "infrastructure": "Kubernetes with GPU support",
                    "tracking": "MLflow experiment tracking",
                },
            },
            "active_learning": {
                "description": "Intelligent sample selection for annotation",
                "endpoints": [
                    "POST /active-learning/query - Run query strategy",
                    "GET /active-learning/queue - Get annotation queue",
                    "GET /active-learning/stats - Get AL statistics",
                ],
                "strategies": ["uncertainty", "diversity", "hybrid", "random"],
                "target": "85% annotation reduction",
            },
            "annotations": {
                "description": "Annotation management and quality control",
                "endpoints": [
                    "POST /annotations - Create annotation",
                    "GET /annotations/wafer/{id} - Get wafer annotations",
                    "POST /annotations/export - Export to COCO JSON",
                ],
                "formats": ["COCO JSON", "Polygon segmentation"],
                "validation": "Polygon quality checks + inter-annotator agreement",
            },
            "authentication": {
                "description": "JWT-based authentication and RBAC",
                "endpoints": [
                    "POST /auth/register - Register user",
                    "POST /auth/login - Login",
                    "GET /auth/me - Get current user",
                ],
                "security": {
                    "method": "JWT with access + refresh tokens",
                    "password_hashing": "bcrypt (cost factor 12)",
                    "roles": ["admin", "engineer", "annotator", "viewer"],
                },
            },
        },
        "performance_targets": {
            "mean_iou": ">95%",
            "latency": "<2s per wafer",
            "annotation_reduction": "85%",
            "throughput": "10K wafers/day",
            "uptime": "99.9%",
        },
        "roi": {
            "annual_benefit": "$3.33M",
            "implementation_cost": "$250K",
            "roi_ratio": "13.3:1",
            "payback_period": "0.9 months",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
