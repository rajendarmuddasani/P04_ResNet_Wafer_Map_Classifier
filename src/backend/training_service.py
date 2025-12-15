"""
Training Service - ML Training Job Orchestration

Features:
- Create and manage training jobs
- Kubernetes Job creation for distributed training
- MLflow experiment tracking integration
- Hyperparameter validation
- Job status monitoring
- Model registry management

Supports:
- Supervised training
- Active learning iterations
- Semi-supervised learning (FixMatch)
- Fine-tuning

Usage:
    POST /training/jobs - Create new training job
    GET /training/jobs/{job_id} - Get job status
    GET /training/jobs - List all jobs
    DELETE /training/jobs/{job_id} - Cancel job
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid
from sqlalchemy.orm import Session
import yaml
import mlflow
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging

from src.database.database import get_db
from src.database.models import TrainingJob, User

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/training", tags=["training"])


# Enums for validation
class TrainingMode(str, Enum):
    SUPERVISED = "supervised"
    ACTIVE_LEARNING = "active_learning"
    SEMI_SUPERVISED = "semi_supervised"
    FINE_TUNING = "fine_tuning"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelArchitecture(str, Enum):
    RESNET50_UNET = "resnet50_unet"
    RESNET101_UNET = "resnet101_unet"
    RESNET18_UNET = "resnet18_unet"


# Request/Response models
class TrainingJobCreate(BaseModel):
    """Request model for creating training job."""
    job_name: str = Field(..., min_length=3, max_length=255, description="Job name")
    model_architecture: ModelArchitecture = Field(ModelArchitecture.RESNET50_UNET, description="Model architecture")
    training_mode: TrainingMode = Field(TrainingMode.SUPERVISED, description="Training mode")
    
    # Training hyperparameters
    num_epochs: int = Field(50, ge=1, le=500, description="Number of epochs")
    batch_size: int = Field(16, ge=1, le=128, description="Batch size")
    learning_rate: float = Field(1e-4, ge=1e-6, le=1e-1, description="Learning rate")
    
    # Loss function
    loss_function: str = Field("combined", description="Loss function type")
    dice_weight: float = Field(0.5, ge=0.0, le=1.0)
    focal_weight: float = Field(0.5, ge=0.0, le=1.0)
    
    # Optimizer
    optimizer: str = Field("adamw", description="Optimizer type (adam, adamw, sgd)")
    scheduler: str = Field("reduce_on_plateau", description="LR scheduler")
    
    # Dataset configuration
    train_annotation_file: str = Field(..., description="Path to training annotations")
    val_annotation_file: str = Field(..., description="Path to validation annotations")
    test_annotation_file: Optional[str] = Field(None, description="Path to test annotations")
    
    # Active learning specific
    active_learning_budget: Optional[int] = Field(50, ge=1, le=1000, description="Samples per AL iteration")
    active_learning_iterations: Optional[int] = Field(15, ge=1, le=50, description="Number of AL iterations")
    query_strategy: Optional[str] = Field("hybrid", description="AL query strategy")
    
    # Semi-supervised specific
    num_unlabeled_samples: Optional[int] = Field(None, description="Number of unlabeled samples")
    fixmatch_threshold: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="FixMatch confidence threshold")
    
    # Compute resources
    gpu_count: int = Field(1, ge=0, le=8, description="Number of GPUs")
    cpu_cores: int = Field(4, ge=1, le=32, description="Number of CPU cores")
    memory_gb: int = Field(16, ge=4, le=128, description="Memory in GB")
    
    # Additional parameters
    early_stopping_patience: int = Field(10, ge=1, le=50)
    precision: int = Field(32, description="Training precision (16 or 32)")
    
    @validator('precision')
    def validate_precision(cls, v):
        if v not in [16, 32]:
            raise ValueError('Precision must be 16 or 32')
        return v
    
    @validator('dice_weight', 'focal_weight')
    def validate_loss_weights(cls, v, values):
        if 'dice_weight' in values and 'focal_weight' in values:
            if values['dice_weight'] + values['focal_weight'] != 1.0:
                raise ValueError('Loss weights must sum to 1.0')
        return v


class TrainingJobResponse(BaseModel):
    """Response model for training job."""
    id: str
    job_name: str
    model_architecture: str
    training_mode: str
    status: str
    
    # Progress
    current_epoch: Optional[int] = None
    best_val_iou: Optional[float] = None
    best_val_dice: Optional[float] = None
    
    # Timestamps
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Kubernetes info
    k8s_job_name: Optional[str] = None
    k8s_namespace: str = "ml-training"
    
    # MLflow tracking
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    
    # Model paths
    model_checkpoint_path: Optional[str] = None
    onnx_model_path: Optional[str] = None
    
    # Error info
    error_message: Optional[str] = None


class TrainingJobList(BaseModel):
    """Response model for listing jobs."""
    jobs: List[TrainingJobResponse]
    total: int
    page: int
    page_size: int


class KubernetesJobManager:
    """
    Manages Kubernetes Job creation and monitoring for training.
    
    Creates GPU-enabled training pods with proper resource limits.
    """
    
    def __init__(self, namespace: str = "ml-training"):
        self.namespace = namespace
        
        try:
            # Try in-cluster config first (when running in K8s)
            config.load_incluster_config()
        except:
            # Fall back to kubeconfig (local development)
            try:
                config.load_kube_config()
            except:
                logger.warning("Kubernetes config not found. K8s features disabled.")
                self.k8s_available = False
                return
        
        self.batch_api = client.BatchV1Api()
        self.core_api = client.CoreV1Api()
        self.k8s_available = True
    
    def create_training_job(
        self,
        job_id: str,
        training_config: TrainingJobCreate,
    ) -> str:
        """
        Create Kubernetes Job for training.
        
        Args:
            job_id: Unique job identifier
            training_config: Training configuration
        
        Returns:
            Kubernetes Job name
        """
        if not self.k8s_available:
            raise RuntimeError("Kubernetes not available")
        
        job_name = f"wafer-train-{job_id[:8]}"
        
        # Container specification
        container = client.V1Container(
            name="trainer",
            image="wafer-classifier-training:latest",  # Docker image with PyTorch
            command=["python", "scripts/train.py"],
            args=[
                "--job-id", job_id,
                "--config", f"/config/training_{job_id}.yaml",
            ],
            resources=client.V1ResourceRequirements(
                requests={
                    "cpu": f"{training_config.cpu_cores}",
                    "memory": f"{training_config.memory_gb}Gi",
                    "nvidia.com/gpu": f"{training_config.gpu_count}",
                },
                limits={
                    "cpu": f"{training_config.cpu_cores}",
                    "memory": f"{training_config.memory_gb}Gi",
                    "nvidia.com/gpu": f"{training_config.gpu_count}",
                }
            ),
            volume_mounts=[
                client.V1VolumeMount(
                    name="data-volume",
                    mount_path="/data"
                ),
                client.V1VolumeMount(
                    name="models-volume",
                    mount_path="/models"
                ),
                client.V1VolumeMount(
                    name="config-volume",
                    mount_path="/config"
                ),
            ],
            env=[
                client.V1EnvVar(name="MLFLOW_TRACKING_URI", value="http://mlflow:5000"),
                client.V1EnvVar(name="JOB_ID", value=job_id),
            ]
        )
        
        # Pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": "wafer-training",
                    "job-id": job_id,
                }
            ),
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[container],
                volumes=[
                    client.V1Volume(
                        name="data-volume",
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name="wafer-data-pvc"
                        )
                    ),
                    client.V1Volume(
                        name="models-volume",
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name="wafer-models-pvc"
                        )
                    ),
                    client.V1Volume(
                        name="config-volume",
                        config_map=client.V1ConfigMapVolumeSource(
                            name=f"training-config-{job_id[:8]}"
                        )
                    ),
                ],
                # GPU node selector
                node_selector={"gpu": "true"} if training_config.gpu_count > 0 else None,
            )
        )
        
        # Job specification
        job_spec = client.V1JobSpec(
            template=template,
            backoff_limit=3,  # Retry up to 3 times
            ttl_seconds_after_finished=86400,  # Clean up after 24 hours
        )
        
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
            ),
            spec=job_spec,
        )
        
        try:
            self.batch_api.create_namespaced_job(
                namespace=self.namespace,
                body=job,
            )
            logger.info(f"Created Kubernetes Job: {job_name}")
            return job_name
        
        except ApiException as e:
            logger.error(f"Failed to create K8s Job: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create training job: {str(e)}")
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """
        Get Kubernetes Job status.
        
        Args:
            job_name: K8s Job name
        
        Returns:
            Job status information
        """
        if not self.k8s_available:
            return {"status": "unknown", "message": "Kubernetes not available"}
        
        try:
            job = self.batch_api.read_namespaced_job_status(
                name=job_name,
                namespace=self.namespace,
            )
            
            status = "unknown"
            if job.status.succeeded:
                status = "completed"
            elif job.status.failed:
                status = "failed"
            elif job.status.active:
                status = "running"
            else:
                status = "pending"
            
            return {
                "status": status,
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
                "start_time": job.status.start_time.isoformat() if job.status.start_time else None,
                "completion_time": job.status.completion_time.isoformat() if job.status.completion_time else None,
            }
        
        except ApiException as e:
            logger.error(f"Failed to get K8s Job status: {e}")
            return {"status": "error", "message": str(e)}
    
    def delete_job(self, job_name: str):
        """Delete Kubernetes Job."""
        if not self.k8s_available:
            return
        
        try:
            self.batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground",
                    grace_period_seconds=5,
                )
            )
            logger.info(f"Deleted Kubernetes Job: {job_name}")
        
        except ApiException as e:
            logger.error(f"Failed to delete K8s Job: {e}")


# Global K8s manager
k8s_manager = KubernetesJobManager()


@router.post("/jobs", response_model=TrainingJobResponse, status_code=201)
async def create_training_job(
    job_request: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Create new training job.
    
    Creates database entry and Kubernetes Job for training.
    """
    job_id = str(uuid.uuid4())
    
    try:
        # Create database entry
        job = TrainingJob(
            id=job_id,
            job_name=job_request.job_name,
            model_architecture=job_request.model_architecture.value,
            training_mode=job_request.training_mode.value,
            num_epochs=job_request.num_epochs,
            batch_size=job_request.batch_size,
            learning_rate=job_request.learning_rate,
            loss_function=job_request.loss_function,
            status=JobStatus.PENDING.value,
            hyperparameters={
                "dice_weight": job_request.dice_weight,
                "focal_weight": job_request.focal_weight,
                "optimizer": job_request.optimizer,
                "scheduler": job_request.scheduler,
                "early_stopping_patience": job_request.early_stopping_patience,
                "precision": job_request.precision,
                "train_annotation_file": job_request.train_annotation_file,
                "val_annotation_file": job_request.val_annotation_file,
                "test_annotation_file": job_request.test_annotation_file,
            },
            k8s_namespace="ml-training",
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Create Kubernetes Job (async in background)
        if k8s_manager.k8s_available:
            k8s_job_name = k8s_manager.create_training_job(job_id, job_request)
            
            # Update job with K8s info
            job.k8s_job_name = k8s_job_name
            job.status = JobStatus.RUNNING.value
            job.started_at = datetime.utcnow()
            db.commit()
        else:
            logger.warning("Kubernetes not available. Job created but not started.")
        
        logger.info(f"Created training job: {job_id}")
        
        return TrainingJobResponse(
            id=str(job.id),
            job_name=job.job_name,
            model_architecture=job.model_architecture,
            training_mode=job.training_mode,
            status=job.status,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            k8s_job_name=job.k8s_job_name,
            k8s_namespace=job.k8s_namespace,
        )
    
    except Exception as e:
        logger.error(f"Failed to create training job: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {str(e)}")


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str, db: Session = Depends(get_db)):
    """Get training job details and status."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Update status from Kubernetes if available
    if job.k8s_job_name and k8s_manager.k8s_available:
        k8s_status = k8s_manager.get_job_status(job.k8s_job_name)
        
        # Update database status
        if k8s_status['status'] in ['completed', 'failed']:
            job.status = k8s_status['status']
            if k8s_status['completion_time']:
                job.completed_at = datetime.fromisoformat(k8s_status['completion_time'])
            db.commit()
    
    return TrainingJobResponse(
        id=str(job.id),
        job_name=job.job_name,
        model_architecture=job.model_architecture,
        training_mode=job.training_mode,
        status=job.status,
        best_val_iou=job.best_val_iou,
        best_val_dice=job.best_val_dice,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        k8s_job_name=job.k8s_job_name,
        k8s_namespace=job.k8s_namespace,
        mlflow_run_id=job.mlflow_run_id,
        mlflow_experiment_id=job.mlflow_experiment_id,
        model_checkpoint_path=job.model_checkpoint_path,
        onnx_model_path=job.onnx_model_path,
        error_message=job.error_message,
    )


@router.get("/jobs", response_model=TrainingJobList)
async def list_training_jobs(
    page: int = 1,
    page_size: int = 20,
    status: Optional[JobStatus] = None,
    training_mode: Optional[TrainingMode] = None,
    db: Session = Depends(get_db),
):
    """List training jobs with pagination and filtering."""
    query = db.query(TrainingJob)
    
    # Apply filters
    if status:
        query = query.filter(TrainingJob.status == status.value)
    if training_mode:
        query = query.filter(TrainingJob.training_mode == training_mode.value)
    
    # Get total count
    total = query.count()
    
    # Pagination
    jobs = query.order_by(TrainingJob.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    
    job_responses = [
        TrainingJobResponse(
            id=str(job.id),
            job_name=job.job_name,
            model_architecture=job.model_architecture,
            training_mode=job.training_mode,
            status=job.status,
            best_val_iou=job.best_val_iou,
            best_val_dice=job.best_val_dice,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            k8s_job_name=job.k8s_job_name,
            k8s_namespace=job.k8s_namespace,
        )
        for job in jobs
    ]
    
    return TrainingJobList(
        jobs=job_responses,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel running training job."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status not in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job.status}")
    
    # Delete Kubernetes Job
    if job.k8s_job_name and k8s_manager.k8s_available:
        k8s_manager.delete_job(job.k8s_job_name)
    
    # Update database
    job.status = JobStatus.CANCELLED.value
    job.completed_at = datetime.utcnow()
    db.commit()
    
    return {"status": "success", "message": f"Training job {job_id} cancelled"}


@router.get("/metrics/{job_id}")
async def get_training_metrics(job_id: str, db: Session = Depends(get_db)):
    """
    Get training metrics from MLflow.
    
    Returns real-time training metrics (loss, IoU, etc.) from MLflow tracking.
    """
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if not job.mlflow_run_id:
        return {"metrics": [], "message": "MLflow tracking not yet started"}
    
    try:
        # Get metrics from MLflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(job.mlflow_run_id)
        
        metrics = run.data.metrics
        params = run.data.params
        
        return {
            "run_id": job.mlflow_run_id,
            "status": run.info.status,
            "metrics": metrics,
            "params": params,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }
    
    except Exception as e:
        logger.error(f"Failed to get MLflow metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
