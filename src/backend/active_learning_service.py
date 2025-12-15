"""
Active Learning Service - Query Strategy API

Features:
- Manage active learning queue
- Run query strategies (uncertainty, diversity, hybrid)
- Compute uncertainty scores and embeddings
- Assign samples to annotators
- Track annotation progress

Target: 85% annotation reduction (5000 → 750 samples)

Usage:
    POST /active-learning/query - Run query strategy
    GET /active-learning/queue - Get pending samples
    POST /active-learning/assign - Assign samples to annotator
    GET /active-learning/stats - Get AL statistics
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
import logging

from src.database.database import get_db
from src.database.models import (
    ActiveLearningQueue,
    DefectEmbedding,
    WaferMap,
    TrainingJob,
    User,
    Annotation,
)
from src.training.active_learning import (
    UncertaintySampling,
    DiversitySampling,
    HybridActiveLearning,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/active-learning", tags=["active-learning"])


class QueryStrategy(str, Enum):
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"
    RANDOM = "random"


class QueueStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    ANNOTATED = "annotated"
    SKIPPED = "skipped"
    REJECTED = "rejected"


# Request/Response models
class QueryRequest(BaseModel):
    """Request for running query strategy."""
    training_job_id: str = Field(..., description="Training job ID")
    query_strategy: QueryStrategy = Field(QueryStrategy.HYBRID, description="Query strategy")
    budget: int = Field(50, ge=1, le=1000, description="Number of samples to select")
    uncertainty_weight: float = Field(0.6, ge=0.0, le=1.0, description="Uncertainty weight (hybrid only)")
    diversity_weight: float = Field(0.4, ge=0.0, le=1.0, description="Diversity weight (hybrid only)")
    iteration: int = Field(1, ge=1, description="AL iteration number")


class QueueSample(BaseModel):
    """Sample in active learning queue."""
    id: str
    wafer_map_id: str
    wafer_id: str
    query_strategy: str
    uncertainty_score: Optional[float]
    diversity_score: Optional[float]
    combined_score: float
    priority: int
    status: str
    assigned_to: Optional[str] = None
    created_at: str


class QueueResponse(BaseModel):
    """Response for queue listing."""
    samples: List[QueueSample]
    total: int
    pending: int
    assigned: int
    annotated: int
    page: int
    page_size: int


class AssignRequest(BaseModel):
    """Request for assigning samples to annotator."""
    sample_ids: List[str] = Field(..., min_items=1, description="Sample IDs to assign")
    annotator_id: str = Field(..., description="Annotator user ID")


class ALStats(BaseModel):
    """Active learning statistics."""
    total_samples: int
    labeled_samples: int
    unlabeled_samples: int
    annotation_reduction: float  # Percentage
    queue_pending: int
    queue_assigned: int
    queue_completed: int
    average_uncertainty: Optional[float]
    average_diversity: Optional[float]
    iterations_completed: int


@router.post("/query")
async def run_query_strategy(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    """
    Run active learning query strategy to select samples for annotation.
    
    Steps:
    1. Get embeddings and uncertainty scores for unlabeled data
    2. Apply query strategy to select top-k samples
    3. Add selected samples to annotation queue
    4. Return selected sample IDs
    """
    try:
        # Verify training job exists
        training_job = db.query(TrainingJob).filter(
            TrainingJob.id == request.training_job_id
        ).first()
        
        if not training_job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        # Get all embeddings for this training job
        embeddings_data = db.query(DefectEmbedding).filter(
            DefectEmbedding.training_job_id == request.training_job_id
        ).all()
        
        if not embeddings_data:
            raise HTTPException(
                status_code=400,
                detail="No embeddings found. Run inference first to generate embeddings."
            )
        
        # Get already labeled wafer IDs
        labeled_wafer_ids = set(
            db.query(Annotation.wafer_map_id).distinct().all()
        )
        labeled_wafer_ids = {str(w[0]) for w in labeled_wafer_ids}
        
        # Prepare data for query strategy
        all_embeddings = []
        all_uncertainties = []
        wafer_map_ids = []
        
        for emb in embeddings_data:
            wafer_map_id = str(emb.wafer_map_id)
            
            # Skip already labeled samples
            if wafer_map_id in labeled_wafer_ids:
                continue
            
            all_embeddings.append(emb.embedding)
            all_uncertainties.append(emb.prediction_entropy or 0.0)
            wafer_map_ids.append(wafer_map_id)
        
        if len(wafer_map_ids) == 0:
            return {
                "message": "All samples are already labeled",
                "selected_samples": [],
            }
        
        # Convert to numpy arrays
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        uncertainty_array = np.array(all_uncertainties, dtype=np.float32)
        
        # Get indices of already labeled samples (for diversity calculation)
        labeled_indices = [
            i for i, wid in enumerate(wafer_map_ids)
            if wid in labeled_wafer_ids
        ]
        
        # Run query strategy
        if request.query_strategy == QueryStrategy.UNCERTAINTY:
            # Select top-k uncertain samples
            top_indices = np.argsort(uncertainty_array)[-request.budget:][::-1]
            selected_indices = top_indices.tolist()
            diversity_scores = [None] * len(selected_indices)
        
        elif request.query_strategy == QueryStrategy.DIVERSITY:
            # CoreSet greedy selection
            diversity_sampler = DiversitySampling()
            selected_indices = diversity_sampler.coreset_greedy(
                embeddings_array,
                labeled_indices,
                request.budget
            )
            diversity_scores = [1.0] * len(selected_indices)  # Placeholder
        
        elif request.query_strategy == QueryStrategy.HYBRID:
            # Hybrid strategy
            hybrid_sampler = HybridActiveLearning(
                uncertainty_weight=request.uncertainty_weight,
                diversity_weight=request.diversity_weight
            )
            selected_indices = hybrid_sampler.query(
                uncertainty_array,
                embeddings_array,
                labeled_indices,
                request.budget
            )
            
            # Compute diversity scores for selected samples
            if len(labeled_indices) > 0:
                from sklearn.metrics import pairwise_distances
                labeled_embs = embeddings_array[labeled_indices]
                selected_embs = embeddings_array[selected_indices]
                distances = pairwise_distances(selected_embs, labeled_embs, metric='euclidean')
                diversity_scores = distances.min(axis=1).tolist()
            else:
                diversity_scores = [1.0] * len(selected_indices)
        
        elif request.query_strategy == QueryStrategy.RANDOM:
            # Random baseline
            selected_indices = np.random.choice(
                len(wafer_map_ids),
                size=min(request.budget, len(wafer_map_ids)),
                replace=False
            ).tolist()
            diversity_scores = [None] * len(selected_indices)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown query strategy: {request.query_strategy}")
        
        # Create batch ID for this query
        batch_id = str(uuid.uuid4())
        
        # Add selected samples to queue
        queue_entries = []
        for idx in selected_indices:
            wafer_map_id = wafer_map_ids[idx]
            uncertainty_score = float(uncertainty_array[idx])
            diversity_score = diversity_scores[idx] if diversity_scores[idx] is not None else None
            
            # Combined score (for hybrid)
            if request.query_strategy == QueryStrategy.HYBRID:
                combined_score = (
                    request.uncertainty_weight * uncertainty_score +
                    request.diversity_weight * (diversity_score or 0)
                )
            else:
                combined_score = uncertainty_score
            
            queue_entry = ActiveLearningQueue(
                wafer_map_id=wafer_map_id,
                training_job_id=request.training_job_id,
                query_strategy=request.query_strategy.value,
                uncertainty_score=uncertainty_score,
                diversity_score=diversity_score,
                combined_score=combined_score,
                priority=request.budget - list(selected_indices).index(idx),  # Higher priority for top samples
                status=QueueStatus.PENDING.value,
                query_iteration=request.iteration,
                batch_id=batch_id,
            )
            queue_entries.append(queue_entry)
        
        # Bulk insert
        db.bulk_save_objects(queue_entries)
        db.commit()
        
        logger.info(
            f"Query strategy '{request.query_strategy.value}' selected {len(selected_indices)} samples "
            f"for iteration {request.iteration}"
        )
        
        return {
            "message": f"Selected {len(selected_indices)} samples for annotation",
            "query_strategy": request.query_strategy.value,
            "selected_samples": [wafer_map_ids[i] for i in selected_indices],
            "batch_id": batch_id,
            "iteration": request.iteration,
            "average_uncertainty": float(uncertainty_array[selected_indices].mean()),
            "average_diversity": float(np.mean([s for s in diversity_scores if s is not None])) if any(diversity_scores) else None,
        }
    
    except Exception as e:
        logger.error(f"Query strategy failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query strategy failed: {str(e)}")


@router.get("/queue", response_model=QueueResponse)
async def get_queue(
    status: Optional[QueueStatus] = None,
    training_job_id: Optional[str] = None,
    iteration: Optional[int] = None,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
):
    """Get active learning queue with filtering and pagination."""
    query = db.query(ActiveLearningQueue).join(WaferMap)
    
    # Apply filters
    if status:
        query = query.filter(ActiveLearningQueue.status == status.value)
    if training_job_id:
        query = query.filter(ActiveLearningQueue.training_job_id == training_job_id)
    if iteration:
        query = query.filter(ActiveLearningQueue.query_iteration == iteration)
    
    # Get counts
    total = query.count()
    pending = db.query(ActiveLearningQueue).filter(ActiveLearningQueue.status == QueueStatus.PENDING.value).count()
    assigned = db.query(ActiveLearningQueue).filter(ActiveLearningQueue.status == QueueStatus.ASSIGNED.value).count()
    annotated = db.query(ActiveLearningQueue).filter(ActiveLearningQueue.status == QueueStatus.ANNOTATED.value).count()
    
    # Pagination with ordering by combined score (descending)
    samples = query.order_by(ActiveLearningQueue.combined_score.desc()).offset((page - 1) * page_size).limit(page_size).all()
    
    # Build response
    sample_responses = []
    for sample in samples:
        wafer_map = db.query(WaferMap).filter(WaferMap.id == sample.wafer_map_id).first()
        
        sample_responses.append(QueueSample(
            id=str(sample.id),
            wafer_map_id=str(sample.wafer_map_id),
            wafer_id=wafer_map.wafer_id if wafer_map else "unknown",
            query_strategy=sample.query_strategy,
            uncertainty_score=sample.uncertainty_score,
            diversity_score=sample.diversity_score,
            combined_score=sample.combined_score,
            priority=sample.priority,
            status=sample.status,
            assigned_to=str(sample.assigned_to) if sample.assigned_to else None,
            created_at=sample.created_at.isoformat(),
        ))
    
    return QueueResponse(
        samples=sample_responses,
        total=total,
        pending=pending,
        assigned=assigned,
        annotated=annotated,
        page=page,
        page_size=page_size,
    )


@router.post("/assign")
async def assign_samples(
    request: AssignRequest,
    db: Session = Depends(get_db),
):
    """Assign queue samples to annotator."""
    # Verify annotator exists
    annotator = db.query(User).filter(User.id == request.annotator_id).first()
    if not annotator:
        raise HTTPException(status_code=404, detail="Annotator not found")
    
    if annotator.role != "annotator":
        raise HTTPException(status_code=400, detail="User is not an annotator")
    
    # Update samples
    updated_count = 0
    for sample_id in request.sample_ids:
        sample = db.query(ActiveLearningQueue).filter(ActiveLearningQueue.id == sample_id).first()
        
        if sample and sample.status == QueueStatus.PENDING.value:
            sample.status = QueueStatus.ASSIGNED.value
            sample.assigned_to = request.annotator_id
            sample.assigned_at = datetime.utcnow()
            updated_count += 1
    
    db.commit()
    
    logger.info(f"Assigned {updated_count} samples to annotator {request.annotator_id}")
    
    return {
        "status": "success",
        "assigned_count": updated_count,
        "annotator_id": request.annotator_id,
    }


@router.get("/stats", response_model=ALStats)
async def get_al_stats(
    training_job_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get active learning statistics."""
    query = db.query(ActiveLearningQueue)
    if training_job_id:
        query = query.filter(ActiveLearningQueue.training_job_id == training_job_id)
    
    total_samples = db.query(WaferMap).count()
    labeled_samples = db.query(Annotation.wafer_map_id).distinct().count()
    unlabeled_samples = total_samples - labeled_samples
    
    annotation_reduction = (1 - labeled_samples / total_samples) * 100 if total_samples > 0 else 0
    
    queue_pending = query.filter(ActiveLearningQueue.status == QueueStatus.PENDING.value).count()
    queue_assigned = query.filter(ActiveLearningQueue.status == QueueStatus.ASSIGNED.value).count()
    queue_completed = query.filter(ActiveLearningQueue.status == QueueStatus.ANNOTATED.value).count()
    
    # Average scores
    avg_uncertainty = db.query(func.avg(ActiveLearningQueue.uncertainty_score)).scalar()
    avg_diversity = db.query(func.avg(ActiveLearningQueue.diversity_score)).scalar()
    
    # Iterations
    max_iteration = db.query(func.max(ActiveLearningQueue.query_iteration)).scalar() or 0
    
    return ALStats(
        total_samples=total_samples,
        labeled_samples=labeled_samples,
        unlabeled_samples=unlabeled_samples,
        annotation_reduction=annotation_reduction,
        queue_pending=queue_pending,
        queue_assigned=queue_assigned,
        queue_completed=queue_completed,
        average_uncertainty=float(avg_uncertainty) if avg_uncertainty else None,
        average_diversity=float(avg_diversity) if avg_diversity else None,
        iterations_completed=max_iteration,
    )


@router.patch("/queue/{sample_id}/status")
async def update_sample_status(
    sample_id: str,
    status: QueueStatus,
    db: Session = Depends(get_db),
):
    """Update queue sample status (e.g., mark as annotated)."""
    sample = db.query(ActiveLearningQueue).filter(ActiveLearningQueue.id == sample_id).first()
    
    if not sample:
        raise HTTPException(status_code=404, detail="Queue sample not found")
    
    sample.status = status.value
    
    if status == QueueStatus.ANNOTATED:
        sample.completed_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "status": "success",
        "sample_id": sample_id,
        "new_status": status.value,
    }
