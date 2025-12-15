"""
SQLAlchemy ORM Models for Wafer Defect Classification Platform

Mirrors the PostgreSQL schema with ORM capabilities:
- Type-safe database operations
- Relationship management
- Query building
- Data validation

Usage:
    from src.database.models import WaferMap, Annotation, TrainingJob
    from src.database.database import SessionLocal
    
    db = SessionLocal()
    wafer = db.query(WaferMap).filter_by(wafer_id="W12345").first()
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, CheckConstraint, Index, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, index=True)  # admin, annotator, engineer, viewer
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Annotation metrics
    total_annotations = Column(Integer, default=0)
    average_annotation_time_seconds = Column(Float)
    inter_annotator_agreement = Column(Float)
    
    # Relationships
    uploaded_wafer_maps = relationship("WaferMap", back_populates="uploaded_by_user", foreign_keys="WaferMap.uploaded_by")
    annotations = relationship("Annotation", back_populates="annotator", foreign_keys="Annotation.annotated_by")
    training_jobs = relationship("TrainingJob", back_populates="creator")
    active_learning_assignments = relationship("ActiveLearningQueue", back_populates="assigned_user")
    
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'annotator', 'engineer', 'viewer')", name="valid_role"),
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"


class WaferMap(Base):
    """Wafer map metadata and storage references."""
    
    __tablename__ = "wafer_maps"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    wafer_id = Column(String(255), unique=True, nullable=False, index=True)
    lot_id = Column(String(100), index=True)
    die_size_x = Column(Float)
    die_size_y = Column(Float)
    
    # Storage
    image_path = Column(String(500), nullable=False)
    image_format = Column(String(10), default="png")
    image_size_bytes = Column(Integer)
    image_width = Column(Integer)
    image_height = Column(Integer)
    
    # Status
    status = Column(String(50), default="uploaded", index=True)
    
    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    processed_at = Column(DateTime(timezone=True))
    
    # Metadata
    metadata = Column(JSONB)
    
    # Foreign keys
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    # Relationships
    uploaded_by_user = relationship("User", back_populates="uploaded_wafer_maps", foreign_keys=[uploaded_by])
    annotations = relationship("Annotation", back_populates="wafer_map", cascade="all, delete-orphan")
    embeddings = relationship("DefectEmbedding", back_populates="wafer_map", cascade="all, delete-orphan")
    active_learning_entries = relationship("ActiveLearningQueue", back_populates="wafer_map", cascade="all, delete-orphan")
    inference_logs = relationship("InferenceLog", back_populates="wafer_map", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint(
            "status IN ('uploaded', 'queued', 'annotated', 'training', 'validated', 'archived')",
            name="valid_status"
        ),
        Index("idx_wafer_maps_metadata", "metadata", postgresql_using="gin"),
    )
    
    def __repr__(self):
        return f"<WaferMap(id={self.id}, wafer_id={self.wafer_id}, status={self.status})>"


class Annotation(Base):
    """Segmentation annotations in COCO format."""
    
    __tablename__ = "annotations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    wafer_map_id = Column(UUID(as_uuid=True), ForeignKey("wafer_maps.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Annotation data
    segmentation = Column(JSONB, nullable=False)  # Polygon coordinates
    category_id = Column(Integer, nullable=False, index=True)  # 0-7
    category_name = Column(String(50))
    
    # Bounding box
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_width = Column(Float)
    bbox_height = Column(Float)
    area = Column(Float)
    
    # Quality
    confidence = Column(Float)
    is_verified = Column(Boolean, default=False, index=True)
    is_active_learning_sample = Column(Boolean, default=False)
    
    # Annotator info
    annotated_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    annotation_time_seconds = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    verified_at = Column(DateTime(timezone=True))
    verified_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    # Version control
    version = Column(Integer, default=1)
    parent_annotation_id = Column(UUID(as_uuid=True), ForeignKey("annotations.id", ondelete="SET NULL"))
    
    # Relationships
    wafer_map = relationship("WaferMap", back_populates="annotations")
    annotator = relationship("User", back_populates="annotations", foreign_keys=[annotated_by])
    verifier = relationship("User", foreign_keys=[verified_by])
    parent_annotation = relationship("Annotation", remote_side=[id])
    
    __table_args__ = (
        CheckConstraint("category_id BETWEEN 0 AND 7", name="valid_category_id"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="valid_confidence"),
        Index("idx_annotations_segmentation", "segmentation", postgresql_using="gin"),
    )
    
    def __repr__(self):
        return f"<Annotation(id={self.id}, category={self.category_name}, verified={self.is_verified})>"


class TrainingJob(Base):
    """ML training job tracking."""
    
    __tablename__ = "training_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_name = Column(String(255), nullable=False)
    
    # Configuration
    model_architecture = Column(String(100), default="resnet50_unet")
    num_epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    loss_function = Column(String(50), default="combined")
    training_mode = Column(String(50), default="supervised", index=True)
    
    # Dataset info
    total_samples = Column(Integer)
    train_samples = Column(Integer)
    val_samples = Column(Integer)
    test_samples = Column(Integer)
    num_labeled_samples = Column(Integer)
    num_unlabeled_samples = Column(Integer)
    
    # Status
    status = Column(String(50), default="pending", index=True)
    
    # Results
    best_val_iou = Column(Float)
    best_val_dice = Column(Float)
    final_train_loss = Column(Float)
    final_val_loss = Column(Float)
    
    # Kubernetes
    k8s_job_name = Column(String(255))
    k8s_namespace = Column(String(100), default="ml-training")
    
    # MLflow
    mlflow_run_id = Column(String(255), index=True)
    mlflow_experiment_id = Column(String(255))
    
    # Storage
    model_checkpoint_path = Column(String(500))
    onnx_model_path = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # User tracking
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    # Hyperparameters
    hyperparameters = Column(JSONB)
    
    # Logs
    error_message = Column(Text)
    training_logs = Column(Text)
    
    # Relationships
    creator = relationship("User", back_populates="training_jobs")
    embeddings = relationship("DefectEmbedding", back_populates="training_job")
    active_learning_queues = relationship("ActiveLearningQueue", back_populates="training_job")
    
    __table_args__ = (
        CheckConstraint(
            "training_mode IN ('supervised', 'active_learning', 'semi_supervised', 'fine_tuning')",
            name="valid_training_mode"
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="valid_status"
        ),
    )
    
    def __repr__(self):
        return f"<TrainingJob(id={self.id}, name={self.job_name}, status={self.status})>"


class DefectEmbedding(Base):
    """Feature embeddings for active learning."""
    
    __tablename__ = "defect_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    wafer_map_id = Column(UUID(as_uuid=True), ForeignKey("wafer_maps.id", ondelete="CASCADE"), nullable=False, index=True)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id", ondelete="SET NULL"), index=True)
    
    # Embedding
    embedding = Column(ARRAY(Float), nullable=False)
    embedding_dim = Column(Integer, nullable=False, default=2048)
    
    # Uncertainty metrics
    prediction_entropy = Column(Float, index=True)
    bald_score = Column(Float)
    variation_ratio = Column(Float)
    
    # Predictions
    class_probabilities = Column(ARRAY(Float))
    predicted_class = Column(Integer, index=True)
    prediction_confidence = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    wafer_map = relationship("WaferMap", back_populates="embeddings")
    training_job = relationship("TrainingJob", back_populates="embeddings")
    
    __table_args__ = (
        CheckConstraint("predicted_class BETWEEN 0 AND 7", name="valid_predicted_class"),
    )
    
    def __repr__(self):
        return f"<DefectEmbedding(id={self.id}, predicted_class={self.predicted_class})>"


class ActiveLearningQueue(Base):
    """Active learning sample queue."""
    
    __tablename__ = "active_learning_queue"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    wafer_map_id = Column(UUID(as_uuid=True), ForeignKey("wafer_maps.id", ondelete="CASCADE"), nullable=False)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False)
    
    # Query strategy
    query_strategy = Column(String(50), default="hybrid")
    
    # Scores
    uncertainty_score = Column(Float)
    diversity_score = Column(Float)
    combined_score = Column(Float, index=True)
    
    # Priority
    priority = Column(Integer, default=0)
    status = Column(String(50), default="pending", index=True)
    
    # Assignment
    assigned_to = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    assigned_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Metadata
    query_iteration = Column(Integer, index=True)
    batch_id = Column(UUID(as_uuid=True))
    
    # Relationships
    wafer_map = relationship("WaferMap", back_populates="active_learning_entries")
    training_job = relationship("TrainingJob", back_populates="active_learning_queues")
    assigned_user = relationship("User", back_populates="active_learning_assignments")
    
    __table_args__ = (
        CheckConstraint(
            "query_strategy IN ('uncertainty', 'diversity', 'hybrid', 'random')",
            name="valid_query_strategy"
        ),
        CheckConstraint(
            "status IN ('pending', 'assigned', 'annotated', 'skipped', 'rejected')",
            name="valid_status"
        ),
    )
    
    def __repr__(self):
        return f"<ActiveLearningQueue(id={self.id}, status={self.status}, score={self.combined_score})>"


class AnnotationMetric(Base):
    """Inter-annotator agreement metrics."""
    
    __tablename__ = "annotation_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Pairwise comparison
    annotator1_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    annotator2_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    wafer_map_id = Column(UUID(as_uuid=True), ForeignKey("wafer_maps.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Agreement metrics
    iou_agreement = Column(Float)
    dice_agreement = Column(Float)
    pixel_agreement = Column(Float)
    
    # Annotation comparison
    annotation1_id = Column(UUID(as_uuid=True), ForeignKey("annotations.id", ondelete="CASCADE"))
    annotation2_id = Column(UUID(as_uuid=True), ForeignKey("annotations.id", ondelete="CASCADE"))
    
    # Timestamps
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint("annotator1_id != annotator2_id", name="different_annotators"),
        CheckConstraint("iou_agreement BETWEEN 0 AND 1", name="valid_iou"),
        CheckConstraint("dice_agreement BETWEEN 0 AND 1", name="valid_dice"),
        CheckConstraint("pixel_agreement BETWEEN 0 AND 1", name="valid_pixel"),
    )
    
    def __repr__(self):
        return f"<AnnotationMetric(id={self.id}, iou={self.iou_agreement:.3f})>"


class InferenceLog(Base):
    """Production inference monitoring."""
    
    __tablename__ = "inference_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    wafer_map_id = Column(UUID(as_uuid=True), ForeignKey("wafer_maps.id", ondelete="CASCADE"), index=True)
    
    # Model info
    model_version = Column(String(100), index=True)
    model_path = Column(String(500))
    
    # Results
    predicted_mask_path = Column(String(500))
    class_distribution = Column(JSONB)
    total_defect_area = Column(Float)
    confidence_scores = Column(ARRAY(Float))
    
    # Performance
    inference_time_ms = Column(Float)
    preprocessing_time_ms = Column(Float)
    postprocessing_time_ms = Column(Float)
    
    # Request metadata
    request_id = Column(String(255))
    client_ip = Column(String(45))
    api_endpoint = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Status
    status = Column(String(50), default="success", index=True)
    error_message = Column(Text)
    
    # Relationships
    wafer_map = relationship("WaferMap", back_populates="inference_logs")
    
    __table_args__ = (
        CheckConstraint("status IN ('success', 'error')", name="valid_status"),
    )
    
    def __repr__(self):
        return f"<InferenceLog(id={self.id}, status={self.status}, latency={self.inference_time_ms}ms)>"
