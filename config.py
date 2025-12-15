"""
Configuration Management - Load settings from environment variables

This module provides centralized configuration management for the application,
loading all sensitive data from environment variables.
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="wafer_classifier", env="POSTGRES_DB")
    
    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # JWT
    secret_key: str = Field(
        default="development-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=30, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    debug: bool = Field(default=False, env="DEBUG")
    
    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:3001,http://localhost:3002",
        env="CORS_ORIGINS"
    )
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="wafer_defect_classification",
        env="MLFLOW_EXPERIMENT_NAME"
    )
    
    # Storage
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    s3_bucket: Optional[str] = Field(default=None, env="S3_BUCKET")
    
    data_dir: str = Field(default="/data", env="DATA_DIR")
    models_dir: str = Field(default="/models", env="MODELS_DIR")
    
    # Kubernetes
    kube_namespace: str = Field(default="default", env="KUBE_NAMESPACE")
    gpu_node_selector: str = Field(default="nvidia.com/gpu=true", env="GPU_NODE_SELECTOR")
    
    # Training
    default_batch_size: int = Field(default=32, env="DEFAULT_BATCH_SIZE")
    default_learning_rate: float = Field(default=0.001, env="DEFAULT_LEARNING_RATE")
    default_epochs: int = Field(default=100, env="DEFAULT_EPOCHS")
    
    # Active Learning
    al_uncertainty_weight: float = Field(default=0.6, env="AL_UNCERTAINTY_WEIGHT")
    al_diversity_weight: float = Field(default=0.4, env="AL_DIVERSITY_WEIGHT")
    al_batch_size: int = Field(default=100, env="AL_BATCH_SIZE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def database_url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def cors_origins_list(self) -> list:
        """Get CORS origins as list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
