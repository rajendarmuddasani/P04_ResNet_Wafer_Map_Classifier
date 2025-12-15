"""
PyTorch Lightning Training Module for Wafer Defect Segmentation

Implements supervised training with:
- Model training and validation
- Metric tracking (IoU, Dice, Loss)
- Model checkpointing
- MLflow experiment tracking
- Learning rate scheduling
- Early stopping

Supports multiple training modes:
- Supervised: Standard training with labeled data
- Active learning: Iterative training with query strategy
- Semi-supervised: FixMatch with labeled + unlabeled data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import MLFlowLogger
from typing import Dict, Optional, Tuple, Any
import mlflow

from src.models.resnet_unet import ResNetUNet, build_model
from src.models.losses import get_loss_function
from src.models.metrics import SegmentationMetrics, PRDMetricsValidator


class SegmentationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for wafer defect segmentation.
    
    Features:
    - Automatic training/validation loops
    - Metric computation and logging
    - Optimizer and scheduler configuration
    - Model checkpointing
    - MLflow integration
    
    Args:
        model_config: Model configuration dict
        loss_config: Loss function configuration dict
        optimizer_config: Optimizer configuration dict
        num_classes: Number of segmentation classes (default: 8)
        learning_rate: Initial learning rate
        monitor_metric: Metric to monitor for checkpointing (default: 'val_iou')
    
    Example:
        >>> model_config = {'architecture': 'resnet50_unet', 'num_classes': 8}
        >>> loss_config = {'type': 'combined', 'dice_weight': 0.5, 'focal_weight': 0.5}
        >>> module = SegmentationLightningModule(model_config, loss_config)
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        optimizer_config: Optional[Dict[str, Any]] = None,
        num_classes: int = 8,
        learning_rate: float = 1e-4,
        monitor_metric: str = "val_iou",
    ):
        super().__init__()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Build model
        self.model = build_model(model_config)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Loss function
        self.loss_fn = get_loss_function(loss_config)
        
        # Metrics
        self.train_metrics = SegmentationMetrics(num_classes)
        self.val_metrics = SegmentationMetrics(num_classes)
        
        # PRD validation
        self.prd_validator = PRDMetricsValidator(
            target_mean_iou=0.95,
            target_class_iou=0.90
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for single batch.
        
        Args:
            batch: Tuple of (images, masks)
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        images, masks = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute loss
        loss = self.loss_fn(logits, masks)
        
        # Update metrics
        self.train_metrics.update(logits.detach(), masks)
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step for single batch.
        
        Args:
            batch: Tuple of (images, masks)
            batch_idx: Batch index
        """
        images, masks = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute loss
        loss = self.loss_fn(logits, masks)
        
        # Update metrics
        self.val_metrics.update(logits, masks)
        
        # Log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_train_epoch_end(self) -> None:
        """Called at end of training epoch."""
        # Compute and log training metrics
        metrics = self.train_metrics.compute_all_metrics()
        
        self.log('train_iou', metrics['mean_iou'], prog_bar=True)
        self.log('train_dice', metrics['mean_dice'])
        self.log('train_pixel_accuracy', metrics['pixel_accuracy'])
        
        # Log per-class IoU
        for i in range(self.num_classes):
            self.log(f'train_iou_class_{i}', metrics[f'iou_class_{i}'])
        
        # Reset metrics
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self) -> None:
        """Called at end of validation epoch."""
        # Compute and log validation metrics
        metrics = self.val_metrics.compute_all_metrics()
        
        self.log('val_iou', metrics['mean_iou'], prog_bar=True)
        self.log('val_dice', metrics['mean_dice'])
        self.log('val_pixel_accuracy', metrics['pixel_accuracy'])
        
        # Log per-class IoU
        for i in range(self.num_classes):
            self.log(f'val_iou_class_{i}', metrics[f'iou_class_{i}'])
        
        # Check PRD compliance
        is_compliant, violations = self.prd_validator.validate(metrics)
        
        if is_compliant:
            self.log('prd_compliant', 1.0)
        else:
            self.log('prd_compliant', 0.0)
            # Log violations
            for key, message in violations.items():
                print(f"⚠️  PRD Violation: {message}")
        
        # Reset metrics
        self.val_metrics.reset()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Optimizer
        optimizer_name = self.hparams.get('optimizer', 'adamw')
        
        if optimizer_name.lower() == 'adam':
            optimizer = Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5
            )
        elif optimizer_name.lower() == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-2
            )
        elif optimizer_name.lower() == 'sgd':
            optimizer = SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler_name = self.hparams.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_name == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',  # Maximize IoU
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_iou',
                    'interval': 'epoch',
                }
            }
        
        elif scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
            }
        
        else:
            return {'optimizer': optimizer}
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Prediction step for inference.
        
        Args:
            batch: Input images
            batch_idx: Batch index
        
        Returns:
            Predicted segmentation masks
        """
        logits = self(batch)
        predictions = torch.argmax(logits, dim=1)
        return predictions


def create_trainer(
    max_epochs: int = 50,
    gpus: int = 1,
    checkpoint_dir: str = "checkpoints",
    experiment_name: str = "wafer_segmentation",
    early_stopping_patience: int = 10,
    precision: int = 32,  # 16 for mixed precision (faster, less memory)
) -> pl.Trainer:
    """
    Create configured PyTorch Lightning Trainer.
    
    Args:
        max_epochs: Maximum number of training epochs
        gpus: Number of GPUs to use (0 for CPU)
        checkpoint_dir: Directory to save model checkpoints
        experiment_name: MLflow experiment name
        early_stopping_patience: Epochs to wait before early stopping
        precision: Training precision (32 or 16)
    
    Returns:
        Configured Trainer instance
    """
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='wafer-segmentation-{epoch:02d}-{val_iou:.4f}',
        monitor='val_iou',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_iou',
        patience=early_stopping_patience,
        mode='max',
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri="mlruns",  # Local MLflow tracking
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        precision=precision,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=mlflow_logger,
        log_every_n_steps=10,
        deterministic=True,  # For reproducibility
        gradient_clip_val=1.0,  # Prevent exploding gradients
    )
    
    return trainer


def train_model(
    train_dataloader,
    val_dataloader,
    model_config: Dict[str, Any],
    loss_config: Dict[str, Any],
    max_epochs: int = 50,
    gpus: int = 1,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
) -> Tuple[SegmentationLightningModule, pl.Trainer]:
    """
    High-level training function.
    
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        model_config: Model configuration
        loss_config: Loss function configuration
        max_epochs: Maximum epochs
        gpus: Number of GPUs
        learning_rate: Learning rate
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Tuple of (trained_module, trainer)
    
    Example:
        >>> model_config = {'architecture': 'resnet50_unet', 'num_classes': 8}
        >>> loss_config = {'type': 'combined'}
        >>> module, trainer = train_model(
        ...     train_loader, val_loader, model_config, loss_config
        ... )
    """
    # Create Lightning module
    module = SegmentationLightningModule(
        model_config=model_config,
        loss_config=loss_config,
        learning_rate=learning_rate,
    )
    
    # Create trainer
    trainer = create_trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Train
    trainer.fit(module, train_dataloader, val_dataloader)
    
    return module, trainer


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Lightning Training Module")
    print("=" * 60)
    
    # Test module initialization
    print("\n1. Testing module initialization...")
    
    model_config = {
        'architecture': 'resnet50_unet',
        'num_classes': 8,
        'encoder_weights': None,  # Random init for testing
    }
    
    loss_config = {
        'type': 'combined',
        'dice_weight': 0.5,
        'focal_weight': 0.5,
    }
    
    module = SegmentationLightningModule(
        model_config=model_config,
        loss_config=loss_config,
        learning_rate=1e-4,
    )
    
    print(f"   ✅ Module created: {module.__class__.__name__}")
    print(f"   Model parameters: {sum(p.numel() for p in module.parameters()):,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    dummy_input = torch.randn(2, 3, 300, 300)
    dummy_mask = torch.randint(0, 8, (2, 300, 300))
    
    with torch.no_grad():
        output = module(dummy_input)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ Forward pass successful")
    
    # Test training step
    print("\n3. Testing training step...")
    loss = module.training_step((dummy_input, dummy_mask), 0)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✅ Training step successful")
    
    print("\n" + "=" * 60)
    print("✅ Training module ready!")
    print("=" * 60)
    print("\nUsage:")
    print("  from src.training.trainer import train_model")
    print("  module, trainer = train_model(train_loader, val_loader, model_config, loss_config)")
