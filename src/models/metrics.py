"""
Evaluation Metrics for Wafer Defect Segmentation

Implements key metrics for assessing segmentation quality:
- IoU (Intersection over Union): Primary metric for PRD requirements (>95% target)
- Dice Coefficient: F1-score for segmentation
- Pixel Accuracy: Basic classification accuracy
- Per-class metrics: Detailed analysis for each defect type
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class SegmentationMetrics:
    """
    Comprehensive metrics calculator for multi-class segmentation.
    
    Computes:
    - IoU (Intersection over Union) per class and mean
    - Dice coefficient per class and mean
    - Pixel accuracy (overall and per-class)
    - Confusion matrix
    
    Args:
        num_classes: Number of segmentation classes (8 for defect types)
        ignore_index: Class index to ignore in metrics (e.g., unlabeled pixels)
    """
    
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)
        self.total_samples = 0
    
    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with new batch.
        
        Args:
            predictions: Model predictions of shape (batch_size, num_classes, H, W) or (batch_size, H, W)
            targets: Ground truth labels of shape (batch_size, H, W) with class indices
        """
        # Convert logits to class predictions if needed
        if predictions.dim() == 4:  # (batch_size, num_classes, H, W)
            predictions = predictions.argmax(dim=1)  # (batch_size, H, W)
        
        # Flatten predictions and targets
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Remove ignore_index pixels
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Update confusion matrix
        # confusion_matrix[i, j] = number of pixels with true class i predicted as class j
        for t, p in zip(targets, predictions):
            self.confusion_matrix[t.long(), p.long()] += 1
        
        self.total_samples += len(predictions)
    
    def compute_iou(self) -> Dict[str, float]:
        """
        Compute IoU (Intersection over Union) per class and mean.
        
        IoU = TP / (TP + FP + FN)
        
        Returns:
            Dictionary with per-class IoU and mean IoU
        """
        # True Positives: diagonal of confusion matrix
        tp = torch.diag(self.confusion_matrix).float()
        
        # False Positives: sum of predicted as class i (column) minus TP
        fp = self.confusion_matrix.sum(dim=0).float() - tp
        
        # False Negatives: sum of true class i (row) minus TP
        fn = self.confusion_matrix.sum(dim=1).float() - tp
        
        # IoU per class
        iou_per_class = tp / (tp + fp + fn + 1e-10)
        
        # Mean IoU (excluding classes with no samples)
        valid_classes = (tp + fn) > 0
        mean_iou = iou_per_class[valid_classes].mean().item()
        
        # Create result dictionary
        result = {"mean_iou": mean_iou}
        for i in range(self.num_classes):
            result[f"iou_class_{i}"] = iou_per_class[i].item()
        
        return result
    
    def compute_dice(self) -> Dict[str, float]:
        """
        Compute Dice coefficient per class and mean.
        
        Dice = 2 * TP / (2 * TP + FP + FN)
        
        Returns:
            Dictionary with per-class Dice and mean Dice
        """
        tp = torch.diag(self.confusion_matrix).float()
        fp = self.confusion_matrix.sum(dim=0).float() - tp
        fn = self.confusion_matrix.sum(dim=1).float() - tp
        
        # Dice per class
        dice_per_class = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-10)
        
        # Mean Dice (excluding classes with no samples)
        valid_classes = (tp + fn) > 0
        mean_dice = dice_per_class[valid_classes].mean().item()
        
        result = {"mean_dice": mean_dice}
        for i in range(self.num_classes):
            result[f"dice_class_{i}"] = dice_per_class[i].item()
        
        return result
    
    def compute_pixel_accuracy(self) -> Dict[str, float]:
        """
        Compute pixel-level accuracy.
        
        Pixel Accuracy = (TP) / (Total Pixels)
        
        Returns:
            Dictionary with overall accuracy and per-class accuracy
        """
        tp = torch.diag(self.confusion_matrix).float()
        total_per_class = self.confusion_matrix.sum(dim=1).float()
        
        # Overall pixel accuracy
        overall_accuracy = tp.sum() / self.confusion_matrix.sum()
        
        # Per-class accuracy (recall)
        accuracy_per_class = tp / (total_per_class + 1e-10)
        
        result = {"pixel_accuracy": overall_accuracy.item()}
        for i in range(self.num_classes):
            result[f"accuracy_class_{i}"] = accuracy_per_class[i].item()
        
        return result
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Returns:
            Dictionary with IoU, Dice, and accuracy metrics
        """
        metrics = {}
        metrics.update(self.compute_iou())
        metrics.update(self.compute_dice())
        metrics.update(self.compute_pixel_accuracy())
        
        return metrics
    
    def get_confusion_matrix(self) -> torch.Tensor:
        """Return current confusion matrix."""
        return self.confusion_matrix.clone()


def compute_batch_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None
) -> float:
    """
    Compute mean IoU for a single batch (fast utility function).
    
    Args:
        predictions: Model predictions of shape (batch_size, num_classes, H, W)
        targets: Ground truth labels of shape (batch_size, H, W)
        num_classes: Number of classes
        ignore_index: Class index to ignore
    
    Returns:
        Mean IoU across all classes
    """
    metrics = SegmentationMetrics(num_classes, ignore_index)
    metrics.update(predictions, targets)
    return metrics.compute_iou()["mean_iou"]


def compute_batch_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None
) -> float:
    """
    Compute mean Dice coefficient for a single batch.
    
    Args:
        predictions: Model predictions of shape (batch_size, num_classes, H, W)
        targets: Ground truth labels of shape (batch_size, H, W)
        num_classes: Number of classes
        ignore_index: Class index to ignore
    
    Returns:
        Mean Dice coefficient across all classes
    """
    metrics = SegmentationMetrics(num_classes, ignore_index)
    metrics.update(predictions, targets)
    return metrics.compute_dice()["mean_dice"]


class PRDMetricsValidator:
    """
    Validator for PRD requirements compliance.
    
    PRD Requirements:
    - Overall IoU > 95%
    - Per-class IoU > 90%
    - Inference latency < 2s per wafer map
    - Annotation reduction > 85% (via active learning)
    
    Args:
        target_mean_iou: Minimum required mean IoU (default: 0.95)
        target_class_iou: Minimum required per-class IoU (default: 0.90)
    """
    
    def __init__(self, target_mean_iou: float = 0.95, target_class_iou: float = 0.90):
        self.target_mean_iou = target_mean_iou
        self.target_class_iou = target_class_iou
    
    def validate(self, metrics: Dict[str, float]) -> Tuple[bool, Dict[str, str]]:
        """
        Validate if metrics meet PRD requirements.
        
        Args:
            metrics: Dictionary of computed metrics
        
        Returns:
            Tuple of (is_compliant, violations_dict)
        """
        violations = {}
        
        # Check mean IoU
        mean_iou = metrics.get("mean_iou", 0.0)
        if mean_iou < self.target_mean_iou:
            violations["mean_iou"] = f"Mean IoU {mean_iou:.3f} < target {self.target_mean_iou:.3f}"
        
        # Check per-class IoU
        for key, value in metrics.items():
            if key.startswith("iou_class_"):
                if value < self.target_class_iou:
                    class_id = key.split("_")[-1]
                    violations[key] = f"Class {class_id} IoU {value:.3f} < target {self.target_class_iou:.3f}"
        
        is_compliant = len(violations) == 0
        
        return is_compliant, violations
    
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            metrics: Dictionary of computed metrics
        
        Returns:
            Formatted report string
        """
        is_compliant, violations = self.validate(metrics)
        
        report = "=" * 60 + "\n"
        report += "PRD Metrics Validation Report\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Mean IoU: {metrics.get('mean_iou', 0.0):.3f} (target: {self.target_mean_iou:.3f})\n"
        report += f"Mean Dice: {metrics.get('mean_dice', 0.0):.3f}\n"
        report += f"Pixel Accuracy: {metrics.get('pixel_accuracy', 0.0):.3f}\n\n"
        
        report += "Per-Class IoU:\n"
        for i in range(8):  # 8 defect classes
            iou = metrics.get(f"iou_class_{i}", 0.0)
            status = "✓" if iou >= self.target_class_iou else "✗"
            report += f"  Class {i}: {iou:.3f} {status}\n"
        
        report += "\n"
        if is_compliant:
            report += "✅ All PRD requirements met!\n"
        else:
            report += "❌ PRD violations detected:\n"
            for key, message in violations.items():
                report += f"  - {message}\n"
        
        report += "=" * 60 + "\n"
        
        return report


if __name__ == "__main__":
    # Test metrics computation
    print("Testing segmentation metrics...")
    
    batch_size = 4
    num_classes = 8
    height, width = 300, 300
    
    # Create dummy predictions and targets
    logits = torch.randn(batch_size, num_classes, height, width)
    predictions = logits.argmax(dim=1)  # (batch_size, H, W)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test SegmentationMetrics
    metrics_calculator = SegmentationMetrics(num_classes)
    metrics_calculator.update(predictions, targets)
    
    all_metrics = metrics_calculator.compute_all_metrics()
    
    print("\nComputed Metrics:")
    print(f"  Mean IoU: {all_metrics['mean_iou']:.4f}")
    print(f"  Mean Dice: {all_metrics['mean_dice']:.4f}")
    print(f"  Pixel Accuracy: {all_metrics['pixel_accuracy']:.4f}")
    
    # Test PRD validation
    validator = PRDMetricsValidator()
    report = validator.generate_report(all_metrics)
    print("\n" + report)
    
    # Test batch utility functions
    batch_iou = compute_batch_iou(logits, targets, num_classes)
    batch_dice = compute_batch_dice(logits, targets, num_classes)
    
    print(f"\nBatch IoU (utility): {batch_iou:.4f}")
    print(f"Batch Dice (utility): {batch_dice:.4f}")
    
    print("\n✅ Metrics test passed!")
