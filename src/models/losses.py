"""
Loss Functions for Wafer Defect Segmentation

Implements combined loss functions optimized for imbalanced multi-class segmentation:
- Dice Loss: Overlap-based metric robust to class imbalance
- Focal Loss: Focuses learning on hard examples
- Combined Loss: Weighted combination for optimal performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    
    Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice coefficient
    
    Advantages:
    - Robust to class imbalance (background vs defects)
    - Directly optimizes IoU-like metric
    - Works well for small objects (defects)
    
    Args:
        smooth: Smoothing constant to avoid division by zero (default: 1.0)
        ignore_index: Class index to ignore in loss calculation (e.g., 255 for unlabeled)
    """
    
    def __init__(self, smooth: float = 1.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            logits: Model predictions of shape (batch_size, num_classes, H, W)
            targets: Ground truth labels of shape (batch_size, H, W) with class indices
        
        Returns:
            Scalar loss value
        """
        num_classes = logits.shape[1]
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Convert targets to one-hot encoding: (batch_size, num_classes, H, W)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Mask out ignore_index if specified
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask
        
        # Flatten spatial dimensions: (batch_size, num_classes, H*W)
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Compute intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=2)  # (batch_size, num_classes)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (batch_size, num_classes)
        
        # Dice coefficient per class
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average across classes and batch
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Focuses learning on hard examples by down-weighting easy ones.
    
    Args:
        alpha: Weighting factor for class imbalance (default: 0.25)
        gamma: Focusing parameter (default: 2.0, higher = more focus on hard examples)
        ignore_index: Class index to ignore in loss calculation
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            logits: Model predictions of shape (batch_size, num_classes, H, W)
            targets: Ground truth labels of shape (batch_size, H, W) with class indices
        
        Returns:
            Scalar loss value
        """
        # Compute cross-entropy loss per pixel
        ce_loss = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index or -100)
        
        # Get predicted probabilities for true class
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        p_t = (probs * targets_one_hot).sum(dim=1)  # (batch_size, H, W)
        
        # Apply focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Average across batch and spatial dimensions
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice + Focal Loss for optimal segmentation performance.
    
    Loss = λ_dice * DiceLoss + λ_focal * FocalLoss
    
    Combines benefits:
    - Dice Loss: Optimizes overlap metrics (IoU)
    - Focal Loss: Handles class imbalance and hard examples
    
    Args:
        dice_weight: Weight for Dice loss (default: 0.5)
        focal_weight: Weight for Focal loss (default: 0.5)
        smooth: Smoothing constant for Dice loss
        alpha: Alpha parameter for Focal loss
        gamma: Gamma parameter for Focal loss
        ignore_index: Class index to ignore
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model predictions of shape (batch_size, num_classes, H, W)
            targets: Ground truth labels of shape (batch_size, H, W)
        
        Returns:
            Scalar loss value
        """
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        
        combined = self.dice_weight * dice + self.focal_weight * focal
        
        return combined


class WeightedCombinedLoss(nn.Module):
    """
    Class-weighted combined loss for severe imbalance.
    
    Applies per-class weights to handle rare defect types:
    - Background: Low weight (common)
    - Rare defects (Lithography, Etching): High weight (uncommon)
    
    Args:
        class_weights: Tensor of shape (num_classes,) with per-class weights
        dice_weight: Weight for Dice loss component
        focal_weight: Weight for Focal loss component
    """
    
    def __init__(
        self,
        class_weights: torch.Tensor,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted combined loss.
        
        Args:
            logits: Model predictions of shape (batch_size, num_classes, H, W)
            targets: Ground truth labels of shape (batch_size, H, W)
        
        Returns:
            Scalar loss value
        """
        # Weighted cross-entropy
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="mean")
        
        # Dice loss (already handles imbalance via overlap metric)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        # Combine
        combined = self.dice_weight * dice_loss + self.focal_weight * ce_loss
        
        return combined


def get_loss_function(config: dict) -> nn.Module:
    """
    Factory function to build loss from configuration.
    
    Args:
        config: Loss configuration with keys:
            - type: 'dice', 'focal', 'combined', 'weighted_combined'
            - dice_weight: Weight for Dice component (default: 0.5)
            - focal_weight: Weight for Focal component (default: 0.5)
            - class_weights: List of per-class weights (for weighted_combined)
    
    Returns:
        Loss function module
    
    Example:
        >>> config = {'type': 'combined', 'dice_weight': 0.6, 'focal_weight': 0.4}
        >>> loss_fn = get_loss_function(config)
    """
    loss_type = config.get("type", "combined")
    
    if loss_type == "dice":
        return DiceLoss(smooth=config.get("smooth", 1.0))
    
    elif loss_type == "focal":
        return FocalLoss(
            alpha=config.get("alpha", 0.25),
            gamma=config.get("gamma", 2.0)
        )
    
    elif loss_type == "combined":
        return CombinedLoss(
            dice_weight=config.get("dice_weight", 0.5),
            focal_weight=config.get("focal_weight", 0.5),
            smooth=config.get("smooth", 1.0),
            alpha=config.get("alpha", 0.25),
            gamma=config.get("gamma", 2.0),
        )
    
    elif loss_type == "weighted_combined":
        class_weights = torch.tensor(config["class_weights"], dtype=torch.float32)
        return WeightedCombinedLoss(
            class_weights=class_weights,
            dice_weight=config.get("dice_weight", 0.5),
            focal_weight=config.get("focal_weight", 0.5),
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    batch_size = 4
    num_classes = 8
    height, width = 300, 300
    
    # Create dummy data
    logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dice_value = dice_loss(logits, targets)
    print(f"Dice Loss: {dice_value.item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    focal_value = focal_loss(logits, targets)
    print(f"Focal Loss: {focal_value.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss(dice_weight=0.6, focal_weight=0.4)
    combined_value = combined_loss(logits, targets)
    print(f"Combined Loss: {combined_value.item():.4f}")
    
    # Test Weighted Combined Loss
    class_weights = torch.tensor([0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 1.0])  # Higher weight for rare classes
    weighted_loss = WeightedCombinedLoss(class_weights=class_weights)
    weighted_value = weighted_loss(logits, targets)
    print(f"Weighted Combined Loss: {weighted_value.item():.4f}")
    
    print("\n✅ Loss functions test passed!")
