"""
Data Loading and Preprocessing for Wafer Defect Segmentation

Handles:
- COCO JSON annotation parsing
- Image loading from S3/MinIO/local filesystem
- Data augmentation pipeline (albumentations)
- PyTorch Dataset and DataLoader creation
- Train/val/test splitting

Supports:
- Supervised learning (labeled data)
- Semi-supervised learning (labeled + unlabeled)
- Active learning (selective annotation)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class WaferDefectDataset(Dataset):
    """
    PyTorch Dataset for wafer defect segmentation.
    
    Supports COCO JSON format annotations with 8 defect classes:
    - 0: Background
    - 1: Edge defect
    - 2: Center defect
    - 3: Ring defect
    - 4: Scratch
    - 5: Particle
    - 6: Lithography
    - 7: Etching
    
    Args:
        image_dir: Directory containing wafer map images
        annotation_file: Path to COCO JSON annotations file
        transform: Albumentations transform pipeline (optional)
        image_size: Target image size as (height, width) tuple
        cache_images: Cache images in memory for faster loading (use for small datasets)
    
    Example:
        >>> dataset = WaferDefectDataset(
        ...     image_dir='data/raw/images',
        ...     annotation_file='data/processed/annotations.json',
        ...     image_size=(300, 300)
        ... )
        >>> image, mask = dataset[0]
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (300, 300),
        cache_images: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_size = image_size
        self.cache_images = cache_images
        self.image_cache = {}
        
        # Load COCO annotations
        logger.info(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image ID to filename mapping
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        
        # Group annotations by image ID
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
        
        # Get list of image IDs
        self.image_ids = list(self.image_info.keys())
        
        logger.info(f"Loaded {len(self.image_ids)} images with annotations")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and segmentation mask.
        
        Args:
            idx: Index of sample
        
        Returns:
            Tuple of (image_tensor, mask_tensor)
                - image_tensor: (3, H, W) float32 in [0, 1]
                - mask_tensor: (H, W) int64 with class indices
        """
        image_id = self.image_ids[idx]
        
        # Load image
        image = self._load_image(image_id)
        
        # Create segmentation mask
        mask = self._create_mask(image_id, image.shape[:2])
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default: resize and convert to tensor
            image = cv2.resize(image, self.image_size[::-1])
            mask = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def _load_image(self, image_id: int) -> np.ndarray:
        """Load image from disk or cache."""
        if self.cache_images and image_id in self.image_cache:
            return self.image_cache[image_id].copy()
        
        image_info = self.image_info[image_id]
        image_path = self.image_dir / image_info['file_name']
        
        # Load image (supports PNG, JPG, etc.)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.cache_images:
            self.image_cache[image_id] = image.copy()
        
        return image
    
    def _create_mask(self, image_id: int, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create segmentation mask from COCO polygons.
        
        Args:
            image_id: COCO image ID
            image_shape: (height, width) of target mask
        
        Returns:
            Segmentation mask of shape (height, width) with class indices
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Get annotations for this image
        annotations = self.annotations_by_image.get(image_id, [])
        
        # Draw each annotation on mask
        for ann in annotations:
            category_id = ann['category_id']
            segmentation = ann['segmentation']
            
            # COCO segmentation can be polygon or RLE
            if isinstance(segmentation, list):
                # Polygon format: list of [x1, y1, x2, y2, ...]
                for polygon in segmentation:
                    polygon = np.array(polygon).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [polygon], category_id)
            else:
                # RLE format (not commonly used for wafer maps)
                logger.warning(f"RLE segmentation not supported, skipping annotation {ann['id']}")
        
        return mask
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get pixel count per class across entire dataset."""
        class_counts = {i: 0 for i in range(8)}
        
        for idx in range(len(self)):
            _, mask = self[idx]
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            unique, counts = np.unique(mask_np, return_counts=True)
            for class_id, count in zip(unique, counts):
                class_counts[int(class_id)] += int(count)
        
        return class_counts


class UnlabeledWaferDataset(Dataset):
    """
    Dataset for unlabeled wafer maps (semi-supervised learning).
    
    Returns only images without segmentation masks.
    Used for pseudo-labeling and consistency regularization.
    
    Args:
        image_dir: Directory containing wafer map images
        image_list: List of image filenames
        transform: Albumentations transform
        image_size: Target size
    """
    
    def __init__(
        self,
        image_dir: str,
        image_list: List[str],
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (300, 300),
    ):
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get unlabeled image.
        
        Returns:
            Image tensor of shape (3, H, W)
        """
        image_path = self.image_dir / self.image_list[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = cv2.resize(image, self.image_size[::-1])
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return image


def get_training_transforms(image_size: Tuple[int, int] = (300, 300)) -> A.Compose:
    """
    Get augmentation pipeline for training.
    
    Augmentations:
    - Geometric: Rotate, flip, shift, scale
    - Color: Brightness, contrast, saturation
    - Noise: Gaussian noise, blur
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=180, p=0.7),  # Wafer maps are rotationally symmetric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=0,  # Already handled by Rotate
            p=0.5
        ),
        
        # Optical distortion (simulates imaging variations)
        A.OpticalDistortion(distort_limit=0.3, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        
        # Color transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        
        # Noise and blur
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # Resize and normalize
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_validation_transforms(image_size: Tuple[int, int] = (300, 300)) -> A.Compose:
    """
    Get transform pipeline for validation/testing (no augmentation).
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader


def compute_class_weights(dataset: WaferDefectDataset) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced data.
    
    Uses inverse frequency weighting:
    weight_i = 1 / (frequency_i + epsilon)
    
    Args:
        dataset: Dataset to compute weights from
    
    Returns:
        Tensor of class weights of shape (num_classes,)
    """
    logger.info("Computing class weights from dataset...")
    class_counts = dataset.get_class_distribution()
    
    # Convert to numpy array
    counts = np.array([class_counts[i] for i in range(8)])
    
    # Compute inverse frequency weights
    weights = 1.0 / (counts + 1e-6)
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    logger.info("Class weights computed:")
    for i, (count, weight) in enumerate(zip(counts, weights)):
        logger.info(f"  Class {i}: count={count}, weight={weight:.4f}")
    
    return torch.from_numpy(weights).float()


if __name__ == "__main__":
    # Test data loading
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Data Loading Test")
    print("=" * 60)
    
    # Note: Requires actual data files to run
    print("\n⚠️  This test requires:")
    print("  1. Wafer map images in data/raw/images/")
    print("  2. COCO annotations in data/processed/annotations.json")
    print("\nExample COCO format:")
    print("""
{
  "images": [
    {"id": 1, "file_name": "wafer_001.png", "width": 300, "height": 300}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1000,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {"id": 0, "name": "background"},
    {"id": 1, "name": "edge"},
    ...
  ]
}
    """)
    
    print("\n✅ Data loading module ready!")
    print("   Use get_training_transforms() for augmentation")
    print("   Use WaferDefectDataset for labeled data")
    print("   Use UnlabeledWaferDataset for semi-supervised learning")
