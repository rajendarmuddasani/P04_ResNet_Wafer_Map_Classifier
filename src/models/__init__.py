"""
Model Package Initialization

Exports key components for easy imports:
- ResNetUNet: Core segmentation model
- Loss functions: DiceLoss, FocalLoss, CombinedLoss
- Metrics: SegmentationMetrics, PRDMetricsValidator
- ONNX inference: ONNXSegmentationInference
"""

from src.models.resnet_unet import (
    ResNetUNet,
    LightweightResNetUNet,
    build_model,
)

from src.models.losses import (
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    WeightedCombinedLoss,
    get_loss_function,
)

from src.models.metrics import (
    SegmentationMetrics,
    PRDMetricsValidator,
    compute_batch_iou,
    compute_batch_dice,
)

from src.models.onnx_inference import (
    ONNXSegmentationInference,
    export_pytorch_to_onnx,
)

__all__ = [
    # Models
    "ResNetUNet",
    "LightweightResNetUNet",
    "build_model",
    # Losses
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "WeightedCombinedLoss",
    "get_loss_function",
    # Metrics
    "SegmentationMetrics",
    "PRDMetricsValidator",
    "compute_batch_iou",
    "compute_batch_dice",
    # Inference
    "ONNXSegmentationInference",
    "export_pytorch_to_onnx",
]
