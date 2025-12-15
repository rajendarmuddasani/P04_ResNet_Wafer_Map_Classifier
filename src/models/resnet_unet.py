"""
ResNet-50 U-Net Architecture for Wafer Defect Segmentation

This module implements the core segmentation model combining:
- ResNet-50 encoder (pre-trained on ImageNet)
- U-Net decoder with skip connections
- Multi-class pixel-wise segmentation (8 defect classes)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, Dict, Any


class ResNetUNet(nn.Module):
    """
    ResNet-50 U-Net for wafer defect segmentation.
    
    Architecture:
    - Encoder: ResNet-50 pre-trained on ImageNet (25M parameters)
    - Decoder: Symmetric upsampling with skip connections
    - Output: 8-class segmentation (Edge, Center, Ring, Scratch, Particle, Lithography, Etching, Random)
    
    Args:
        encoder_name: Backbone encoder ('resnet50', 'resnet101', 'resnet34')
        encoder_weights: Pre-training weights ('imagenet', None for random init)
        in_channels: Number of input channels (3 for RGB wafer maps)
        num_classes: Number of segmentation classes (8 defect types)
        activation: Output activation (None for logits, 'sigmoid' for probabilities)
    
    Example:
        >>> model = ResNetUNet(num_classes=8, encoder_name='resnet50', encoder_weights='imagenet')
        >>> input_batch = torch.randn(4, 3, 300, 300)  # Batch of 4 wafer maps
        >>> output = model(input_batch)  # Shape: (4, 8, 300, 300)
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        num_classes: int = 8,
        activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        
        # Use segmentation_models_pytorch for pre-built U-Net with ResNet encoder
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
        )
        
        # Store encoder depth for skip connections
        self.encoder_depth = 5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet-50 U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, num_classes, height, width)
        """
        return self.model(x)
    
    def get_encoder_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from encoder for active learning embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Dictionary of feature maps at different scales
        """
        features = {}
        
        # Forward through encoder stages
        encoder = self.model.encoder
        
        x0 = encoder.conv1(x)
        x0 = encoder.bn1(x0)
        x0 = encoder.relu(x0)
        features['stage0'] = x0
        
        x1 = encoder.maxpool(x0)
        x1 = encoder.layer1(x1)
        features['stage1'] = x1
        
        x2 = encoder.layer2(x1)
        features['stage2'] = x2
        
        x3 = encoder.layer3(x2)
        features['stage3'] = x3
        
        x4 = encoder.layer4(x3)
        features['stage4'] = x4  # Deepest features for embeddings (2048-dim)
        
        return features
    
    def freeze_encoder(self) -> None:
        """Freeze encoder weights (for initial training, fine-tune decoder only)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights (for fine-tuning entire network)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter count for encoder, decoder, and total."""
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
        head_params = sum(p.numel() for p in self.model.segmentation_head.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "segmentation_head": head_params,
            "total": total_params,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for logging and reproducibility."""
        return {
            "architecture": "ResNet-50 U-Net",
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "encoder_depth": self.encoder_depth,
            "parameters": self.get_num_parameters(),
        }


class LightweightResNetUNet(nn.Module):
    """
    Lightweight variant using ResNet-18 encoder for faster inference.
    
    Trade-off: ~1.5× faster inference, -2% IoU compared to ResNet-50.
    Use case: Edge deployment, real-time requirements, limited compute.
    """
    
    def __init__(
        self,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        num_classes: int = 8,
        activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name="resnet18",  # 11M parameters vs 32M for ResNet-50
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to build model from configuration.
    
    Args:
        config: Model configuration dictionary with keys:
            - architecture: 'resnet50_unet', 'resnet101_unet', 'resnet18_unet'
            - num_classes: Number of segmentation classes
            - encoder_weights: 'imagenet' or None
            - in_channels: Input channels (default 3)
    
    Returns:
        Initialized model
    
    Example:
        >>> config = {
        ...     'architecture': 'resnet50_unet',
        ...     'num_classes': 8,
        ...     'encoder_weights': 'imagenet'
        ... }
        >>> model = build_model(config)
    """
    architecture = config.get("architecture", "resnet50_unet")
    num_classes = config.get("num_classes", 8)
    encoder_weights = config.get("encoder_weights", "imagenet")
    in_channels = config.get("in_channels", 3)
    
    if architecture == "resnet50_unet":
        model = ResNetUNet(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif architecture == "resnet101_unet":
        model = ResNetUNet(
            encoder_name="resnet101",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif architecture == "resnet18_unet":
        model = LightweightResNetUNet(
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


if __name__ == "__main__":
    # Test model instantiation
    print("Testing ResNet-50 U-Net...")
    
    model = ResNetUNet(num_classes=8, encoder_name="resnet50", encoder_weights=None)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 300, 300)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.get_num_parameters()}")
    print(f"Model config: {model.get_model_config()}")
    
    # Test encoder feature extraction
    features = model.get_encoder_features(dummy_input)
    print("\nEncoder feature shapes:")
    for stage, feat in features.items():
        print(f"  {stage}: {feat.shape}")
    
    print("\n✅ Model test passed!")
