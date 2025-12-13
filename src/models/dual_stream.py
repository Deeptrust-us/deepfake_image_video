"""Dual-stream deepfake detection model architecture."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class FrequencyStream(nn.Module):
    """Frequency domain stream using ResNet50 backbone."""
    
    def __init__(self, input_channels: int = 1, feature_dim: int = 256, 
                 pretrained: bool = True):
        """
        Initialize frequency stream with ResNet50 backbone.
        
        Args:
            input_channels: Number of input channels (1 for magnitude, 2 for magnitude+phase)
            feature_dim: Output feature dimension
            pretrained: Whether to use pretrained weights (may not be optimal for frequency domain)
        """
        super(FrequencyStream, self).__init__()
        
        # Load ResNet50 backbone
        # For frequency domain, pretrained ImageNet weights may not be optimal
        # Use pretrained=False for frequency stream, or use pretrained=True but reinitialize first layer
        backbone = models.resnet50(pretrained=pretrained)
        
        # Replace first conv layer to accept input_channels instead of 3
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        backbone.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize first conv layer with proper initialization for frequency domain
        # Use Kaiming initialization (He initialization) which works well for ReLU activations
        nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # If using pretrained weights, we could try to adapt them, but for frequency domain
        # it's better to start fresh or use a lower learning rate for pretrained layers
        
        # Remove final fully connected layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # ResNet50 outputs 2048 features
        backbone_dim = 2048
        self.fc = nn.Linear(backbone_dim, feature_dim)
        
        # Initialize FC layer
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through frequency stream.
        
        Args:
            x: Input frequency spectrum (B, C, H, W)
            
        Returns:
            Feature vector (B, feature_dim)
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SpatialStream(nn.Module):
    """Spatial domain stream using pretrained backbone."""
    
    def __init__(self, backbone_name: str = "resnet18", feature_dim: int = 256, 
                 pretrained: bool = True):
        """
        Initialize spatial stream.
        
        Args:
            backbone_name: Name of backbone architecture ('resnet18' or 'efficientnet_b0')
            feature_dim: Output feature dimension
            pretrained: Whether to use pretrained weights
        """
        super(SpatialStream, self).__init__()
        
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            # Remove final fully connected layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 512
        elif backbone_name == "efficientnet_b0":
            from torchvision.models import efficientnet_b0
            backbone = efficientnet_b0(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.fc = nn.Linear(backbone_dim, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial stream.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            Feature vector (B, feature_dim)
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DualStreamModel(nn.Module):
    """Quad-stream deepfake detection model (face + frame, each with RGB + frequency)."""
    
    def __init__(self, spatial_backbone: str = "resnet18", 
                 spatial_feature_dim: int = 256,
                 frequency_channels: int = 1,
                 fusion_dim: int = 512,
                 dropout: float = 0.5,
                 pretrained: bool = True,
                 use_attention: bool = False):
        """
        Initialize quad-stream model.
        
        Args:
            spatial_backbone: Backbone for spatial stream
            spatial_feature_dim: Feature dimension for spatial stream
            frequency_channels: Input channels for frequency stream (1 for magnitude, 2 for magnitude+phase)
            fusion_dim: Dimension after fusion
            dropout: Dropout probability
            pretrained: Whether to use pretrained weights for both streams
            use_attention: Whether to use attention-based fusion
        """
        super(DualStreamModel, self).__init__()
        
        # Face streams
        self.face_spatial_stream = SpatialStream(
            backbone_name=spatial_backbone,
            feature_dim=spatial_feature_dim,
            pretrained=pretrained
        )
        
        self.face_frequency_stream = FrequencyStream(
            input_channels=frequency_channels,
            feature_dim=spatial_feature_dim,
            pretrained=False  # Don't use pretrained weights for frequency domain
        )
        
        # Frame streams (whole frame)
        self.frame_spatial_stream = SpatialStream(
            backbone_name=spatial_backbone,
            feature_dim=spatial_feature_dim,
            pretrained=pretrained
        )
        
        self.frame_frequency_stream = FrequencyStream(
            input_channels=frequency_channels,
            feature_dim=spatial_feature_dim,
            pretrained=False  # Don't use pretrained weights for frequency domain
        )
        
        self.use_attention = use_attention
        
        if use_attention:
            # Attention-based fusion for 4 streams
            self.attention_weights = nn.Linear(spatial_feature_dim * 4, 4)
        
        # Fusion layers - now fusing 4 streams
        self.fusion = nn.Sequential(
            nn.Linear(spatial_feature_dim * 4, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize fusion and classifier layers
        self._initialize_fusion_layers()
    
    def _initialize_fusion_layers(self):
        """Initialize fusion and classifier layers with proper weights."""
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Initialize classifier with smaller weights to start near 0.5 (random guess)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # Start near 0.5 after sigmoid
    
    def forward(self, face_spatial: torch.Tensor, face_frequency: torch.Tensor,
                frame_spatial: torch.Tensor, frame_frequency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quad-stream model.
        
        Args:
            face_spatial: Face crop RGB image (B, 3, H, W)
            face_frequency: Face crop frequency spectrum (B, C, H, W)
            frame_spatial: Whole frame RGB image (B, 3, H, W)
            frame_frequency: Whole frame frequency spectrum (B, C, H, W)
            
        Returns:
            Binary classification logits (B, 1)
        """
        # Extract features from all 4 streams
        face_spatial_features = self.face_spatial_stream(face_spatial)
        face_frequency_features = self.face_frequency_stream(face_frequency)
        frame_spatial_features = self.frame_spatial_stream(frame_spatial)
        frame_frequency_features = self.frame_frequency_stream(frame_frequency)
        
        # Fusion
        if self.use_attention:
            # Compute attention weights for 4 streams
            concat_features = torch.cat([
                face_spatial_features, 
                face_frequency_features,
                frame_spatial_features,
                frame_frequency_features
            ], dim=1)
            attention_logits = self.attention_weights(concat_features)
            attention_weights = torch.softmax(attention_logits, dim=1)
            
            # Apply attention
            face_spatial_attended = face_spatial_features * attention_weights[:, 0:1]
            face_frequency_attended = face_frequency_features * attention_weights[:, 1:2]
            frame_spatial_attended = frame_spatial_features * attention_weights[:, 2:3]
            frame_frequency_attended = frame_frequency_features * attention_weights[:, 3:4]
            
            fused = torch.cat([
                face_spatial_attended,
                face_frequency_attended,
                frame_spatial_attended,
                frame_frequency_attended
            ], dim=1)
        else:
            # Simple concatenation of all 4 streams
            fused = torch.cat([
                face_spatial_features,
                face_frequency_features,
                frame_spatial_features,
                frame_frequency_features
            ], dim=1)
        
        # Pass through fusion layers
        fused = self.fusion(fused)
        
        # Classification
        output = self.classifier(fused)
        
        return output


