"""Modular Multi-Stream deepfake detection model architecture."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List


class FrequencyStream(nn.Module):
    """Frequency domain stream using ResNet50 backbone."""
    
    def __init__(self, input_channels: int = 1, feature_dim: int = 256, 
                 pretrained: bool = False):
        """
        Initialize frequency stream with ResNet50 backbone.
        
        Args:
            input_channels: Number of input channels (1 for magnitude, 3 for replicated magnitude)
            feature_dim: Output feature dimension
            pretrained: Whether to use pretrained weights
        """
        super(FrequencyStream, self).__init__()
        
        # Load ResNet50 backbone (frequency stream is typically trained from scratch)
        backbone = models.resnet50(pretrained=pretrained)
        
        # Replace first conv layer to accept input_channels instead of 3 if input_channels != 3
        if input_channels != 3:
            backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        
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
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial stream.
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class UnifiedMultiStreamModel(nn.Module):
    """
    Unified multi-stream model supporting arbitrary subsets of streams:
    - face_spatial (RGB face)
    - face_frequency (FFT face)
    - frame_spatial (RGB full frame)
    - frame_frequency (FFT full-frame)
    """
    
    def __init__(self, model_name: str, config: dict):
        """
        Initialize multi-stream model.
        """
        super(UnifiedMultiStreamModel, self).__init__()
        from src.models.registry import MODEL_REGISTRY
        
        name_clean = model_name.lower().replace("-", "_")
        if name_clean not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model name: {model_name}")
            
        self.model_info = MODEL_REGISTRY[name_clean]
        self.active_streams = self.model_info["streams"]
        
        model_cfg = config.get('model', {})
        spatial_backbone = model_cfg.get('spatial_backbone', 'resnet18')
        spatial_feature_dim = model_cfg.get('spatial_feature_dim', 256)
        frequency_channels = model_cfg.get('frequency_channels', 1)
        fusion_dim = model_cfg.get('fusion_dim', 512)
        dropout = model_cfg.get('dropout', 0.5)
        pretrained = model_cfg.get('pretrained', True)
        
        # Face streams
        self.face_spatial_stream = None
        if 'face_spatial' in self.active_streams:
            self.face_spatial_stream = SpatialStream(
                backbone_name=spatial_backbone,
                feature_dim=spatial_feature_dim,
                pretrained=pretrained
            )
            
        self.face_frequency_stream = None
        if 'face_frequency' in self.active_streams:
            self.face_frequency_stream = FrequencyStream(
                input_channels=frequency_channels,
                feature_dim=spatial_feature_dim,
                pretrained=False  # Trained from scratch
            )
            
        # Frame streams
        self.frame_spatial_stream = None
        if 'frame_spatial' in self.active_streams:
            self.frame_spatial_stream = SpatialStream(
                backbone_name=spatial_backbone,
                feature_dim=spatial_feature_dim,
                pretrained=pretrained
            )
            
        self.frame_frequency_stream = None
        if 'frame_frequency' in self.active_streams:
            self.frame_frequency_stream = FrequencyStream(
                input_channels=frequency_channels,
                feature_dim=spatial_feature_dim,
                pretrained=False  # Trained from scratch
            )
            
        num_active = len(self.active_streams)
        
        if num_active > 1:
            self.fusion = nn.Sequential(
                nn.Linear(spatial_feature_dim * num_active, fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim // 2, 1)
            )
        else:
            self.fusion = None
            self.classifier = nn.Sequential(
                nn.Linear(spatial_feature_dim, 1)
            )
            
        self._initialize_weights()
        
    def _initialize_weights(self):
        if self.fusion is not None:
            for module in self.fusion.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                        
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                    
    def forward(self, face_spatial: torch.Tensor, face_frequency: torch.Tensor,
                frame_spatial: torch.Tensor, frame_frequency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through active streams.
        """
        features = []
        if self.face_spatial_stream is not None:
            features.append(self.face_spatial_stream(face_spatial))
        if self.face_frequency_stream is not None:
            features.append(self.face_frequency_stream(face_frequency))
        if self.frame_spatial_stream is not None:
            features.append(self.frame_spatial_stream(frame_spatial))
        if self.frame_frequency_stream is not None:
            features.append(self.frame_frequency_stream(frame_frequency))
            
        fused = torch.cat(features, dim=1)
        if self.fusion is not None:
            fused = self.fusion(fused)
        output = self.classifier(fused)
        return output


class DualStreamModel(UnifiedMultiStreamModel):
    """Backward compatibility alias for UnifiedMultiStreamModel configured as full quad_stream."""
    
    def __init__(self, **kwargs):
        config = {
            'model': {
                'spatial_backbone': kwargs.get('spatial_backbone', 'resnet18'),
                'spatial_feature_dim': kwargs.get('spatial_feature_dim', 256),
                'frequency_channels': kwargs.get('frequency_channels', 1),
                'fusion_dim': kwargs.get('fusion_dim', 512),
                'dropout': kwargs.get('dropout', 0.5),
                'pretrained': kwargs.get('pretrained', True)
            }
        }
        super(DualStreamModel, self).__init__(model_name="quad_stream", config=config)
