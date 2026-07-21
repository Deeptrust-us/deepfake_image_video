"""Xception baseline model architecture for deepfake detection."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class XceptionModel(nn.Module):
    """
    Xception deepfake detection baseline model.
    Operates on spatial RGB images (face crops or full frames).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.5, feature_dim: int = 512):
        """
        Initialize Xception baseline.

        Args:
            pretrained: Whether to load pretrained weights
            dropout: Dropout probability
            feature_dim: Feature dimension before classifier
        """
        super(XceptionModel, self).__init__()

        self.backbone_type = "timm"
        try:
            import timm
            # Use timm Xception model
            self.backbone = timm.create_model('legacy_xception', pretrained=pretrained, num_classes=0)
            backbone_out_dim = self.backbone.num_features
        except Exception:
            # Fallback to ResNet50 or EfficientNet if timm not available
            self.backbone_type = "resnet50"
            resnet = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_out_dim = 2048

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        self._initialize_head()

    def _initialize_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        face_spatial: torch.Tensor,
        face_frequency: Optional[torch.Tensor] = None,
        frame_spatial: Optional[torch.Tensor] = None,
        frame_frequency: Optional[torch.Tensor] = None,
        spatial: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Xception baseline.
        Accepts face_spatial or spatial parameter.
        """
        x = face_spatial if face_spatial is not None else spatial
        if x is None and frame_spatial is not None:
            x = frame_spatial

        if self.backbone_type == "timm":
            features = self.backbone(x)
        else:
            features = self.backbone(x)
            features = features.view(features.size(0), -1)

        output = self.classifier(features)
        return output
