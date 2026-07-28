"""RGB+FFT Dual-Stream deepfake detection model architecture (2 streams: 1 Spatial RGB + 1 Frequency FFT)."""

import torch
import torch.nn as nn
from typing import Optional

from src.models.dual_stream import SpatialStream, FrequencyStream


class RGBFFTDualStreamModel(nn.Module):
    """
    2-stream deepfake detection model combining Spatial RGB domain and Frequency FFT domain.
    """

    def __init__(
        self,
        spatial_backbone: str = "resnet18",
        spatial_feature_dim: int = 256,
        frequency_channels: int = 1,
        fusion_dim: int = 512,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Initialize RGB+FFT Dual-Stream model.

        Args:
            spatial_backbone: Backbone for spatial stream ('resnet18' or 'efficientnet_b0')
            spatial_feature_dim: Output dimension of each stream (256)
            frequency_channels: Input channels for frequency stream (1 for magnitude)
            fusion_dim: Intermediate fusion dimension
            dropout: Dropout probability
            pretrained: Whether to use pretrained weights for spatial stream
        """
        super(RGBFFTDualStreamModel, self).__init__()

        # Spatial Stream (RGB domain)
        self.spatial_stream = SpatialStream(
            backbone_name=spatial_backbone,
            feature_dim=spatial_feature_dim,
            pretrained=pretrained
        )

        # Frequency Stream (FFT log-magnitude domain)
        self.frequency_stream = FrequencyStream(
            input_channels=frequency_channels,
            feature_dim=spatial_feature_dim,
            pretrained=False
        )

        # Fusion & Classification (2 streams = spatial_feature_dim * 2)
        self.fusion = nn.Sequential(
            nn.Linear(spatial_feature_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
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

    def forward(
        self,
        face_spatial: torch.Tensor,
        face_frequency: torch.Tensor,
        frame_spatial: Optional[torch.Tensor] = None,
        frame_frequency: Optional[torch.Tensor] = None,
        spatial: Optional[torch.Tensor] = None,
        frequency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for RGB+FFT Dual-Stream model.
        """
        rgb_input = face_spatial if face_spatial is not None else spatial
        fft_input = face_frequency if face_frequency is not None else frequency

        spatial_feats = self.spatial_stream(rgb_input)
        freq_feats = self.frequency_stream(fft_input)

        fused = torch.cat([spatial_feats, freq_feats], dim=1)
        fused_features = self.fusion(fused)
        output = self.classifier(fused_features)

        return output
