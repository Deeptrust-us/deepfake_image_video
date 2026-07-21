"""Model architectures and factory."""

from src.models.dual_stream import DualStreamModel
from src.models.rgb_fft_dual_stream import RGBFFTDualStreamModel
from src.models.xception import XceptionModel


def get_model(model_name: str, config: dict):
    """
    Factory function to instantiate deepfake detection models.

    Args:
        model_name: Name of model architecture ('xception', 'rgb_fft_dual_stream', 'two_stream', 'quad_stream')
        config: Configuration dictionary

    Returns:
        PyTorch nn.Module instance
    """
    name = model_name.lower().replace("-", "_")
    model_cfg = config.get('model', {})

    if name in ["xception", "baseline"]:
        return XceptionModel(
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.5),
            feature_dim=model_cfg.get('spatial_feature_dim', 256) * 2
        )
    elif name in ["rgb_fft_dual_stream", "rgb_fft", "two_stream"]:
        return RGBFFTDualStreamModel(
            spatial_backbone=model_cfg.get('spatial_backbone', 'resnet18'),
            spatial_feature_dim=model_cfg.get('spatial_feature_dim', 256),
            frequency_channels=model_cfg.get('frequency_channels', 1),
            fusion_dim=model_cfg.get('fusion_dim', 512),
            dropout=model_cfg.get('dropout', 0.5),
            pretrained=model_cfg.get('pretrained', True)
        )
    elif name in ["quad_stream", "quad", "dual_stream"]:
        return DualStreamModel(
            spatial_backbone=model_cfg.get('spatial_backbone', 'resnet18'),
            spatial_feature_dim=model_cfg.get('spatial_feature_dim', 256),
            frequency_channels=model_cfg.get('frequency_channels', 1),
            fusion_dim=model_cfg.get('fusion_dim', 512),
            dropout=model_cfg.get('dropout', 0.6),
            pretrained=model_cfg.get('pretrained', True)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported choices: 'xception', 'rgb_fft_dual_stream', 'quad_stream'")


__all__ = ['DualStreamModel', 'RGBFFTDualStreamModel', 'XceptionModel', 'get_model']
