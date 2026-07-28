"""Model architectures and factory."""

from src.models.dual_stream import UnifiedMultiStreamModel, DualStreamModel
from src.models.rgb_fft_dual_stream import RGBFFTDualStreamModel
from src.models.xception import XceptionModel
from src.models.registry import MODEL_REGISTRY


def get_model(model_name: str, config: dict):
    """
    Factory function to instantiate deepfake detection models.

    Args:
        model_name: Name of model architecture (keys from MODEL_REGISTRY)
        config: Configuration dictionary

    Returns:
        PyTorch nn.Module instance
    """
    name_clean = model_name.lower().replace("-", "_")
    
    if name_clean not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Supported choices: {list(MODEL_REGISTRY.keys())}")
        
    model_cfg = config.get('model', {})

    if name_clean in ["xception", "baseline"]:
        return XceptionModel(
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.5),
            feature_dim=model_cfg.get('spatial_feature_dim', 256) * 2
        )
    else:
        # For multi-stream models, determine default dropout based on whether it is dual-stream (0.3) or quad-stream (0.5)
        # unless overridden in model_cfg
        streams = MODEL_REGISTRY[name_clean]["streams"]
        num_streams = len(streams)
        
        default_dropout = 0.5
        if num_streams <= 2:
            default_dropout = 0.3
            
        # Create a deep copy or update config model parameters temporarily for initialization
        from copy import deepcopy
        cfg_copy = deepcopy(config)
        if 'model' not in cfg_copy:
            cfg_copy['model'] = {}
        if 'dropout' not in cfg_copy['model']:
            cfg_copy['model']['dropout'] = default_dropout
            
        return UnifiedMultiStreamModel(model_name=name_clean, config=cfg_copy)


__all__ = ['DualStreamModel', 'RGBFFTDualStreamModel', 'XceptionModel', 'UnifiedMultiStreamModel', 'get_model', 'MODEL_REGISTRY']
