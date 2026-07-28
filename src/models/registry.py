"""Central model registry for deepfake detection models."""

MODEL_REGISTRY = {
    # Baseline model
    "xception": {
        "streams": ["face_spatial"],
        "backbone_types": {"face_spatial": "xception"},
        "feature_dims": {"face_spatial": 512},  # 512 output after linear projection or timm pooling
    },
    # Spatial Single Stream models
    "face_spatial": {
        "streams": ["face_spatial"],
        "backbone_types": {"face_spatial": "resnet18"},
        "feature_dims": {"face_spatial": 256},
    },
    "frame_spatial": {
        "streams": ["frame_spatial"],
        "backbone_types": {"frame_spatial": "resnet18"},
        "feature_dims": {"frame_spatial": 256},
    },
    # Frequency Single Stream models
    "face_frequency": {
        "streams": ["face_frequency"],
        "backbone_types": {"face_frequency": "resnet50"},
        "feature_dims": {"face_frequency": 256},
    },
    "frame_frequency": {
        "streams": ["frame_frequency"],
        "backbone_types": {"frame_frequency": "resnet50"},
        "feature_dims": {"frame_frequency": 256},
    },
    # Dual Stream / Ablation variants
    "face_only": {
        "streams": ["face_spatial", "face_frequency"],
        "backbone_types": {"face_spatial": "resnet18", "face_frequency": "resnet50"},
        "feature_dims": {"face_spatial": 256, "face_frequency": 256},
    },
    "frame_only": {
        "streams": ["frame_spatial", "frame_frequency"],
        "backbone_types": {"frame_spatial": "resnet18", "frame_frequency": "resnet50"},
        "feature_dims": {"frame_spatial": 256, "frame_frequency": 256},
    },
    "spatial_only": {
        "streams": ["face_spatial", "frame_spatial"],
        "backbone_types": {"face_spatial": "resnet18", "frame_spatial": "resnet18"},
        "feature_dims": {"face_spatial": 256, "frame_spatial": 256},
    },
    "frequency_only": {
        "streams": ["face_frequency", "frame_frequency"],
        "backbone_types": {"face_frequency": "resnet50", "frame_frequency": "resnet50"},
        "feature_dims": {"face_frequency": 256, "frame_frequency": 256},
    },
    # Alias mapping for legacy names
    "rgb_fft_dual_stream": {
        "streams": ["face_spatial", "face_frequency"],
        "backbone_types": {"face_spatial": "resnet18", "face_frequency": "resnet50"},
        "feature_dims": {"face_spatial": 256, "face_frequency": 256},
    },
    "two_stream": {
        "streams": ["face_spatial", "face_frequency"],
        "backbone_types": {"face_spatial": "resnet18", "face_frequency": "resnet50"},
        "feature_dims": {"face_spatial": 256, "face_frequency": 256},
    },
    # Proposed complete Quad-Stream model
    "quad_stream": {
        "streams": ["face_spatial", "face_frequency", "frame_spatial", "frame_frequency"],
        "backbone_types": {
            "face_spatial": "resnet18",
            "face_frequency": "resnet50",
            "frame_spatial": "resnet18",
            "frame_frequency": "resnet50"
        },
        "feature_dims": {
            "face_spatial": 256,
            "face_frequency": 256,
            "frame_spatial": 256,
            "frame_frequency": 256
        },
    }
}
