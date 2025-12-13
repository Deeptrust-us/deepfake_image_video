# Quad-Stream Architecture

## Overview

The model uses a **quad-stream** architecture that analyzes videos through four complementary streams:

1. **Face crop RGB** - Cropped and aligned face image
2. **Face crop frequency** - FFT of face crop
3. **Whole frame RGB** - Complete frame resized to 224×224
4. **Whole frame frequency** - FFT of whole frame

This architecture combines spatial and frequency domain analysis for both facial details and contextual scene information.

## Architecture Details

### Input Streams

```
Input:
  - Face RGB (3, 224, 224) → face_spatial_stream → (256,)
  - Face Frequency (1, 224, 224) → face_frequency_stream → (256,)
  - Frame RGB (3, 224, 224) → frame_spatial_stream → (256,)
  - Frame Frequency (1, 224, 224) → frame_frequency_stream → (256,)
```

### Feature Extraction

Each stream uses a separate backbone network:

- **Face Spatial Stream**: ResNet18 or EfficientNet-B0 (pretrained on ImageNet)
- **Face Frequency Stream**: ResNet50 (modified first layer for 1-channel input)
- **Frame Spatial Stream**: ResNet18 or EfficientNet-B0 (pretrained on ImageNet)
- **Frame Frequency Stream**: ResNet50 (modified first layer for 1-channel input)

### Fusion & Classification

```
Fusion:
  - Concatenate: (256 × 4) = (1024,)
  - Fusion layers: (1024,) → (512,) → (256,)
  - Dropout: 0.6
  - Classifier: (256,) → (1,)
```

## Implementation Details

### Dataset (`src/data/dataset.py`)

The dataset returns four inputs per sample:

- `face_spatial`: Face crop RGB (3, 224, 224)
- `face_frequency`: Face crop frequency (1 or 2, 224, 224)
- `frame_spatial`: Whole frame RGB (3, 224, 224)
- `frame_frequency`: Whole frame frequency (1 or 2, 224, 224)

**Key changes:**
- `__getitem__` returns 4 inputs instead of 2
- `_is_valid_sample` checks for both face and frame directories
- `_compute_frequency_stats` computes stats from both face and frame images

### Model (`src/models/dual_stream.py`)

The model (still named `DualStreamModel` for compatibility) implements four streams:

- `face_spatial_stream`: ResNet18/EfficientNet for face RGB
- `face_frequency_stream`: ResNet50 for face frequency
- `frame_spatial_stream`: ResNet18/EfficientNet for frame RGB
- `frame_frequency_stream`: ResNet50 for frame frequency

**Key changes:**
- Four separate stream backbones
- Fusion layer concatenates all 4 feature vectors (1024 features total)
- Forward pass accepts 4 inputs: `forward(face_spatial, face_frequency, frame_spatial, frame_frequency)`

### Training (`train.py`)

Training loop updated to handle four inputs:

```python
face_spatial, face_frequency, frame_spatial, frame_frequency, label = batch
output = model(face_spatial, face_frequency, frame_spatial, frame_frequency)
```

**Key changes:**
- Batch unpacking handles 4 inputs
- Validation loop updated similarly
- Backbone freezing handles all 4 streams

### Preprocessing (`src/data/preprocessing.py`)

Preprocessing saves both face crops and whole frames:

- **Face crops**: Detected, aligned, and cropped to 224×224 (saved to `data/faces/`)
- **Whole frames**: Resized to 224×224 (saved to `data/frames/`)

**Key changes:**
- `process_video` saves resized whole frames alongside face crops
- Whole frames are resized maintaining aspect ratio (may distort)
- Frequency spectra computed on-the-fly during dataset loading

### Evaluation (`evaluate.py`)

Evaluation script updated to use 4-input model format.

### Example Usage (`example_usage.py`)

Example script demonstrates 4-input inference with visualization for all 4 inputs.

## Benefits

1. **More Context**: Whole frames provide background and context information
2. **Complementary Information**: Face crops focus on facial features, frames provide scene context
3. **Better Detection**: Can detect artifacts in both face region and surrounding areas
4. **Robustness**: Less dependent on perfect face detection
5. **Frequency Analysis**: Both face and frame frequency streams capture different artifacts

## Data Requirements

The dataset must contain:

- `data/faces/{video_id}/` - Face crops (224×224)
- `data/frames/{video_id}/` - Whole frames resized to 224×224

Both directories should contain matching frame files (e.g., `frame_0001.jpg`).

## Model Parameters

The quad-stream model has approximately **4× more parameters** in the feature extraction stage compared to the original dual-stream model:

- **Original dual-stream**: ~12M parameters
- **Quad-stream**: ~48M parameters (4 streams × ~12M each)

However, the fusion layers remain similar in size.

## Usage

The model interface requires 4 inputs:

```python
import torch
from src.models.dual_stream import DualStreamModel

model = DualStreamModel(...)
model.eval()

# Prepare inputs
face_spatial = torch.randn(1, 3, 224, 224)
face_frequency = torch.randn(1, 1, 224, 224)
frame_spatial = torch.randn(1, 3, 224, 224)
frame_frequency = torch.randn(1, 1, 224, 224)

# Forward pass
output = model(face_spatial, face_frequency, frame_spatial, frame_frequency)
prediction = torch.sigmoid(output)
```

## Configuration

In `config.yaml`:

```yaml
model:
  spatial_backbone: "resnet18"  # or "efficientnet_b0"
  spatial_feature_dim: 256
  frequency_channels: 1  # 1 for magnitude only, 2 for magnitude+phase
  fusion_dim: 512
  dropout: 0.6
```

## Notes

- Whole frames are resized to 224×224 (may distort aspect ratio)
- Frequency spectra are computed on-the-fly during dataset loading
- All 4 streams use the same augmentation (consistent transformations)
- Model uses more GPU memory due to 4× feature extraction
- Training time increases proportionally with model size

## Migration from Dual-Stream

If migrating from a dual-stream model:

1. Ensure preprocessing generates both `data/faces/` and `data/frames/` directories
2. Update any custom scripts to handle 4 inputs instead of 2
3. Retrain the model (pretrained dual-stream weights are not compatible)
4. Update evaluation scripts to use 4-input format
