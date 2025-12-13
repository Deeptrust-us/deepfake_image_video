# Quad-Stream Model Changes

## Overview
The model has been upgraded from a dual-stream (face RGB + face frequency) to a **quad-stream** model that uses:
1. **Face crop RGB** - Cropped and aligned face image
2. **Face crop frequency** - FFT of face crop
3. **Whole frame RGB** - Complete frame resized to 224x224
4. **Whole frame frequency** - FFT of whole frame

## Changes Made

### 1. Dataset (`src/data/dataset.py`)
- Modified `__getitem__` to return 4 inputs instead of 2:
  - `face_spatial`: Face crop RGB (3, 224, 224)
  - `face_frequency`: Face crop frequency (1 or 2, 224, 224)
  - `frame_spatial`: Whole frame RGB (3, 224, 224)
  - `frame_frequency`: Whole frame frequency (1 or 2, 224, 224)
- Updated `_is_valid_sample` to check for both face and frame directories
- Updated `_compute_frequency_stats` to compute stats from both face and frame images

### 2. Model (`src/models/dual_stream.py`)
- Renamed conceptually to "Quad-Stream Model" (still called DualStreamModel for compatibility)
- Added 4 separate streams:
  - `face_spatial_stream`: ResNet18/EfficientNet for face RGB
  - `face_frequency_stream`: ResNet50 for face frequency
  - `frame_spatial_stream`: ResNet18/EfficientNet for frame RGB
  - `frame_frequency_stream`: ResNet50 for frame frequency
- Updated fusion layer to concatenate all 4 feature vectors (4 × 256 = 1024 features)
- Updated forward pass to accept 4 inputs

### 3. Training (`train.py`)
- Updated training loop to unpack 4 inputs from batch
- Updated validation loop to unpack 4 inputs from batch
- Updated backbone freezing to handle all 4 streams

### 4. Preprocessing (`src/data/preprocessing.py`)
- Modified `process_video` to save resized whole frames (224x224) alongside face crops
- Whole frames are resized and saved in RGB format

### 5. Evaluation (`evaluate.py`)
- Updated to use 4-input model format

### 6. Example Usage (`example_usage.py`)
- Updated to demonstrate 4-input inference
- Added visualization for all 4 inputs

## Model Architecture

```
Input:
  - Face RGB (3, 224, 224) → face_spatial_stream → (256,)
  - Face Frequency (1, 224, 224) → face_frequency_stream → (256,)
  - Frame RGB (3, 224, 224) → frame_spatial_stream → (256,)
  - Frame Frequency (1, 224, 224) → frame_frequency_stream → (256,)

Fusion:
  - Concatenate: (256 × 4) = (1024,)
  - Fusion layers: (1024,) → (512,) → (256,)
  - Classifier: (256,) → (1,)
```

## Benefits

1. **More Context**: Whole frames provide background and context information
2. **Complementary Information**: Face crops focus on facial features, frames provide scene context
3. **Better Detection**: Can detect artifacts in both face region and surrounding areas
4. **Robustness**: Less dependent on perfect face detection

## Usage

The model interface remains similar, but now requires 4 inputs:

```python
output = model(face_spatial, face_frequency, frame_spatial, frame_frequency)
```

## Data Requirements

- `data/faces/{video_id}/` - Face crops (already exists)
- `data/frames/{video_id}/` - Whole frames resized to 224x224 (created during preprocessing)

## Notes

- Whole frames are resized to 224x224 (may distort aspect ratio)
- Frequency spectra are computed on-the-fly during dataset loading
- All 4 streams use the same augmentation (consistent transformations)
- Model now has ~4× more parameters in the feature extraction stage

