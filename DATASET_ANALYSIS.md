# Dataset Analysis

This document provides detailed statistics and analysis of the processed dataset.

## Dataset Overview

### Video to Frame Conversion Process

#### 1. Frame Extraction
- **Method**: Uses `ffmpeg` to extract frames from videos
- **Sampling Rate**: **3 frames per second** (configured in `config.yaml` as `frame_sampling_rate: 3`)
- **Output Format**: JPG images named `frame_0001.jpg`, `frame_0002.jpg`, etc.
- **Location**: Frames saved to `data/frames/{video_id}/`

#### 2. Face Detection & Cropping
- **Detector**: MTCNN face detector
- **Process**: 
  - Each extracted frame → Face detection → Face alignment → Crop to 224×224
- **Output**: Face crops saved to `data/faces/{video_id}/`
- **Format**: Same naming convention (`frame_0001.jpg`, etc.)

#### 3. Frequency Domain Processing
- **Process**: Each face crop and whole frame → FFT (Fast Fourier Transform) → Frequency spectrum
- **Output**: Frequency representations computed on-the-fly during dataset loading
- **Format**: NumPy arrays (`.npy` files) or computed in memory

## Dataset Statistics

### Video Distribution

**Total Videos**: 6,528 videos (Celeb-DF v2)

- **Training**: 4,569 videos (70%)
- **Validation**: 978 videos (15%)
- **Test**: 981 videos (15%)

### Class Distribution

**Training Set**:
- Real videos: 622 (13.6%)
- Fake videos: 3,947 (86.4%)
- **Ratio**: 6.3:1 (highly imbalanced)

**Validation Set**:
- Real videos: 133 (13.6%)
- Fake videos: 845 (86.4%)
- **Ratio**: 6.3:1

**Test Set**:
- Real videos: 134 (13.7%)
- Fake videos: 847 (86.3%)
- **Ratio**: 6.3:1

**Note**: The class imbalance is intentional and expected for Celeb-DF v2. The model uses Focal Loss and oversampling to handle this imbalance.

### Frame Statistics

- **Average frames per video**: ~38 frames
- **Range**: 11-74 frames per video
- **Total frames**:
  - Training: **173,257 frames**
  - Validation: **37,189 frames**
  - Test: **37,503 frames**
  - **Grand Total: ~247,949 frames**

### Frame Extraction Details

- Videos are sampled at **3 FPS** (3 frames per second)
- For a typical 10-second video → ~30 frames
- For a typical 12-second video → ~36 frames
- Frame count varies based on video length

## Training Process

### Frame Selection During Training

#### Training Mode (`is_training=True`)
- **Random frame selection**: Each epoch, a random frame is selected from each video
- This provides data augmentation through temporal diversity
- Same video can contribute different frames in different epochs
- Increases effective dataset size

#### Validation/Test Mode (`is_training=False`)
- **Fixed frame selection**: Always uses the first frame (`frame_0001.jpg`)
- Ensures consistent evaluation
- Reproducible results

### Data Flow

```
Video (MP4)
  ↓ [ffmpeg @ 3 FPS]
Raw Frames (data/frames/{video_id}/)
  ↓ [MTCNN Face Detection]
Face Crops (data/faces/{video_id}/) ← Used for training
  ↓ [FFT]
Frequency Spectra (computed on-the-fly) ← Used for training
```

### During Training

- Each batch contains frames from different videos
- Each sample = 1 frame from 1 video
- **Batch size**: 32 (from config)
- **Total training samples per epoch**: 4,569 (one random frame per video)
- **Effective samples**: Much higher due to random frame selection across epochs

## Directory Structure

```
data/
├── frames/           # Raw extracted frames (whole frames, 224×224)
│   ├── video_0001/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── ...
├── faces/            # Face crops (USED FOR TRAINING)
│   ├── video_0001/
│   │   ├── frame_0001.jpg  (224×224 face crop)
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── ...
├── frequency/        # Frequency domain representations (optional, computed on-the-fly)
│   ├── video_0001/
│   │   ├── frame_0001.npy
│   │   └── ...
│   └── ...
└── *_metadata.json   # Video metadata (train/val/test)
```

## Key Points

1. **Frame-level training**: Model trains on individual frames, not entire videos
2. **Temporal diversity**: Random frame selection during training increases diversity
3. **Quad-stream input**: Each sample provides:
   - **Face spatial stream**: Face crop image (RGB, 224×224)
   - **Face frequency stream**: FFT magnitude spectrum of face crop (1 channel, 224×224)
   - **Frame spatial stream**: Whole frame image (RGB, 224×224)
   - **Frame frequency stream**: FFT magnitude spectrum of whole frame (1 channel, 224×224)
4. **Class imbalance**: Strong bias toward fake videos (86% fake, 14% real)
5. **Dataset size**: ~248K frames total, but training uses 4,569 videos with random frame selection

## Configuration

From `config.yaml`:

```yaml
data:
  frame_sampling_rate: 3  # 3 frames per second
  face_size: 224          # Face crops are 224×224 pixels
  frequency_channels: 1   # Only magnitude spectrum (not phase)
  train_split: 0.7        # 70% training
  val_split: 0.15         # 15% validation
  test_split: 0.15        # 15% test

training:
  batch_size: 32          # 32 frames per batch
```

## Verification

To verify your dataset:

```bash
# Count videos
ls data/faces/ | wc -l

# Check frames in a video
ls data/faces/video_0001/ | wc -l

# View metadata
python3 -c "import json; data=json.load(open('data/train_metadata.json')); print(f'Training videos: {len(data)}')"

# Check class distribution
python3 -c "import json; data=json.load(open('data/train_metadata.json')); real=sum(1 for v in data.values() if v['label']==0); fake=sum(1 for v in data.values() if v['label']==1); print(f'Real: {real}, Fake: {fake}, Ratio: {fake/real:.1f}:1')"
```

## Handling Class Imbalance

The dataset has a severe class imbalance (6.3:1 ratio). The model addresses this through:

1. **Focal Loss**: Focuses learning on hard examples
2. **Oversampling**: Real videos are oversampled 2× during training
3. **Data Augmentation**: Increases effective real video count
4. **Weighted Sampling**: Ensures balanced batches

See `config.yaml` for configuration:

```yaml
training:
  use_focal_loss: true
  focal_loss_alpha: 0.25
  focal_loss_gamma: 2.0
  oversample_real: true
  real_oversample_ratio: 2.0
```

## Expected Performance

With proper training on this dataset:

- **AUC-ROC**: Should approach 70-80%+ (closer to benchmarks)
- **Balanced Predictions**: Model should learn both classes effectively
- **Reduced False Positives**: Oversampling helps model see more real videos

## Notes

- Frame extraction rate can be adjusted in `config.yaml` (`frame_sampling_rate`)
- Lower sampling rates reduce dataset size but may lose temporal information
- Higher sampling rates increase dataset size but require more storage
- The 3 FPS rate is a good balance for most use cases
