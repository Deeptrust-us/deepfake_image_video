# Dataset Guide

This guide provides comprehensive information about downloading, setting up, and using datasets for deepfake detection training.

## Table of Contents

1. [Supported Datasets](#supported-datasets)
2. [Celeb-DF v2](#celeb-df-v2)
3. [FaceForensics++](#faceforensics)
4. [Hugging Face Datasets](#hugging-face-datasets)
5. [Custom Datasets](#custom-datasets)
6. [Preprocessing](#preprocessing)
7. [Dataset Statistics](#dataset-statistics)

## Supported Datasets

The following datasets are supported:

| Dataset | Videos | Real | Fake | Status |
|---------|--------|------|------|--------|
| **Celeb-DF v2** | 6,229 | 590 | 5,639 | ✅ Recommended |
| **FaceForensics++** | 5,000 | 1,000 | 4,000 | ✅ Supported |
| **Hugging Face Preview** | 10 | ~5 | ~5 | ⚠️ Testing only |
| **DeeperForensics-1.0** | 60,000 | - | - | ⬇️ Download needed |

## Celeb-DF v2

### Overview

Celeb-DF v2 is a large-scale deepfake detection dataset containing 590 real videos and 5,639 deepfake videos, totaling 6,229 videos.

**Download**: https://github.com/yuezunli/celeb-deepfakeforensics

### Download Instructions

1. Visit the [Celeb-DF repository](https://github.com/yuezunli/celeb-deepfakeforensics)
2. Request access (usually free for research purposes)
3. Download the videos following their instructions
4. Extract videos to the project directory

### Directory Structure

After downloading, organize videos as follows:

```
data/raw/
├── Celeb-real/          # Real celebrity videos
│   ├── video_0001.mp4
│   └── ...
├── Celeb-synthesis/     # Deepfake videos
│   ├── video_0001.mp4
│   └── ...
└── YouTube-real/        # Real YouTube videos
    ├── video_0001.mp4
    └── ...
```

### Preprocessing

```bash
python scripts/preprocess.py --dataset-type celebdf --videos-dir data/raw
```

### Expected Statistics

- **Total videos**: 6,229
- **Real videos**: 590 (9.5%)
- **Fake videos**: 5,639 (90.5%)
- **Class distribution**: Highly imbalanced (expected for this dataset)

## FaceForensics++

### Overview

FaceForensics++ is a comprehensive dataset with multiple manipulation methods including Deepfakes, Face2Face, FaceSwap, and NeuralTextures.

**Download**: https://github.com/ondyari/FaceForensics

### Download Instructions

#### Option 1: Automated Script (Recommended)

```bash
# Download 100 videos per dataset with c23 compression
bash download_and_setup_faceforensics.sh data/faceforensics_raw c23 100 EU

# Or download all videos (will take a long time and use lots of space!)
bash download_and_setup_faceforensics.sh data/faceforensics_raw c23
```

#### Option 2: Manual Download

```bash
# Download original (real) videos
python download_faceforensics.py data/faceforensics_raw \
    -d original \
    -c c23 \
    -t videos \
    -n 100 \
    --server EU

# Download Deepfakes (fake) videos
python download_faceforensics.py data/faceforensics_raw \
    -d Deepfakes \
    -c c23 \
    -t videos \
    -n 100 \
    --server EU

# Download other manipulation methods
python download_faceforensics.py data/faceforensics_raw -d Face2Face -c c23 -t videos -n 100 --server EU
python download_faceforensics.py data/faceforensics_raw -d FaceSwap -c c23 -t videos -n 100 --server EU
python download_faceforensics.py data/faceforensics_raw -d NeuralTextures -c c23 -t videos -n 100 --server EU
```

### Compression Options

- **`raw`**: Lossless compression, highest quality (~500GB for full dataset)
- **`c23`**: High quality, good balance (~50GB for full dataset) ⭐ Recommended
- **`c40`**: Lower quality, smaller size (~10GB for full dataset)

### Server Options

- **`EU`**: European server (default)
- **`EU2`**: Alternative European server
- **`CA`**: Canadian server (try if EU is slow)

### Directory Structure

After downloading:

```
data/faceforensics_raw/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
│               ├── video1.mp4
│               └── ...
└── manipulated_sequences/
    ├── Deepfakes/
    │   └── c23/
    │       └── videos/
    ├── Face2Face/
    ├── FaceSwap/
    └── NeuralTextures/
```

### Preprocessing

```bash
python scripts/preprocess.py \
    --dataset-type faceforensics \
    --videos-dir data/faceforensics_raw \
    --max_videos 1000
```

### Download Strategies

#### For Testing (Small dataset)
```bash
python download_faceforensics.py data/faceforensics_raw \
    -d all \
    -c c23 \
    -t videos \
    -n 50 \
    --server EU
```

#### For Training (Medium dataset)
```bash
python download_faceforensics.py data/faceforensics_raw \
    -d all \
    -c c23 \
    -t videos \
    -n 500 \
    --server EU
```

#### For Full Dataset (Large)
```bash
python download_faceforensics.py data/faceforensics_raw \
    -d all \
    -c c23 \
    -t videos \
    --server EU
```

## Hugging Face Datasets

### Overview

The project supports datasets hosted on Hugging Face. The default dataset `UniDataPro/deepfake-videos-dataset` provides a 10-video preview for testing.

**Note**: The preview dataset is too small for actual training. For production use, download Celeb-DF v2 or FaceForensics++.

### Configuration

In `config.yaml`:

```yaml
data:
  dataset_name: "UniDataPro/deepfake-videos-dataset"
  dataset_type: "huggingface"
```

### Authentication

Some datasets may require authentication:

```bash
huggingface-cli login
```

### Preprocessing

```bash
python preprocess.py --config config.yaml
```

## Custom Datasets

### Directory Structure

For custom datasets, organize videos as follows:

```
data/raw/
├── real/          # Real videos
│   ├── video1.mp4
│   └── ...
└── fake/          # Fake videos
    ├── video1.mp4
    └── ...
```

### Label Detection

The preprocessing script automatically detects labels based on:

1. **Directory names**: 
   - `real`, `original` → label 0 (real)
   - `fake`, `deepfake`, `synthesis` → label 1 (fake)

2. **File names**: Keywords in filenames are checked

3. **Manual mapping**: Provide a custom label mapping dictionary if needed

### Preprocessing

```bash
python scripts/preprocess.py --dataset-type local --videos-dir data/raw
```

## Preprocessing

### Overview

The preprocessing pipeline performs the following steps:

1. **Frame Extraction**: Extracts frames from videos at a configurable rate (default: 3 FPS)
2. **Face Detection**: Uses MTCNN to detect and align faces
3. **Face Cropping**: Crops faces to 224×224 pixels
4. **Frame Resizing**: Resizes whole frames to 224×224 pixels
5. **Frequency Domain**: Computes FFT and log-magnitude spectrum
6. **Train/Val/Test Split**: Creates stratified splits (default: 70/15/15)

### Configuration

Edit `config.yaml` to customize preprocessing:

```yaml
data:
  frame_sampling_rate: 3  # Frames per second
  face_size: 224          # Face crop size
  train_split: 0.7        # Training split ratio
  val_split: 0.15         # Validation split ratio
  test_split: 0.15        # Test split ratio
```

### Processing Large Datasets

For large datasets, you can:

1. **Process in batches**:
```bash
python scripts/preprocess.py --dataset-type celebdf --max_videos 1000
```

2. **Resume processing**: The script automatically skips already processed videos

3. **Adjust frame sampling rate**: Lower values reduce the number of frames per video

### Output Structure

After preprocessing:

```
data/
├── frames/           # Raw extracted frames
│   ├── video_0001/
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── ...
├── faces/            # Face crops (224×224)
│   ├── video_0001/
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── ...
├── frequency/        # Frequency domain representations
│   ├── video_0001/
│   │   ├── frame_0001.npy
│   │   └── ...
│   └── ...
└── *_metadata.json   # Dataset metadata and splits
```

## Dataset Statistics

### Expected Sizes After Preprocessing

- **Celeb-DF v2**: ~6,229 videos → ~247,000 frames (at 3 FPS)
- **FaceForensics++**: ~5,000 videos → ~200,000 frames
- **Hugging Face Preview**: 10 videos → ~30-50 frames

### Frame Statistics

- **Average frames per video**: ~38 frames (at 3 FPS)
- **Range**: 11-74 frames per video (varies by video length)
- **Frame format**: JPG images (224×224 for faces and frames)

### Class Distribution

**Celeb-DF v2**:
- Training: 13.6% real, 86.4% fake (highly imbalanced)
- Validation: 13.6% real, 86.4% fake
- Test: 13.7% real, 86.3% fake

**Note**: The class imbalance is intentional and expected for Celeb-DF v2. The model uses Focal Loss and oversampling to handle this imbalance.

## Tips

1. **Start small**: Download 50-100 videos first to test the pipeline
2. **Use c23 compression**: Good balance between quality and size for FaceForensics++
3. **Monitor disk space**: Full datasets can be 50GB+ (c23) or 500GB+ (raw)
4. **Resume downloads**: Scripts skip already downloaded files
5. **GPU acceleration**: Use GPU for face detection (much faster)
6. **Frame sampling**: Lower `frame_sampling_rate` for very large datasets

## Troubleshooting

### Slow Downloads
- Try different server: `--server EU2` or `--server CA`
- Download fewer videos: `-n 50`
- Use lower compression: `-c c40` (smaller files)

### Out of Disk Space
- Use `c40` compression instead of `c23` or `raw`
- Download fewer videos: `-n 100`
- Clean up old downloads

### Connection Errors
- Try a different server
- Check your internet connection
- Servers may be temporarily unavailable

### Face Detection Fails
- The model will fall back to resizing the original image if no face is detected
- Ensure videos contain clear faces
- Check `min_face_size` parameter in config

## Terms of Use

By downloading datasets, you agree to their respective terms of use:

- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics
- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- The download scripts will prompt you to confirm before downloading
