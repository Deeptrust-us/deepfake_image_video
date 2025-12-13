# Dataset Guide

## Current Dataset Status

The Hugging Face dataset `UniDataPro/deepfake-videos-dataset` **only provides a 10-video preview**. The full dataset (10,000+ videos) requires contacting UniDataPro.

## Recommended Datasets for Training

For proper training, you should use larger datasets:

### 1. Celeb-DF v2 (Recommended)
- **Size**: 590 real videos + 5,639 deepfake videos
- **Download**: https://github.com/yuezunli/celeb-deepfakeforensics
- **Structure**:
  ```
  data/raw/
    Celeb-real/
    Celeb-synthesis/
    YouTube-real/
    ...
  ```

### 2. FaceForensics++
- **Size**: 1,000 original + 4,000 manipulated videos
- **Download**: https://github.com/ondyari/FaceForensics
- **Structure**:
  ```
  data/raw/
    original_sequences/
    manipulated_sequences/
      Deepfakes/
      Face2Face/
      FaceSwap/
      NeuralTextures/
  ```

### 3. DeeperForensics-1.0
- **Size**: 60,000 videos with 17.6 million frames
- **Download**: https://github.com/EndlessSora/DeeperForensics

## Using Local Datasets

### Step 1: Download Dataset

**Celeb-DF v2:**
```bash
# Follow instructions at: https://github.com/yuezunli/celeb-deepfakeforensics
# Place videos in data/raw/
```

**FaceForensics++:**
```bash
# Follow instructions at: https://github.com/ondyari/FaceForensics
# Place videos in data/raw/
```

### Step 2: Preprocess

**For Celeb-DF v2:**
```bash
python preprocess.py --dataset-type celebdf --videos-dir data/raw
```

**For FaceForensics++:**
```bash
python preprocess.py --dataset-type faceforensics --videos-dir data/raw
```

**For custom structure:**
```bash
python preprocess.py --dataset-type local --videos-dir data/raw
```

### Step 3: Train

```bash
python train.py --config config.yaml
```

## Dataset Structure Detection

The preprocessing script automatically detects labels based on:

1. **Directory names**: 
   - `real`, `original` → label 0 (real)
   - `fake`, `deepfake`, `synthesis` → label 1 (fake)

2. **File names**:
   - Keywords in filename are checked

3. **Manual mapping**:
   - You can provide a custom label mapping dictionary

## Processing Large Datasets

For large datasets, you can:

1. **Process in batches:**
```bash
python preprocess.py --dataset-type celebdf --max_videos 1000
```

2. **Resume processing** (if interrupted, re-run with same command - it skips already processed videos)

3. **Adjust frame sampling rate** in `config.yaml`:
```yaml
data:
  frame_sampling_rate: 3  # Lower = fewer frames per video
```

## Expected Dataset Sizes After Preprocessing

- **Celeb-DF v2**: ~6,229 videos → ~18,000-30,000 frames (at 3 fps)
- **FaceForensics++**: ~5,000 videos → ~15,000-25,000 frames
- **Current (preview)**: 10 videos → ~30-50 frames

## Tips

- Start with a subset (`--max_videos 100`) to test the pipeline
- Use GPU for face detection (much faster)
- Ensure sufficient disk space (each video generates multiple frames)
- Consider using a lower `frame_sampling_rate` for very large datasets


