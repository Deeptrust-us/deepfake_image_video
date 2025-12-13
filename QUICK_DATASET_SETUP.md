# Quick Dataset Setup Guide

## Current Status: Only 10 Videos Available

The Hugging Face dataset `UniDataPro/deepfake-videos-dataset` **only provides 10 videos** as a preview. This is not enough for proper training.

## ✅ Solution: Download Larger Datasets

### Option 1: Celeb-DF v2 (Recommended - 6,229 videos)

```bash
# 1. Download Celeb-DF v2
# Visit: https://github.com/yuezunli/celeb-deepfakeforensics
# Follow their download instructions

# 2. Extract videos to data/raw/
mkdir -p data/raw
# Place videos in: data/raw/Celeb-real/, data/raw/Celeb-synthesis/, etc.

# 3. Preprocess
python preprocess.py --dataset-type celebdf --videos-dir data/raw

# 4. Train
python train.py --config config.yaml
```

### Option 2: FaceForensics++ (5,000 videos)

```bash
# 1. Download FaceForensics++
# Visit: https://github.com/ondyari/FaceForensics
# Follow their download instructions

# 2. Extract videos to data/raw/
mkdir -p data/raw
# Place videos in: data/raw/original_sequences/, data/raw/manipulated_sequences/, etc.

# 3. Preprocess
python preprocess.py --dataset-type faceforensics --videos-dir data/raw

# 4. Train
python train.py --config config.yaml
```

### Option 3: Test with Current 10 Videos

If you just want to test the pipeline:

```bash
# Already done - 10 videos processed
# Training is running with limited data
# Results will be limited due to small dataset size
```

## Dataset Comparison

| Dataset | Videos | Real | Fake | Status |
|---------|--------|------|------|--------|
| **Current (HF Preview)** | 10 | ~5 | ~5 | ✅ Ready (limited) |
| **Celeb-DF v2** | 6,229 | 590 | 5,639 | ⬇️ Download needed |
| **FaceForensics++** | 5,000 | 1,000 | 4,000 | ⬇️ Download needed |
| **DeeperForensics** | 60,000 | - | - | ⬇️ Download needed |

## Quick Start with More Videos

1. **Choose a dataset** (Celeb-DF v2 recommended)
2. **Download** following their official instructions
3. **Place videos** in `data/raw/` with proper structure
4. **Run**: `python preprocess.py --dataset-type celebdf`
5. **Train**: `python train.py`

## Note

The current 10-video dataset is useful for:
- ✅ Testing the pipeline
- ✅ Verifying code works
- ✅ Understanding the workflow

But for **actual model training**, you need **hundreds or thousands** of videos.


