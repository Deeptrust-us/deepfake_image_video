# Fix: Dataset Too Small

## ❌ Current Problem

- **Training videos**: 6 (NEED 100+)
- **Validation videos**: 2 (NEED 20+)
- **Result**: Model cannot learn, AUC = 0.0000, early stopping

## ✅ Solution: Get More Videos

### Quick Fix Options:

### 1. **Celeb-DF v2** (Best Option - 6,229 videos)

```bash
# Download instructions:
# 1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics
# 2. Request access (usually free for research)
# 3. Download videos
# 4. Extract to data/raw/

# Then preprocess:
python preprocess.py --dataset-type celebdf --videos-dir data/raw
```

### 2. **FaceForensics++** (5,000 videos)

```bash
# Download instructions:
# 1. Visit: https://github.com/ondyari/FaceForensics
# 2. Fill out Google form for access
# 3. Download videos
# 4. Extract to data/raw/

# Then preprocess:
python preprocess.py --dataset-type faceforensics --videos-dir data/raw
```

### 3. **Use Your Own Videos**

If you have videos locally:

```bash
# Place videos in data/raw/ with structure:
# data/raw/
#   real/          (real videos)
#   fake/          (fake videos)

python preprocess.py --dataset-type local --videos-dir data/raw
```

## Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Training videos | 100 | 1,000+ |
| Validation videos | 20 | 200+ |
| Test videos | 20 | 200+ |
| Total | 140 | 1,400+ |

## Why More Videos Are Needed

- **Deep learning models** need diverse examples to learn patterns
- **6 videos** = only ~18-30 frames (at 3 fps)
- **Model has 12M parameters** - needs thousands of examples
- **Current result**: Model memorizes instead of learning general patterns

## Quick Test (Current Setup)

The current 10 videos are useful for:
- ✅ Testing the pipeline works
- ✅ Verifying code runs
- ❌ **NOT for actual training**

## Next Steps

1. **Download a larger dataset** (Celeb-DF v2 recommended)
2. **Preprocess** with the appropriate dataset type
3. **Train** with sufficient data
4. **Evaluate** on test set

Run `./get_more_videos.sh` for interactive help!


