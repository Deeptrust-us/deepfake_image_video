# FaceForensics++ Download Guide

## Quick Start

### Option 1: Use the helper script (Recommended)

```bash
# Download 100 videos per dataset with c23 compression
bash download_and_setup_faceforensics.sh data/faceforensics_raw c23 100 EU

# Or download all videos (will take a long time and use lots of space!)
bash download_and_setup_faceforensics.sh data/faceforensics_raw c23
```

### Option 2: Manual download

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

## Parameters Explained

### Compression (`-c`)
- **`raw`**: Lossless compression, highest quality (~500GB for full dataset)
- **`c23`**: High quality, good balance (~50GB for full dataset) ⭐ Recommended
- **`c40`**: Lower quality, smaller size (~10GB for full dataset)

### Datasets (`-d`)
- **`original`**: Real videos (YouTube)
- **`Deepfakes`**: Deepfake manipulations
- **`Face2Face`**: Face2Face manipulations
- **`FaceSwap`**: FaceSwap manipulations
- **`NeuralTextures`**: NeuralTextures manipulations
- **`FaceShifter`**: FaceShifter manipulations
- **`DeepFakeDetection`**: DeepFakeDetection dataset
- **`all`**: Download all datasets

### Number of Videos (`-n`)
- Omit `-n` to download all videos
- Use `-n 100` to download first 100 videos per dataset (good for testing)

### Server (`--server`)
- **`EU`**: European server (default)
- **`EU2`**: Alternative European server
- **`CA`**: Canadian server (try if EU is slow)

## Recommended Download Strategy

### For Testing (Small dataset)
```bash
# Download 50 videos per dataset with c23 compression
python download_faceforensics.py data/faceforensics_raw \
    -d all \
    -c c23 \
    -t videos \
    -n 50 \
    --server EU
```

### For Training (Medium dataset)
```bash
# Download 500 videos per dataset with c23 compression
python download_faceforensics.py data/faceforensics_raw \
    -d all \
    -c c23 \
    -t videos \
    -n 500 \
    --server EU
```

### For Full Dataset (Large)
```bash
# Download all videos with c23 compression (~50GB)
python download_faceforensics.py data/faceforensics_raw \
    -d all \
    -c c23 \
    -t videos \
    --server EU
```

## After Downloading

### 1. Preprocess the videos

```bash
python preprocess.py \
    --dataset-type faceforensics \
    --videos-dir data/faceforensics_raw \
    --max_videos 1000
```

### 2. Start Training

```bash
python train.py --config config.yaml
```

## Dataset Structure

After downloading, the structure will be:
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

## Tips

1. **Start small**: Download 50-100 videos first to test the pipeline
2. **Use c23 compression**: Good balance between quality and size
3. **Try different servers**: If download is slow, try EU2 or CA
4. **Monitor disk space**: Full dataset can be 50GB+ (c23) or 500GB+ (raw)
5. **Resume downloads**: The script skips already downloaded files

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
- The servers may be temporarily unavailable

## Terms of Use

By downloading FaceForensics++, you agree to their terms of use:
- https://github.com/ondyari/FaceForensics
- The script will prompt you to confirm before downloading

