# Quick Start Guide

This guide will help you get started with the quad-stream deepfake detection system in minutes.

## Prerequisites

1. **Python 3.8+** installed
2. **ffmpeg** installed (for video processing)
3. **CUDA-capable GPU** (optional but recommended for training)
4. **Sufficient disk space** (50GB+ recommended for datasets)

## Step 1: Installation

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

### Verify Installation

```bash
# Check Python dependencies
python -c "import torch; import datasets; print('Dependencies OK')"

# Check GPU availability (if using CUDA)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check ffmpeg
ffmpeg -version
```

## Step 2: Dataset Setup

### Option 1: Use Hugging Face Dataset (Testing Only)

The default configuration uses a 10-video preview from Hugging Face. This is suitable for testing the pipeline but **not for actual training**.

```bash
# Authenticate with Hugging Face (if required)
huggingface-cli login

# Preprocess the dataset
python preprocess.py --config config.yaml
```

### Option 2: Download Celeb-DF v2 (Recommended)

For actual training, download Celeb-DF v2:

1. Visit https://github.com/yuezunli/celeb-deepfakeforensics
2. Request access and download videos
3. Extract to `data/raw/` with proper structure
4. Preprocess:

```bash
python preprocess.py --dataset-type celebdf --videos-dir data/raw
```

See [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed instructions.

## Step 3: Preprocessing

The preprocessing script will:
- Extract frames from videos (3 FPS by default)
- Detect and align faces using MTCNN
- Crop faces to 224×224 pixels
- Resize whole frames to 224×224 pixels
- Compute frequency domain representations (FFT)
- Create train/val/test splits (70/15/15)

```bash
# Process all videos
python scripts/preprocess.py --config config/config.yaml

# Or process a subset for testing
python scripts/preprocess.py --max_videos 100
```

**Note**: Preprocessing can take significant time depending on dataset size. For testing, use `--max_videos` to limit the number of videos.

## Step 4: Training

### Basic Training

```bash
python scripts/train.py --config config/config.yaml
```

### Monitor Training

**TensorBoard:**
```bash
tensorboard --logdir logs
# Open browser to http://localhost:6006
```

**Log File:**
```bash
tail -f training_output.log
```

### Training Configuration

Edit `config.yaml` to customize:
- `batch_size`: Number of samples per batch (default: 32)
- `learning_rate`: Learning rate (default: 5e-5)
- `num_epochs`: Maximum number of epochs (default: 50)
- `spatial_backbone`: Model backbone ("resnet18" or "efficientnet_b0")

## Step 5: Evaluation

After training, evaluate on the test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

### Evaluation Options

```bash
# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/best_model.pth --split val

# Evaluate on training set
python evaluate.py --checkpoint checkpoints/best_model.pth --split train
```

## Step 6: Inference

See `example_usage.py` for a simple example of running inference on a single image:

```bash
python scripts/example_usage.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
```

## Common Workflows

### Full Pipeline

```bash
# 1. Preprocess
python scripts/preprocess.py --config config/config.yaml

# 2. Train
python scripts/train.py --config config/config.yaml

# 3. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

### Resume Training

```bash
python scripts/train.py --config config/config.yaml --resume checkpoints/latest.pth
```

### Check Dataset Status

```bash
# Count processed videos
ls data/faces/ | wc -l

# Check frames in a video
ls data/faces/video_0001/ | wc -l

# View metadata
python3 -c "import json; data=json.load(open('data/train_metadata.json')); print(f'Training videos: {len(data)}')"
```

## Troubleshooting

### Issue: "No module named 'src'"
**Solution:** Make sure you're running scripts from the project root directory.

### Issue: "ffmpeg not found"
**Solution:** Install ffmpeg (see Step 1) and ensure it's in your PATH.

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `config/config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: Face detection fails
**Solution:** The model will fall back to resizing the original image if no face is detected. Ensure videos contain clear faces.

### Issue: Dataset download fails
**Solution:** 
1. Check your internet connection
2. Verify Hugging Face authentication if required
3. Check dataset name in `config.yaml`
4. See [DATASET_GUIDE.md](DATASET_GUIDE.md) for dataset-specific troubleshooting

### Issue: Training stops early
**Solution:** 
- Check if early stopping patience is too low
- Verify dataset size (need 100+ videos minimum)
- Check training logs for errors

## Next Steps

- Experiment with different model architectures (ResNet18 vs EfficientNet-B0)
- Try including phase spectrum (`frequency_channels: 2` in config)
- Adjust augmentation parameters
- Test on different datasets
- See [COMMANDS.md](COMMANDS.md) for quick command reference
- See [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed dataset information

## Configuration Tips

Edit `config.yaml` to customize:

- **Model**: Change `spatial_backbone` to `"efficientnet_b0"` for potentially better performance
- **Training**: Adjust `learning_rate`, `batch_size`, `num_epochs`
- **Data**: Change `frame_sampling_rate` to extract more/fewer frames per second
- **Frequency**: Set `frequency_channels: 2` to include phase spectrum
- **Augmentation**: Adjust `noise_std` and `gaussian_blur_prob` for robustness

For more details, see the main [README.md](README.md).
