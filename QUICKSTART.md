# Quick Start Guide

This guide will help you get started with the dual-stream deepfake detection system in minutes.

## Prerequisites

1. **Python 3.8+** installed
2. **ffmpeg** installed (for video processing)
3. **CUDA-capable GPU** (optional but recommended for training)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Install ffmpeg

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
Download from https://ffmpeg.org/download.html

## Step 3: Authenticate with Hugging Face (if required)

Some datasets require authentication. If needed:

```bash
huggingface-cli login
```

## Step 4: Preprocess Dataset

The preprocessing script will:
- Download the dataset from Hugging Face
- Extract frames from videos
- Detect and align faces
- Compute frequency domain representations
- Create train/val/test splits

```bash
# Process all videos (may take a while)
python preprocess.py

# Or process a subset for testing
python preprocess.py --max_videos 100
```

**Note:** Preprocessing can take significant time depending on dataset size. For testing, use `--max_videos` to limit the number of videos.

## Step 5: Train the Model

```bash
python train.py --config config.yaml
```

Training progress will be logged to TensorBoard. Monitor it with:

```bash
tensorboard --logdir logs
```

## Step 6: Evaluate the Model

After training, evaluate on the test set:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

## Example: Single Image Inference

See `example_usage.py` for a simple example of running inference on a single image.

## Troubleshooting

### Issue: "No module named 'src'"
**Solution:** Make sure you're running scripts from the project root directory.

### Issue: "ffmpeg not found"
**Solution:** Install ffmpeg (see Step 2) and ensure it's in your PATH.

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `config.yaml`:
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

## Next Steps

- Experiment with different model architectures (ResNet18 vs EfficientNet-B0)
- Try including phase spectrum (`frequency_channels: 2` in config)
- Adjust augmentation parameters
- Test on different datasets

## Configuration Tips

Edit `config.yaml` to customize:

- **Model**: Change `spatial_backbone` to `"efficientnet_b0"` for potentially better performance
- **Training**: Adjust `learning_rate`, `batch_size`, `num_epochs`
- **Data**: Change `frame_sampling_rate` to extract more/fewer frames per second
- **Frequency**: Set `frequency_channels: 2` to include phase spectrum

For more details, see the main [README.md](README.md).


