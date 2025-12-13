# Command Reference

Quick reference for common commands and operations.

## Pipeline Commands

### Full Pipeline (All Steps)

```bash
# Option 1: Use the script (if available)
bash scripts/shell/run_pipeline.sh

# Option 2: Run manually
python scripts/preprocess.py --config config/config.yaml
python scripts/train.py --config config/config.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

## Preprocessing

### Basic Preprocessing

```bash
# With Hugging Face dataset (default)
python scripts/preprocess.py --config config/config.yaml

# With Celeb-DF v2
python scripts/preprocess.py --dataset-type celebdf --videos-dir data/raw

# With FaceForensics++
python scripts/preprocess.py --dataset-type faceforensics --videos-dir data/faceforensics_raw

# With custom local videos
python scripts/preprocess.py --dataset-type local --videos-dir data/raw
```

### Preprocessing Options

```bash
# Limit number of videos (for testing)
python preprocess.py --max_videos 100

# Specify custom output directory
python preprocess.py --config config.yaml --output-dir custom_data
```

## Training

### Basic Training

```bash
# Standard training
python scripts/train.py --config config/config.yaml

# Resume from checkpoint
python scripts/train.py --config config/config.yaml --resume checkpoints/latest.pth

# Train with custom config
python scripts/train.py --config config/custom_config.yaml
```

### Training in Background

```bash
# Using screen (recommended)
screen -S deepfake_training
python train.py --config config.yaml
# Press Ctrl+A then D to detach

# Attach to screen session
screen -r deepfake_training

# Using nohup
nohup python train.py --config config.yaml > training_output.log 2>&1 &
```

## Evaluation

### Evaluate Model

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test

# Evaluate on validation set
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split val

# Evaluate on training set
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split train
```

### Evaluation Options

```bash
# Save detailed results
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test --save-results

# Generate confusion matrix
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test --save-plots
```

## Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs

# Access at http://localhost:6006

# TensorBoard with custom port
tensorboard --logdir logs --port 6007
```

### Log Files

```bash
# View training output
tail -f training_output.log

# View last 20 lines
tail -20 training_output.log

# Search for errors
grep -i error training_output.log
```

### GPU Monitoring

```bash
# Check GPU usage (NVIDIA)
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

## Dataset Management

### Check Dataset Status

```bash
# Count processed videos
ls data/faces/ | wc -l

# Check frames in a video
ls data/faces/video_0001/ | wc -l

# View metadata
python3 -c "import json; data=json.load(open('data/train_metadata.json')); print(f'Training videos: {len(data)}')"

# Check dataset structure
tree data/ -L 2
```

### Verify Dataset

```bash
# Run dataset verification script (if available)
python scripts/utils/check_dataset.py

# Check for missing files
python scripts/utils/check_dataset.py --verify-files
```

## Download Commands

### FaceForensics++

```bash
# Download 100 videos per dataset
bash scripts/shell/download_and_setup_faceforensics.sh data/faceforensics_raw c23 100 EU

# Download all videos
bash scripts/shell/download_and_setup_faceforensics.sh data/faceforensics_raw c23

# Manual download
python scripts/download/download_faceforensics.py data/faceforensics_raw \
    -d all -c c23 -t videos -n 100 --server EU
```

### Celeb-DF

```bash
# Use the download script (if available)
python scripts/download/download_celebdf.py

# Or follow manual download instructions in DATASET_GUIDE.md
```

## Utility Commands

### Check Dependencies

```bash
# Verify Python packages
python -c "import torch; import datasets; import torchvision; print('All dependencies OK')"

# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

### Clean Up

```bash
# Remove checkpoints (keep best)
rm checkpoints/latest.pth

# Remove old logs
rm -rf logs/run_*

# Remove processed data (careful!)
# rm -rf data/faces/ data/frames/ data/frequency/
```

### View Results

```bash
# View results directory
ls -lh results/

# View confusion matrix
open results/val_confusion_matrix.png  # macOS
xdg-open results/val_confusion_matrix.png  # Linux

# Use view script (if available)
./view_results.sh
```

## Configuration

### Edit Configuration

```bash
# Edit main config
nano config/config.yaml
# or
vim config/config.yaml

# Create custom config
cp config/config.yaml config/custom_config.yaml
nano config/custom_config.yaml
```

### Common Config Changes

```yaml
# Reduce batch size for limited GPU memory
training:
  batch_size: 16

# Change model backbone
model:
  spatial_backbone: "efficientnet_b0"

# Adjust frame sampling rate
data:
  frame_sampling_rate: 2  # Lower = fewer frames
```

## Troubleshooting Commands

### Check System Resources

```bash
# Disk space
df -h

# Memory usage
free -h

# CPU info
lscpu
```

### Debug Training

```bash
# Run with verbose output
python train.py --config config.yaml --verbose

# Run single epoch for testing
# Edit config.yaml: num_epochs: 1

# Check data loading
python -c "from src.data.dataset import DeepfakeDataset; print('Dataset OK')"
```

### Common Issues

```bash
# Fix import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check file permissions
ls -la data/

# Verify ffmpeg installation
which ffmpeg
ffmpeg -version
```

## Quick Reference Table

| Task | Command |
|------|---------|
| Preprocess | `python scripts/preprocess.py --config config/config.yaml` |
| Train | `python scripts/train.py --config config/config.yaml` |
| Evaluate | `python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test` |
| TensorBoard | `tensorboard --logdir logs` |
| Check videos | `ls data/faces/ \| wc -l` |
| View logs | `tail -f training_output.log` |
| GPU status | `nvidia-smi` |

## Additional Resources

- [README.md](README.md) - Project overview and architecture
- [QUICKSTART.md](QUICKSTART.md) - Detailed setup guide
- [DATASET_GUIDE.md](DATASET_GUIDE.md) - Dataset download and setup
- [QUAD_STREAM_CHANGES.md](QUAD_STREAM_CHANGES.md) - Technical architecture details
