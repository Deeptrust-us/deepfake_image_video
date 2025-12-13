# Quick Command Reference

## 🚀 Run Full Pipeline (All Steps)

```bash
# Option 1: Use the script (recommended)
./run_pipeline.sh

# Option 2: Run manually
python preprocess.py --max_videos 10
python train.py --config config.yaml
python evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

## 📊 Individual Steps

### 1. Preprocess Dataset
```bash
# With Hugging Face dataset (10 videos preview)
python preprocess.py --max_videos 10

# With Celeb-DF v2 (if downloaded)
python preprocess.py --dataset-type celebdf --videos-dir data/raw

# With FaceForensics++ (if downloaded)
python preprocess.py --dataset-type faceforensics --videos-dir data/raw

# With custom local videos
python preprocess.py --dataset-type local --videos-dir data/raw
```

### 2. Train Model
```bash
# Standard training
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/latest.pth
```

### 3. Evaluate Model
```bash
# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pth --split test

# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/best_model.pth --split val

# Evaluate on training set
python evaluate.py --checkpoint checkpoints/best_model.pth --split train
```

## 📈 Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir logs

# Then open browser to: http://localhost:6006
```

## 🔍 Check Status

```bash
# Check processed videos
ls data/faces/ | wc -l

# Check training progress
tail -f training_output.log

# Check checkpoints
ls -lh checkpoints/

# Check results
ls results/
```

## ⚙️ Configuration

Edit `config.yaml` to change:
- Batch size
- Learning rate
- Number of epochs
- Model architecture
- Data paths

## 🐛 Troubleshooting

```bash
# Check if dependencies are installed
python -c "import torch; import datasets; print('OK')"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify data structure
ls data/faces/
ls data/frames/
ls data/frequency/
```


