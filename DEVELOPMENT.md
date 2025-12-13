# Development Guide

This guide provides information for developers working on the project, including development workflows, debugging, and best practices.

## Development Environment Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git for version control

### Setup

```bash
# Clone repository
git clone <repository-url>
cd deepfake_image_video

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if any)
pip install -r requirements-dev.txt  # If exists
```

## Running Long Training Sessions

### Using Screen (Recommended)

Screen allows you to detach from long-running processes and reattach later.

#### Start Training in Screen

```bash
# Create a new screen session
screen -S deepfake_training

# Run training
python train.py --config config.yaml

# Detach: Press Ctrl+A then D
```

#### Manage Screen Sessions

```bash
# List all screen sessions
screen -ls

# Attach to existing session
screen -r deepfake_training

# Attach to session by ID
screen -r <session_id>

# Kill a screen session
screen -X -S deepfake_training quit
```

#### Screen Tips

- **Detach**: `Ctrl+A` then `D`
- **Scroll**: `Ctrl+A` then `[` (use arrow keys, `q` to quit)
- **Split screen**: `Ctrl+A` then `S` (vertical), `Ctrl+A` then `|` (horizontal)
- **Switch windows**: `Ctrl+A` then `Tab`

### Using nohup

Alternative to screen for running processes in background:

# Run training in background
nohup python scripts/train.py --config config/config.yaml > training_output.log 2>&1 &

# View output
tail -f training_output.log

# Check if process is running
ps aux | grep train.py
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs

# Access at http://localhost:6006

# With custom port
tensorboard --logdir logs --port 6007
```

### Log Files

```bash
# Follow training output
tail -f training_output.log

# View last N lines
tail -n 100 training_output.log

# Search for errors
grep -i error training_output.log

# Search for specific metrics
grep "val_auc" training_output.log
```

### GPU Monitoring

```bash
# Check GPU usage (NVIDIA)
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Debugging

### Common Debugging Techniques

#### Enable Verbose Logging

# Run with verbose output
python scripts/train.py --config config/config.yaml --verbose

# Or modify logging level in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Data Loading

```bash
# Test dataset loading
python -c "from src.data.dataset import DeepfakeDataset; ds = DeepfakeDataset('data', 'train'); print(f'Dataset size: {len(ds)}'); sample = ds[0]; print(f'Sample keys: {sample.keys()}')"
```

#### Debug Model Forward Pass

```python
# Add to training script
import torch
model.eval()
with torch.no_grad():
    # Test forward pass
    output = model(*batch_inputs)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
```

### Common Issues

#### CUDA Out of Memory

```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # Reduce from 32

# Or use gradient accumulation
# (requires code modification)
```

#### Import Errors

```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### Data Loading Errors

```bash
# Verify data structure
ls -la data/faces/
ls -la data/frames/

# Check metadata files
python -c "import json; print(json.load(open('data/train_metadata.json')))"
```

## Code Organization

### Project Structure

```
deepfake_image_video/
├── src/
│   ├── models/          # Model architectures
│   ├── data/            # Dataset and preprocessing
│   └── utils/           # Utility functions
├── config.yaml          # Configuration file
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── preprocess.py        # Preprocessing script
└── example_usage.py     # Example inference script
```

### Adding New Features

1. **New Model Architecture**: Add to `src/models/`
2. **New Dataset**: Add loader to `src/data/`
3. **New Utilities**: Add to `src/utils/`
4. **New Metrics**: Add to `src/utils/metrics.py`

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

### Unit Tests

```bash
# Run tests (if test suite exists)
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Manual Testing

```bash
# Test preprocessing
python scripts/preprocess.py --max_videos 10

# Test training (single epoch)
# Edit config/config.yaml: num_epochs: 1
python scripts/train.py --config config/config.yaml

# Test evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

## Version Control

### Git Workflow

```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Description of changes"

# Push
git push origin main
```

### Best Practices

- Commit frequently with descriptive messages
- Create branches for new features
- Test before committing
- Keep commits focused and atomic

## Performance Optimization

### Profiling

```python
# Profile training loop
import cProfile
cProfile.run('train_model()', 'profile.stats')

# Analyze results
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

### Optimization Tips

1. **Data Loading**: Use `num_workers > 0` for parallel loading
2. **Mixed Precision**: Use `torch.cuda.amp` for faster training
3. **Gradient Accumulation**: For larger effective batch sizes
4. **Model Pruning**: Reduce model size if needed

## Contributing

### Before Submitting

1. Test your changes thoroughly
2. Update documentation if needed
3. Ensure code follows style guidelines
4. Add tests for new features

### Pull Request Process

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit pull request with description

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [Git Documentation](https://git-scm.com/doc)

## Getting Help

- Check existing documentation
- Review code comments
- Search for similar issues
- Ask questions in issues or discussions

