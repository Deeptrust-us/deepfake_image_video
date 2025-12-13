# Quad-Stream Deepfake Detection

A PyTorch implementation of a quad-stream deepfake detection system that analyzes videos in both spatial and frequency domains, using both face crops and whole frames for improved robustness and generalization.

## Overview

This project implements a quad-stream architecture that combines:
- **Face Spatial Stream**: Uses pretrained ResNet18 or EfficientNet-B0 to extract texture and appearance features from cropped faces
- **Face Frequency Stream**: Uses ResNet50 to analyze frequency-domain artifacts in face crops
- **Frame Spatial Stream**: Uses pretrained ResNet18 or EfficientNet-B0 to extract features from whole video frames
- **Frame Frequency Stream**: Uses ResNet50 to analyze frequency-domain artifacts in whole frames

By combining all four streams, the model captures complementary cues: texture inconsistencies in pixel space, spectral artifacts in the frequency domain, and both facial details and contextual scene information.

## Features

- **Quad-stream architecture** with spatial and frequency domain analysis for both face crops and whole frames
- **Face detection and alignment** using MTCNN
- **Frequency domain transformation** with FFT and log-magnitude spectrum
- **Comprehensive evaluation metrics** (frame-level and video-level)
- **Multiple dataset support** (Celeb-DF v2, FaceForensics++, Hugging Face datasets)
- **Flexible configuration** via YAML files
- **TensorBoard logging** for training visualization
- **Focal Loss** for handling class imbalance
- **Advanced data augmentation** including noise and blur

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd deepfake_image_video

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (required for video processing)
# Ubuntu/Debian:
sudo apt-get install ffmpeg
# macOS:
brew install ffmpeg
```

### Basic Usage

```bash
# 1. Preprocess dataset
python scripts/preprocess.py --config config/config.yaml

# 2. Train model
python scripts/train.py --config config/config.yaml

# 3. Evaluate model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

## Project Structure

```
deepfake_image_video/
├── config/
│   └── config.yaml          # Configuration file
├── scripts/                 # Executable scripts
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── preprocess.py        # Data preprocessing script
│   ├── example_usage.py     # Example inference script
│   ├── download/            # Dataset download scripts
│   │   ├── download_celebdf.py
│   │   ├── download_celebdf_complete.py
│   │   └── download_faceforensics.py
│   ├── utils/               # Utility scripts
│   │   ├── check_dataset.py
│   │   ├── add_whole_frames.py
│   │   └── fix_class_imbalance.py
│   └── shell/               # Shell scripts
│       ├── run_pipeline.sh
│       ├── start_training.sh
│       └── ...
├── src/                     # Source code package
│   ├── models/
│   │   └── dual_stream.py   # Quad-stream model architecture
│   ├── data/
│   │   ├── dataset.py       # Dataset loader
│   │   └── preprocessing.py  # Data preprocessing utilities
│   └── utils/
│       ├── face_detection.py    # Face detection utilities
│       ├── frequency_domain.py  # Frequency domain transformations
│       ├── augmentations.py     # Data augmentations
│       └── metrics.py           # Evaluation metrics
├── data/                    # Processed data (created during preprocessing)
│   ├── frames/              # Extracted frames
│   ├── faces/               # Detected and aligned faces
│   ├── frequency/           # Frequency domain representations
│   └── *_metadata.json      # Dataset metadata and splits
├── checkpoints/             # Model checkpoints
├── logs/                    # TensorBoard logs
├── results/                 # Evaluation results
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Configuration

Edit `config/config.yaml` to customize:

- **Data settings**: Dataset name, splits, frame sampling rate
- **Model architecture**: Backbone choice, feature dimensions, fusion method
- **Training parameters**: Batch size, learning rate, optimizer, scheduler
- **Preprocessing**: Augmentation parameters, face detection settings

## Model Architecture

### Face Spatial Stream
- Backbone: ResNet18 or EfficientNet-B0 (pretrained on ImageNet)
- Input: Cropped and aligned face images (224×224)
- Output: 256-dimensional feature vector

### Face Frequency Stream
- Backbone: ResNet50
- Input: Log-magnitude spectrum of face crops (1 channel) or magnitude+phase (2 channels)
- Output: 256-dimensional feature vector

### Frame Spatial Stream
- Backbone: ResNet18 or EfficientNet-B0 (pretrained on ImageNet)
- Input: Whole video frames resized to 224×224
- Output: 256-dimensional feature vector

### Frame Frequency Stream
- Backbone: ResNet50
- Input: Log-magnitude spectrum of whole frames (1 channel) or magnitude+phase (2 channels)
- Output: 256-dimensional feature vector

### Fusion & Classification
- Concatenation of all four feature vectors (1024 dimensions total)
- Fusion layers: 1024 → 512 → 256
- Dropout (0.6) for regularization
- Binary classification output

## Evaluation Metrics

The model reports both frame-level and video-level metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score
- **Recall**: Recall score
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **EER**: Equal Error Rate

Video-level predictions are aggregated using mean pooling across all frames in a video.

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Step-by-step setup and usage guide
- **[DATASET_GUIDE.md](DATASET_GUIDE.md)**: Comprehensive guide for downloading and using datasets
- **[COMMANDS.md](COMMANDS.md)**: Quick reference for common commands
- **[QUAD_STREAM_CHANGES.md](QUAD_STREAM_CHANGES.md)**: Technical details about the quad-stream architecture
- **[DATASET_ANALYSIS.md](DATASET_ANALYSIS.md)**: Dataset statistics and analysis
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Development guide for contributors

## Research Background

This implementation is based on research showing that:
- High-frequency components in the frequency domain are particularly informative for deepfake detection
- Combining spatial and frequency features improves robustness across different manipulation methods
- Frequency-domain analysis can reveal artifacts introduced by generative models that are less visible in pixel space
- Using both face crops and whole frames provides complementary information for detection

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{quad-stream-deepfake-detection,
  title={Quad-Stream Deepfake Detection},
  author={Your Name},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face for dataset hosting
- FaceNet-PyTorch for MTCNN implementation
- PyTorch and torchvision communities
- Celeb-DF and FaceForensics++ dataset creators
