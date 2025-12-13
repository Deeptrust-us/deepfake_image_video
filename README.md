# Dual-Stream Deepfake Detection

A PyTorch implementation of a dual-stream deepfake detection system that analyzes videos in both spatial and frequency domains for improved robustness and generalization.

## 🔍 Overview

This project implements a dual-stream architecture that combines:
- **Spatial Stream**: Uses pretrained ResNet18 or EfficientNet-B0 to extract texture and appearance features
- **Frequency Stream**: Uses a lightweight CNN to analyze frequency-domain artifacts introduced by generative models

By combining both streams, the model can capture complementary cues: texture inconsistencies in pixel space and spectral artifacts in the frequency domain.

## 📋 Features

- **Dual-stream architecture** with spatial and frequency domain analysis
- **Face detection and alignment** using MTCNN
- **Frequency domain transformation** with FFT and log-magnitude spectrum
- **Comprehensive evaluation metrics** (frame-level and video-level)
- **Hugging Face dataset integration** for easy data loading
- **Flexible configuration** via YAML files
- **TensorBoard logging** for training visualization

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deepfake_image_video
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure ffmpeg is installed (for video processing):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Data Preprocessing

1. Configure the dataset in `config.yaml` (default uses `UniDataPro/deepfake-videos-dataset`)

2. Run preprocessing:
```bash
python preprocess.py --config config.yaml
```

This will:
- Download the dataset from Hugging Face
- Extract frames from videos
- Detect and align faces
- Compute frequency domain representations
- Create train/val/test splits

### Training

Train the model:
```bash
python train.py --config config.yaml
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

## 📁 Project Structure

```
deepfake_image_video/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── preprocess.py            # Data preprocessing script
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── src/
│   ├── models/
│   │   └── dual_stream.py   # Dual-stream model architecture
│   ├── data/
│   │   ├── dataset.py       # Dataset loader
│   │   └── preprocessing.py # Data preprocessing utilities
│   └── utils/
│       ├── face_detection.py    # Face detection utilities
│       ├── frequency_domain.py   # Frequency domain transformations
│       ├── augmentations.py     # Data augmentations
│       └── metrics.py           # Evaluation metrics
├── data/                    # Processed data (created during preprocessing)
│   ├── frames/              # Extracted frames
│   ├── faces/               # Detected and aligned faces
│   ├── frequency/           # Frequency domain representations
│   └── *_metadata.json      # Dataset metadata and splits
├── checkpoints/             # Model checkpoints
├── logs/                    # TensorBoard logs
└── results/                 # Evaluation results
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Data settings**: Dataset name, splits, frame sampling rate
- **Model architecture**: Backbone choice, feature dimensions, fusion method
- **Training parameters**: Batch size, learning rate, optimizer, scheduler
- **Preprocessing**: Augmentation parameters, face detection settings

## 🏗️ Model Architecture

### Spatial Stream
- Backbone: ResNet18 or EfficientNet-B0 (pretrained on ImageNet)
- Output: 256-dimensional feature vector

### Frequency Stream
- Input: Log-magnitude spectrum (1 channel) or magnitude+phase (2 channels)
- Architecture: Lightweight CNN with 3-5 convolutional blocks
- Output: 256-dimensional feature vector

### Fusion & Classification
- Concatenation or attention-based fusion
- Dense layers with dropout
- Binary classification output

## 📊 Evaluation Metrics

The model reports both frame-level and video-level metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score
- **Recall**: Recall score
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **EER**: Equal Error Rate

## 🔬 Research Background

This implementation is based on research showing that:
- High-frequency components in the frequency domain are particularly informative for deepfake detection
- Combining spatial and frequency features improves robustness across different manipulation methods
- Frequency-domain analysis can reveal artifacts introduced by generative models that are less visible in pixel space

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{dual-stream-deepfake-detection,
  title={Dual-Stream Deepfake Detection},
  author={Your Name},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Hugging Face for dataset hosting
- FaceNet-PyTorch for MTCNN implementation
- PyTorch and torchvision communities
