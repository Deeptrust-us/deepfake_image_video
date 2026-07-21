#!/usr/bin/env bash

# Automated execution script for running FaceForensics++ evaluation on Google Colab
# Models: Xception baseline & RGB+FFT Dual-Stream model

set -e

NUM_VIDEOS=${1:-50}
SERVER=${2:-"EU"}

echo "========================================================================"
echo " Starting Deepfake Detection Evaluation for FaceForensics++ (Colab)    "
echo "========================================================================"

# 1. Install Dependencies
echo "1. Installing Python dependencies..."
pip install -q -r requirements.txt

# 2. Download FaceForensics++ Subset
echo "2. Downloading FaceForensics++ (Deepfakes, c23 compression, num_videos=${NUM_VIDEOS})..."
mkdir -p data/faceforensics_raw

python scripts/download/download_faceforensics.py data/faceforensics_raw \
    -d Deepfakes \
    -c c23 \
    -t videos \
    -n ${NUM_VIDEOS} \
    --server ${SERVER} \
    -y

python scripts/download/download_faceforensics.py data/faceforensics_raw \
    -d original \
    -c c23 \
    -t videos \
    -n ${NUM_VIDEOS} \
    --server ${SERVER} \
    -y

# 3. Preprocess Dataset
echo "3. Preprocessing FaceForensics++ frames and faces..."
python scripts/preprocess.py \
    --dataset-type faceforensics \
    --videos-dir data/faceforensics_raw \
    --max_videos $((NUM_VIDEOS * 2))

# 4. Evaluate Xception Baseline Model
echo "4. Running Evaluation: Xception (baseline)..."
python scripts/evaluate.py \
    --model xception \
    --config config/config.yaml \
    --split test \
    --optimal_threshold \
    --output_dir results

# 5. Evaluate RGB+FFT Dual-Stream Model
echo "5. Running Evaluation: RGB+FFT Dual-Stream..."
python scripts/evaluate.py \
    --model rgb_fft_dual_stream \
    --config config/config.yaml \
    --split test \
    --optimal_threshold \
    --output_dir results

# 6. Package Logs and Results
echo "6. Packaging logs and evaluation results..."
mkdir -p evaluation_logs
cp -r results/ evaluation_logs/
cp -r logs/ evaluation_logs/ 2>/dev/null || true
tar -czf faceforensics_evaluation_logs.tar.gz evaluation_logs/

echo "========================================================================"
echo " Evaluation Complete! Results saved to results/ and faceforensics_evaluation_logs.tar.gz"
echo "========================================================================"
