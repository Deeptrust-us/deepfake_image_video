#!/usr/bin/env bash

# Automated execution script for running FaceForensics++ evaluation on Google Colab
# Models: Xception baseline & RGB+FFT Dual-Stream model

set -e

NUM_VIDEOS=${1:-50}
SERVER=${2:-"EU2"}
WORK_DIR="/content/deepfake_image_video"
export NONINTERACTIVE=1

echo "========================================================================"
echo " Starting Deepfake Detection Evaluation for FaceForensics++ (Colab)    "
echo "========================================================================"

# 0. Clone repository if not already in directory
if [ ! -d "$WORK_DIR" ]; then
    echo "0. Cloning repository https://github.com/Deeptrust-us/deepfake_image_video.git..."
    git clone https://github.com/Deeptrust-us/deepfake_image_video.git "$WORK_DIR"
fi

cd "$WORK_DIR"

# 1. Install Dependencies
echo "1. Installing Python dependencies..."
pip install -q -r "$WORK_DIR/requirements.txt"

# 2. Download FaceForensics++ Subset
echo "2. Downloading FaceForensics++ (Deepfakes, c23 compression, num_videos=${NUM_VIDEOS})..."
mkdir -p "$WORK_DIR/data/faceforensics_raw"

python "$WORK_DIR/scripts/download/download_faceforensics.py" "$WORK_DIR/data/faceforensics_raw" \
    -d Deepfakes \
    -c c23 \
    -t videos \
    -n ${NUM_VIDEOS} \
    --server ${SERVER}

python "$WORK_DIR/scripts/download/download_faceforensics.py" "$WORK_DIR/data/faceforensics_raw" \
    -d original \
    -c c23 \
    -t videos \
    -n ${NUM_VIDEOS} \
    --server ${SERVER}

# 3. Preprocess Dataset
echo "3. Preprocessing FaceForensics++ frames and faces..."
python "$WORK_DIR/scripts/preprocess.py" \
    --dataset-type faceforensics \
    --videos-dir "$WORK_DIR/data/faceforensics_raw" \
    --max_videos $((NUM_VIDEOS * 2))

# 4. Evaluate Xception Baseline Model
echo "4. Running Evaluation: Xception (baseline)..."
python "$WORK_DIR/scripts/evaluate.py" \
    --model xception \
    --config "$WORK_DIR/config/config.yaml" \
    --split test \
    --optimal_threshold \
    --output_dir "$WORK_DIR/results"

# 5. Evaluate RGB+FFT Dual-Stream Model
echo "5. Running Evaluation: RGB+FFT Dual-Stream..."
python "$WORK_DIR/scripts/evaluate.py" \
    --model rgb_fft_dual_stream \
    --config "$WORK_DIR/config/config.yaml" \
    --split test \
    --optimal_threshold \
    --output_dir "$WORK_DIR/results"

# 6. Package Logs and Results
echo "6. Packaging logs and evaluation results..."
mkdir -p "$WORK_DIR/evaluation_logs"
cp -r "$WORK_DIR/results/" "$WORK_DIR/evaluation_logs/"
cp -r "$WORK_DIR/logs/" "$WORK_DIR/evaluation_logs/" 2>/dev/null || true
tar -czf "$WORK_DIR/faceforensics_evaluation_logs.tar.gz" "$WORK_DIR/evaluation_logs/"

echo "========================================================================"
echo " Evaluation Complete! Results saved to $WORK_DIR/results and $WORK_DIR/faceforensics_evaluation_logs.tar.gz"
echo "========================================================================"
