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

# 2. Run Integrated Pipeline
echo "2. Running Integrated Pipeline..."
python "$WORK_DIR/scripts/run_pipeline.py" \
    --model all \
    --epochs 1 \
    --num_videos "$NUM_VIDEOS" \
    --server "$SERVER"

# 3. Package Logs and Results
echo "3. Packaging logs and evaluation results..."
mkdir -p "$WORK_DIR/evaluation_logs"
cp -r "$WORK_DIR/results/" "$WORK_DIR/evaluation_logs/"
cp -r "$WORK_DIR/logs/" "$WORK_DIR/evaluation_logs/" 2>/dev/null || true
tar -czf "$WORK_DIR/faceforensics_evaluation_logs.tar.gz" "$WORK_DIR/evaluation_logs/"

echo "========================================================================"
echo " Evaluation Complete! Summary results are in results/pipeline_summary.txt"
echo " Package saved to $WORK_DIR/faceforensics_evaluation_logs.tar.gz"
echo "========================================================================"
