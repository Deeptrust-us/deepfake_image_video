#!/bin/bash
# Helper script to download and setup FaceForensics++ dataset

set -e

# Configuration
DOWNLOAD_DIR="${1:-data/faceforensics_raw}"
COMPRESSION="${2:-c23}"  # Options: raw, c23, c40 (c23 is good balance)
NUM_VIDEOS="${3:-100}"   # Number of videos per dataset (None for all)
SERVER="${4:-EU}"        # Server: EU, EU2, CA

echo "=========================================="
echo "FaceForensics++ Download Helper"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Download directory: $DOWNLOAD_DIR"
echo "  Compression: $COMPRESSION"
echo "  Videos per dataset: ${NUM_VIDEOS:-all}"
echo "  Server: $SERVER"
echo ""
echo "Note: This will download videos from FaceForensics++"
echo "      You must agree to their terms of use."
echo ""
read -p "Press Enter to continue or CTRL-C to cancel..."

# Create download directory
mkdir -p "$DOWNLOAD_DIR"

# Download datasets
# You can customize which datasets to download
# Options: original, Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, DeepFakeDetection

echo ""
echo "Downloading FaceForensics++ datasets..."
echo "This may take a while depending on your internet speed..."
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Download original (real) videos
echo "1. Downloading original (real) videos..."
python scripts/download/download_faceforensics.py "$DOWNLOAD_DIR" \
    -d original \
    -c "$COMPRESSION" \
    -t videos \
    ${NUM_VIDEOS:+-n $NUM_VIDEOS} \
    --server "$SERVER"

# Download manipulated (fake) videos
for dataset in Deepfakes Face2Face FaceSwap NeuralTextures; do
    echo ""
    echo "2. Downloading $dataset (fake) videos..."
    python scripts/download/download_faceforensics.py "$DOWNLOAD_DIR" \
        -d "$dataset" \
        -c "$COMPRESSION" \
        -t videos \
        ${NUM_VIDEOS:+-n $NUM_VIDEOS} \
        --server "$SERVER"
done

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Preprocess the downloaded videos:"
echo "   python scripts/preprocess.py --dataset-type faceforensics --videos-dir $DOWNLOAD_DIR"
echo ""
echo "2. Or process manually:"
echo "   python scripts/preprocess.py --dataset-type faceforensics --videos-dir $DOWNLOAD_DIR --max_videos 1000"
echo ""

