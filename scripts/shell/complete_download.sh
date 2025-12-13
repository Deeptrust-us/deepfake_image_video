#!/bin/bash
# Complete and verify Celeb-DF v2 download

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

echo "============================================"
echo "Completing Celeb-DF v2 Download"
echo "============================================"
echo ""

# Run the download script (will resume if needed)
python scripts/download/download_celebdf_complete.py

echo ""
echo "============================================"
echo "Download Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Check the dataset location shown above"
echo "2. Extract any zip files if needed"
echo "3. Run preprocessing:"
echo "   python scripts/preprocess.py --dataset-type celebdf --videos-dir <dataset_path>"
echo ""




