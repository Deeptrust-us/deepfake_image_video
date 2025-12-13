#!/bin/bash
# Complete and verify Celeb-DF v2 download

cd /home/felipeEngin/Documents/deepfake_image_video

echo "============================================"
echo "Completing Celeb-DF v2 Download"
echo "============================================"
echo ""

# Run the download script (will resume if needed)
python download_celebdf_complete.py

echo ""
echo "============================================"
echo "Download Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Check the dataset location shown above"
echo "2. Extract any zip files if needed"
echo "3. Run preprocessing:"
echo "   python preprocess.py --dataset-type celebdf --videos-dir <dataset_path>"
echo ""




