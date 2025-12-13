#!/bin/bash
# Script to help download larger deepfake datasets

echo "============================================"
echo "GETTING MORE VIDEOS FOR TRAINING"
echo "============================================"
echo ""
echo "Current status: Only 10 videos (6 for training)"
echo "Required: 100+ videos minimum, 1000+ recommended"
echo ""

# Check available disk space
echo "Checking disk space..."
df -h . | tail -1
echo ""

echo "=== Option 1: Celeb-DF v2 (Recommended) ==="
echo "Size: ~6,229 videos (~100GB)"
echo "Steps:"
echo "1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics"
echo "2. Follow their download instructions"
echo "3. Extract videos to: data/raw/"
echo "4. Run: python scripts/preprocess.py --dataset-type celebdf --videos-dir data/raw"
echo ""

echo "=== Option 2: FaceForensics++ ==="
echo "Size: ~5,000 videos (~50GB)"
echo "Steps:"
echo "1. Visit: https://github.com/ondyari/FaceForensics"
echo "2. Follow their download instructions"
echo "3. Extract videos to: data/raw/"
echo "4. Run: python scripts/preprocess.py --dataset-type faceforensics --videos-dir data/raw"
echo ""

echo "=== Option 3: Quick Test with Synthetic Data ==="
echo "We can create a script to download sample videos from YouTube"
echo "Would you like me to create a script for that? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Creating YouTube download script..."
    cat > download_youtube_samples.sh << 'EOF'
#!/bin/bash
# Download sample videos from YouTube for testing
# Note: This is for testing only. For real training, use proper datasets.

mkdir -p data/raw/youtube_samples

echo "Downloading sample videos..."
# Add YouTube URLs here
# yt-dlp -f "best[height<=720]" -o "data/raw/youtube_samples/%(title)s.%(ext)s" <URL>

echo "Done. Then run: python scripts/preprocess.py --dataset-type local --videos-dir data/raw/youtube_samples"
EOF
    chmod +x download_youtube_samples.sh
    echo "Created download_youtube_samples.sh"
fi

echo ""
echo "=== Recommendation ==="
echo "For proper training, download Celeb-DF v2 or FaceForensics++"
echo "The current 10 videos are only good for testing the pipeline."


