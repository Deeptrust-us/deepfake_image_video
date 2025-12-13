"""Download Celeb-DF v2 dataset using kagglehub."""

import kagglehub
import os
from pathlib import Path

print("=" * 60)
print("Downloading Celeb-DF v2 Dataset")
print("=" * 60)
print()

# Download latest version
print("Downloading dataset (this may take a while)...")
print("Dataset size: ~100GB, please be patient...")
print()

try:
    path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
    print(f"\n✅ Download complete!")
    print(f"Path to dataset files: {path}")
    print()
    
    # Check what's in the downloaded directory
    if os.path.exists(path):
        print("Checking downloaded files...")
        items = list(Path(path).iterdir())
        print(f"Found {len(items)} items in dataset directory")
        
        # Look for video directories
        video_dirs = []
        for item in items:
            if item.is_dir():
                # Check if it contains videos
                videos = list(item.rglob("*.mp4")) + list(item.rglob("*.avi"))
                if videos:
                    video_dirs.append(str(item))
                    print(f"  ✓ Found videos in: {item.name} ({len(videos)} videos)")
        
        print()
        print("=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print(f"1. Dataset downloaded to: {path}")
        print("2. Run preprocessing:")
        print(f"   python preprocess.py --dataset-type celebdf --videos-dir {path}")
        print()
        print("Or if videos are in a subdirectory:")
        if video_dirs:
            print(f"   python preprocess.py --dataset-type celebdf --videos-dir {video_dirs[0]}")
        
except Exception as e:
    print(f"\n❌ Error downloading dataset: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have Kaggle API credentials set up")
    print("2. Run: pip install kaggle")
    print("3. Set up Kaggle API token: https://www.kaggle.com/docs/api")
    print("4. Or download manually from: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2")
    raise


