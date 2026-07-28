"""Download FaceForensics++ C23 dataset using kagglehub and link it to data/faceforensics_raw."""

import kagglehub
import os
from pathlib import Path
import sys
import shutil

print("=" * 60)
print("Downloading FaceForensics++ C23 Dataset")
print("=" * 60)
print()

# Download latest version
print("Downloading dataset (this may take a while)...")
print("Dataset size: ~17.9GB, please be patient...")
print()

try:
    path = kagglehub.dataset_download("xdxd003/ff-c23")
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
        
        # Set up symlink to data/faceforensics_raw
        target_link = os.path.join("data", "faceforensics_raw")
        os.makedirs(os.path.dirname(target_link), exist_ok=True)
        
        if os.path.exists(target_link) or os.path.islink(target_link):
            print(f"\nRemoving existing path/link at {target_link}...")
            if os.path.islink(target_link):
                os.unlink(target_link)
            elif os.path.isdir(target_link):
                shutil.rmtree(target_link)
            else:
                os.remove(target_link)
        
        print(f"Creating symlink from {path} to {target_link}...")
        os.symlink(path, target_link)
        print("✓ Symlink created successfully!")
        
        print()
        print("=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("1. Run preprocessing (using the linked directory):")
        print(f"   python scripts/preprocess.py --config config/config.yaml --dataset-type faceforensics --videos-dir {target_link}")
        print()
        
except Exception as e:
    print(f"\n❌ Error downloading dataset: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have Kaggle API credentials set up")
    print("2. Run: pip install kaggle kagglehub")
    print("3. Set up Kaggle API token: https://www.kaggle.com/docs/api")
    print("4. Or download manually from: https://www.kaggle.com/datasets/xdxd003/ff-c23")
    raise
