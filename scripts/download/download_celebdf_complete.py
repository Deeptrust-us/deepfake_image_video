"""Complete download of Celeb-DF v2 dataset using kagglehub."""

import kagglehub
import os
from pathlib import Path
import sys

print("=" * 60)
print("Downloading Celeb-DF v2 Dataset")
print("=" * 60)
print()
print("Dataset: reubensuju/celeb-df-v2")
print("Size: ~9.29GB (compressed)")
print("This may take 30-60 minutes depending on your connection")
print()

try:
    # Download latest version
    print("Starting download...")
    print("(This will resume if previously interrupted)")
    print()
    
    path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
    
    print(f"\n✅ Download complete!")
    print(f"Path to dataset files: {path}")
    print()
    
    # Check what's in the downloaded directory
    if os.path.exists(path):
        print("Checking downloaded files...")
        items = list(Path(path).iterdir())
        print(f"Found {len(items)} items in dataset directory")
        
        # Look for video directories or zip files
        video_dirs = []
        zip_files = []
        
        for item in items:
            if item.is_dir():
                # Check if it contains videos
                videos = list(item.rglob("*.mp4")) + list(item.rglob("*.avi"))
                if videos:
                    video_dirs.append(str(item))
                    print(f"  ✓ Found videos in: {item.name} ({len(videos)} videos)")
            elif item.suffix.lower() in ['.zip', '.tar', '.gz']:
                zip_files.append(str(item))
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  ✓ Found archive: {item.name} ({size_mb:.1f} MB)")
        
        print()
        print("=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print(f"1. Dataset downloaded to: {path}")
        
        if zip_files:
            print("\n2. Extract the archive files:")
            for zip_file in zip_files:
                print(f"   unzip -q {zip_file} -d data/raw/")
            print("\n3. Then run preprocessing:")
            print("   python preprocess.py --dataset-type celebdf --videos-dir data/raw")
        elif video_dirs:
            print("\n2. Videos are already extracted. Run preprocessing:")
            print(f"   python preprocess.py --dataset-type celebdf --videos-dir {path}")
        else:
            print("\n2. Check the directory structure:")
            print(f"   ls -la {path}")
            print("\n3. Then run preprocessing:")
            print(f"   python preprocess.py --dataset-type celebdf --videos-dir {path}")
        
except KeyboardInterrupt:
    print("\n\n⚠️  Download interrupted by user")
    print("You can resume by running this script again")
    print("kagglehub will automatically resume from where it stopped")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error downloading dataset: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have Kaggle API credentials set up")
    print("2. Install: pip install kaggle kagglehub")
    print("3. Set up Kaggle API token:")
    print("   - Go to: https://www.kaggle.com/account")
    print("   - Create API token")
    print("   - Place in: ~/.kaggle/kaggle.json")
    print("4. Or download manually from: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2")
    raise




