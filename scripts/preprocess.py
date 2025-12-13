"""Data preprocessing script for downloading and processing deepfake datasets."""

import os
import sys
import argparse
import yaml
import torch

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import download_and_preprocess_huggingface_dataset
from src.data.local_preprocessing import (
    process_local_dataset,
    process_celebdf_structure,
    process_faceforensics_structure
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess deepfake dataset")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name from config")
    parser.add_argument("--dataset-type", type=str, default=None, 
                       choices=["huggingface", "local", "celebdf", "faceforensics"],
                       help="Dataset type")
    parser.add_argument("--videos-dir", type=str, default=None, 
                       help="Directory containing videos (for local datasets)")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--device", type=str, default=None, help="Device for face detection (cpu/cuda)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Determine dataset type
    dataset_type = args.dataset_type or config['data'].get('dataset_type', 'huggingface')
    
    if dataset_type == "huggingface":
        # Hugging Face dataset
        dataset_name = args.dataset or config['data']['dataset_name']
        print(f"\n⚠️  NOTE: The Hugging Face dataset '{dataset_name}' only provides a 10-video preview.")
        print("   For larger datasets, use local videos with --dataset-type local")
        print("   or download Celeb-DF v2 / FaceForensics++ separately.\n")
        
        metadata_file = download_and_preprocess_huggingface_dataset(
            dataset_name=dataset_name,
            output_root=config['data']['data_root'],
            fps=config['data']['frame_sampling_rate'],
            use_phase=(config['model']['frequency_channels'] == 2),
            device=device,
            max_videos=args.max_videos
        )
    
    elif dataset_type in ["local", "celebdf", "faceforensics"]:
        # Local dataset
        videos_dir = args.videos_dir or config['data'].get('raw_videos_dir', 'data/raw')
        
        if not os.path.exists(videos_dir):
            print(f"Error: Videos directory not found: {videos_dir}")
            print("\nTo use local datasets:")
            print("1. Download Celeb-DF v2 or FaceForensics++")
            print("2. Place videos in data/raw/")
            print("3. Run: python preprocess.py --dataset-type celebdf --videos-dir data/raw")
            return
        
        if dataset_type == "celebdf":
            metadata_file = process_celebdf_structure(
                videos_dir=videos_dir,
                output_root=config['data']['data_root'],
                fps=config['data']['frame_sampling_rate'],
                use_phase=(config['model']['frequency_channels'] == 2),
                device=device,
                max_videos=args.max_videos
            )
        elif dataset_type == "faceforensics":
            metadata_file = process_faceforensics_structure(
                videos_dir=videos_dir,
                output_root=config['data']['data_root'],
                fps=config['data']['frame_sampling_rate'],
                use_phase=(config['model']['frequency_channels'] == 2),
                device=device,
                max_videos=args.max_videos
            )
        else:
            metadata_file = process_local_dataset(
                videos_dir=videos_dir,
                output_root=config['data']['data_root'],
                label_mapping=None,
                fps=config['data']['frame_sampling_rate'],
                use_phase=(config['model']['frequency_channels'] == 2),
                device=device,
                max_videos=args.max_videos
            )
    else:
        print(f"Unknown dataset type: {dataset_type}")
        return
    
    print(f"\n✅ Preprocessing complete! Metadata saved to: {metadata_file}")
    print("You can now run training with: python train.py")


if __name__ == "__main__":
    main()

