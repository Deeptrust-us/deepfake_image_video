"""HuggingFace dataset download and preprocessing utilities."""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

from src.data.local_preprocessing import process_local_dataset


def download_and_preprocess_huggingface_dataset(
    dataset_name: str = "UniDataPro/deepfake-videos-dataset",
    output_root: str = "data",
    fps: int = 3,
    use_phase: bool = False,
    device: str = "cpu",
    max_videos: Optional[int] = None
) -> str:
    """Download HuggingFace video preview dataset and preprocess it."""
    print(f"Downloading HuggingFace dataset '{dataset_name}'...")
    raw_dir = os.path.join(output_root, "raw_huggingface")
    os.makedirs(raw_dir, exist_ok=True)

    try:
        from datasets import load_dataset
        hf_dataset = load_dataset(dataset_name, split="train")

        # Save downloaded videos to disk
        for idx, item in enumerate(hf_dataset):
            if max_videos is not None and idx >= max_videos:
                break

            video_bytes = item.get('video', None) or item.get('file', None)
            label = item.get('label', 1)

            video_filename = os.path.join(raw_dir, f"video_{idx:04d}_{'fake' if label == 1 else 'real'}.mp4")

            if video_bytes is not None and not os.path.exists(video_filename):
                if isinstance(video_bytes, bytes):
                    with open(video_filename, 'wb') as f:
                        f.write(video_bytes)
                elif isinstance(video_bytes, str) and os.path.exists(video_bytes):
                    import shutil
                    shutil.copy(video_bytes, video_filename)

    except Exception as e:
        print(f"HuggingFace dataset download note: {e}")
        print("Will attempt local processing on any available videos in raw directory.")

    return process_local_dataset(
        videos_dir=raw_dir,
        output_root=output_root,
        fps=fps,
        use_phase=use_phase,
        device=device,
        max_videos=max_videos
    )
