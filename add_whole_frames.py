#!/usr/bin/env python3
"""Add resized whole frames to existing preprocessed dataset."""

import os
import cv2
import json
from tqdm import tqdm
from pathlib import Path

def add_whole_frames_to_existing_dataset(data_root="data"):
    """
    Add resized whole frames (224x224) to existing dataset that only has face crops.
    This assumes frames were extracted but not resized.
    """
    frames_dir = os.path.join(data_root, "frames")
    faces_dir = os.path.join(data_root, "faces")
    
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    
    if not os.path.exists(faces_dir):
        print(f"Error: Faces directory not found: {faces_dir}")
        return
    
    # Get all video IDs from faces directory
    video_ids = [d for d in os.listdir(faces_dir) 
                 if os.path.isdir(os.path.join(faces_dir, d))]
    
    print(f"Found {len(video_ids)} videos")
    print("Adding resized whole frames...")
    
    processed = 0
    skipped = 0
    
    for video_id in tqdm(video_ids):
        video_frames_dir = os.path.join(frames_dir, video_id)
        video_faces_dir = os.path.join(faces_dir, video_id)
        
        if not os.path.exists(video_frames_dir):
            skipped += 1
            continue
        
        # Get face files to know which frames to process
        face_files = sorted([f for f in os.listdir(video_faces_dir) if f.endswith('.jpg')])
        
        for face_file in face_files:
            # Check if corresponding frame exists
            frame_path = os.path.join(video_frames_dir, face_file)
            
            if not os.path.exists(frame_path):
                continue
            
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Check if already resized (224x224)
            if frame.shape[:2] == (224, 224):
                continue  # Already resized, skip
            
            # Resize to 224x224
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            
            # Save resized frame (overwrite)
            cv2.imwrite(frame_path, cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))
            processed += 1
    
    print(f"\n✓ Processed {processed} frames")
    print(f"  Skipped {skipped} videos (no frames directory)")
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add resized whole frames to existing dataset")
    parser.add_argument("--data-root", type=str, default="data", help="Root directory of dataset")
    args = parser.parse_args()
    
    add_whole_frames_to_existing_dataset(args.data_root)

