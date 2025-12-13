#!/usr/bin/env python3
"""Complete dataset analysis and verification script."""

import json
import os
import numpy as np
from pathlib import Path
from collections import Counter

def analyze_dataset():
    """Perform complete dataset analysis."""
    
    data_root = "data"
    
    print("=" * 80)
    print("COMPLETE DATASET ANALYSIS")
    print("=" * 80)
    
    # Load metadata files
    splits = {}
    for split in ['train', 'val', 'test']:
        metadata_file = os.path.join(data_root, f"{split}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                splits[split] = json.load(f)
        else:
            splits[split] = []
    
    # Overall statistics
    print("\n📊 OVERALL STATISTICS")
    print("-" * 80)
    total_videos = sum(len(splits[s]) for s in splits)
    print(f"Total Videos: {total_videos:,}")
    for split in ['train', 'val', 'test']:
        count = len(splits[split])
        pct = (count / total_videos * 100) if total_videos > 0 else 0
        print(f"  {split.capitalize():12s}: {count:5,} videos ({pct:5.1f}%)")
    
    # Class distribution
    print("\n📈 CLASS DISTRIBUTION")
    print("-" * 80)
    for split in ['train', 'val', 'test']:
        if len(splits[split]) == 0:
            continue
        labels = [v['label'] for v in splits[split]]
        real_count = sum(1 for l in labels if l == 0)
        fake_count = sum(1 for l in labels if l == 1)
        total = len(labels)
        print(f"\n{split.capitalize()} Split:")
        print(f"  Real videos: {real_count:5,} ({real_count/total*100:5.1f}%)")
        print(f"  Fake videos: {fake_count:5,} ({fake_count/total*100:5.1f}%)")
        print(f"  Imbalance ratio: {fake_count/real_count if real_count > 0 else 'N/A':.2f}:1")
    
    # Frame statistics
    print("\n🎬 FRAME STATISTICS")
    print("-" * 80)
    for split in ['train', 'val', 'test']:
        if len(splits[split]) == 0:
            continue
        frame_counts = [v['num_frames'] for v in splits[split]]
        total_frames = sum(frame_counts)
        print(f"\n{split.capitalize()} Split:")
        print(f"  Mean frames/video: {np.mean(frame_counts):.1f}")
        print(f"  Median frames/video: {np.median(frame_counts):.1f}")
        print(f"  Min frames: {min(frame_counts)}")
        print(f"  Max frames: {max(frame_counts)}")
        print(f"  Total frames: {total_frames:,}")
        print(f"  Std deviation: {np.std(frame_counts):.1f}")
    
    # Video duration estimation (assuming 3 FPS)
    fps = 3
    print("\n⏱️  VIDEO DURATION ESTIMATION")
    print("-" * 80)
    print(f"(Based on {fps} FPS frame sampling rate)")
    for split in ['train', 'val', 'test']:
        if len(splits[split]) == 0:
            continue
        frame_counts = [v['num_frames'] for v in splits[split]]
        durations = [f / fps for f in frame_counts]
        print(f"\n{split.capitalize()} Split:")
        print(f"  Mean duration: {np.mean(durations):.1f} seconds ({np.mean(durations)/60:.1f} minutes)")
        print(f"  Min duration: {min(durations):.1f} seconds")
        print(f"  Max duration: {max(durations):.1f} seconds ({max(durations)/60:.1f} minutes)")
    
    # Verify actual files on disk
    print("\n🔍 FILE SYSTEM VERIFICATION")
    print("-" * 80)
    faces_dir = os.path.join(data_root, "faces")
    frames_dir = os.path.join(data_root, "frames")
    frequency_dir = os.path.join(data_root, "frequency")
    
    if os.path.exists(faces_dir):
        video_dirs = [d for d in os.listdir(faces_dir) 
                     if os.path.isdir(os.path.join(faces_dir, d))]
        print(f"\nFace directories found: {len(video_dirs):,}")
        
        # Sample check
        if len(video_dirs) > 0:
            sample_video = video_dirs[0]
            sample_faces_dir = os.path.join(faces_dir, sample_video)
            face_files = [f for f in os.listdir(sample_faces_dir) if f.endswith('.jpg')]
            print(f"Sample video '{sample_video}': {len(face_files)} face crops")
            
            # Check if metadata matches actual files
            mismatches = []
            checked = 0
            for split in ['train', 'val', 'test']:
                for video in splits[split][:100]:  # Check first 100 per split
                    video_id = video['video_id']
                    expected_frames = video['num_frames']
                    video_faces_dir = os.path.join(faces_dir, video_id)
                    if os.path.exists(video_faces_dir):
                        actual_frames = len([f for f in os.listdir(video_faces_dir) 
                                           if f.endswith('.jpg')])
                        if actual_frames != expected_frames:
                            mismatches.append((video_id, expected_frames, actual_frames))
                        checked += 1
            
            if mismatches:
                print(f"\n⚠️  Found {len(mismatches)} mismatches (expected vs actual frames):")
                for vid, exp, act in mismatches[:5]:
                    print(f"  {vid}: expected {exp}, found {act}")
            else:
                print(f"\n✓ Checked {checked} videos: All frame counts match metadata")
    
    # Training sample calculation
    print("\n🎯 TRAINING DETAILS")
    print("-" * 80)
    train_count = len(splits['train'])
    batch_size = 32  # From config
    batches_per_epoch = train_count // batch_size
    print(f"Training videos: {train_count:,}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {batches_per_epoch:,}")
    print(f"Samples per epoch: {train_count:,} (one random frame per video)")
    print("\nNote: During training, a RANDOM frame is selected from each video each epoch.")
    print("      This provides temporal diversity and effective data augmentation.")
    
    # Data pipeline summary
    print("\n🔄 DATA PIPELINE SUMMARY")
    print("-" * 80)
    print("1. Video (MP4) → ffmpeg extracts frames @ 3 FPS")
    print("2. Raw frames → MTCNN detects and crops faces (224x224)")
    print("3. Face crops → FFT computes frequency spectrum")
    print("4. During training:")
    print("   - Training: Random frame selected from each video")
    print("   - Validation/Test: First frame (frame_0001.jpg) used")
    print("5. Each sample provides:")
    print("   - Spatial stream: Face crop image (3 channels, 224x224)")
    print("   - Frequency stream: FFT magnitude (1 channel, 224x224)")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    analyze_dataset()

