"""Local dataset preprocessing utilities for FaceForensics++, Celeb-DF, and custom local datasets."""

import os
import json
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

from src.utils.face_detection import FaceDetector
from src.utils.frequency_domain import prepare_frequency_input


def extract_frames_from_video(video_path: str, target_fps: int = 3, max_frames: int = 50) -> List[Tuple[int, np.ndarray]]:
    """Extract frames from video file at target FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0

    frame_interval = max(1, int(round(original_fps / target_fps)))
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_count, frame_rgb))
            if len(frames) >= max_frames:
                break

        frame_count += 1

    cap.release()
    return frames


def save_processed_sample(
    video_id: str,
    frame_idx: int,
    frame_rgb: np.ndarray,
    face_detector: Optional[FaceDetector],
    output_root: str,
    use_phase: bool = False
) -> Dict[str, Any]:
    """Process single frame: detect face, compute frequency, save files, return metadata entry."""
    frames_dir = os.path.join(output_root, "frames", video_id)
    faces_dir = os.path.join(output_root, "faces", video_id)
    freq_dir = os.path.join(output_root, "frequency", video_id)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)
    os.makedirs(freq_dir, exist_ok=True)

    frame_name = f"frame_{frame_idx:04d}"
    frame_file = os.path.join(frames_dir, f"{frame_name}.jpg")
    face_file = os.path.join(faces_dir, f"{frame_name}.jpg")
    freq_file = os.path.join(freq_dir, f"{frame_name}.npy")

    # Resize full frame to 224x224
    frame_224 = cv2.resize(frame_rgb, (224, 224))
    cv2.imwrite(frame_file, cv2.cvtColor(frame_224, cv2.COLOR_RGB2BGR))

    # Detect face or fallback to full frame
    face_crop = None
    if face_detector is not None:
        try:
            face_crop = face_detector.detect_and_align(frame_rgb)
        except Exception:
            face_crop = None

    if face_crop is not None:
        # face_crop is float32 [0, 1]
        face_img_u8 = (face_crop * 255.0).astype(np.uint8)
    else:
        face_crop = frame_224.astype(np.float32) / 255.0
        face_img_u8 = frame_224

    cv2.imwrite(face_file, cv2.cvtColor(face_img_u8, cv2.COLOR_RGB2BGR))

    # Compute FFT on face crop
    freq_data = prepare_frequency_input(face_crop, use_phase=use_phase, normalize=True)
    np.save(freq_file, freq_data)

    # Return relative paths for dataset metadata
    rel_frame = os.path.relpath(frame_file, output_root)
    rel_face = os.path.relpath(face_file, output_root)
    rel_freq = os.path.relpath(freq_file, output_root)

    return {
        'video_id': video_id,
        'frame_id': frame_name,
        'frame_path': rel_frame,
        'face_path': rel_face,
        'face_frequency_path': rel_freq,
        'frequency_path': rel_freq
    }


def split_and_save_metadata(sample_entries: List[Dict[str, Any]], output_root: str, train_ratio=0.7, val_ratio=0.15):
    """Group samples by video_id and save stratified train/val/test metadata JSON files."""
    video_map = {}
    for sample in sample_entries:
        vid = sample['video_id']
        if vid not in video_map:
            video_map[vid] = []
        video_map[vid].append(sample)

    video_ids = list(video_map.keys())
    np.random.seed(42)
    np.random.shuffle(video_ids)

    n_total = len(video_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_vids = set(video_ids[:n_train])
    val_vids = set(video_ids[n_train:n_train + n_val])
    test_vids = set(video_ids[n_train + n_val:])

    train_samples, val_samples, test_samples = [], [], []

    for vid, samples in video_map.items():
        if vid in train_vids:
            train_samples.extend(samples)
        elif vid in val_vids:
            val_samples.extend(samples)
        else:
            test_samples.extend(samples)

    # Save to JSON files
    for split_name, split_data in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        out_file = os.path.join(output_root, f"{split_name}_metadata.json")
        with open(out_file, 'w') as f:
            json.dump(split_data, f, indent=2)

    print(f"Dataset split saved to {output_root}: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)} frames.")
    return os.path.join(output_root, "train_metadata.json")


def process_faceforensics_structure(
    videos_dir: str,
    output_root: str = "data",
    fps: int = 3,
    use_phase: bool = False,
    device: str = "cpu",
    max_videos: Optional[int] = None
) -> str:
    """
    Process FaceForensics++ video directory structure:
    - original_sequences/youtube/c23/videos/ (real, label 0)
    - manipulated_sequences/Deepfakes/c23/videos/ (fake, label 1)
    or flat directory containing mp4 files.
    """
    print(f"Preprocessing FaceForensics++ dataset from: {videos_dir}")

    # Search for real videos
    real_patterns = [
        os.path.join(videos_dir, "original_sequences", "youtube", "*", "videos", "*.mp4"),
        os.path.join(videos_dir, "original_sequences", "*", "*", "videos", "*.mp4"),
        os.path.join(videos_dir, "original", "*.mp4"),
        os.path.join(videos_dir, "real", "*.mp4"),
    ]
    real_files = []
    for pat in real_patterns:
        real_files.extend(glob.glob(pat))

    # Search for fake videos (Deepfakes subset)
    fake_patterns = [
        os.path.join(videos_dir, "manipulated_sequences", "Deepfakes", "*", "videos", "*.mp4"),
        os.path.join(videos_dir, "manipulated_sequences", "*", "*", "videos", "*.mp4"),
        os.path.join(videos_dir, "Deepfakes", "*.mp4"),
        os.path.join(videos_dir, "fake", "*.mp4"),
    ]
    fake_files = []
    for pat in fake_patterns:
        fake_files.extend(glob.glob(pat))

    # Fallback: if flat structure
    if not real_files and not fake_files:
        all_videos = glob.glob(os.path.join(videos_dir, "**", "*.mp4"), recursive=True)
        for v in all_videos:
            v_lower = v.lower()
            if "original" in v_lower or "real" in v_lower or "youtube" in v_lower:
                real_files.append(v)
            else:
                fake_files.append(v)

    video_list = [(v, 0) for v in real_files] + [(v, 1) for v in fake_files]

    if max_videos is not None and max_videos > 0:
        video_list = video_list[:max_videos]

    print(f"Found {len(real_files)} real videos and {len(fake_files)} fake videos (processing {len(video_list)} total).")

    # Initialize FaceDetector
    try:
        detector = FaceDetector(device=device)
    except Exception as e:
        print(f"Warning: Could not initialize FaceDetector ({e}). Falling back to resizing.")
        detector = None

    all_sample_entries = []

    for video_path, label in tqdm(video_list, desc="Processing FF++ videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_id = f"ffpp_{label}_{video_name}"

        frames = extract_frames_from_video(video_path, target_fps=fps)
        for frame_idx, frame_rgb in frames:
            entry = save_processed_sample(
                video_id=video_id,
                frame_idx=frame_idx,
                frame_rgb=frame_rgb,
                face_detector=detector,
                output_root=output_root,
                use_phase=use_phase
            )
            entry['label'] = label
            all_sample_entries.append(entry)

    return split_and_save_metadata(all_sample_entries, output_root)


def process_celebdf_structure(
    videos_dir: str,
    output_root: str = "data",
    fps: int = 3,
    use_phase: bool = False,
    device: str = "cpu",
    max_videos: Optional[int] = None
) -> str:
    """Process Celeb-DF dataset structure."""
    return process_local_dataset(
        videos_dir=videos_dir,
        output_root=output_root,
        label_mapping={'Celeb-real': 0, 'YouTube-real': 0, 'Celeb-synthesis': 1},
        fps=fps,
        use_phase=use_phase,
        device=device,
        max_videos=max_videos
    )


def process_local_dataset(
    videos_dir: str,
    output_root: str = "data",
    label_mapping: Optional[Dict[str, int]] = None,
    fps: int = 3,
    use_phase: bool = False,
    device: str = "cpu",
    max_videos: Optional[int] = None
) -> str:
    """Process generic local dataset directory."""
    print(f"Preprocessing local dataset from: {videos_dir}")

    all_videos = glob.glob(os.path.join(videos_dir, "**", "*.mp4"), recursive=True) + \
                 glob.glob(os.path.join(videos_dir, "**", "*.avi"), recursive=True)

    if max_videos is not None and max_videos > 0:
        all_videos = all_videos[:max_videos]

    video_list = []
    for v_path in all_videos:
        label = 1  # Default fake
        if label_mapping:
            for folder_name, l_val in label_mapping.items():
                if folder_name in v_path:
                    label = l_val
                    break
        else:
            v_lower = v_path.lower()
            if "real" in v_lower or "original" in v_lower:
                label = 0

        video_list.append((v_path, label))

    try:
        detector = FaceDetector(device=device)
    except Exception:
        detector = None

    all_sample_entries = []

    for video_path, label in tqdm(video_list, desc="Processing videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_id = f"local_{label}_{video_name}"

        frames = extract_frames_from_video(video_path, target_fps=fps)
        for frame_idx, frame_rgb in frames:
            entry = save_processed_sample(
                video_id=video_id,
                frame_idx=frame_idx,
                frame_rgb=frame_rgb,
                face_detector=detector,
                output_root=output_root,
                use_phase=use_phase
            )
            entry['label'] = label
            all_sample_entries.append(entry)

    return split_and_save_metadata(all_sample_entries, output_root)
