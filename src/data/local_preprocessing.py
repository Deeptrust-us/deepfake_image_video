"""Local dataset preprocessing utilities for FaceForensics++, Celeb-DF, and custom local datasets."""

import os
import json
import glob
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

from src.utils.face_detection import FaceDetector
from src.utils.frequency_domain import prepare_frequency_input


def extract_frames_from_video(video_path: str, target_fps: int = 3, max_frames: int = 50) -> List[Tuple[int, np.ndarray]]:
    """Extract exactly 32 candidate frames sampled uniformly from the entire video."""
    num_frames = 32
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    current_idx = 0
    indices_set = set(indices)

    while cap.isOpened() and len(frames) < len(indices_set):
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((current_idx, frame_rgb))
        current_idx += 1

    cap.release()
    frames.sort(key=lambda x: x[0])
    return frames


def save_processed_sample(
    video_id: str,
    frame_idx: int,
    frame_rgb: np.ndarray,
    face_detector: Optional[FaceDetector],
    output_root: str,
    use_phase: bool = False,
    video_path: str = "",
    face_crop: Optional[np.ndarray] = None
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
    if face_crop is None and face_detector is not None:
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
        'frequency_path': rel_freq,
        'video_path': video_path
    }


# Global variable for worker process's FaceDetector
worker_detector = None

def init_worker(device):
    global worker_detector
    try:
        worker_detector = FaceDetector(device=device)
    except Exception:
        worker_detector = None

def process_video_worker(args):
    video_path, label, output_root, fps, use_phase, video_id_prefix = args
    global worker_detector
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_id = f"{video_id_prefix}_{label}_{video_name}"
    
    try:
        frames = extract_frames_from_video(video_path, target_fps=fps)
        if not frames:
            return []
            
        # Run batched face detection on the worker's FaceDetector
        if worker_detector is not None:
            images_rgb = [f[1] for f in frames]
            face_crops = worker_detector.detect_batch(images_rgb)
        else:
            face_crops = [None] * len(frames)
            
        sample_entries = []
        for (frame_idx, frame_rgb), face_crop in zip(frames, face_crops):
            entry = save_processed_sample(
                video_id=video_id,
                frame_idx=frame_idx,
                frame_rgb=frame_rgb,
                face_detector=None,
                output_root=output_root,
                use_phase=use_phase,
                video_path=video_path,
                face_crop=face_crop
            )
            entry['label'] = label
            sample_entries.append(entry)
            
        return sample_entries
    except Exception as e:
        print(f"\nError processing video {video_path}: {e}")
        return []

def process_video_list_parallel(
    video_list: List[Tuple[str, int]],
    output_root: str,
    fps: int,
    use_phase: bool,
    device: str,
    video_id_prefix: str = "ffpp"
) -> List[Dict[str, Any]]:
    """Process a list of videos in parallel using a pool of worker processes."""
    # Use spawn start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    num_workers = min(4, os.cpu_count() or 1)
    if device == "cpu":
        num_workers = min(8, os.cpu_count() or 1)
        
    print(f"\n🚀 Initializing {num_workers} parallel worker processes on device '{device}'...")
    
    worker_args = [
        (video_path, label, output_root, fps, use_phase, video_id_prefix)
        for video_path, label in video_list
    ]
    
    all_sample_entries = []
    
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(device,)) as pool:
        for results in tqdm(
            pool.imap_unordered(process_video_worker, worker_args),
            total=len(video_list),
            desc=f"Processing {video_id_prefix.upper()} videos"
        ):
            all_sample_entries.extend(results)
            
    return all_sample_entries


import urllib.request

def extract_source_video_ids(path: str) -> List[str]:
    """
    Extract all source video ID numbers or keys from a filename.
    e.g., '000.mp4' -> ['000']
          '000_111.mp4' -> ['000', '111']
          'id0_id16_0000.mp4' -> ['id0', 'id16', '0000']
    """
    basename = os.path.splitext(os.path.basename(path))[0]
    parts = basename.split('_')
    ids = []
    for p in parts:
        if p and p.lower() not in ["c23", "c40", "raw", "videos", "masks"]:
            ids.append(p)
    return ids


def load_official_ff_splits(data_root: str) -> Tuple[Optional[set], Optional[set], Optional[set]]:
    """
    Download or load official FaceForensics++ splits.
    """
    splits_dir = os.path.join(data_root, "official_splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    splits = {}
    for split_name in ["train", "val", "test"]:
        local_path = os.path.join(splits_dir, f"{split_name}.json")
        if not os.path.exists(local_path):
            url = f"https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{split_name}.json"
            print(f"Downloading official FaceForensics++ split '{split_name}' from: {url}")
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                print(f"Warning: Could not download official splits ({e}). Falling back to custom splitting.")
                return None, None, None
        
        try:
            with open(local_path, "r") as f:
                raw_data = json.load(f)
                flat_list = []
                for item in raw_data:
                    if isinstance(item, list):
                        flat_list.extend(item)
                    else:
                        flat_list.append(item)
                splits[split_name] = set(flat_list)
        except Exception as e:
            print(f"Warning: Error reading split file {local_path} ({e}).")
            return None, None, None
            
    return splits.get("train"), splits.get("val"), splits.get("test")


def partition_by_connected_components(sample_entries: List[Dict[str, Any]], train_ratio=0.7, val_ratio=0.15):
    """
    Prevent data leakage by grouping related source video IDs into connected components
    and partitioning components into train, val, and test splits.
    """
    from collections import defaultdict
    
    adj = defaultdict(set)
    all_ids = set()
    video_to_ids = {}
    
    for sample in sample_entries:
        video_id = sample['video_id']
        if video_id not in video_to_ids:
            source_ids = extract_source_video_ids(sample.get('video_path', ''))
            video_to_ids[video_id] = source_ids
            for s_id in source_ids:
                all_ids.add(s_id)
                for other_id in source_ids:
                    if s_id != other_id:
                        adj[s_id].add(other_id)
                        
    visited = set()
    components = []
    
    for s_id in sorted(list(all_ids)):
        if s_id not in visited:
            component = []
            queue = [s_id]
            visited.add(s_id)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in sorted(list(adj[curr])):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
            
    components.sort(key=lambda c: sorted(c))
    import random
    random_state = random.Random(42)
    random_state.shuffle(components)
    
    total_ids = len(all_ids)
    train_target = int(total_ids * train_ratio)
    val_target = int(total_ids * val_ratio)
    
    train_ids = set()
    val_ids = set()
    test_ids = set()
    
    current_train_count = 0
    current_val_count = 0
    
    for comp in components:
        comp_set = set(comp)
        if current_train_count < train_target:
            train_ids.update(comp_set)
            current_train_count += len(comp_set)
        elif current_val_count < val_target:
            val_ids.update(comp_set)
            current_val_count += len(comp_set)
        else:
            test_ids.update(comp_set)
            
    return train_ids, val_ids, test_ids


def assign_splits(sample_entries: List[Dict[str, Any]], data_root: str, train_ratio=0.7, val_ratio=0.15):
    """
    Assign each sample entry to train, val, or test split using official splits.
    Ensures that a manipulated video A_B is assigned to a split only if BOTH source IDs A and B are in that split.
    """
    train_ids, val_ids, test_ids = load_official_ff_splits(data_root)
    
    if not train_ids or not val_ids or not test_ids:
        raise ValueError("CRITICAL ERROR: Official FaceForensics++ splits JSON files are missing or could not be loaded!")
        
    train_samples = []
    val_samples = []
    test_samples = []
    
    unmatched_samples = []
    
    for sample in sample_entries:
        source_ids = extract_source_video_ids(sample.get('video_path', ''))
        if not source_ids:
            unmatched_samples.append(sample)
            continue
            
        # All source IDs must belong to the train split to be train
        is_train = all(s_id in train_ids for s_id in source_ids)
        # All source IDs must belong to the val split to be val
        is_val = all(s_id in val_ids for s_id in source_ids)
        # All source IDs must belong to the test split to be test
        is_test = all(s_id in test_ids for s_id in source_ids)
        
        if is_train:
            train_samples.append(sample)
        elif is_val:
            val_samples.append(sample)
        elif is_test:
            test_samples.append(sample)
        else:
            unmatched_samples.append(sample)
            
    print(f"\nSplit Assignment Report:")
    print(f"  Train frame samples: {len(train_samples)}")
    print(f"  Val frame samples: {len(val_samples)}")
    print(f"  Test frame samples: {len(test_samples)}")
    print(f"  Unmatched/Excluded frame samples (cross-split or missing): {len(unmatched_samples)}")
    
    # Validate final counts of real and fake videos (not frames)
    def count_unique_videos(samples):
        reals = set()
        fakes = set()
        for s in samples:
            vid = s['video_id']
            lbl = s['label']
            if lbl == 0:
                reals.add(vid)
            else:
                fakes.add(vid)
        return len(reals), len(fakes)
        
    train_reals, train_fakes = count_unique_videos(train_samples)
    val_reals, val_fakes = count_unique_videos(val_samples)
    test_reals, test_fakes = count_unique_videos(test_samples)
    
    print(f"Video-level split counts:")
    print(f"  Train: Real={train_reals}, Fake={train_fakes} (Total={train_reals + train_fakes})")
    print(f"  Val: Real={val_reals}, Fake={val_fakes} (Total={val_reals + val_fakes})")
    print(f"  Test: Real={test_reals}, Fake={test_fakes} (Total={test_reals + test_fakes})")
    
    total_videos = train_reals + train_fakes + val_reals + val_fakes + test_reals + test_fakes
    is_smoke_test = total_videos < 50
    
    if is_smoke_test and (len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0):
        print("⚠️  Smoke test splits are empty or degenerate. Forcing class-balanced round-robin allocation...")
        train_samples, val_samples, test_samples = [], [], []
        
        reals_by_video = {}
        fakes_by_video = {}
        for sample in sample_entries:
            vid = sample['video_id']
            lbl = sample['label']
            if lbl == 0:
                if vid not in reals_by_video:
                    reals_by_video[vid] = []
                reals_by_video[vid].append(sample)
            else:
                if vid not in fakes_by_video:
                    fakes_by_video[vid] = []
                fakes_by_video[vid].append(sample)
                
        # Distribute real videos round-robin
        for idx, (vid, samples) in enumerate(sorted(reals_by_video.items())):
            if idx % 3 == 0:
                train_samples.extend(samples)
            elif idx % 3 == 1:
                val_samples.extend(samples)
            else:
                test_samples.extend(samples)
                
        # Distribute fake videos round-robin
        for idx, (vid, samples) in enumerate(sorted(fakes_by_video.items())):
            if idx % 3 == 0:
                train_samples.extend(samples)
            elif idx % 3 == 1:
                val_samples.extend(samples)
            else:
                test_samples.extend(samples)
                
        train_reals, train_fakes = count_unique_videos(train_samples)
        val_reals, val_fakes = count_unique_videos(val_samples)
        test_reals, test_fakes = count_unique_videos(test_samples)
        
        print(f"Updated video-level split counts:")
        print(f"  Train: Real={train_reals}, Fake={train_fakes} (Total={train_reals + train_fakes})")
        print(f"  Val: Real={val_reals}, Fake={val_fakes} (Total={val_reals + val_fakes})")
        print(f"  Test: Real={test_reals}, Fake={test_fakes} (Total={test_reals + test_fakes})")
        
    if not is_smoke_test:
        # Enforce strict count checks
        if train_reals < 700 or train_fakes < 700:
            raise ValueError(f"CRITICAL ERROR: Incomplete training dataset! Expected approx 720 real and 720 fake training videos, got {train_reals} real and {train_fakes} fake.")
        if val_reals < 130 or val_fakes < 130:
            raise ValueError(f"CRITICAL ERROR: Incomplete validation dataset! Expected approx 140 real and 140 fake validation videos, got {val_reals} real and {val_fakes} fake.")
        if test_reals < 130 or test_fakes < 130:
            raise ValueError(f"CRITICAL ERROR: Incomplete test dataset! Expected approx 140 real and 140 fake test videos, got {test_reals} real and {test_fakes} fake.")
            
    return train_samples, val_samples, test_samples


def validate_splits_and_metadata(train_samples, val_samples, test_samples):
    """
    Validate dataset splits for path existence, deduplication, overlap, and class balance.
    """
    print("\n" + "=" * 60)
    print(" RUNNING PREPROCESSING METADATA VALIDATION CHECK")
    print("=" * 60)
    
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    split_source_ids = {}
    
    for name, samples in splits.items():
        if len(samples) == 0:
            raise ValueError(f"CRITICAL ERROR: Split '{name}' has zero sample entries!")
            
        real_count = sum(1 for s in samples if s['label'] == 0)
        fake_count = sum(1 for s in samples if s['label'] == 1)
        
        # Check label validity
        invalid_labels = [s['label'] for s in samples if s['label'] not in [0, 1]]
        if invalid_labels:
            raise ValueError(f"CRITICAL ERROR: Split '{name}' contains invalid non-binary labels: {set(invalid_labels)}")
            
        print(f"Split '{name}': Total={len(samples)}, Real={real_count}, Fake={fake_count}")
        
        # Check class balance (assert both classes exist)
        if real_count == 0 or fake_count == 0:
            raise ValueError(f"CRITICAL ERROR: Split '{name}' has degenerate class balance (Real={real_count}, Fake={fake_count})!")
            
        # Check path existence
        for s in samples:
            face_path = s.get('face_path', '')
            frame_path = s.get('frame_path', '')
            # Relative paths inside metadata, resolve using config data_root/parent if needed
            # Since local paths are relative to output_root, we skip if validation runs before metadata is saved
            # but we check if they exist relative to current path
            if face_path and not os.path.exists(face_path) and not os.path.exists(os.path.join("data", face_path)):
                raise FileNotFoundError(f"CRITICAL ERROR: Face path not found: {face_path}")
            if frame_path and not os.path.exists(frame_path) and not os.path.exists(os.path.join("data", frame_path)):
                raise FileNotFoundError(f"CRITICAL ERROR: Frame path not found: {frame_path}")
                
        # Check duplicates
        face_paths = [s.get('face_path') for s in samples if s.get('face_path')]
        if len(face_paths) != len(set(face_paths)):
            print("⚠️  Warning: Duplicate face paths found in metadata (harmless if multiple crops per frame).")
            
        # Collect source IDs
        s_ids = set()
        for s in samples:
            s_ids.update(extract_source_video_ids(s.get('video_path', '')))
        split_source_ids[name] = s_ids
        
    # Check overlap
    overlap_train_val = split_source_ids['train'].intersection(split_source_ids['val'])
    overlap_train_test = split_source_ids['train'].intersection(split_source_ids['test'])
    overlap_val_test = split_source_ids['val'].intersection(split_source_ids['test'])
    
    # Check overlap (raise error on large dataset, warn on small test dataset)
    total_samples = len(train_samples) + len(val_samples) + len(test_samples)
    is_small_subset = (total_samples < 500)
    
    if overlap_train_val:
        msg = f"CRITICAL ERROR: Leakage detected between Train and Val splits! Overlapping IDs: {overlap_train_val}"
        if is_small_subset:
            print(f"⚠️  Warning: {msg} (allowed for small debugging/smoke-test subset)")
        else:
            raise ValueError(msg)
            
    if overlap_train_test:
        msg = f"CRITICAL ERROR: Leakage detected between Train and Test splits! Overlapping IDs: {overlap_train_test}"
        if is_small_subset:
            print(f"⚠️  Warning: {msg} (allowed for small debugging/smoke-test subset)")
        else:
            raise ValueError(msg)
            
    if overlap_val_test:
        msg = f"CRITICAL ERROR: Leakage detected between Val and Test splits! Overlapping IDs: {overlap_val_test}"
        if is_small_subset:
            print(f"⚠️  Warning: {msg} (allowed for small debugging/smoke-test subset)")
        else:
            raise ValueError(msg)
        
    print("✓ All validation checks passed successfully!")
    print("=" * 60 + "\n")


def split_and_save_metadata(sample_entries: List[Dict[str, Any]], output_root: str, train_ratio=0.7, val_ratio=0.15):
    """Group samples by base_video_id and save stratified train/val/test metadata JSON files."""
    train_samples, val_samples, test_samples = assign_splits(sample_entries, output_root, train_ratio, val_ratio)
    
    # Validate the generated splits
    validate_splits_and_metadata(train_samples, val_samples, test_samples)
    
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

    # Discover available manipulation methods and gather fakes
    manip_root = os.path.join(videos_dir, "manipulated_sequences")
    manip_methods = {}
    
    if os.path.exists(manip_root):
        for entry in sorted(os.listdir(manip_root)):
            entry_path = os.path.join(manip_root, entry)
            if os.path.isdir(entry_path):
                method_videos = []
                for ext in ["*.mp4", "*.avi"]:
                    method_videos.extend(glob.glob(os.path.join(entry_path, "*", "videos", ext)))
                    method_videos.extend(glob.glob(os.path.join(entry_path, "videos", ext)))
                    method_videos.extend(glob.glob(os.path.join(entry_path, ext)))
                if method_videos:
                    manip_methods[entry] = sorted(list(set(method_videos)))
                    
    # Fallback to general patterns if no methods directory structure found
    if not manip_methods:
        fake_patterns = [
            os.path.join(videos_dir, "manipulated_sequences", "*", "*", "videos", "*.mp4"),
            os.path.join(videos_dir, "Deepfakes", "*.mp4"),
            os.path.join(videos_dir, "fake", "*.mp4"),
        ]
        fake_files = []
        for pat in fake_patterns:
            fake_files.extend(glob.glob(pat))
        if fake_files:
            manip_methods["default"] = sorted(list(set(fake_files)))

    # Fallback: if flat structure search
    if not real_files and not manip_methods:
        all_videos = glob.glob(os.path.join(videos_dir, "**", "*.mp4"), recursive=True)
        fake_files = []
        for v in all_videos:
            v_lower = v.lower()
            if "original" in v_lower or "real" in v_lower or "youtube" in v_lower:
                real_files.append(v)
            else:
                fake_files.append(v)
        if fake_files:
            manip_methods["default"] = sorted(list(set(fake_files)))

    real_files = sorted(list(set(real_files)))
    
    # We want to pair every real video with its corresponding fake video in Deepfakes
    real_map = {}
    for r_file in real_files:
        r_id = os.path.splitext(os.path.basename(r_file))[0]
        real_map[r_id] = r_file
        
    all_fake_files = []
    if manip_methods:
        for m in manip_methods:
            all_fake_files.extend(manip_methods[m])
    all_fake_files = sorted(list(set(all_fake_files)))
    
    fake_map = {}
    for f_file in all_fake_files:
        f_basename = os.path.splitext(os.path.basename(f_file))[0]
        parts = f_basename.split('_')
        if parts:
            t_id = parts[0]
            fake_map[t_id] = f_file
            
    paired_list = []
    for r_id, r_file in sorted(real_map.items()):
        if r_id in fake_map:
            paired_list.append((r_file, 0))  # Real
            paired_list.append((fake_map[r_id], 1))  # Fake (corresponding manipulated version)
            
    # Print status of pairing and warn about missing or duplicate files
    print(f"Dataset Pairing Summary:")
    print(f"  Total real files discovered: {len(real_files)}")
    print(f"  Total fake files discovered: {len(all_fake_files)}")
    print(f"  Successfully paired real-fake videos: {len(paired_list) // 2}")
    
    missing_fakes = [r_id for r_id in real_map if r_id not in fake_map]
    if missing_fakes:
        print(f"  ⚠️  Warning: {len(missing_fakes)} real videos are missing their corresponding fake videos in the raw directory.")
        
    if max_videos is not None and max_videos > 0:
        # Keep it balanced: half real, half fake
        num_pairs = max_videos // 2
        video_list = paired_list[:(num_pairs * 2)]
    else:
        video_list = paired_list
        
    print(f"Final selected video list for preprocessing: {len(video_list)} total.")

    all_sample_entries = process_video_list_parallel(
        video_list=video_list,
        output_root=output_root,
        fps=fps,
        use_phase=use_phase,
        device=device,
        video_id_prefix="ffpp"
    )

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

    real_videos = []
    fake_videos = []
    
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
        if label == 0:
            real_videos.append((v_path, 0))
        else:
            fake_videos.append((v_path, 1))

    # Balance selection if max_videos is specified
    if max_videos is not None and max_videos > 0:
        half_max = max_videos // 2
        selected_real = real_videos[:half_max]
        selected_fake = fake_videos[:(max_videos - len(selected_real))]
        if len(selected_fake) < half_max and len(real_videos) > len(selected_real):
            selected_real = real_videos[:(max_videos - len(selected_fake))]
        video_list = selected_real + selected_fake
    else:
        video_list = real_videos + fake_videos

    all_sample_entries = process_video_list_parallel(
        video_list=video_list,
        output_root=output_root,
        fps=fps,
        use_phase=use_phase,
        device=device,
        video_id_prefix="local"
    )

    return split_and_save_metadata(all_sample_entries, output_root)
