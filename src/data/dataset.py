"""Dataset loader for deepfake detection models."""

import os
import json
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import torchvision.transforms as transforms

from src.utils.frequency_domain import prepare_frequency_input


class DeepfakeDataset(Dataset):
    """
    Dataset class for loading face/frame images and frequency domain data.
    Supports single-stream (Xception), 2-stream (RGB+FFT), and 4-stream (Quad-Stream) models.
    """

    def __init__(
        self,
        data_root: str,
        metadata_file: str,
        face_detector=None,
        augmentations=None,
        use_phase: bool = False,
        normalize_frequency: bool = True,
        is_training: bool = False,
        spatial_size: int = 224,
        frequency_size: int = 224,
        frequency_channels: int = 1
    ):
        """
        Initialize DeepfakeDataset.
        """
        self.data_root = data_root
        self.metadata_file = metadata_file
        self.face_detector = face_detector
        self.augmentations = augmentations
        self.use_phase = use_phase
        self.normalize_frequency = normalize_frequency
        self.is_training = is_training
        self.spatial_size = spatial_size
        self.frequency_size = frequency_size
        self.frequency_channels = frequency_channels

        # Default image transforms (RGB standard ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load samples from metadata
        self.all_samples = self._load_metadata()
        self.active_samples = []
        
        # Initial population of active samples
        self.epoch_init(seed=42)

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load sample entries from metadata JSON file."""
        if not os.path.exists(self.metadata_file):
            print(f"Warning: Metadata file {self.metadata_file} not found. Returning empty dataset.")
            return []

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        return metadata

    def epoch_init(self, seed: int):
        """
        Populate active_samples list:
        - If is_training=True: Group by video_id and randomly select exactly 24 frames using a seeded generator.
        - If is_training=False: Keep all 32 candidate frames.
        """
        if not self.all_samples:
            self.active_samples = []
            return
            
        if self.is_training:
            # Group all samples by video_id
            by_video = {}
            for s in self.all_samples:
                vid = s.get('video_id', 'unknown')
                if vid not in by_video:
                    by_video[vid] = []
                by_video[vid].append(s)
                
            import random
            rng = random.Random(seed)
            
            self.active_samples = []
            for vid, v_samples in sorted(by_video.items()):
                # Sort to ensure deterministic order before sampling
                v_samples.sort(key=lambda x: x.get('frame_id', ''))
                n = len(v_samples)
                if n <= 24:
                    self.active_samples.extend(v_samples)
                else:
                    selected = rng.sample(v_samples, 24)
                    selected.sort(key=lambda x: x.get('frame_id', ''))
                    self.active_samples.extend(selected)
        else:
            self.active_samples = list(self.all_samples)

    def __len__(self) -> int:
        return len(self.active_samples)

    def _load_image(self, rel_or_abs_path: str, size: int) -> np.ndarray:
        """Load RGB image as float32 array in [0, 1] resized to target size."""
        full_path = rel_or_abs_path if os.path.isabs(rel_or_abs_path) else os.path.join(self.data_root, rel_or_abs_path)

        if not os.path.exists(full_path):
            return np.zeros((size, size, 3), dtype=np.float32)

        image_bgr = cv2.imread(full_path)
        if image_bgr is None:
            return np.zeros((size, size, 3), dtype=np.float32)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (size, size))
        return image_resized.astype(np.float32) / 255.0

    def _load_frequency(self, rel_or_abs_path: str, rgb_image: np.ndarray, size: int) -> np.ndarray:
        """Load pre-computed frequency .npy file or compute frequency spectrum on the fly."""
        if rel_or_abs_path:
            full_path = rel_or_abs_path if os.path.isabs(rel_or_abs_path) else os.path.join(self.data_root, rel_or_abs_path)
            if os.path.exists(full_path):
                try:
                    freq = np.load(full_path)
                    if freq.shape[0] != size or freq.shape[1] != size:
                        freq = cv2.resize(freq, (size, size))
                    if len(freq.shape) == 2:
                        freq = np.expand_dims(freq, axis=-1)
                    return freq.astype(np.float32)
                except Exception:
                    pass

        # Compute FFT from RGB image on the fly (it is already resized to size x size)
        return prepare_frequency_input(
            rgb_image,
            use_phase=self.use_phase,
            normalize=self.normalize_frequency
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.active_samples[idx]

        # Load RGB face & frame
        face_path = item.get('face_path', '')
        frame_path = item.get('frame_path', '')

        # Fall back if one is missing
        if not face_path and frame_path:
            face_path = frame_path
        elif not frame_path and face_path:
            frame_path = face_path

        face_rgb = self._load_image(face_path, self.spatial_size)
        frame_rgb = self._load_image(frame_path, self.spatial_size)

        # Apply spatial data augmentations during training if specified
        is_augmented = False
        if self.is_training and self.augmentations is not None:
            face_rgb, frame_rgb = self.augmentations(face_rgb, frame_rgb)
            is_augmented = True

        # Load frequency representations (after spatial augmentations)
        # If augmented during training, bypass pre-computed file to compute matching FFT on the fly!
        face_freq_path = '' if is_augmented else item.get('face_frequency_path', '')
        frame_freq_path = '' if is_augmented else item.get('frame_frequency_path', item.get('frequency_path', ''))

        face_freq = self._load_frequency(face_freq_path, face_rgb, self.frequency_size)
        frame_freq = self._load_frequency(frame_freq_path, frame_rgb, self.frequency_size)

        # Convert RGB images to Tensors & apply normalization and cast to float32
        face_spatial_tensor = self.transform(face_rgb).float()
        frame_spatial_tensor = self.transform(frame_rgb).float()

        # Convert Frequency arrays to Tensors (C, H, W)
        face_freq_tensor = torch.from_numpy(face_freq).permute(2, 0, 1).float()
        frame_freq_tensor = torch.from_numpy(frame_freq).permute(2, 0, 1).float()

        # Replicate log-magnitude to 3 channels if requested
        if self.frequency_channels == 3:
            if face_freq_tensor.shape[0] == 1:
                face_freq_tensor = face_freq_tensor.repeat(3, 1, 1)
            if frame_freq_tensor.shape[0] == 1:
                frame_freq_tensor = frame_freq_tensor.repeat(3, 1, 1)

        label = torch.tensor(item['label'], dtype=torch.long)
        video_id = item.get('video_id', 'unknown')

        return {
            'face_spatial': face_spatial_tensor,
            'face_frequency': face_freq_tensor,
            'frame_spatial': frame_spatial_tensor,
            'frame_frequency': frame_freq_tensor,
            'spatial': face_spatial_tensor,       # Alias for single-stream / dual-stream
            'frequency': face_freq_tensor,      # Alias for dual-stream
            'label': label,
            'video_id': video_id
        }
