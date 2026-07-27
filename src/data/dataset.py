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
        target_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize DeepfakeDataset.

        Args:
            data_root: Root directory of dataset
            metadata_file: Path to metadata JSON file
            face_detector: Face detector instance (optional, for on-the-fly detection)
            augmentations: Augmentation instance (optional)
            use_phase: Whether frequency input includes phase spectrum (2 channels vs 1 channel)
            normalize_frequency: Whether to normalize frequency spectrums
            is_training: Whether dataset is used for training
            target_size: Target image size (H, W)
        """
        self.data_root = data_root
        self.metadata_file = metadata_file
        self.face_detector = face_detector
        self.augmentations = augmentations
        self.use_phase = use_phase
        self.normalize_frequency = normalize_frequency
        self.is_training = is_training
        self.target_size = target_size

        # Default image transforms (RGB standard ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load samples from metadata
        self.samples = self._load_metadata()

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load sample entries from metadata JSON file."""
        if not os.path.exists(self.metadata_file):
            print(f"Warning: Metadata file {self.metadata_file} not found. Returning empty dataset.")
            return []

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        return metadata

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_or_abs_path: str) -> np.ndarray:
        """Load RGB image as float32 array in [0, 1]."""
        full_path = rel_or_abs_path if os.path.isabs(rel_or_abs_path) else os.path.join(self.data_root, rel_or_abs_path)

        if not os.path.exists(full_path):
            # Return blank black image if missing
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)

        image_bgr = cv2.imread(full_path)
        if image_bgr is None:
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.target_size)
        return image_resized.astype(np.float32) / 255.0

    def _load_frequency(self, rel_or_abs_path: str, rgb_image: np.ndarray) -> np.ndarray:
        """Load pre-computed frequency .npy file or compute frequency spectrum on the fly."""
        if rel_or_abs_path:
            full_path = rel_or_abs_path if os.path.isabs(rel_or_abs_path) else os.path.join(self.data_root, rel_or_abs_path)
            if os.path.exists(full_path):
                try:
                    freq = np.load(full_path)
                    if len(freq.shape) == 2:
                        freq = np.expand_dims(freq, axis=-1)
                    return freq.astype(np.float32)
                except Exception:
                    pass

        # Compute FFT from RGB image on the fly
        return prepare_frequency_input(
            rgb_image,
            use_phase=self.use_phase,
            normalize=self.normalize_frequency
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # Load RGB face & frame
        face_path = item.get('face_path', '')
        frame_path = item.get('frame_path', '')

        # Fall back if one is missing
        if not face_path and frame_path:
            face_path = frame_path
        elif not frame_path and face_path:
            frame_path = face_path

        face_rgb = self._load_image(face_path)
        frame_rgb = self._load_image(frame_path)

        # Apply spatial data augmentations during training if specified
        if self.is_training and self.augmentations is not None:
            face_rgb, frame_rgb = self.augmentations(face_rgb, frame_rgb)

        # Load frequency representations (after spatial augmentations)
        face_freq = self._load_frequency(item.get('face_frequency_path', ''), face_rgb)
        frame_freq = self._load_frequency(item.get('frame_frequency_path', item.get('frequency_path', '')), frame_rgb)

        # Convert RGB images to Tensors & apply normalization and cast to float32
        face_spatial_tensor = self.transform(face_rgb).float()
        frame_spatial_tensor = self.transform(frame_rgb).float()

        # Convert Frequency arrays to Tensors (C, H, W)
        face_freq_tensor = torch.from_numpy(face_freq).permute(2, 0, 1).float()
        frame_freq_tensor = torch.from_numpy(frame_freq).permute(2, 0, 1).float()

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
