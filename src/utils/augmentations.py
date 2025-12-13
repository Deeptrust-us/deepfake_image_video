"""Data augmentation utilities for spatial and frequency domains."""

import numpy as np
import cv2
from typing import Tuple
import random


class DualStreamAugmentation:
    """Augmentation that applies consistent transformations to both spatial and frequency streams."""
    
    def __init__(self, horizontal_flip_prob: float = 0.5, rotation_range: int = 15,
                 brightness_range: float = 0.2, contrast_range: float = 0.2,
                 noise_std: float = 0.02, gaussian_blur_prob: float = 0.3):
        """
        Initialize augmentation parameters.
        
        Args:
            horizontal_flip_prob: Probability of horizontal flip
            rotation_range: Maximum rotation angle in degrees
            brightness_range: Brightness adjustment range
            contrast_range: Contrast adjustment range
            noise_std: Standard deviation for Gaussian noise (0 to disable)
            gaussian_blur_prob: Probability of applying Gaussian blur
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.gaussian_blur_prob = gaussian_blur_prob
    
    def apply_spatial(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to spatial image.
        
        Args:
            image: Input image (H, W, C) with values in [0, 1]
            
        Returns:
            Augmented image
        """
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = np.fliplr(image)
        
        # Rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Brightness and contrast
        if self.brightness_range > 0 or self.contrast_range > 0:
            brightness = random.uniform(-self.brightness_range, self.brightness_range)
            contrast = random.uniform(1.0 - self.contrast_range, 1.0 + self.contrast_range)
            
            image = image * contrast + brightness
            image = np.clip(image, 0, 1)
        
        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # Gaussian blur (simulates compression artifacts)
        if random.random() < self.gaussian_blur_prob:
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image
    
    def apply_frequency(self, frequency: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to frequency spectrum.
        Note: Only geometric transformations (flip, rotation) are applied.
        
        Args:
            frequency: Frequency spectrum (H, W, C)
            
        Returns:
            Augmented frequency spectrum
        """
        # Horizontal flip (consistent with spatial)
        if random.random() < self.horizontal_flip_prob:
            frequency = np.fliplr(frequency)
        
        # Rotation (consistent with spatial)
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            h, w = frequency.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to each channel
            rotated_channels = []
            for c in range(frequency.shape[2]):
                rotated = cv2.warpAffine(frequency[:, :, c], M, (w, h), borderMode=cv2.BORDER_REFLECT)
                rotated_channels.append(rotated)
            frequency = np.stack(rotated_channels, axis=-1)
        
        return frequency
    
    def __call__(self, spatial: np.ndarray, frequency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply consistent augmentations to both streams.
        
        Args:
            spatial: Spatial image (H, W, C)
            frequency: Frequency spectrum (H, W, C)
            
        Returns:
            Augmented spatial and frequency images
        """
        # Use same random seed for consistent transformations
        seed = random.randint(0, 2**32 - 1)
        
        random.seed(seed)
        spatial_aug = self.apply_spatial(spatial)
        
        random.seed(seed)
        frequency_aug = self.apply_frequency(frequency)
        
        return spatial_aug, frequency_aug


