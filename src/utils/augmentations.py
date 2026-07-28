"""Data augmentation utilities for spatial and frequency domains."""

import numpy as np
import cv2
from typing import Tuple
import random


class DualStreamAugmentation:
    """Augmentation that applies consistent spatial transformations to both face and frame images."""
    
    def __init__(self, horizontal_flip_prob: float = 0.5, rotation_range: float = 5.0,
                 brightness_range: float = 0.1, contrast_range: float = 0.1,
                 noise_std: float = 0.01, gaussian_blur_prob: float = 0.3,
                 jpeg_prob: float = 0.5):
        """
        Initialize augmentation parameters.
        
        Args:
            horizontal_flip_prob: Probability of horizontal flip
            rotation_range: Maximum rotation angle in degrees (approx ±5)
            brightness_range: Brightness adjustment range (approx ±10%)
            contrast_range: Contrast adjustment range (approx ±10%)
            noise_std: Standard deviation for Gaussian noise (approx 0.01)
            gaussian_blur_prob: Probability of applying Gaussian blur
            jpeg_prob: Probability of applying JPEG compression augmentation
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.gaussian_blur_prob = gaussian_blur_prob
        self.jpeg_prob = jpeg_prob
        
    def __call__(self, face_image: np.ndarray, frame_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply consistent augmentations to both face and frame images.
        """
        # Determine shared geometric parameters to keep face and frame synchronized
        flip = random.random() < self.horizontal_flip_prob
        rotate = self.rotation_range > 0 and random.random() < 0.5
        angle = random.uniform(-self.rotation_range, self.rotation_range) if rotate else 0.0
        
        # Shared color/corruption parameters (so they look like they are from the same camera/compression state)
        apply_jpeg = random.random() < self.jpeg_prob
        quality = random.randint(70, 90) if apply_jpeg else 100
        
        apply_blur = random.random() < self.gaussian_blur_prob
        kernel_size = random.choice([3, 5]) if apply_blur else 3
        
        apply_noise = self.noise_std > 0 and random.random() < 0.5
        noise_params = np.random.normal(0, self.noise_std, face_image.shape) if apply_noise else None
        noise_params_frame = np.random.normal(0, self.noise_std, frame_image.shape) if apply_noise else None
        
        adjust_color = self.brightness_range > 0 or self.contrast_range > 0
        brightness = random.uniform(-self.brightness_range, self.brightness_range) if adjust_color else 0.0
        contrast = random.uniform(1.0 - self.contrast_range, 1.0 + self.contrast_range) if adjust_color else 1.0

        def apply_to_img(image, noise_array):
            # 1. Flip
            if flip:
                image = np.fliplr(image)
            
            # 2. Rotate
            if rotate:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                
            # 3. Brightness and Contrast
            if adjust_color:
                image = image * contrast + brightness
                image = np.clip(image, 0, 1)
                
            # 4. JPEG compression
            if apply_jpeg:
                img_u8 = (image * 255).astype(np.uint8)
                _, enc = cv2.imencode('.jpg', img_u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                image = cv2.imdecode(enc, 1).astype(np.float32) / 255.0
                
            # 5. Gaussian blur
            if apply_blur:
                image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
                
            # 6. Gaussian noise
            if apply_noise and noise_array is not None:
                image = image + noise_array
                image = np.clip(image, 0, 1)
                
            return image

        return apply_to_img(face_image, noise_params), apply_to_img(frame_image, noise_params_frame)
