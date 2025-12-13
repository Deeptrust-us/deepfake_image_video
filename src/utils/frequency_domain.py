"""Frequency domain transformation utilities for deepfake detection."""

import numpy as np
import cv2
from typing import Tuple, Optional


def compute_fft(image: np.ndarray, use_phase: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute 2D FFT of an image.
    
    Args:
        image: Input image (H, W, C) or (H, W) with values in [0, 1]
        use_phase: Whether to return phase spectrum
        
    Returns:
        magnitude_spectrum: Log-magnitude spectrum
        phase_spectrum: Phase spectrum (if use_phase=True), else None
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    # Convert to float32 for FFT
    gray_float = gray.astype(np.float32)
    
    # Compute 2D FFT
    fft = np.fft.fft2(gray_float)
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute magnitude and phase
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted) if use_phase else None
    
    # Apply log scaling to magnitude
    magnitude_log = np.log1p(magnitude)
    
    return magnitude_log, phase


def normalize_frequency_spectrum(spectrum: np.ndarray, mean: float = None, std: float = None) -> Tuple[np.ndarray, float, float]:
    """
    Normalize frequency spectrum to zero mean and unit variance.
    
    Args:
        spectrum: Frequency spectrum to normalize
        mean: Pre-computed mean (if None, computed from spectrum)
        std: Pre-computed std (if None, computed from spectrum)
        
    Returns:
        normalized_spectrum: Normalized spectrum
        mean: Mean value used
        std: Std value used
    """
    if mean is None:
        mean = np.mean(spectrum)
    if std is None:
        std = np.std(spectrum)
    
    if std > 0:
        normalized = (spectrum - mean) / std
    else:
        normalized = spectrum - mean
    
    return normalized, mean, std


def prepare_frequency_input(image: np.ndarray, use_phase: bool = False, 
                           normalize: bool = True, mean: float = None, 
                           std: float = None) -> np.ndarray:
    """
    Prepare frequency domain input for model.
    
    Args:
        image: Input image (H, W, C) with values in [0, 1]
        use_phase: Whether to include phase spectrum
        normalize: Whether to normalize the spectrum
        mean: Pre-computed mean for normalization
        std: Pre-computed std for normalization
        
    Returns:
        frequency_input: Frequency domain representation
        - If use_phase=False: (H, W, 1) magnitude spectrum
        - If use_phase=True: (H, W, 2) magnitude + phase spectrum
    """
    magnitude, phase = compute_fft(image, use_phase=use_phase)
    
    if normalize:
        magnitude, mean, std = normalize_frequency_spectrum(magnitude, mean, std)
        if phase is not None:
            phase, _, _ = normalize_frequency_spectrum(phase)
    
    # Stack channels
    if use_phase and phase is not None:
        frequency_input = np.stack([magnitude, phase], axis=-1)
    else:
        frequency_input = np.expand_dims(magnitude, axis=-1)
    
    return frequency_input


