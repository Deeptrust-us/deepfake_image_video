"""Face detection and alignment utilities using MTCNN or Dlib."""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import torch
from facenet_pytorch import MTCNN


class FaceDetector:
    """Face detector using MTCNN for detection and alignment."""
    
    def __init__(self, min_face_size: int = 40, device: str = "cpu"):
        """
        Initialize face detector.
        
        Args:
            min_face_size: Minimum face size to detect
            device: Device to run detection on ('cpu' or 'cuda')
        """
        self.device = device
        self.mtcnn = MTCNN(
            image_size=320,
            margin=0,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=device
        )
    
    def detect_and_align(self, image: np.ndarray, is_bgr: bool = False) -> Optional[np.ndarray]:
        """
        Detect and align face in image.
        
        Args:
            image: Input image as numpy array (BGR or RGB)
            is_bgr: Whether the input image is in BGR format
            
        Returns:
            Aligned face image (320x320) or None if no face detected
        """
        # Convert BGR to RGB if needed
        if is_bgr:
            if len(image.shape) == 3 and image.shape[2] == 3 and isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Detect and align face
        face_tensor = self.mtcnn(pil_image)
        
        if face_tensor is None:
            return None
        
        # Convert tensor to numpy array
        face_array = face_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Normalize from [-1, 1] to [0, 1] if needed
        if face_array.min() < 0:
            face_array = (face_array + 1) / 2.0
        
        # Ensure values are in [0, 1] range
        face_array = np.clip(face_array, 0, 1)
        
        # Resize to 320x320
        face_array = cv2.resize(face_array, (320, 320))
        
        return face_array
    
    def detect_batch(self, images: list, is_bgr: bool = False) -> list:
        """
        Detect faces in a batch of images.
        
        Args:
            images: List of images as numpy arrays
            is_bgr: Whether the input images are in BGR format
            
        Returns:
            List of aligned face images (or None for failed detections)
        """
        if not images:
            return []
            
        pil_images = []
        for img in images:
            if is_bgr:
                if len(img.shape) == 3 and img.shape[2] == 3 and isinstance(img, np.ndarray):
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img
            else:
                img_rgb = img
            pil_images.append(Image.fromarray(img_rgb))
            
        try:
            # MTCNN forward method accepts list of PIL images and runs batched detection on GPU
            face_tensors = self.mtcnn(pil_images)
        except Exception as e:
            print(f"Warning: Batch face detection failed ({e}). Falling back to sequential.")
            return [self.detect_and_align(img, is_bgr=is_bgr) for img in images]
            
        if face_tensors is None:
            return [None] * len(images)
            
        face_crops = []
        for face_tensor in face_tensors:
            if face_tensor is None:
                face_crops.append(None)
                continue
                
            # Convert tensor to numpy array
            face_array = face_tensor.permute(1, 2, 0).cpu().numpy()
            
            # Normalize from [-1, 1] to [0, 1] if needed
            if face_array.min() < 0:
                face_array = (face_array + 1) / 2.0
            
            # Ensure values are in [0, 1] range
            face_array = np.clip(face_array, 0, 1)
            
            # Resize to 320x320
            face_array = cv2.resize(face_array, (320, 320))
            
            face_crops.append(face_array)
            
        return face_crops


