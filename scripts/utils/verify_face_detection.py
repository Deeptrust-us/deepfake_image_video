import os
import sys
import numpy as np
import torch
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.face_detection import FaceDetector

def test_face_detection():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for verification: {device}")
    
    detector = FaceDetector(device=device)
    print("FaceDetector successfully initialized.")
    
    # Create a dummy image (e.g. solid white block or random noise)
    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Test single frame detection
    print("Testing single frame detection...")
    res = detector.detect_and_align(dummy_img, is_bgr=False)
    # For solid white image, MTCNN should return None (no face)
    assert res is None, "Solid white image should not have a face"
    print("✓ Single frame detection test passed.")
    
    # Test batch detection
    print("Testing batch detection...")
    dummy_batch = [dummy_img] * 5
    res_batch = detector.detect_batch(dummy_batch, is_bgr=False)
    assert len(res_batch) == 5, f"Expected 5 results, got {len(res_batch)}"
    assert all(r is None for r in res_batch), "All results should be None"
    print("✓ Batch detection test passed.")
    
    print("\n✅ Verification script completed successfully!")

if __name__ == "__main__":
    test_face_detection()
