"""Example usage script for dual-stream deepfake detection."""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dual_stream import DualStreamModel
from src.utils.face_detection import FaceDetector
from src.utils.frequency_domain import prepare_frequency_input


def example_inference():
    """Example of running inference on a single image."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model (using default config values)
    model = DualStreamModel(
        spatial_backbone="resnet18",
        spatial_feature_dim=256,
        frequency_channels=1,
        fusion_dim=512,
        dropout=0.5,
        pretrained=True
    ).to(device)
    
    model.eval()
    
    # Initialize face detector
    face_detector = FaceDetector(device=device)
    
    # Example: Load an image (replace with your image path)
    # image_path = "path/to/your/image.jpg"
    # image = np.array(Image.open(image_path)) / 255.0
    
    # For demonstration, create dummy images
    print("Creating dummy images for demonstration...")
    whole_frame = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Detect and align face
    print("Detecting face...")
    face = face_detector.detect_and_align((whole_frame * 255).astype(np.uint8))
    
    if face is None:
        print("No face detected, using resized original image")
        face = whole_frame
    
    # Prepare frequency inputs for both face and frame
    print("Computing frequency domain representations...")
    face_frequency_input = prepare_frequency_input(
        face,
        use_phase=False,
        normalize=True
    )
    
    frame_frequency_input = prepare_frequency_input(
        whole_frame,
        use_phase=False,
        normalize=True
    )
    
    # Convert to tensors
    face_spatial_tensor = torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0).to(device)
    if face_spatial_tensor.shape[1] == 1:
        face_spatial_tensor = face_spatial_tensor.repeat(1, 3, 1, 1)
    
    frame_spatial_tensor = torch.from_numpy(whole_frame).permute(2, 0, 1).float().unsqueeze(0).to(device)
    if frame_spatial_tensor.shape[1] == 1:
        frame_spatial_tensor = frame_spatial_tensor.repeat(1, 3, 1, 1)
    
    face_frequency_tensor = torch.from_numpy(face_frequency_input).permute(2, 0, 1).float().unsqueeze(0).to(device)
    frame_frequency_tensor = torch.from_numpy(frame_frequency_input).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(face_spatial_tensor, face_frequency_tensor, 
                      frame_spatial_tensor, frame_frequency_tensor)
        probability = output.item()
    
    print(f"\nPrediction: {probability:.4f}")
    print(f"Predicted class: {'Fake' if probability > 0.5 else 'Real'}")
    
    # Visualize inputs and prediction
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 5, 1)
    plt.imshow(face)
    plt.title("Face Crop (RGB)")
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(face_frequency_input[:, :, 0], cmap='hot')
    plt.title("Face Frequency")
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(whole_frame)
    plt.title("Whole Frame (RGB)")
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.imshow(frame_frequency_input[:, :, 0], cmap='hot')
    plt.title("Frame Frequency")
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.bar(['Real', 'Fake'], [1 - probability, probability])
    plt.title(f"Prediction: {probability:.2%}")
    plt.ylabel("Probability")
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig("example_inference.png")
    print("\nVisualization saved to example_inference.png")


if __name__ == "__main__":
    example_inference()


