"""Evaluation script for dual-stream deepfake detection."""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dual_stream import DualStreamModel
from src.data.dataset import DeepfakeDataset
from src.utils.metrics import compute_frame_metrics, compute_video_metrics, compute_confusion_matrix


def evaluate_model(model, dataloader, device, video_level=True):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []
    video_predictions = defaultdict(list)
    video_labels = {}
    
    with torch.no_grad():
        for batch in dataloader:
            face_spatial = batch['face_spatial'].to(device)
            face_frequency = batch['face_frequency'].to(device)
            frame_spatial = batch['frame_spatial'].to(device)
            frame_frequency = batch['frame_frequency'].to(device)
            labels = batch['label'].float().to(device)
            video_ids = batch['video_id']
            
            outputs = model(face_spatial, face_frequency, frame_spatial, frame_frequency).squeeze()
            probas = outputs.cpu().numpy()
            preds = (probas > 0.5).astype(int)
            labels_np = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probas.extend(probas)
            
            # Aggregate by video
            if video_level:
                for vid_id, proba, label in zip(video_ids, probas, labels_np):
                    video_predictions[vid_id].append(proba)
                    video_labels[vid_id] = int(label)
    
    # Frame-level metrics
    frame_metrics = compute_frame_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    
    # Video-level metrics
    video_metrics = None
    if video_level:
        video_metrics = compute_video_metrics(
            video_predictions,
            video_labels,
            aggregation="mean"
        )
    
    return frame_metrics, video_metrics, all_labels, all_preds, all_probas


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate dual-stream deepfake detection model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], 
                       help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    data_root = config['data']['data_root']
    dataset = DeepfakeDataset(
        data_root=data_root,
        metadata_file=os.path.join(data_root, f"{args.split}_metadata.json"),
        face_detector=None,
        augmentations=None,
        use_phase=(config['model']['frequency_channels'] == 2),
        normalize_frequency=config['preprocessing']['frequency_normalize'],
        is_training=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Initialize model
    model = DualStreamModel(
        spatial_backbone=config['model']['spatial_backbone'],
        spatial_feature_dim=config['model']['spatial_feature_dim'],
        frequency_channels=config['model']['frequency_channels'],
        fusion_dim=config['model']['fusion_dim'],
        dropout=config['model']['dropout'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate
    print(f"Evaluating on {args.split} split...")
    frame_metrics, video_metrics, y_true, y_pred, y_proba = evaluate_model(
        model, dataloader, device, video_level=True
    )
    
    # Print results
    print("\n" + "="*50)
    print(f"Frame-level Metrics ({args.split}):")
    print("="*50)
    for key, value in frame_metrics.items():
        print(f"  {key.upper()}: {value:.4f}")
    
    if video_metrics:
        print("\n" + "="*50)
        print(f"Video-level Metrics ({args.split}):")
        print("="*50)
        for key, value in video_metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, f"{args.split}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Frame-level Metrics ({args.split}):\n")
        for key, value in frame_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        
        if video_metrics:
            f.write(f"\nVideo-level Metrics ({args.split}):\n")
            for key, value in video_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, f"{args.split}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()


