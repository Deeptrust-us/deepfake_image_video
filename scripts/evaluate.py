"""Evaluation script for deepfake detection models (Xception, RGB+FFT Dual-Stream, Quad-Stream)."""

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

from src.models import get_model
from src.data.dataset import DeepfakeDataset
from src.utils.metrics import compute_frame_metrics, compute_video_metrics, find_optimal_threshold


def evaluate_model(model, dataloader, device, video_level=True, threshold=0.5):
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

            outputs = model(
                face_spatial=face_spatial,
                face_frequency=face_frequency,
                frame_spatial=frame_spatial,
                frame_frequency=frame_frequency
            ).squeeze()

            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            probas = outputs.cpu().numpy()
            if probas.ndim == 0:
                probas = np.array([probas.item()])

            labels_np = labels.cpu().numpy()
            if labels_np.ndim == 0:
                labels_np = np.array([labels_np.item()])

            all_labels.extend(labels_np)
            all_probas.extend(probas)

            if video_level:
                for vid_id, proba, label in zip(video_ids, probas, labels_np):
                    video_predictions[vid_id].append(proba)
                    video_labels[vid_id] = int(label)

    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)
    all_preds = (all_probas >= threshold).astype(int)

    # Frame-level metrics
    frame_metrics = compute_frame_metrics(all_labels, all_preds, all_probas)

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
    if len(y_true) == 0:
        return
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
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="quad_stream",
                        choices=["xception", "rgb_fft_dual_stream", "two_stream", "quad_stream"],
                        help="Model architecture to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--optimal_threshold", action="store_true", help="Determine F1-optimal threshold on val set")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating Model: {args.model}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model using model factory
    model = get_model(args.model, config).to(device)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print("Loaded checkpoint state dict.")
    else:
        print("Note: No checkpoint specified or file not found. Running inference with initialized model weights.")

    data_root = config['data']['data_root']

    # F1-optimal threshold search on validation set if requested
    optimal_thresh = args.threshold
    if args.optimal_threshold:
        val_meta_path = os.path.join(data_root, "val_metadata.json")
        if os.path.exists(val_meta_path):
            print("Calculating F1-optimal threshold on validation split...")
            val_dataset = DeepfakeDataset(
                data_root=data_root,
                metadata_file=val_meta_path,
                use_phase=(config['model']['frequency_channels'] == 2),
                normalize_frequency=config['preprocessing']['frequency_normalize'],
                is_training=False
            )
            val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
            if len(val_dataset) > 0:
                _, _, val_labels, _, val_probas = evaluate_model(model, val_loader, device, video_level=False, threshold=0.5)
                optimal_thresh = find_optimal_threshold(val_labels, val_probas)
                print(f"Optimal threshold found: {optimal_thresh:.4f}")

    # Create target dataset split
    metadata_file = os.path.join(data_root, f"{args.split}_metadata.json")
    dataset = DeepfakeDataset(
        data_root=data_root,
        metadata_file=metadata_file,
        use_phase=(config['model']['frequency_channels'] == 2),
        normalize_frequency=config['preprocessing']['frequency_normalize'],
        is_training=False
    )

    if len(dataset) == 0:
        print(f"Warning: Dataset split '{args.split}' is empty or metadata file {metadata_file} not found.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=config['training'].get('pin_memory', False)
    )

    # Evaluate
    print(f"\nEvaluating {args.model} on {args.split} split with threshold {optimal_thresh:.4f}...")
    frame_metrics, video_metrics, y_true, y_pred, y_proba = evaluate_model(
        model, dataloader, device, video_level=True, threshold=optimal_thresh
    )

    # Print results formatted for Paper table comparison
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS: {args.model.upper()} ({args.split} split)")
    print("="*60)
    print(f"  Accuracy : {frame_metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score : {frame_metrics['f1']*100:.2f}%")
    print(f"  AUC      : {frame_metrics['auc']:.4f}")
    print(f"  EER      : {frame_metrics['eer']*100:.2f}%")
    print(f"  Precision: {frame_metrics['precision']*100:.2f}%")
    print(f"  Recall   : {frame_metrics['recall']*100:.2f}%")
    print("="*60)

    if video_metrics:
        print(f"\nVideo-level Metrics ({args.split}):")
        for k, v in video_metrics.items():
            print(f"  {k.upper()}: {v:.4f}")

    # Save results
    model_tag = args.model.lower().replace("-", "_")
    results_file = os.path.join(args.output_dir, f"{model_tag}_{args.split}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Threshold: {optimal_thresh:.4f}\n\n")
        f.write("Frame-level Metrics:\n")
        f.write(f"Accuracy: {frame_metrics['accuracy']*100:.2f}%\n")
        f.write(f"F1-score: {frame_metrics['f1']*100:.2f}%\n")
        f.write(f"AUC: {frame_metrics['auc']:.4f}\n")
        f.write(f"EER: {frame_metrics['eer']*100:.2f}%\n")
        for key, value in frame_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    cm_path = os.path.join(args.output_dir, f"{model_tag}_{args.split}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)

    print(f"\nSaved evaluation metrics to: {results_file}")


if __name__ == "__main__":
    main()
