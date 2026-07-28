"""Evaluation script for deepfake detection models (Xception, Multi-stream, Quad-Stream)."""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import get_model, MODEL_REGISTRY
from src.data.dataset import DeepfakeDataset
from src.utils.metrics import compute_frame_metrics, compute_video_metrics, find_optimal_threshold


def evaluate_model(model, dataloader, device, video_level=True, threshold=0.5):
    """
    Evaluate model on dataset using raw logits (handles Sigmoid internally).
    Consistently averages frame probabilities for video-level aggregation.
    """
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

            # shape alignment
            outputs = outputs.view_as(labels)

            # Apply sigmoid exactly once here to get frame-level probability
            probas = torch.sigmoid(outputs).cpu().numpy()
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
        # Probabilities averaging logic (consistent with train/val validation)
        video_probas = []
        video_labels_list = []
        for video_id, predictions in video_predictions.items():
            video_probas.append(np.mean(predictions))
            video_labels_list.append(video_labels[video_id])
            
        video_probas_arr = np.array(video_probas)
        video_labels_arr = np.array(video_labels_list)
        video_preds_arr = (video_probas_arr >= threshold).astype(int)
        
        video_metrics = compute_frame_metrics(video_labels_arr, video_preds_arr, video_probas_arr)

    return frame_metrics, video_metrics, all_labels, all_preds, all_probas, video_predictions, video_labels


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
    parser.add_argument("--model", type=str, default="quad_stream", help="Model architecture to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--threshold", type=float, default=None, help="Force a classification threshold (overrides checkpoint threshold)")
    parser.add_argument("--optimal_threshold", action="store_true", help="Determine F1-optimal threshold on val set videos")
    parser.add_argument("--metadata_file", type=str, default=None, help="Override metadata file path (e.g. for cross-dataset test)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--allow_random_weights", action="store_true", help="Allow running evaluation with initialized weights for debugging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replication")
    args = parser.parse_args()

    # Resolve config path robustly
    config_path = args.config
    if not os.path.exists(config_path):
        alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.config))
        if os.path.exists(alt_path):
            config_path = alt_path
        elif os.path.exists(os.path.join('/content/deepfake_image_video', args.config)):
            config_path = os.path.join('/content/deepfake_image_video', args.config)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating Model: {args.model}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve default checkpoint path if not provided
    model_name_clean = args.model.lower().replace("-", "_")
    if not args.checkpoint:
        default_ckpt = os.path.join(config['paths']['checkpoint_dir'], f"best_model_{model_name_clean}_seed{args.seed}.pth")
        if os.path.exists(default_ckpt):
            args.checkpoint = default_ckpt
            print(f"✓ Using default seed-specific checkpoint path: {args.checkpoint}")
        else:
            std_ckpt = os.path.join(config['paths']['checkpoint_dir'], f"best_model_{model_name_clean}.pth")
            if os.path.exists(std_ckpt):
                args.checkpoint = std_ckpt
                print(f"✓ Using default checkpoint path: {args.checkpoint}")

    # Set spatial size per model native choices
    if args.model.lower() == "xception":
        spatial_size = 299
        frequency_size = 224
    else:
        spatial_size = 224
        frequency_size = 224

    # Initialize model using model factory
    model = get_model(args.model, config).to(device)

    # Load checkpoint
    optimal_thresh = 0.5
    model_metadata = {}
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"CRITICAL ERROR: Specified checkpoint file not found at: {args.checkpoint}")
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"✓ Successfully loaded checkpoint from {args.checkpoint}")
        
        # Load optimal threshold if stored
        if isinstance(checkpoint, dict) and 'optimal_threshold' in checkpoint:
            optimal_thresh = checkpoint['optimal_threshold']
            print(f"✓ Loaded optimal threshold from checkpoint: {optimal_thresh:.4f}")
        model_metadata = checkpoint.get('model_metadata', {})
    else:
        if not args.allow_random_weights:
            raise ValueError("CRITICAL ERROR: No checkpoint specified! Set --checkpoint or run with --allow_random_weights for debugging.")
        print("Note: Running inference with newly initialized model weights.")

    data_root = config['data']['data_root']

    # F1-optimal threshold search on validation videos
    if args.optimal_threshold:
        val_meta_path = os.path.join(data_root, "val_metadata.json")
        if os.path.exists(val_meta_path):
            print("Calculating F1-optimal threshold on validation videos...")
            val_dataset = DeepfakeDataset(
                data_root=data_root,
                metadata_file=val_meta_path,
                use_phase=(config['model']['frequency_channels'] == 2),
                normalize_frequency=config['preprocessing']['frequency_normalize'],
                is_training=False,
                spatial_size=spatial_size,
                frequency_size=frequency_size,
                frequency_channels=config['model'].get('frequency_channels', 1)
            )
            val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
            if len(val_dataset) > 0:
                _, _, _, _, _, val_video_preds, val_video_labels = evaluate_model(
                    model, val_loader, device, video_level=True, threshold=0.5
                )
                
                # Compute video-level average probabilities
                video_probas = []
                video_labels_list = []
                for video_id, predictions in val_video_preds.items():
                    video_probas.append(np.mean(predictions))
                    video_labels_list.append(val_video_labels[video_id])
                    
                optimal_thresh = find_optimal_threshold(np.array(video_labels_list), np.array(video_probas))
                print(f"Optimal threshold found on validation videos: {optimal_thresh:.4f}")

    # Override threshold if explicitly forced via CLI
    if args.threshold is not None:
        optimal_thresh = args.threshold
        print(f"✓ Overriding threshold to forced CLI value: {optimal_thresh:.4f}")

    # Create target dataset split
    if args.metadata_file:
        metadata_file = args.metadata_file
    else:
        metadata_file = os.path.join(data_root, f"{args.split}_metadata.json")
        
    dataset = DeepfakeDataset(
        data_root=data_root,
        metadata_file=metadata_file,
        use_phase=(config['model']['frequency_channels'] == 2),
        normalize_frequency=config['preprocessing']['frequency_normalize'],
        is_training=False,
        spatial_size=spatial_size,
        frequency_size=frequency_size,
        frequency_channels=config['model'].get('frequency_channels', 1)
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
    frame_metrics, video_metrics, y_true, y_pred, y_proba, video_predictions, video_labels = evaluate_model(
        model, dataloader, device, video_level=True, threshold=optimal_thresh
    )

    # Print results formatted at the video-level as primary
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS (VIDEO-LEVEL): {args.model.upper()} ({args.split} split)")
    print("="*60)
    print(f"  Accuracy : {video_metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score : {video_metrics['f1']*100:.2f}%")
    print(f"  AUC      : {video_metrics['auc']:.4f}")
    print(f"  EER      : {video_metrics['eer']*100:.2f}%")
    print(f"  Precision: {video_metrics['precision']*100:.2f}%")
    print(f"  Recall   : {video_metrics['recall']*100:.2f}%")
    print(f"  Number of Evaluated Videos: {len(video_predictions)}")
    print("="*60)

    # Save exact dataset manifest for this run
    manifest_data = {
        "preprocessing": {
            "spatial_size": spatial_size,
            "frequency_size": frequency_size,
            "frequency_channels": config['model'].get('frequency_channels', 1),
            "normalize_frequency": config['preprocessing']['frequency_normalize'],
        },
        "split": args.split,
        "seed": args.seed,
        "model": args.model,
        "samples": []
    }
    for item in dataset.active_samples:
        manifest_data["samples"].append({
            "video_id": item.get('video_id'),
            "label": int(item.get('label')),
            "face_path": item.get('face_path'),
            "frame_path": item.get('frame_path'),
            "video_path": item.get('video_path')
        })
    # Compute checksum of manifest content to verify integrity
    manifest_str = json.dumps(manifest_data, sort_keys=True)
    manifest_md5 = hashlib.md5(manifest_str.encode('utf-8')).hexdigest()
    manifest_data["metadata_checksum"] = manifest_md5
    
    model_tag = args.model.lower().replace("-", "_")
    manifest_path = os.path.join(args.output_dir, f"{model_tag}_{args.split}_manifest_seed{args.seed}.json")
    with open(manifest_path, 'w') as mf:
        json.dump(manifest_data, mf, indent=2)
    print(f"Saved dataset manifest to: {manifest_path}")

    # Save results
    results_file = os.path.join(args.output_dir, f"{model_tag}_{args.split}_results_seed{args.seed}.txt")
    legacy_file = os.path.join(args.output_dir, f"{model_tag}_{args.split}_results.txt")
    
    # Save seed-specific metrics file
    for filepath in [results_file, legacy_file]:
        with open(filepath, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Threshold: {optimal_thresh:.4f}\n")
            f.write(f"Active Streams: {model_metadata.get('streams', MODEL_REGISTRY.get(model_tag, {}).get('streams', []))}\n")
            f.write(f"Backbone Type: {model_metadata.get('backbone_type', 'unknown')}\n")
            f.write(f"Parameter Count: {model_metadata.get('param_count', 'unknown')}\n\n")
            
            f.write("Video-level Metrics (Primary):\n")
            f.write(f"Accuracy: {video_metrics['accuracy']*100:.2f}%\n")
            f.write(f"F1-score: {video_metrics['f1']*100:.2f}%\n")
            f.write(f"AUC: {video_metrics['auc']:.4f}\n")
            f.write(f"EER: {video_metrics['eer']*100:.2f}%\n")
            for key, value in video_metrics.items():
                f.write(f"video_{key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("Frame-level Metrics (Supplementary):\n")
            for key, value in frame_metrics.items():
                f.write(f"frame_{key}: {value:.4f}\n")

    # Save video-level predictions for thresholds verification
    preds_output = os.path.join(args.output_dir, f"{model_tag}_{args.split}_predictions_seed{args.seed}.json")
    with open(preds_output, 'w') as pf:
        # Convert numpy lists to standard lists for json serialization
        serializable_preds = {vid: [float(p) for p in probs] for vid, probs in video_predictions.items()}
        json.dump({
            "video_predictions": serializable_preds,
            "video_labels": video_labels
        }, pf, indent=2)

    cm_path = os.path.join(args.output_dir, f"{model_tag}_{args.split}_confusion_matrix_seed{args.seed}.png")
    
    # Generate frame-level predictions array for confusion matrix
    plot_confusion_matrix(y_true, y_pred, cm_path)

    print(f"Saved evaluation metrics to: {results_file}")
    print(f"Saved predictions to: {preds_output}")


if __name__ == "__main__":
    main()
