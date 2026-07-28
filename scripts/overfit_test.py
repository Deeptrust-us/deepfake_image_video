#!/usr/bin/env python3
"""Standalone script to run overfitting verification on models."""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import hashlib
import json
import subprocess
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import get_model
from src.data.dataset import DeepfakeDataset
from src.utils.metrics import compute_frame_metrics


def get_git_commit():
    """Retrieve current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return "no_git_commit"


def get_config_fingerprint(config_path):
    """Compute MD5 hash of config file."""
    try:
        with open(config_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return "no_config_hash"


def main():
    parser = argparse.ArgumentParser(description="Overfitting Test Verification")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="xception", help="Model to overfit test")
    parser.add_argument("--epochs", type=int, default=25, help="Number of overfitting epochs")
    parser.add_argument("--device", type=str, default=None, help="Device to run test on")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Overfit Test: Model={args.model.upper()} | Device={device}")

    # Set up data paths and load training metadata
    data_root = config['data']['data_root']
    train_metadata_file = os.path.join(data_root, "train_metadata.json")

    # Instantiate dataset with target sizes based on model choice
    if args.model.lower() == "xception":
        spatial_size = 299
        frequency_size = 224
    else:
        spatial_size = 224
        frequency_size = 224

    dataset = DeepfakeDataset(
        data_root=data_root,
        metadata_file=train_metadata_file,
        augmentations=None,  # Disable augmentation
        use_phase=(config['model']['frequency_channels'] == 2),
        normalize_frequency=config['preprocessing']['frequency_normalize'],
        is_training=False,  # Disable random subsets
        spatial_size=spatial_size,
        frequency_size=frequency_size,
        frequency_channels=config['model'].get('frequency_channels', 1)
    )

    if len(dataset) == 0:
        print("ERROR: Training metadata is empty or missing. Run preprocessing first.")
        sys.exit(1)

    # Redesign: select a fixed balanced subset of 32 frames from different videos
    reals_by_video = {}
    fakes_by_video = {}
    for sample in dataset.all_samples:
        vid = sample['video_id']
        lbl = sample['label']
        if lbl == 0:
            if vid not in reals_by_video:
                reals_by_video[vid] = []
            reals_by_video[vid].append(sample)
        else:
            if vid not in fakes_by_video:
                fakes_by_video[vid] = []
            fakes_by_video[vid].append(sample)

    overfit_samples = []
    # Pick 16 real videos, 1 frame from each = 16 real frames
    for vid in sorted(list(reals_by_video.keys()))[:16]:
        overfit_samples.append(reals_by_video[vid][0])
    # Pick 16 fake videos, 1 frame from each = 16 fake frames
    for vid in sorted(list(fakes_by_video.keys()))[:16]:
        overfit_samples.append(fakes_by_video[vid][0])

    # Assert exactly 32 frames, balanced
    real_cnt = sum(1 for s in overfit_samples if s['label'] == 0)
    fake_cnt = sum(1 for s in overfit_samples if s['label'] == 1)
    print(f"Selected {len(overfit_samples)} frames for overfit test (Real={real_cnt}, Fake={fake_cnt}) from unique videos.")

    if real_cnt == 0 or fake_cnt == 0 or len(overfit_samples) < 10:
        # Fallback to simple slice if we have very few videos (e.g. smoke test / tiny subsets)
        print("Warning: Insufficient unique videos. Falling back to simple slice.")
        reals = [s for s in dataset.all_samples if s['label'] == 0][:16]
        fakes = [s for s in dataset.all_samples if s['label'] == 1][:16]
        overfit_samples = reals + fakes

    # Force active samples
    dataset.active_samples = overfit_samples
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Disable dropout in configuration for this test
    config_overfit = yaml.safe_load(yaml.dump(config))
    config_overfit['model']['dropout'] = 0.0
    config_overfit['model']['pretrained'] = True  # Keep pretrained flag

    # Instantiate model
    model = get_model(args.model, config_overfit).to(device)
    model.train()

    # Disable early stopping, scheduling, weight decay, and backbone freezing
    # We train all parameters with a learning rate of 1e-4 using standard Adam
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()

    print("\nRunning Overfitting Test (25 epochs max)...")
    success = False
    final_acc = 0.0
    final_auc = 0.0

    for epoch in range(args.epochs):
        all_probas = []
        all_labels = []
        all_preds = []
        running_loss = 0.0

        for batch in dataloader:
            face_spatial = batch['face_spatial'].to(device)
            face_frequency = batch['face_frequency'].to(device)
            frame_spatial = batch['frame_spatial'].to(device)
            frame_frequency = batch['frame_frequency'].to(device)
            labels = batch['label'].float().to(device)

            optimizer.zero_grad()
            outputs = model(face_spatial, face_frequency, frame_spatial, frame_frequency).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            # Align shapes
            outputs = outputs.view_as(labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probas = torch.sigmoid(outputs).detach().cpu().numpy()
            if probas.ndim == 0:
                probas = np.array([probas.item()])
            preds = (probas >= 0.5).astype(int)

            all_probas.extend(probas)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)

        metrics = compute_frame_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probas))
        print(f"  Epoch {epoch:02d}: Loss={running_loss/len(dataloader):.4f} | Accuracy={metrics['accuracy']*100:.2f}% | AUC={metrics['auc']:.4f}")

        if metrics['accuracy'] >= 0.98 and metrics['auc'] >= 0.98:
            success = True
            final_acc = metrics['accuracy']
            final_auc = metrics['auc']
            print(f"\n✓ Overfitting test PASSED on epoch {epoch}!")
            break

    # Save overfit status cache file
    status_dir = Path("results")
    status_dir.makedirs_p() if hasattr(status_dir, 'makedirs_p') else os.makedirs(status_dir, exist_ok=True)
    status_file = status_dir / "overfit_test_status.json"

    # Save to json file
    model_clean = args.model.lower().replace("-", "_")
    
    # Try to load existing cache or start new
    cache = {}
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    commit_hash = get_git_commit()
    config_md5 = get_config_fingerprint(args.config)

    cache[model_clean] = {
        "commit": commit_hash,
        "config_md5": config_md5,
        "status": "passed" if success else "failed",
        "accuracy": float(final_acc),
        "auc": float(final_auc),
        "timestamp": str(torch.typename(model))
    }

    with open(status_file, 'w') as f:
        json.dump(cache, f, indent=2)

    if not success:
        print(f"\n❌ Overfitting test FAILED. Model could not overfit.")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
