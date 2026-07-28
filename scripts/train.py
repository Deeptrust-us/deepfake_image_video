"""Training script for deepfake detection baseline and multi-stream models."""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json
import hashlib
from collections import defaultdict
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import get_model, UnifiedMultiStreamModel
from src.data.dataset import DeepfakeDataset
from src.utils.augmentations import DualStreamAugmentation
from src.utils.metrics import compute_frame_metrics, find_optimal_threshold


def get_git_commit():
    """Retrieve current git commit hash."""
    try:
        import subprocess
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, accum_steps):
    """Train for one epoch with corrected gradient accumulation."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in pbar:
        face_spatial = batch['face_spatial'].to(device)
        face_frequency = batch['face_frequency'].to(device)
        frame_spatial = batch['frame_spatial'].to(device)
        frame_frequency = batch['frame_frequency'].to(device)
        labels = batch['label'].float().to(device)
        
        # Forward pass
        outputs = model(face_spatial, face_frequency, frame_spatial, frame_frequency).squeeze()
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
            
        # Shape alignment check
        outputs = outputs.view_as(labels)
        
        # Assertions
        assert labels.dtype == torch.float32, f"Labels must be FloatTensor, got {labels.dtype}"
        assert outputs.shape == labels.shape, f"Outputs/labels shape mismatch: {outputs.shape} vs {labels.shape}"
        
        # Loss scale division by accumulation steps
        loss = criterion(outputs, labels)
        loss = loss / accum_steps
        loss.backward()
        
        # Weight step after accumulation steps or final batch
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate running loss & predictions
        running_loss += (loss.item() * accum_steps)
        probas = torch.sigmoid(outputs).detach().cpu().numpy()
        if probas.ndim == 0:
            probas = np.array([probas.item()])
        preds = (probas >= 0.5).astype(int)
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probas.extend(probas)
        
        pbar.set_postfix({'loss': loss.item() * accum_steps})
    
    metrics = compute_frame_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def validate(model, dataloader, device):
    """Validate model and return both frame-level and video-level metrics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    video_predictions = defaultdict(list)
    video_labels = {}
    
    val_criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            face_spatial = batch['face_spatial'].to(device)
            face_frequency = batch['face_frequency'].to(device)
            frame_spatial = batch['frame_spatial'].to(device)
            frame_frequency = batch['frame_frequency'].to(device)
            labels = batch['label'].float().to(device)
            video_ids = batch['video_id']
            
            outputs = model(face_spatial, face_frequency, frame_spatial, frame_frequency).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            outputs = outputs.view_as(labels)
            loss = val_criterion(outputs, labels)
            running_loss += loss.item()
            
            # Continuous probabilities via sigmoid
            probas = torch.sigmoid(outputs).cpu().numpy()
            if probas.ndim == 0:
                probas = np.array([probas.item()])
            preds = (probas >= 0.5).astype(int)
            
            labels_np = labels.cpu().numpy()
            if labels_np.ndim == 0:
                labels_np = np.array([labels_np.item()])
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probas.extend(probas)
            
            # Consistent Probabilities Averaging aggregation method
            for vid_id, prob, lbl in zip(video_ids, probas, labels_np):
                video_predictions[vid_id].append(prob)
                video_labels[vid_id] = int(lbl)
                
    frame_metrics = compute_frame_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    frame_metrics['loss'] = running_loss / len(dataloader)
    
    # Validation Video-level predictions threshold search
    video_probas = []
    video_labels_list = []
    for vid_id, preds_list in video_predictions.items():
        video_probas.append(np.mean(preds_list))
        video_labels_list.append(video_labels[vid_id])
        
    video_probas_arr = np.array(video_probas)
    video_labels_arr = np.array(video_labels_list)
    
    opt_val_thresh = find_optimal_threshold(video_labels_arr, video_probas_arr)
    video_preds_arr = (video_probas_arr >= opt_val_thresh).astype(int)
    
    video_metrics = compute_frame_metrics(video_labels_arr, video_preds_arr, video_probas_arr)
    video_metrics['optimal_threshold'] = opt_val_thresh
    
    return frame_metrics, video_metrics


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection baseline or multi-stream model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="quad_stream", help="Model architecture to train")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replication")
    parser.add_argument("--smoke_test", action="store_true", help="Run in smoke test mode")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"✓ Initialized random seed: {args.seed}")
    
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
        
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
        print(f"✓ Overriding num_epochs to {args.epochs}")
        
    # Check if overfitting test has run and passed for this configuration (except in smoke tests)
    model_name_clean = args.model.lower().replace("-", "_")
    status_file = os.path.join("results", "overfit_test_status.json")
    commit_hash = get_git_commit()
    config_md5 = get_config_fingerprint(config_path)
    
    passed_overfit = False
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                cache = json.load(f)
            if model_name_clean in cache:
                entry = cache[model_name_clean]
                if entry.get("commit") == commit_hash and entry.get("config_md5") == config_md5 and entry.get("status") == "passed":
                    passed_overfit = True
        except Exception:
            pass
            
    if not args.smoke_test and not passed_overfit:
        raise ValueError(f"CRITICAL ERROR: Overfitting test has not been run or did not pass for model '{args.model}' and current configuration. Run: python scripts/overfit_test.py --config {args.config} --model {args.model}")
    print("✓ Overfitting test verification check passed.")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Determine batch size and gradient accumulation steps based on model Choice
    if args.model.lower() == "xception":
        batch_size = 32
        accum_steps = 1
        spatial_size = 299
        frequency_size = 224
    elif args.model.lower() in ["rgb_fft_dual_stream", "two_stream", "face_only", "frame_only", "spatial_only", "frequency_only"]:
        batch_size = 16
        accum_steps = 2
        spatial_size = 224
        frequency_size = 224
    elif args.model.lower() in ["quad_stream"]:
        batch_size = 8
        accum_steps = 4
        spatial_size = 224
        frequency_size = 224
    else:
        batch_size = config['training'].get('batch_size', 32)
        accum_steps = 1
        spatial_size = 224
        frequency_size = 224
        
    print(f"Effective batch size calculation: batch_size={batch_size} * accum_steps={accum_steps} = {batch_size * accum_steps}")
    
    # Initialize augmentations
    aug_config = config['preprocessing']['augmentations']
    augmentations = DualStreamAugmentation(
        horizontal_flip_prob=aug_config['horizontal_flip'],
        rotation_range=aug_config.get('rotation_range', 5.0),
        brightness_range=aug_config.get('brightness_range', 0.1),
        contrast_range=aug_config.get('contrast_range', 0.1),
        noise_std=aug_config.get('noise_std', 0.01),
        gaussian_blur_prob=aug_config.get('gaussian_blur_prob', 0.3)
    )
    
    # Create datasets
    data_root = config['data']['data_root']
    train_dataset = DeepfakeDataset(
        data_root=data_root,
        metadata_file=os.path.join(data_root, "train_metadata.json"),
        face_detector=None,
        augmentations=augmentations,
        use_phase=(config['model']['frequency_channels'] == 2),
        normalize_frequency=config['preprocessing']['frequency_normalize'],
        is_training=True,
        spatial_size=spatial_size,
        frequency_size=frequency_size,
        frequency_channels=config['model'].get('frequency_channels', 1)
    )
    
    # Verify training dataset counts
    train_labels = [item['label'] for item in train_dataset.all_samples]
    train_real = train_labels.count(0)
    train_fake = train_labels.count(1)
    print(f"Verified training split total frames: Real={train_real}, Fake={train_fake}")
    
    # Check validation set
    val_metadata_path = os.path.join(data_root, "val_metadata.json")
    if os.path.exists(val_metadata_path):
        val_dataset = DeepfakeDataset(
            data_root=data_root,
            metadata_file=val_metadata_path,
            face_detector=None,
            augmentations=None,
            use_phase=(config['model']['frequency_channels'] == 2),
            normalize_frequency=config['preprocessing']['frequency_normalize'],
            is_training=False,
            spatial_size=spatial_size,
            frequency_size=frequency_size,
            frequency_channels=config['model'].get('frequency_channels', 1)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config['training'].get('num_workers', 0),
            pin_memory=config['training'].get('pin_memory', False)
        )
    else:
        val_loader = None
        val_dataset = None
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=config['training'].get('pin_memory', False)
    )
    
    # Initialize model using factory
    model = get_model(args.model, config).to(device)
    
    # Loss: raw logits BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    
    # Set up parameter groups & validation
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable_params}
    
    if args.model.lower() == "xception":
        # Phase 1: Freeze backbone, optimize classifier only
        for param in model.backbone.parameters():
            param.requires_grad = False
            
        param_groups = [
            {'params': list(model.classifier.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5}
        ]
        optimizer = optim.AdamW(param_groups)
        print("✓ Phase 1: Xception backbone FROZEN. Optimizing classifier only.")
    else:
        # Multi-stream models
        pretrained_params = []
        scratch_params = []
        
        pretrained_backbones = []
        if hasattr(model, 'face_spatial_stream') and model.face_spatial_stream is not None:
            pretrained_backbones.append(model.face_spatial_stream.backbone)
        if hasattr(model, 'frame_spatial_stream') and model.frame_spatial_stream is not None:
            pretrained_backbones.append(model.frame_spatial_stream.backbone)
            
        pretrained_ids = set()
        for bb in pretrained_backbones:
            for p in bb.parameters():
                pretrained_ids.add(id(p))
                pretrained_params.append(p)
                
        for p in model.parameters():
            if p.requires_grad and id(p) not in pretrained_ids:
                scratch_params.append(p)
                
        param_groups = [
            {'params': pretrained_params, 'lr': 1e-5, 'weight_decay': 1e-5},
            {'params': scratch_params, 'lr': 1e-4, 'weight_decay': 1e-5}
        ]
        
        # Validate optimizer parameter groups strictly
        grouped_ids = []
        for group in param_groups:
            for p in group['params']:
                grouped_ids.append(id(p))
                
        # Check duplicates
        from collections import Counter
        counts = Counter(grouped_ids)
        duplicates = [pid for pid, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError("CRITICAL ERROR: Trainable parameters assigned to multiple optimizer groups!")
            
        # Check omissions
        grouped_set = set(grouped_ids)
        omitted = trainable_ids - grouped_set
        if omitted:
            raise ValueError("CRITICAL ERROR: Trainable parameters omitted from optimizer groups!")
            
        optimizer = optim.AdamW(param_groups)
        print(f"✓ Optimizer parameter groups validated successfully.")
        print(f"  Pretrained backbone parameters: {sum(p.numel() for p in pretrained_params):,}")
        print(f"  Newly initialized parameters: {sum(p.numel() for p in scratch_params):,}")

    # LR Scheduler (plateau based on validation video AUC)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Tensorboard
    writer = SummaryWriter(log_dir=config['paths']['log_dir'])
    
    best_val_auc = 0.0
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        print(f"Resumed from epoch {start_epoch}")
        
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        # Seeded dynamic epoch frame selection
        train_dataset.epoch_init(seed=args.seed + epoch)
        
        # Xception baseline: Phase 2 unfreezing at epoch 3
        if args.model.lower() == "xception" and epoch == 3:
            print("\n✓ Phase 2: Unfreezing Xception backbone. Fine-tuning complete model.")
            for param in model.backbone.parameters():
                param.requires_grad = True
                
            trainable_params_p2 = [p for p in model.parameters() if p.requires_grad]
            trainable_ids_p2 = {id(p) for p in trainable_params_p2}
            
            param_groups_p2 = [
                {'params': list(model.backbone.parameters()), 'lr': 1e-5, 'weight_decay': 1e-5},
                {'params': list(model.classifier.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5}
            ]
            
            # Validate Phase 2 parameter groups
            grouped_ids_p2 = []
            for group in param_groups_p2:
                for p in group['params']:
                    grouped_ids_p2.append(id(p))
            assert set(grouped_ids_p2) == trainable_ids_p2, "Optimizer groups validation failed at Phase 2 unfreezing!"
            
            # Recreate optimizer and scheduler
            optimizer = optim.AdamW(param_groups_p2)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=2
            )
            
        # Train one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, accum_steps)
        
        # Validate
        if val_loader is not None and len(val_dataset) > 0:
            val_frame_metrics, val_video_metrics = validate(model, val_loader, device)
        else:
            val_frame_metrics = train_metrics.copy()
            val_video_metrics = train_metrics.copy()
            val_video_metrics['optimal_threshold'] = 0.5
            
        # Step ReduceLROnPlateau using validation video-level AUC
        val_auc = val_video_metrics['auc']
        scheduler.step(val_auc)
        
        # Log to TensorBoard
        for key, value in train_metrics.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_video_metrics.items():
            writer.add_scalar(f'Val_Video/{key}', value, epoch)
            
        print(f"\nEpoch {epoch}:")
        print(f"  Train     - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val Video - Loss: {val_frame_metrics['loss']:.4f}, AUC: {val_auc:.4f}, F1: {val_video_metrics['f1']:.4f} (Thresh={val_video_metrics['optimal_threshold']:.4f})")
        
        # Checkpoint paths
        best_filename = f'best_model_{model_name_clean}_seed{args.seed}.pth'
        latest_filename = f'latest_{model_name_clean}_seed{args.seed}.pth'
        best_path = os.path.join(config['paths']['checkpoint_dir'], best_filename)
        latest_path = os.path.join(config['paths']['checkpoint_dir'], latest_filename)
        
        # Save best checkpoint (based on validation video AUC)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save resolved registry metadata inside checkpoint
            streams = model.active_streams if hasattr(model, 'active_streams') else ["face_spatial"]
            backbone_type = getattr(model, 'backbone_type', config['model'].get('spatial_backbone', 'resnet18'))
            param_count = sum(p.numel() for p in model.parameters())
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': best_val_auc,
                'val_metrics': val_video_metrics,
                'optimal_threshold': val_video_metrics['optimal_threshold'],
                'model_metadata': {
                    'model_name': args.model,
                    'streams': streams,
                    'backbone_type': backbone_type,
                    'param_count': param_count
                }
            }
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model checkpoint to: {best_path} (Video AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            
        # Save latest checkpoint
        latest_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
            'val_metrics': val_video_metrics,
            'optimal_threshold': val_video_metrics.get('optimal_threshold', 0.5)
        }
        torch.save(latest_checkpoint, latest_path)
        
        # Early stopping patience check (based on validation video AUC)
        if patience_counter >= 7:
            print(f"Early stopping triggered after 7 epochs of no video AUC improvement.")
            break
            
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
