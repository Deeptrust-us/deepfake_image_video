"""Training script for dual-stream deepfake detection."""

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
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dual_stream import DualStreamModel
from src.data.dataset import DeepfakeDataset
from src.utils.augmentations import DualStreamAugmentation
from src.utils.face_detection import FaceDetector
from src.utils.metrics import compute_frame_metrics


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, class_weights=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        face_spatial = batch['face_spatial'].to(device)
        face_frequency = batch['face_frequency'].to(device)
        frame_spatial = batch['frame_spatial'].to(device)
        frame_frequency = batch['frame_frequency'].to(device)
        labels = batch['label'].float().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(face_spatial, face_frequency, frame_spatial, frame_frequency).squeeze()
        
        # Compute loss
        if isinstance(criterion, nn.BCELoss) and criterion.reduction == 'none':
            # Weighted BCE loss
            per_sample_loss = criterion(outputs, labels)
            sample_weights = class_weights[labels.long()]
            loss = (per_sample_loss * sample_weights).mean()
        else:
            # Focal loss or standard loss (already handles reduction)
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        preds = (outputs > 0.5).cpu().numpy()
        probas = outputs.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels_np)
        all_probas.extend(probas)
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute epoch metrics
    metrics = compute_frame_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    # Use standard BCE loss for validation (not weighted)
    val_criterion = nn.BCELoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            face_spatial = batch['face_spatial'].to(device)
            face_frequency = batch['face_frequency'].to(device)
            frame_spatial = batch['frame_spatial'].to(device)
            frame_frequency = batch['frame_frequency'].to(device)
            labels = batch['label'].float().to(device)
            
            outputs = model(face_spatial, face_frequency, frame_spatial, frame_frequency).squeeze()
            loss = val_criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (outputs > 0.5).cpu().numpy()
            probas = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probas.extend(probas)
    
    metrics = compute_frame_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train dual-stream deepfake detection model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Initialize face detector (optional, for on-the-fly detection)
    face_detector = None  # Assume faces are preprocessed
    
    # Initialize augmentations
    aug_config = config['preprocessing']['augmentations']
    augmentations = DualStreamAugmentation(
        horizontal_flip_prob=aug_config['horizontal_flip'],
        rotation_range=aug_config['rotation_range'],
        brightness_range=aug_config['brightness_range'],
        contrast_range=aug_config['contrast_range'],
        noise_std=aug_config.get('noise_std', 0.0),
        gaussian_blur_prob=aug_config.get('gaussian_blur_prob', 0.0)
    )
    
    # Create datasets
    data_root = config['data']['data_root']
    train_dataset = DeepfakeDataset(
        data_root=data_root,
        metadata_file=os.path.join(data_root, "train_metadata.json"),
        face_detector=face_detector,
        augmentations=augmentations,
        use_phase=(config['model']['frequency_channels'] == 2),
        normalize_frequency=config['preprocessing']['frequency_normalize'],
        is_training=True
    )
    
    # Check if validation set exists and has data
    val_metadata_path = os.path.join(data_root, "val_metadata.json")
    if os.path.exists(val_metadata_path):
        val_dataset = DeepfakeDataset(
            data_root=data_root,
            metadata_file=val_metadata_path,
            face_detector=face_detector,
            augmentations=None,
            use_phase=(config['model']['frequency_channels'] == 2),
            normalize_frequency=config['preprocessing']['frequency_normalize'],
            is_training=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        ) if len(val_dataset) > 0 else None
    else:
        val_loader = None
        val_dataset = None
    
    # Create dataloaders with optional oversampling for real videos
    oversample_real = config['training'].get('oversample_real', False)
    if oversample_real:
        from torch.utils.data import WeightedRandomSampler
        # Calculate sample weights: higher weight for real videos (label=0)
        real_oversample_ratio = config['training'].get('real_oversample_ratio', 2.0)
        sample_weights = []
        for item in train_dataset.samples:
            if item['label'] == 0:  # Real video
                sample_weights.append(real_oversample_ratio)
            else:  # Fake video
                sample_weights.append(1.0)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"Using WeightedRandomSampler: Real videos oversampled {real_oversample_ratio}x")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
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
    
    # Freeze backbones if requested
    freeze_spatial = config['model'].get('freeze_spatial_backbone', False)
    freeze_frequency = config['model'].get('freeze_frequency_backbone', False)
    
    if freeze_spatial:
        for param in model.face_spatial_stream.backbone.parameters():
            param.requires_grad = False
        for param in model.frame_spatial_stream.backbone.parameters():
            param.requires_grad = False
        print("✓ Spatial backbones FROZEN (face + frame, only FC layers trainable)")
    else:
        print("✓ Spatial backbones UNFROZEN (face + frame, all parameters trainable)")
    
    if freeze_frequency:
        for param in model.face_frequency_stream.backbone.parameters():
            param.requires_grad = False
        for param in model.frame_frequency_stream.backbone.parameters():
            param.requires_grad = False
        print("✓ Frequency backbones FROZEN (face + frame, only FC layers trainable)")
    else:
        print("✓ Frequency backbones UNFROZEN (face + frame, all parameters trainable)")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    
    # Loss and optimizer
    # Calculate class weights for imbalanced dataset
    import json
    with open(os.path.join(data_root, "train_metadata.json"), 'r') as f:
        train_metadata = json.load(f)
    
    train_labels = [item['label'] for item in train_metadata]
    class_counts = np.bincount(train_labels)
    total = len(train_labels)
    
    # Calculate weights: inverse frequency (more weight to minority class)
    class_weights = total / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class distribution: Real={class_counts[0]}, Fake={class_counts[1]}")
    print(f"Class weights: Real={class_weights[0]:.4f}, Fake={class_weights[1]:.4f}")
    
    # Choose loss function
    use_focal_loss = config['training'].get('use_focal_loss', False)
    if use_focal_loss:
        from src.utils.focal_loss import FocalLoss
        focal_alpha = config['training'].get('focal_loss_alpha', 0.25)
        focal_gamma = config['training'].get('focal_loss_gamma', 2.0)
        # Use inverse frequency as alpha for positive class
        alpha_fake = class_weights[1] / (class_weights[0] + class_weights[1])
        criterion = FocalLoss(alpha=alpha_fake, gamma=focal_gamma, reduction='mean')
        print(f"Using Focal Loss (alpha={alpha_fake:.4f}, gamma={focal_gamma})")
    else:
        print("Using weighted BCE loss to handle class imbalance")
        # Use standard BCE loss with no reduction - we'll apply weights manually in training loop
        criterion = nn.BCELoss(reduction='none')  # No reduction, we'll weight manually
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if config['training']['optimizer'] == "adam":
        optimizer = optim.Adam(
            trainable_params,
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
    else:
        optimizer = optim.SGD(
            trainable_params,
            lr=float(config['training']['learning_rate']),
            momentum=float(config['training']['momentum']),
            weight_decay=float(config['training']['weight_decay'])
        )
    
    # Learning rate scheduler
    if config['training']['scheduler'] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['scheduler_params']['step_size'],
            gamma=config['training']['scheduler_params']['gamma']
        )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=config['paths']['log_dir'])
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, class_weights=class_weights_tensor)
        
        # Validate
        if val_loader is not None and len(val_dataset) > 0:
            val_metrics = validate(model, val_loader, criterion, device)
        else:
            # Use train metrics as validation if no validation set
            val_metrics = train_metrics.copy()
            print("Warning: No validation set available, using training metrics")
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
            'val_metrics': val_metrics
        }
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(checkpoint, os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
            patience_counter = 0
            print(f"  ✓ Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(config['paths']['checkpoint_dir'], 'latest.pth'))
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()

