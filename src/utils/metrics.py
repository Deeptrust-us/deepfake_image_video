"""Evaluation metrics for deepfake detection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from typing import Dict, Tuple


def compute_frame_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute frame-level evaluation metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # ROC-AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics['auc'] = 0.0
    
    # EER (Equal Error Rate)
    metrics['eer'] = compute_eer(y_true, y_proba)
    
    return metrics


def compute_eer(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        EER value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    fnr = 1 - tpr
    
    # Find threshold where FPR = FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    return float(eer)


def compute_video_metrics(video_predictions: Dict[str, np.ndarray],
                         video_labels: Dict[str, int],
                         aggregation: str = "mean") -> Dict[str, float]:
    """
    Compute video-level metrics by aggregating frame predictions.
    
    Args:
        video_predictions: Dictionary mapping video_id to array of frame predictions
        video_labels: Dictionary mapping video_id to label
        aggregation: Aggregation method ('mean', 'max', 'median')
        
    Returns:
        Dictionary of video-level metrics
    """
    video_probas = []
    video_labels_list = []
    
    for video_id, predictions in video_predictions.items():
        if video_id not in video_labels:
            continue
        
        # Aggregate frame predictions
        if aggregation == "mean":
            video_proba = np.mean(predictions)
        elif aggregation == "max":
            video_proba = np.max(predictions)
        elif aggregation == "median":
            video_proba = np.median(predictions)
        else:
            video_proba = np.mean(predictions)
        
        video_probas.append(video_proba)
        video_labels_list.append(video_labels[video_id])
    
    video_probas = np.array(video_probas)
    video_labels_list = np.array(video_labels_list)
    video_preds = (video_probas > 0.5).astype(int)
    
    return compute_frame_metrics(video_labels_list, video_preds, video_probas)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


