import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn

# Configure matplotlib to use Times New Roman.
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['figure.titlesize'] = 18

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.factory import available_models, build_model  # noqa: E402
from train.config import FEATURE_SUBSETS, get_config  # noqa: E402
from train.utils import build_dataloaders  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ChainBMD/LSTM models on the test split.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name to evaluate. Options: {', '.join(available_models())}",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (defaults to TrainingConfig.checkpoint_path).",
    )
    parser.add_argument(
        "--subject-split",
        action="store_true",
        help="Match training: split train/val/test by subject.",
    )
    parser.add_argument(
        "--feature-subset",
        type=str,
        choices=tuple(FEATURE_SUBSETS.keys()),
        help="Evaluate only the specified node feature subset.",
    )
    return parser.parse_args()


def collect_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect predictions, labels, and probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            labels = batch["label"].to(device)
            logits, _, _ = model(sequences)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3, save_path: Path = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Compute percentages.
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Use a cleaner color palette.
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    
    # Add colorbar.
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.set_ylabel('Count', rotation=90, va='bottom', fontsize=12)
    
    # Set labels.
    class_labels = [f'Class {i}' for i in range(num_classes)]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_labels,
           yticklabels=class_labels)
    
    # Set titles and axis labels
    # ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Annotate cells with count and percentage
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=11, fontweight='bold')
    
    # Add gridlines
    ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int = 3, save_path: Path = None):
    """Plot ROC curves (one-vs-rest, one curve per class)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Define colors and line styles for the three classes.
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # blue, purple, orange
    markers = ['o', 's', '^']  # distinct markers
    linestyles = ['-', '-', '-']
    
    # Compute and plot ROC curves per class.
    for i in range(num_classes):
        # Convert multiclass to binary (one-vs-rest).
        y_true_binary = (y_true == i).astype(int)
        y_prob_binary = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
        
        # Plot with distinct colors, markers, and thicker lines.
        ax.plot(fpr, tpr, 
               color=colors[i],
               linestyle=linestyles[i],
               marker=markers[i],
               markersize=5,
               markevery=max(1, len(fpr) // 15),  # show markers at intervals
               lw=2.5, 
               label=f'Class {i}',
               alpha=0.9)
    
    # Plot diagonal (random classifier).
    ax.plot([0, 1], [0, 1], 
           color='gray', 
           linestyle='--', 
           lw=2, 
           label='Random',
           alpha=0.7,
           zorder=0)  # bottom layer
    
    # Configure axes.
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves', fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=12, ncol=1)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Axis ticks
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"ROC curves saved to: {save_path}")
    plt.close()


def plot_auc_scores(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int = 3, save_path: Path = None):
    """Plot AUC bar chart."""
    # Compute AUC per class.
    auc_scores = []
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_prob_binary = y_probs[:, i]
        auc_score = roc_auc_score(y_true_binary, y_prob_binary)
        auc_scores.append(auc_score)
    
    # Macro average AUC.
    macro_auc = np.mean(auc_scores)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Use a softer color palette.
    colors = ['#5B9BD5', '#70AD47', '#FFC000']  # soft blue, green, yellow
    macro_color = '#4472C4'  # dark blue for macro average
    
    # Prepare data: per-class + macro average.
    class_labels = [f'Class {i}' for i in range(num_classes)] + ['Macro Average']
    all_auc = auc_scores + [macro_auc]
    bar_colors = colors + [macro_color]
    
    # Plot vertical bars.
    bars = ax.bar(class_labels, all_auc, color=bar_colors, alpha=0.8, width=0.6, edgecolor='white', linewidth=2)
    
    # Add value labels.
    for i, (bar, value) in enumerate(zip(bars, all_auc)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Y-axis range.
    ax.set_ylim([0.0, 1.1])
    ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    # ax.set_title('AUC Scores', fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Y-axis ticks
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"AUC chart saved to: {save_path}")
    plt.close()
    
    return auc_scores, macro_auc


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, num_classes: int = 3):
    """Print all evaluation metrics."""
    print("\n" + "="*60)
    print("Evaluation Metrics")
    print("="*60)
    
    # Base metrics (macro average is common for multiclass).
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print("\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # AUC (one-vs-rest, per-class and macro average).
    try:
        auc_scores = []
        print("\nAUC (per class, one-vs-rest):")
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_prob_binary = y_probs[:, i]
            auc_score = roc_auc_score(y_true_binary, y_prob_binary)
            auc_scores.append(auc_score)
            print(f"  Class {i}: {auc_score:.4f}")
        
        macro_auc = np.mean(auc_scores)
        print(f"\nMacro AUC: {macro_auc:.4f}")
    except Exception as e:
        print(f"\nError computing AUC: {e}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("\nConfusion Matrix:")
    print(cm)
    
    print("="*60 + "\n")


def main():
    args = _parse_args()
    env_model = os.getenv("CHAINBMD_MODEL")
    selected_model = args.model or env_model
    cfg = get_config(model_name=selected_model)
    if args.subject_split:
        cfg.split_by_subject = True
    if args.feature_subset:
        cfg.feature_subset = args.feature_subset
        cfg.feature_indices = FEATURE_SUBSETS[cfg.feature_subset]
        cfg.raw_feature_dim = len(cfg.feature_indices)
    if cfg.feature_indices:
        if hasattr(cfg.model_config, "raw_feature_dim"):
            cfg.model_config.raw_feature_dim = len(cfg.feature_indices)
        if hasattr(cfg.model_config, "input_dim"):
            cfg.model_config.input_dim = len(cfg.feature_indices)
    checkpoint = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_path
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} not found. Train the model first."
        )
    _, _, test_loader = build_dataloaders(
        hdf5_path=str(cfg.data_path),
        batch_size=cfg.batch_size,
        val_split=cfg.subject_val_split if cfg.split_by_subject else cfg.val_split,
        test_split=cfg.subject_test_split if cfg.split_by_subject else cfg.test_split,
        target_field=cfg.target_field,
        random_seed=cfg.random_seed,
        split_by_subject=cfg.split_by_subject,
        feature_indices=cfg.feature_indices,
    )
    device = torch.device(cfg.device)
    model = build_model(cfg.model_name, cfg.model_config).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    
    # Collect predictions
    y_true, y_pred, y_probs = collect_predictions(model, test_loader, device)
    
    # Print metrics
    num_classes = getattr(cfg.model_config, "num_classes", 3)
    print_metrics(y_true, y_pred, y_probs, num_classes=num_classes)
    
    # Plot confusion matrix
    output_dir = PROJECT_ROOT / "outputs" / "figures" / "test_metrics"
    plot_confusion_matrix(y_true, y_pred, num_classes=num_classes, 
                        save_path=output_dir / "confusion_matrix.png")
    
    # Plot ROC curves
    plot_roc_curves(y_true, y_probs, num_classes=num_classes, 
                   save_path=output_dir / "roc_curves.png")
    
    # Plot AUC bar chart
    plot_auc_scores(y_true, y_probs, num_classes=num_classes, 
                   save_path=output_dir / "auc_scores.png")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
