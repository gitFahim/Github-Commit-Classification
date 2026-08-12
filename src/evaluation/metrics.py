"""Visualization and reporting utilities for model evaluation."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> None:
    """Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class names for axis labels.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size tuple.
    """
    if labels is None:
        labels = ["Corrective", "Perfective", "Adaptive"]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> None:
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
        save_path: Path to save the figure.
        figsize: Figure size tuple.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_title("Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc")
    axes[1].set_title("Accuracy Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")
    plt.close(fig)
