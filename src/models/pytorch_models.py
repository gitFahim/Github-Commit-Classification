"""PyTorch-based neural network for commit classification.

A fully-connected feedforward network with batch normalization and dropout
for classifying commits into maintenance activities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class CommitClassifier(nn.Module):
    """Fully-connected neural network for commit classification.

    Architecture:
        Input (65) -> FC(128) -> BN -> ReLU -> Dropout(0.3)
                   -> FC(64)  -> BN -> ReLU -> Dropout(0.3)
                   -> FC(3)   -> Softmax
    """

    def __init__(
        self,
        input_dim: int = 65,
        hidden_dims: List[int] = None,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (no softmax)."""
        return self.network(x)


class CommitDatasetTorch(Dataset):
    """PyTorch Dataset wrapper for commit features."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int = 65,
    hidden_dims: Optional[List[int]] = None,
    num_classes: int = 3,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    dropout: float = 0.3,
    device: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> Dict:
    """Train and evaluate the PyTorch commit classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        input_dim: Number of input features.
        hidden_dims: List of hidden layer dimensions.
        num_classes: Number of output classes.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Adam learning rate.
        dropout: Dropout probability.
        device: "cuda", "cpu", or None (auto).
        save_dir: Directory to save best model checkpoint.

    Returns:
        Dictionary with training history and final metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Datasets and loaders
    train_dataset = CommitDatasetTorch(X_train, y_train)
    val_dataset = CommitDatasetTorch(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = CommitClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_dir / "best_model.pt")

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
    )

    final_acc = accuracy_score(all_labels, all_preds)
    final_kappa = cohen_kappa_score(all_labels, all_preds)

    results = {
        "best_val_accuracy": float(best_val_acc),
        "best_val_accuracy_pct": round(float(best_val_acc) * 100, 3),
        "final_accuracy": float(final_acc),
        "final_kappa": float(final_kappa),
        "final_kappa_pct": round(float(final_kappa) * 100, 3),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "classification_report": classification_report(
            all_labels, all_preds, output_dict=True
        ),
        "history": history,
    }

    return results
