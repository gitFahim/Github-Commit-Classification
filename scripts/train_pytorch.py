#!/usr/bin/env python
"""CLI script to train and evaluate the PyTorch neural network.

Usage:
    python scripts/train_pytorch.py --data data/raw/dataset.csv --strategy combined --epochs 40 --output results/
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CommitDataset
from src.evaluation.metrics import plot_confusion_matrix, plot_training_history
from src.models.pytorch_models import train_pytorch


def main():
    parser = argparse.ArgumentParser(
        description="Train PyTorch neural network for commit classification"
    )
    parser.add_argument(
        "--data", type=str, default="data/raw/dataset.csv",
        help="Path to dataset CSV file"
    )
    parser.add_argument(
        "--strategy", type=str, default="combined",
        choices=["keywords", "changes", "combined"],
        help="Feature strategy to use"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15,
        help="Fraction of data for validation"
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Directory to save results"
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}...")
    dataset = CommitDataset(args.data)
    print(f"Loaded {len(dataset)} commits")
    print(f"Feature strategy: {args.strategy}")

    X, y = dataset.to_sklearn_arrays(feature_strategy=args.strategy)
    input_dim = X.shape[1]
    print(f"Feature matrix shape: {X.shape}")
    print(f"Input dimension: {input_dim}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"\nTraining PyTorch model for {args.epochs} epochs...")
    results = train_pytorch(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        save_dir=Path(args.output) / "checkpoints",
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"pytorch_{args.strategy}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot training history
    plot_training_history(
        results["history"],
        save_path=output_dir / f"figures/training_history_{args.strategy}.png",
    )

    # Plot confusion matrix using validation predictions from last epoch
    # We need to re-run inference to get predictions
    import torch
    from src.models.pytorch_models import CommitClassifier, CommitDatasetTorch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CommitClassifier(input_dim=input_dim, dropout=args.dropout).to(device)
    checkpoint_path = output_dir / "checkpoints/best_model.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    val_dataset = CommitDatasetTorch(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    plot_confusion_matrix(
        all_labels,
        all_preds,
        title=f"PyTorch NN — {args.strategy.capitalize()} Features",
        save_path=output_dir / f"figures/confusion_matrix_pytorch_{args.strategy}.png",
    )

    print(f"\nResults saved to {output_dir}")
    print(f"Best validation accuracy: {results['best_val_accuracy_pct']:.3f}%")


if __name__ == "__main__":
    main()
