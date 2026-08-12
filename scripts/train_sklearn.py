#!/usr/bin/env python
"""CLI script to train and evaluate sklearn classifiers.

Usage:
    python scripts/train_sklearn.py --data data/raw/dataset.csv --strategy combined --output results/
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CommitDataset
from src.evaluation.metrics import plot_confusion_matrix
from src.models.sklearn_models import SklearnTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train sklearn classifiers for commit classification"
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
        help="Fraction of data for testing"
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
    print(f"  Keywords: {len(dataset.get_keyword_features())}")
    print(f"  Changes:  {len(dataset.get_code_change_features())}")
    print(f"  Combined: {len(dataset.get_feature_names())}")

    X, y = dataset.to_sklearn_arrays(feature_strategy=args.strategy)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train-test split for PyTorch comparison consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    trainer = SklearnTrainer(X_train, y_train, test_size=args.test_size, random_state=42)
    # Override with our pre-split data
    trainer.X_train = X_train
    trainer.X_test = X_test
    trainer.y_train = y_train
    trainer.y_test = y_test

    results = trainer.run_all()
    trainer.print_summary()

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_results(output_dir / f"sklearn_{args.strategy}_results.json")

    # Plot confusion matrices for best performer (usually RF)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    plot_confusion_matrix(
        y_test.tolist(),
        y_pred.tolist(),
        title=f"Random Forest — {args.strategy.capitalize()} Features",
        save_path=output_dir / f"figures/confusion_matrix_sklearn_{args.strategy}.png",
    )

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
