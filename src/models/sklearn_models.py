"""Scikit-learn based classifiers for commit classification.

Implements Random Forest, Gradient Boosting, and Decision Tree (J48)
with unified training and evaluation interface.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


class SklearnTrainer:
    """Unified trainer for sklearn-based commit classifiers.

    Supports three feature strategies:
        - keywords: 19 keyword binary features
        - changes: 46 code change binary features
        - combined: 65 fused features

    And three classifiers:
        - Random Forest
        - Gradient Boosting (GBM)
        - Decision Tree (J48 equivalent)
    """

    CLASSIFIERS = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
    }

    def __init__(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.15, random_state: int = 42) -> None:
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.results: Dict = {}
        self._split_data()

    def _split_data(self) -> None:
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )

    def train_and_evaluate(self, classifier_name: str = "random_forest") -> Dict:
        """Train a classifier and compute all evaluation metrics.

        Args:
            classifier_name: One of "random_forest", "gradient_boosting", "decision_tree".

        Returns:
            Dictionary with accuracy, kappa, precision, recall, f1, confusion matrix,
            classification report, and cross-validation scores.
        """
        if classifier_name not in self.CLASSIFIERS:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        clf = self.CLASSIFIERS[classifier_name]
        clf.fit(self.X_train, self.y_train)

        y_pred_train = clf.predict(self.X_train)
        y_pred_test = clf.predict(self.X_test)

        # Basic metrics
        accuracy = accuracy_score(self.y_test, y_pred_test)
        kappa = cohen_kappa_score(self.y_test, y_pred_test)

        # Per-class and macro metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, y_pred_test, average=None
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred_test, average="macro"
        )

        # Cross-validation
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        cv_scores = cross_val_score(clf, self.X, self.y, scoring="accuracy", cv=cv, n_jobs=1)

        result = {
            "classifier": classifier_name,
            "accuracy": float(accuracy),
            "accuracy_pct": round(float(accuracy) * 100, 3),
            "kappa": float(kappa),
            "kappa_pct": round(float(kappa) * 100, 3),
            "precision": {
                "per_class": precision.tolist(),
                "macro": float(macro_precision),
            },
            "recall": {
                "per_class": recall.tolist(),
                "macro": float(macro_recall),
            },
            "f1": {
                "per_class": f1.tolist(),
                "macro": float(macro_f1),
            },
            "support": support.tolist(),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred_test).tolist(),
            "classification_report": classification_report(self.y_test, y_pred_test, output_dict=True),
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "cv_accuracy_mean_pct": round(float(np.mean(cv_scores)) * 100, 3),
        }

        self.results[classifier_name] = result
        return result

    def run_all(self) -> Dict[str, Dict]:
        """Train and evaluate all classifiers."""
        for name in self.CLASSIFIERS:
            print(f"\nTraining {name}...")
            self.train_and_evaluate(name)
        return self.results

    def save_results(self, output_path: Path) -> None:
        """Save all results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")

    def print_summary(self, classifier_name: Optional[str] = None) -> None:
        """Print a human-readable summary of results."""
        names = [classifier_name] if classifier_name else list(self.results.keys())
        for name in names:
            if name not in self.results:
                continue
            r = self.results[name]
            print(f"\n{'='*50}")
            print(f"Classifier: {name}")
            print(f"{'='*50}")
            print(f"Accuracy:      {r['accuracy_pct']:.3f}%")
            print(f"Cohen's Kappa: {r['kappa_pct']:.3f}%")
            print(f"CV Accuracy:   {r['cv_accuracy_mean_pct']:.3f}% (+/- {r['cv_accuracy_std']*100:.3f}%)")
            print(f"Macro Precision: {r['precision']['macro']*100:.3f}%")
            print(f"Macro Recall:    {r['recall']['macro']*100:.3f}%")
            print(f"Macro F1:        {r['f1']['macro']*100:.3f}%")
            print(f"\nConfusion Matrix:")
            print(np.array(r["confusion_matrix"]))
