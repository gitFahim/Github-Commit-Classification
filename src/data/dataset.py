"""Dataset loader and Commit object factory."""

import csv
from pathlib import Path
from typing import List, Union

import pandas as pd

from .commit import Commit


class CommitDataset:
    """Loads commit data from CSV and produces Commit objects.

    Expects a comma-delimited file with columns:
        commitId, project, comment, label, <feature_columns...>

    Commit messages in the source dataset are not consistently quoted.  The
    parser therefore rebuilds the comment from every field between ``project``
    and ``label`` while keeping the fixed trailing label and feature columns.
    """

    def __init__(self, csv_path: Union[str, Path], delimiter: str = ",") -> None:
        self.csv_path = Path(csv_path)
        self.delimiter = delimiter
        self.df = self._read_csv()
        self.commits: List[Commit] = []
        self._build_commits()

    def _read_csv(self) -> pd.DataFrame:
        """Load rows while preserving unquoted delimiters in commit messages."""
        with self.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file, delimiter=self.delimiter)
            try:
                columns = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Dataset is empty: {self.csv_path}") from exc

            if len(columns) < 5:
                raise ValueError("Dataset must contain commit metadata and at least one feature column")

            trailing_fields = len(columns) - 3  # label plus every feature column
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if len(row) < len(columns):
                    raise ValueError(
                        f"Row {line_number} has {len(row)} fields; expected at least {len(columns)}"
                    )

                comment = self.delimiter.join(row[2:-trailing_fields])
                rows.append(row[:2] + [comment] + row[-trailing_fields:])

        return pd.DataFrame(rows, columns=columns)

    def _build_commits(self) -> None:
        """Convert DataFrame rows into Commit objects."""
        for _, row in self.df.iterrows():
            features = row.iloc[4:].tolist()
            commit = Commit(
                commit_id=row["commitId"],
                project=row["project"],
                comment=row["comment"],
                label=row["label"],
                features=features,
            )
            self.commits.append(commit)

    def __len__(self) -> int:
        return len(self.commits)

    def __getitem__(self, idx: int) -> Commit:
        return self.commits[idx]

    def get_feature_names(self) -> List[str]:
        """Return list of feature column names."""
        return self.df.columns[4:].tolist()

    def get_keyword_features(self) -> List[str]:
        """Return keyword feature column names (first 19 after label)."""
        # Based on original paper: 19 keyword features
        all_features = self.get_feature_names()
        # Heuristic: keyword features are lowercase, code changes are UPPER_SNAKE_CASE
        keywords = [f for f in all_features if f.islower()]
        return keywords

    def get_code_change_features(self) -> List[str]:
        """Return code change feature column names (UPPER_SNAKE_CASE)."""
        all_features = self.get_feature_names()
        code_changes = [f for f in all_features if not f.islower()]
        return code_changes

    def to_sklearn_arrays(self, feature_strategy: str = "combined"):
        """Return X, y arrays suitable for scikit-learn.

        Args:
            feature_strategy: One of "keywords", "changes", "combined".

        Returns:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        import numpy as np

        if feature_strategy == "keywords":
            feature_names = self.get_keyword_features()
        elif feature_strategy == "changes":
            feature_names = self.get_code_change_features()
        elif feature_strategy == "combined":
            feature_names = self.get_feature_names()
        else:
            raise ValueError(f"Unknown strategy: {feature_strategy}")

        X = self.df[feature_names].values.astype(float)
        y = self.df["label"].values

        # Encode string labels to integers if needed
        if not pd.api.types.is_numeric_dtype(self.df["label"]):
            label_map = Commit.LABEL_MAP
            unknown_labels = sorted({str(label).strip() for label in y} - set(label_map))
            if unknown_labels:
                raise ValueError(f"Unknown commit labels: {', '.join(unknown_labels)}")
            y = np.array([label_map[str(label).strip()] for label in y])

        return X, y
