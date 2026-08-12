"""Commit data model for representing a single commit instance."""

from typing import List, Union
import torch


class Commit:
    """Represents a single software commit with features and label.

    Attributes:
        commit_id: Unique commit hash.
        project: Source project name.
        comment: Commit message text.
        label_str: Raw label string (e.g., "Corrective", "Perfective", "Adaptive").
        label: Integer-encoded label (0, 1, 2).
        features: List of binary float features (keywords + code changes).
    """

    LABEL_MAP = {
        "Corrective": 0,
        "Perfective": 1,
        "Adaptive": 2,
        "c": 0,
        "p": 1,
        "a": 2,
    }
    REVERSE_LABEL_MAP = {
        0: "Corrective",
        1: "Perfective",
        2: "Adaptive",
    }

    def __init__(
        self,
        commit_id: str,
        project: str,
        comment: str,
        label: Union[str, int],
        features: List[Union[int, float]],
    ) -> None:
        self.commit_id = str(commit_id)
        self.project = str(project)
        self.comment = str(comment)

        if isinstance(label, str):
            self.label_str = label.strip()
            self.label = self.LABEL_MAP.get(self.label_str, 0)
        else:
            self.label = int(label)
            self.label_str = self.REVERSE_LABEL_MAP.get(self.label, "Unknown")

        self.features = [float(x) for x in features]

    def get_all_features_list(self) -> List[float]:
        """Return all features as a list of floats."""
        return self.features

    def get_all_features_tensor(self) -> torch.Tensor:
        """Return all features as a PyTorch float tensor."""
        return torch.tensor(self.features, dtype=torch.float32)

    def get_label(self) -> int:
        """Return integer class label."""
        return self.label

    def get_label_str(self) -> str:
        """Return string class label."""
        return self.label_str

    def get_labels_list(self) -> List[float]:
        """Return one-hot encoded label list for 3 classes."""
        one_hot = [0.0, 0.0, 0.0]
        one_hot[self.label] = 1.0
        return one_hot

    def get_labels_tensor(self) -> torch.Tensor:
        """Return one-hot encoded label as PyTorch tensor."""
        return torch.tensor(self.get_labels_list(), dtype=torch.float32)

    def __repr__(self) -> str:
        return (
            f"Commit(id={self.commit_id[:8]!r}, "
            f"project={self.project!r}, "
            f"label={self.label_str!r}, "
            f"features={len(self.features)})"
        )

    @classmethod
    def prepare_text_vectorizer(cls) -> None:
        """Placeholder for text vectorizer initialization.

        In the original paper, this would initialize a TF-IDF or CountVectorizer
        for commit message text. The current implementation uses pre-computed
        keyword binary features extracted from commit messages.
        """
        pass
