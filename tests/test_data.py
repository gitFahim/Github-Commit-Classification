"""Tests for data layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from data.commit import Commit
from data.dataset import CommitDataset


def test_commit_creation():
    commit = Commit(
        commit_id="abc123",
        project="test-project",
        comment="fix bug in login",
        label="Corrective",
        features=[1.0] * 65,
    )
    assert commit.commit_id == "abc123"
    assert commit.label == 0
    assert commit.label_str == "Corrective"
    assert len(commit.features) == 65
    assert len(commit.get_all_features_list()) == 65
    assert commit.get_label() == 0


def test_commit_integer_label():
    commit = Commit(
        commit_id="def456",
        project="test-project",
        comment="add new feature",
        label=2,
        features=[0.0] * 65,
    )
    assert commit.label == 2
    assert commit.label_str == "Adaptive"


def test_commit_one_hot():
    commit = Commit(
        commit_id="ghi789",
        project="test-project",
        comment="refactor code",
        label="Perfective",
        features=[0.5] * 65,
    )
    one_hot = commit.get_labels_list()
    assert one_hot == [0.0, 1.0, 0.0]


def test_commit_tensor():
    import torch

    commit = Commit(
        commit_id="jkl012",
        project="test-project",
        comment="test update",
        label="Corrective",
        features=[1.0, 0.0] * 33,
    )
    tensor = commit.get_all_features_tensor()
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape[0] == 66  # 33 * 2


def test_dataset_loads_comma_delimited_rows_with_commas_in_comments(tmp_path):
    csv_path = tmp_path / "commits.csv"
    csv_path.write_text(
        "commitId,project,comment,label,feature_a,feature_b\n"
        "abc123,demo,Fix parser, preserve commas,c,1,0\n",
        encoding="utf-8",
    )

    dataset = CommitDataset(csv_path)

    assert len(dataset) == 1
    assert dataset[0].comment == "Fix parser, preserve commas"
    assert dataset[0].label == 0
    X, y = dataset.to_sklearn_arrays()
    assert X.tolist() == [[1.0, 0.0]]
    assert y.tolist() == [0]
