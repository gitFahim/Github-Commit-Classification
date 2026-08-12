"""Tests for model architectures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import torch

from models.pytorch_models import CommitClassifier, CommitDatasetTorch
from models.sklearn_models import SklearnTrainer


def test_commit_classifier_forward():
    model = CommitClassifier(input_dim=65, num_classes=3)
    x = torch.randn(4, 65)
    out = model(x)
    assert out.shape == (4, 3)


def test_commit_classifier_weights_initialized():
    model = CommitClassifier(input_dim=10, hidden_dims=[8], num_classes=3)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            assert m.weight is not None
            assert m.bias is not None


def test_torch_dataset():
    X = np.random.rand(10, 65).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = CommitDatasetTorch(X, y)
    assert len(dataset) == 10
    x_item, y_item = dataset[0]
    assert isinstance(x_item, torch.Tensor)
    assert x_item.shape[0] == 65
    assert y_item.item() in [0, 1, 2]


def test_sklearn_trainer_split():
    X = np.random.rand(100, 65)
    y = np.array([0] * 34 + [1] * 33 + [2] * 33)
    trainer = SklearnTrainer(X, y, test_size=0.2, random_state=42)
    assert trainer.X_train is not None
    assert trainer.X_test is not None
    assert len(trainer.X_train) == 80
    assert len(trainer.X_test) == 20
