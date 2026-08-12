"""Tests for NLP preprocessing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.nlp import TextPreprocessor


def test_preprocessor_pipeline():
    preprocessor = TextPreprocessor()
    tokens = preprocessor.preprocess("Fixed a critical BUG in the login system!!!")
    assert isinstance(tokens, list)
    assert "fix" in tokens
    assert "bug" in tokens
    assert "critical" in tokens
    # Stopwords should be removed
    assert "a" not in tokens
    assert "the" not in tokens
    # Punctuation should be removed
    assert "!!!" not in tokens


def test_preprocessor_lemmatization():
    preprocessor = TextPreprocessor()
    tokens = preprocessor.preprocess("running runs runner")
    # Lemmatizer should reduce to base forms
    assert "run" in tokens or "running" in tokens


def test_preprocessor_batch():
    preprocessor = TextPreprocessor()
    texts = ["Fix bug", "Add feature", "Refactor code"]
    batch = preprocessor.preprocess_batch(texts)
    assert len(batch) == 3
    assert all(isinstance(t, list) for t in batch)
