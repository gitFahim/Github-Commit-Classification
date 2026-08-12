"""Tests for feature extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.keywords import KeywordExtractor
from features.code_changes import CodeChangeExtractor


def test_keyword_extractor():
    extractor = KeywordExtractor()
    features = extractor.extract("Fixed a critical bug in the login system")
    assert len(features) == len(KeywordExtractor.KEYWORDS)
    assert features[KeywordExtractor.KEYWORDS.index("fix")] == 1
    assert features[KeywordExtractor.KEYWORDS.index("bug")] == 1


def test_keyword_extractor_no_match():
    extractor = KeywordExtractor()
    features = extractor.extract("Updated documentation")
    assert len(features) == len(KeywordExtractor.KEYWORDS)
    # "doc" is not in the keyword list, so all should be 0
    assert sum(features) == 0


def test_code_change_types():
    types = CodeChangeExtractor.get_change_types()
    assert len(types) == 48
    assert "STATEMENT_DELETE" in types
    assert "ADDITIONAL_CLASS" in types


def test_code_change_count():
    assert CodeChangeExtractor.count() == 48
