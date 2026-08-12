"""Keyword-based feature extraction from commit messages.

Implements the keyword strategy from Levin & Yehudai (2017):
19 stemmed keywords extracted from commit messages.
"""

from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Ensure NLTK data is available
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)


class KeywordExtractor:
    """Extracts stemmed keyword binary features from commit messages.

    The 19 keywords from the original paper (after stemming):
        add, allow, bug, chang, error, fail, fix, implement, improv,
        issu, method, new, npe, refactor, remov, report, set, support, test, use
    """

    KEYWORDS = [
        "add", "allow", "bug", "chang", "error", "fail", "fix",
        "implement", "improv", "issu", "method", "new", "npe",
        "refactor", "remov", "report", "set", "support", "test", "use",
    ]

    def __init__(self) -> None:
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text: str) -> List[str]:
        """Tokenize, lowercase, remove punctuation, remove stopwords, lemmatize."""
        # Remove punctuation and tokenize
        tokens = self.tokenizer.tokenize(text.lower())
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        # Lemmatize then stem to match the paper's keyword vocabulary.
        tokens = [self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in tokens]
        return tokens

    def extract(self, text: str) -> List[int]:
        """Return binary vector indicating presence of each keyword.

        Args:
            text: Raw commit message.

        Returns:
            List of 0/1 integers, one per keyword.
        """
        tokens = self.preprocess(text)
        tokens_set = set(tokens)
        return [1 if kw in tokens_set else 0 for kw in self.KEYWORDS]

    def extract_batch(self, texts: List[str]) -> List[List[int]]:
        """Extract keyword features for a batch of commit messages."""
        return [self.extract(t) for t in texts]
