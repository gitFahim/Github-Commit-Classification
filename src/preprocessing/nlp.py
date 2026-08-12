"""NLP preprocessing pipeline for commit messages.

Provides text cleaning, tokenization, stopword removal, and lemmatization
using NLTK.
"""

import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)


class TextPreprocessor:
    """Preprocesses commit messages for feature extraction.

    Pipeline:
        1. Lowercase
        2. Punctuation removal
        3. Tokenization (word-level)
        4. Stopword removal
        5. Lemmatization
    """

    def __init__(self) -> None:
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def remove_punctuation(self, text: str) -> str:
        """Remove all punctuation characters from text."""
        return "".join([c for c in text if c not in string.punctuation])

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase word tokens."""
        return self.tokenizer.tokenize(text.lower())

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out English stopwords."""
        return [t for t in tokens if t not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their base form."""
        return [self.lemmatizer.lemmatize(t, pos="v") for t in tokens]

    def preprocess(self, text: str) -> List[str]:
        """Run full preprocessing pipeline on a single commit message.

        Args:
            text: Raw commit message string.

        Returns:
            List of cleaned, lemmatized tokens.
        """
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """Run preprocessing on a batch of commit messages."""
        return [self.preprocess(t) for t in texts]
