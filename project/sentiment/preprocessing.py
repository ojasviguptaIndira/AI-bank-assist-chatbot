"""Text preprocessing for sentiment modeling."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize


FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "there", "they", "this", "to", "was",
    "we", "were", "with", "you", "your",
}


@dataclass
class TextPreprocessor:
    """Normalize review text consistently across training and inference."""

    stop_words: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.stop_words:
            self.stop_words = self._load_stop_words()
        self._punctuation_table = str.maketrans("", "", string.punctuation)

    def preprocess(self, text: object) -> str:
        """Lowercase, tokenize, remove stopwords, and normalize spaces."""
        value = "" if text is None else str(text)
        value = value.lower().translate(self._punctuation_table)
        value = re.sub(r"\s+", " ", value).strip()
        tokens = [
            token for token in wordpunct_tokenize(value)
            if token.isalpha() and len(token) > 2 and token not in self.stop_words
        ]
        return " ".join(tokens)

    def batch_preprocess(self, texts: list[str]) -> list[str]:
        """Preprocess a collection of texts."""
        return [self.preprocess(text) for text in texts]

    @staticmethod
    def _load_stop_words() -> set[str]:
        try:
            nltk.data.find("corpora/stopwords")
            return set(stopwords.words("english"))
        except LookupError:
            return FALLBACK_STOPWORDS
