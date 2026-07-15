"""Text cleaning and n-gram analysis utilities."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "so", "that", "the", "their",
    "there", "they", "this", "to", "was", "we", "were", "with", "you", "your",
}


class TextProcessor:
    """Normalize free text and extract token statistics."""

    def __init__(self) -> None:
        self.stop_words = self._load_stopwords()
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    def clean_text(self, text: object) -> str:
        """Lowercase, remove punctuation, remove stopwords, and short words."""
        normalized = "" if text is None else str(text).lower()
        normalized = normalized.translate(self.punctuation_table)
        normalized = re.sub(r"\s+", " ", normalized)
        tokens = [
            token for token in normalized.split()
            if len(token) >= 3 and token not in self.stop_words
        ]
        return " ".join(tokens)

    def word_frequency(self, texts: Iterable[str], top_n: int) -> list[tuple[str, int]]:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text.split())
        return counter.most_common(top_n)

    def ngrams(
        self,
        texts: Iterable[str],
        ngram_range: tuple[int, int],
        top_n: int,
    ) -> list[tuple[str, int]]:
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(list(texts))
        counts = matrix.sum(axis=0).A1
        names = vectorizer.get_feature_names_out()
        pairs = sorted(zip(names, counts), key=lambda item: item[1], reverse=True)
        return [(name, int(count)) for name, count in pairs[:top_n]]

    def _load_stopwords(self) -> set[str]:
        try:
            nltk.data.find("corpora/stopwords")
            return set(stopwords.words("english"))
        except LookupError:
            return FALLBACK_STOPWORDS
