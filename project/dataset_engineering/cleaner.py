"""Shared cleaning helpers."""

from __future__ import annotations

import html
import logging
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class BaseCleaner:
    """Common dataset-cleaning behavior."""

    frame: pd.DataFrame
    removed_columns: list[str] = field(default_factory=list)
    renamed_columns: dict[str, str] = field(default_factory=dict)
    derived_columns: list[str] = field(default_factory=list)

    def drop_columns(self, columns: list[str]) -> None:
        present = [column for column in columns if column in self.frame.columns]
        if not present:
            return
        self.frame = self.frame.drop(columns=present)
        self.removed_columns.extend(present)
        LOGGER.info("Dropped %s columns", len(present))

    def drop_high_missing(self, threshold: float) -> list[str]:
        missing_ratio = self.frame.isna().mean()
        columns = missing_ratio[missing_ratio > threshold].index.tolist()
        self.drop_columns(columns)
        return columns

    def rename_columns_snake_case(self) -> None:
        mapping = {column: self._to_snake_case(column) for column in self.frame.columns}
        self.frame = self.frame.rename(columns=mapping)
        self.renamed_columns.update(mapping)

    @staticmethod
    def parse_dates(series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, errors="coerce", format="%b-%y")
        if parsed.notna().sum() > 0:
            return parsed
        return pd.to_datetime(series, errors="coerce")

    @staticmethod
    def clean_text(text: object) -> str:
        raw = "" if text is None else str(text)
        cleaned = html.unescape(raw)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = cleaned.lower()
        cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_snake_case(value: str) -> str:
        normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized.lower()
