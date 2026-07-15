"""Matplotlib plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from bank_loan_eda.config import PLOTS_DPI


class PlotFactory:
    """Generate consistent plots without seaborn."""

    @staticmethod
    def histogram(series: pd.Series, title: str, path: Path, bins: int = 30) -> None:
        clean = series.dropna()
        if clean.empty:
            return
        plt.figure(figsize=(10, 6))
        plt.hist(clean, bins=bins, color="#2a6f97", edgecolor="white")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=PLOTS_DPI)
        plt.close()

    @staticmethod
    def bar(series: pd.Series, title: str, path: Path, top_n: int = 15) -> None:
        clean = series.dropna().astype(str).value_counts().head(top_n)
        if clean.empty:
            return
        plt.figure(figsize=(11, 6))
        clean.sort_values().plot(kind="barh", color="#d62828")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=PLOTS_DPI)
        plt.close()

    @staticmethod
    def bar_from_series(series: pd.Series, title: str, path: Path, top_n: int = 15) -> None:
        clean = series.dropna().head(top_n)
        if clean.empty:
            return
        plt.figure(figsize=(11, 6))
        clean.sort_values().plot(kind="barh", color="#f77f00")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=PLOTS_DPI)
        plt.close()

    @staticmethod
    def heatmap(frame: pd.DataFrame, title: str, path: Path) -> None:
        if frame.empty:
            return
        plt.figure(figsize=(12, 9))
        plt.imshow(frame, cmap="coolwarm", aspect="auto")
        plt.title(title)
        plt.xticks(range(len(frame.columns)), frame.columns, rotation=90)
        plt.yticks(range(len(frame.index)), frame.index)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, dpi=PLOTS_DPI)
        plt.close()

    @staticmethod
    def wordcloud(text: str, title: str, path: Path) -> None:
        if not text.strip():
            return
        cloud = WordCloud(width=1200, height=700, background_color="white").generate(text)
        plt.figure(figsize=(12, 7))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=PLOTS_DPI)
        plt.close()
