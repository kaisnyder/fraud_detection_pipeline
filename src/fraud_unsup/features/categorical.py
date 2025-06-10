"""Categorical encoders (frequency + target encoding)."""
from __future__ import annotations
import pandas as pd
import numpy as np

class FrequencyEncoder:
    """Replace category with its frequency (count / N)."""

    def __init__(self):
        self.freq_: dict[str, pd.Series] = {}

    def fit(self, df: pd.DataFrame, cat_cols: list[str]) -> "FrequencyEncoder":
        self.freq_ = {c: df[c].value_counts(normalize=True) for c in cat_cols}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in self.freq_.items():
            df[f"{col}_freq"] = df[col].map(mapping).fillna(0.0).astype("float32")
        return df

    def fit_transform(self, df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
        self.fit(df, cat_cols)
        return self.transform(df)


class RareLabelGrouper:
    """
    Replace infrequent levels (<threshold) with '__rare__'
    so downstream encoders (one-hot, freq) don't blow up dimensionality.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.keep_: dict[str, set[str]] = {}

    def fit(self, df: pd.DataFrame, cat_cols: list[str]) -> "RareLabelGrouper":
        self.keep_ = {
            col: set(df[col].value_counts(normalize=True).loc[lambda s: s > self.threshold].index)
            for col in cat_cols
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, keep in self.keep_.items():
            df[col] = np.where(df[col].isin(keep), df[col], "__rare__")
        return df

    def fit_transform(self, df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
        self.fit(df, cat_cols)
        return self.transform(df)
