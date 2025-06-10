"""
Clean & cast the joined IEEE-CIS tables.

Steps
-----
1.  Ensure `data/interim/<split>.parquet` exists.
    – If not, call `join_and_save(split)` to generate it.
2.  Down-cast float64 → float32, int64 → int32 where safe
3.  Impute missing values
    – numeric  → median   (configurable)
    – categorical → "missing"  (configurable)
4.  Write cleaned DataFrame to `data/processed/<split>.parquet`
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from ..config import CFG
from ..utils.logging import get_logger
from ..utils.timer import timeit
from .join_tables import join_and_save

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Down-cast 64-bit floats / ints to 32-bit where it fits."""
    for col, dtype in df.dtypes.items():
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64":
            if df[col].min() >= -2**31 and df[col].max() < 2**31:
                df[col] = df[col].astype("int32")
    return df


def _impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Median for numeric, fixed token for categoricals."""
    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes("object").columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(CFG.train.categorical_impute_value)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────────
def clean(split: str = "train") -> Path:
    """
    Parameters
    ----------
    split : "train" or "test"
    Returns
    -------
    Path to the cleaned parquet file.
    """
    raw_path = CFG.data.interim / f"{split}.parquet"
    if not raw_path.exists():
        logger.info(f"{raw_path} not found → running join_and_save('{split}')")
        join_and_save(split)

    out_path = CFG.data.processed / f"{split}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with timeit(f"Clean {split}"):
        df = pd.read_parquet(raw_path)
        df = _cast_dtypes(df)
        df = _impute_nulls(df)
        df.to_parquet(out_path, index=False)
        logger.info(f"Wrote cleaned file → {out_path}")

    return out_path


if __name__ == "__main__":
    clean("train")
    clean("test")
