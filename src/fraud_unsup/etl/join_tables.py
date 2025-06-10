"""
Merge *identity* and *transaction* CSVs → single DataFrame
and save to Parquet for faster downstream IO.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..config import CFG
from ..utils.logging import get_logger
from ..utils.timer import timeit

logger = get_logger(__name__)

def load_csv(name: str, nrows: int | None = None) -> pd.DataFrame:
    path = CFG.data.raw / f"{name}.csv"
    logger.info(f"Loading {path} ...")
    return pd.read_csv(path, nrows=nrows)

def join_and_save(split: str = "train", nrows: int | None = None) -> Path:
    """
    Parameters
    ----------
    split  : 'train' or 'test'
    nrows  : optionally limit rows for faster experiments
    """
    with timeit(f"Join {split} sets"):
        trans = load_csv(f"{split}_transaction", nrows)
        identity = load_csv(f"{split}_identity", nrows)
        logger.info(f"Shapes: transaction={trans.shape}, identity={identity.shape}")

        df = trans.merge(identity, how="left", on="TransactionID")
        logger.info(f"Joined shape: {df.shape}")

        out_path = CFG.data.interim / f"{split}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved joined data → {out_path}")
        return out_path

if __name__ == "__main__":
    join_and_save("train", nrows=CFG.train.max_rows)
    join_and_save("test", nrows=CFG.train.max_rows)
