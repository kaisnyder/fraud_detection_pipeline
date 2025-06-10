"""
Temporal feature generation from `TransactionDT`.

The original column is “seconds since X” (per the IEEE competition).  
We treat *min value* as epoch = 2017-11-30 and derive calendar features.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

_EPOCH = pd.Timestamp("2017-11-30")  # competition forum hint

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = _EPOCH + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["dow"]   = dt.dt.dayofweek.astype("int8")
    df["hour"]  = dt.dt.hour.astype("int8")
    df["month"] = dt.dt.month.astype("int8")

    # cyclic encoding ----------------------------------------------- #
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    return df
