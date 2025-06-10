# src/fraud_unsup/pipelines/infer.py
"""
Batch-scoring pipeline for new data.

Usage
-----
python -m fraud_unsup.pipelines.infer <parquet_path> --out_csv scored.csv
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from ..config import CFG
from ..features.temporal import add_time_features
from ..models.isolation_forest import IFDetector
from ..models.gmm_detector     import GMMDetector
from ..models.autoencoder      import AEDetector

# ─── Artifact paths ────────────────────────────────────────────────────────
MODEL_DIR  = Path(CFG.model_dir)
ENC_RARE   = MODEL_DIR / "rare_encoder.joblib"
ENC_FREQ   = MODEL_DIR / "freq_encoder.joblib"
SCL_IF_GMM = MODEL_DIR / "scaler_if_gmm.joblib"
SCL_AE     = MODEL_DIR / "scaler_ae.joblib"
W_JSON     = MODEL_DIR / "ensemble_weights.json"

# ─── Helpers ───────────────────────────────────────────────────────────────
def load_weights() -> tuple[float, float, float]:
    if W_JSON.exists():
        with W_JSON.open() as fp:
            js = json.load(fp)
        return js["iso"], js["ae"], js["gmm"]
    # fallback
    return 0.4, 0.3, 0.3


def prepare(df: pd.DataFrame):
    """Apply temporal, categorical encoders, and numeric scalers."""
    df = add_time_features(df)

    # load fitted objects
    rare   = joblib.load(ENC_RARE)
    freq   = joblib.load(ENC_FREQ)
    scl_if = joblib.load(SCL_IF_GMM)
    scl_ae = joblib.load(SCL_AE)

    # categorical → rare / freq
    df = rare.transform(df)          # <- SINGLE arg
    df = freq.transform(df)          # <- SINGLE arg

    num_cols = df.select_dtypes("number").columns.tolist()
    num_df   = df[num_cols]

    # numeric matrices
    X_if_gmm = scl_if.transform(num_df)
    X_ae     = scl_ae.transform(num_df)

    return X_if_gmm, X_ae, df


# ─── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=Path, help="Input parquet (raw or cleaned)")
    ap.add_argument("--out_csv", default="scored.csv")
    args = ap.parse_args()

    # 1) load data
    df_in = pd.read_parquet(args.parquet)

    # 2) build feature matrices
    X_if, X_ae, df = prepare(df_in)

    # 3) load detectors
    iso = IFDetector.load(MODEL_DIR / "iso_forest.joblib")
    gmm = GMMDetector.load(MODEL_DIR / "gmm.joblib")
    ae  = AEDetector.load(MODEL_DIR / "autoencoder.pth")

    # 4) score
    df["iso_score"] = iso.score(X_if)
    df["gmm_score"] = gmm.score(X_if)
    df["ae_score"]  = ae.score(X_ae)

    # 5) ensemble
    w_iso, w_ae, w_gmm = load_weights()
    df["ensemble_score"] = (
        w_iso * df["iso_score"]
      + w_ae  * df["ae_score"]
      + w_gmm * df["gmm_score"]
    )

    # 6) write
    df.to_csv(args.out_csv, index=False)
    print(f"✓ Saved scored file → {args.out_csv}")


if __name__ == "__main__":
    main()
