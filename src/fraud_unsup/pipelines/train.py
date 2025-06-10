"""
End-to-end training pipeline.

  • Ensures parquet exists → joins/cleans if needed
  • Builds feature matrix (temporal + categorical encoders + scalers)
  • Fits Isolation-Forest, GMM, Auto-Encoder
  • Logs metrics + best ensemble weights to MLflow
  • Saves models + encoders/scalers under artifacts/models/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd

from ..config import CFG, save as save_cfg
from ..etl.clean_cast import clean
from ..features.temporal import add_time_features
from ..features.categorical import RareLabelGrouper, FrequencyEncoder
from ..features.scale_reduce import ScalerReducer
from ..models.isolation_forest import IFDetector
from ..models.gmm_detector import GMMDetector
from ..models.autoencoder import AEDetector
from ..models.evaluate import (
    precision_at_k,
    pr_auc,
    best_weights_for_precision,
    save_weights_json,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(CFG.model_dir)  # artifacts/models/
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────────────────────────────────────
def prepare_features(split: str):
    """Return matrices + fitted preprocessors for saving."""
    df = pd.read_parquet(clean(split))
    y = df[CFG.train.target_col].values

    # temporal -----------------------------------------------------------
    df = add_time_features(df)

    cat_cols = df.select_dtypes("object").columns.tolist()
    num_cols = df.select_dtypes("number").columns.tolist()

    # categorical encoders ----------------------------------------------
    rare = RareLabelGrouper().fit(df, cat_cols)
    df = rare.transform(df)                 # ← fixed: only df argument

    freq = FrequencyEncoder().fit(df, cat_cols)
    df = freq.transform(df)                 # ← fixed: only df argument

    num_df = df[num_cols + [c for c in df.columns if c.endswith("_freq")]]

    # numeric scalers / PCA ---------------------------------------------
    latent_dim = CFG.model.autoencoder["latent_dim"]
    scaler_if_gmm = ScalerReducer(n_components=latent_dim * 2).fit(num_df)
    X_if_gmm = scaler_if_gmm.transform(num_df)

    scaler_ae = ScalerReducer(minmax=True).fit(num_df)
    X_ae = scaler_ae.transform(num_df)

    return (
        X_if_gmm,
        X_ae,
        y,
        rare,
        freq,
        scaler_if_gmm,
        scaler_ae,
    )


# ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train"])
    ap.add_argument("--grid_step", type=float, default=0.05)
    args = ap.parse_args()

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("fraud-unsup")

    with mlflow.start_run():
        save_cfg()  # snapshot config → artifacts/config.yaml

        (
            X_if,
            X_ae,
            y,
            rare_enc,
            freq_enc,
            scl_if,
            scl_ae,
        ) = prepare_features(args.split)

        # save preprocessors --------------------------------------------
        joblib.dump(rare_enc, MODEL_DIR / "rare_encoder.joblib")
        joblib.dump(freq_enc, MODEL_DIR / "freq_encoder.joblib")
        joblib.dump(scl_if, MODEL_DIR / "scaler_if_gmm.joblib")
        joblib.dump(scl_ae, MODEL_DIR / "scaler_ae.joblib")

        # 1️⃣ Isolation-Forest -----------------------------------------
        iso = IFDetector().fit(X_if)
        iso_scores = iso.score(X_if)
        iso.save()

        # 2️⃣ GMM -------------------------------------------------------
        gmm = GMMDetector().fit(X_if)
        gmm_scores = gmm.score(X_if)
        gmm.save()

        # 3️⃣ Auto-Encoder ---------------------------------------------
        ae = AEDetector().fit(X_ae)
        ae_scores = ae.score(X_ae)
        ae.save()

        # log individual metrics ---------------------------------------
        mlflow.log_metrics(
            {
                "iso_pr_auc": pr_auc(y, iso_scores),
                "iso_prec_at_1pct": precision_at_k(y, iso_scores),
                "gmm_pr_auc": pr_auc(y, gmm_scores),
                "gmm_prec_at_1pct": precision_at_k(y, gmm_scores),
                "ae_pr_auc": pr_auc(y, ae_scores),
                "ae_prec_at_1pct": precision_at_k(y, ae_scores),
            }
        )

        # ensemble grid-search -----------------------------------------
        score_mat = np.vstack([iso_scores, ae_scores, gmm_scores]).T
        best_w, best_prec = best_weights_for_precision(
            y, score_mat, step=args.grid_step, k_frac=0.01
        )
        ens_scores = (score_mat * np.array(best_w)).sum(axis=1)

        mlflow.log_metric("ensemble_prec_at_1pct", best_prec)
        mlflow.log_metric("ensemble_pr_auc", pr_auc(y, ens_scores))
        mlflow.log_params(
            {"w_iso": best_w[0], "w_ae": best_w[1], "w_gmm": best_w[2]}
        )
        save_weights_json(best_w, MODEL_DIR / "ensemble_weights.json")

        logger.info(
            f"Training complete – best weights iso/ae/gmm = {best_w} "
            f"(precision@1%={best_prec:.4f})"
        )


if __name__ == "__main__":
    main()
