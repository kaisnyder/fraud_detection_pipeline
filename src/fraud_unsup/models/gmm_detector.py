"""Gaussian-Mixture-Model anomaly detector wrapper."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.mixture import GaussianMixture

from ..config import CFG, MODEL_DIR
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GMMDetector:
    """Wrapper that converts GMM likelihood → anomaly score."""

    def __init__(self, **override):
        params = {**CFG.model.gmm, **override}
        self.model = GaussianMixture(**params)

    # ────────────────────────────────
    def fit(self, X: pd.DataFrame):
        logger.info("Fitting GaussianMixture …")
        self.model.fit(X)
        return self

    def score(self, X: pd.DataFrame):
        """Return *negative* log-likelihood so that **higher = more anomalous**."""
        return -self.model.score_samples(X)

    # ────────────────────────────────
    def save(self, path: Path | None = None) -> Path:
        path = Path(path or MODEL_DIR / "gmm.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"GMM saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | str):
        obj = cls.__new__(cls)
        obj.model = joblib.load(Path(path))
        return obj
