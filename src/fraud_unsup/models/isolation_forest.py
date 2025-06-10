"""Isolation-Forest anomaly detector wrapper."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import CFG, MODEL_DIR
from ..utils.logging import get_logger

logger = get_logger(__name__)


class IFDetector:
    """Thin wrapper around `sklearn.ensemble.IsolationForest`."""

    def __init__(self, **override):
        params = {**CFG.model.isolation_forest, **override}
        self.model = IsolationForest(**params)

    # ────────────────────────────────
    # Fit / score
    # ────────────────────────────────
    def fit(self, X: pd.DataFrame):
        logger.info("Fitting IsolationForest …")
        self.model.fit(X)
        return self

    def score(self, X: pd.DataFrame):
        """Return *negative* decision_function so that **higher = more anomalous**."""
        return -self.model.decision_function(X)

    # ────────────────────────────────
    # Persistence helpers
    # ────────────────────────────────
    def save(self, path: Path | None = None) -> Path:
        path = Path(path or MODEL_DIR / "iso_forest.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"IsolationForest saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | str):
        obj = cls.__new__(cls)  # bypass __init__
        obj.model = joblib.load(Path(path))
        return obj
