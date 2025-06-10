"""
Centralised configuration & hyper-parameters.
Only edit this file (or override via env-vars / CLI) to tune models.
"""
from __future__ import annotations
import yaml, json, os
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any

# ─── repo paths ────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data"
ARTIFACT_DIR= ROOT / "artifacts"
MODEL_DIR   = ARTIFACT_DIR / "models"

# ─── sub-configs ───────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    raw:       Path = DATA_DIR / "raw"
    interim:   Path = DATA_DIR / "interim"
    processed: Path = DATA_DIR / "processed"

@dataclass
class TrainConfig:
    random_seed: int   = 42
    test_size:   float = 0.2
    target_col:  str   = "isFraud"
    id_col:      str   = "TransactionID"
    numerical_impute_strategy: str       = "median"
    categorical_impute_value:  str       = "missing"
    max_rows:                 int | None = None   # dev-subset

@dataclass
class ModelConfig:
    # ① Isolation Forest tweaked
    isolation_forest: dict[str, Any] = field(
        default_factory=lambda: dict(
            n_estimators=600,
            contamination=0.03,
            max_features=0.6,
            random_state=42,
            n_jobs=-1,
        )
    )
    # ② GMM tweaked
    gmm: dict[str, Any] = field(
        default_factory=lambda: dict(
            n_components=24,
            covariance_type="diag",
            random_state=42,
        )
    )
    # ③ Auto-Encoder deeper, longer training
    autoencoder: dict[str, Any] = field(
        default_factory=lambda: dict(
            hidden_dims=[256, 128, 64],
            latent_dim=32,
            lr=1e-3,
            epochs=100,        # early-stop in code
            batch_size=4096,
        )
    )

@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    model_dir: Path    = MODEL_DIR  # expose for convenience

CFG = Config()

# ─── serialization helper ─────────────────────────────────────────────────
def _to_serialisable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        typ = type(obj)
        return typ(_to_serialisable(v) for v in obj)
    return obj

def save(cfg: Config = CFG, path: Path|str = ARTIFACT_DIR / "config.yaml") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(_to_serialisable(cfg), f)
