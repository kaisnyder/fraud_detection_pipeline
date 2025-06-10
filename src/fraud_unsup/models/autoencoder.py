"""Improved Auto-Encoder detector (SELU + BatchNorm + Dropout + early stop)."""

from __future__ import annotations
from pathlib import Path
from typing import Any, List

import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ..config import CFG, MODEL_DIR
from ..utils.logging import get_logger

logger = get_logger(__name__)

# ─── helpers ───────────────────────────────────────────────────────────────
def _to_tensor(x) -> torch.Tensor:
    """DataFrame / ndarray / Tensor → float32 tensor."""
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, pd.DataFrame):
        x = x.values
    return torch.tensor(x, dtype=torch.float32)

# ─── network ───────────────────────────────────────────────────────────────
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], lat: int):
        super().__init__()
        enc, prev = [], input_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SELU(), nn.Dropout(0.2)]
            prev = h
        enc += [nn.Linear(prev, lat)]
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], lat
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SELU(), nn.Dropout(0.2)]
            prev = h
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):               # type: ignore
        return self.decoder(self.encoder(x))

# ─── detector wrapper ──────────────────────────────────────────────────────
class AEDetector:
    def __init__(self, **override):
        self.params = {**CFG.model.autoencoder, **override}
        self.model: AutoEncoder|None = None

    # build once we know input_dim
    def _build(self, d): self.model = AutoEncoder(d, self.params["hidden_dims"], self.params["latent_dim"])

    def fit(self, X):
        data = _to_tensor(X)
        self._build(data.shape[1]); m = self.model
        crit, opt = nn.MSELoss(), torch.optim.Adam(m.parameters(), lr=self.params["lr"])
        # simple 90/10 split for early stop
        n_val = int(0.1 * len(data))
        train_ds, val_ds = random_split(data, [len(data)-n_val, n_val], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=self.params["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.params["batch_size"])

        best_loss, patience, patience_left = float("inf"), 3, 3
        for epoch in range(1, self.params["epochs"]+1):
            m.train(); tloss = 0.
            for batch in train_loader:
                opt.zero_grad(); recon = m(batch); loss = crit(recon, batch)
                loss.backward(); opt.step(); tloss += loss.item()*len(batch)
            tloss /= len(train_ds)

            # val
            m.eval(); vloss = 0.
            with torch.no_grad():
                for batch in val_loader:
                    vloss += crit(m(batch), batch).item()*len(batch)
            vloss /= len(val_ds)
            logger.info(f"AE epoch {epoch:03d} train={tloss:.5f}  val={vloss:.5f}")

            if vloss < best_loss - 1e-4:
                best_loss, patience_left = vloss, patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    logger.info("Early stop.")
                    break
        return self

    def score(self, X):
        data = _to_tensor(X); self.model.eval()
        with torch.no_grad():
            err = ((data - self.model(data))**2).mean(dim=1)
        return err.cpu().numpy()

    # ─ persistence ─
    def save(self, path:Path|None=None):
        path = Path(path or MODEL_DIR / "autoencoder.pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": self.params,
                    "input_dim": self.model.encoder[0].in_features,
                    "state": self.model.state_dict()}, path)
        logger.info(f"AE saved → {path}"); return path

    @classmethod
    def load(cls, path):
        ckpt = torch.load(Path(path), map_location="cpu")
        obj  = cls(**ckpt["params"]); obj._build(ckpt["input_dim"])
        obj.model.load_state_dict(ckpt["state"]); obj.model.eval(); return obj
