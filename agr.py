"""AGR — Attractor Geometry Repeller for MLX.

Maintains a bank of attractor centers in the latent space and computes a
repeller loss that pushes hidden states away from frequently visited
regions, preventing mode collapse during LoRA fine-tuning.

Usage with mlx_lm::

    from agr import AttractorRepeller

    agr = AttractorRepeller(dim=2560, num_centers=32, ema=0.99)

    # inside custom loss:
    hidden = model._last_hidden_state.mean(axis=1)   # (batch, dim)
    repeller_loss = agr.mlx_repeller_loss(hidden)
    agr.update(hidden)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx


class AttractorRepeller:
    """Maintain attractor centers and compute an MLX-compatible repeller loss.

    Parameters
    ----------
    dim : int
        Hidden-state dimensionality (e.g. 2560 for Qwen3-4B).
    num_centers : int
        Number of attractor centers to maintain.
    ema : float
        Exponential moving average decay for center updates.
    repeller_weight : float
        Base weight applied to the repeller loss component.
    min_distance : float
        Soft minimum distance — centers closer than this incur quadratic
        penalty.
    """

    def __init__(
        self,
        dim: int = 2560,
        num_centers: int = 32,
        ema: float = 0.99,
        repeller_weight: float = 0.01,
        min_distance: float = 1.0,
    ):
        self.dim = dim
        self.num_centers = num_centers
        self.ema = ema
        self.repeller_weight = repeller_weight
        self.min_distance = min_distance
        self._step = 0

        self.centers: mx.array = mx.zeros((num_centers, dim))
        self.visits: mx.array = mx.zeros(num_centers)

    # ------------------------------------------------------------------
    # Core loss — pure MX, no Python floats in the hot path
    # ------------------------------------------------------------------

    def mlx_repeller_loss(self, hidden: mx.array) -> mx.array:
        """Compute repeller loss for a batch of hidden states.

        Parameters
        ----------
        hidden : mx.array, shape (batch, dim)
            Mean-pooled hidden states from the last transformer layer.

        Returns
        -------
        mx.array
            Scalar repeller loss (differentiable).
        """
        if self._step == 0:
            return mx.array(0.0)

        # hidden: (B, D), centers: (K, D)
        # distances: (B, K)
        diff = hidden[:, None, :] - self.centers[None, :, :]
        distances = mx.sqrt((diff ** 2).sum(axis=-1) + 1e-8)

        # Soft minimum: push hidden away from nearest attractor
        # Use log-sum-exp trick for numerical stability
        neg_dist = -distances
        alpha = 10.0
        log_weights = alpha * neg_dist
        max_lw = log_weights.max(axis=-1, keepdims=True)
        weights = mx.exp(log_weights - max_lw)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)
        nearest_weighted_dist = (weights * distances).sum(axis=-1)

        # Quadratic penalty for being too close to any center
        closeness = mx.maximum(self.min_distance - distances, 0.0)
        proximity_penalty = (closeness ** 2).sum(axis=-1).mean()

        # Repeller: maximize weighted distance, minimize proximity
        loss = -nearest_weighted_dist.mean() + proximity_penalty

        return loss * self.repeller_weight

    # ------------------------------------------------------------------
    # Center update — called after loss computation
    # ------------------------------------------------------------------

    def update(self, hidden: mx.array) -> None:
        """Update attractor centers with exponential moving average.

        Parameters
        ----------
        hidden : mx.array, shape (batch, dim)
            Mean-pooled hidden states from the current batch.
        """
        self._step += 1

        batch_mean = hidden.mean(axis=0)

        # Find nearest center for each sample, then update those centers
        diff = hidden[:, None, :] - self.centers[None, :, :]
        distances = mx.sqrt((diff ** 2).sum(axis=-1) + 1e-8)
        nearest_idx = distances.argmin(axis=-1)

        # Update each visited center with EMA
        new_centers = mx.array(self.centers)
        new_visits = mx.array(self.visits)

        for i in range(hidden.shape[0]):
            idx = int(nearest_idx[i].item())
            decay = self.ema ** (1.0 / max(1.0, new_visits[idx].item()))
            new_centers[idx] = decay * new_centers[idx] + (1.0 - decay) * hidden[i]
            new_visits[idx] += 1

        self.centers = new_centers
        self.visits = new_visits

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return serialisable state."""
        return {
            "dim": self.dim,
            "num_centers": self.num_centers,
            "ema": self.ema,
            "repeller_weight": self.repeller_weight,
            "min_distance": self.min_distance,
            "step": self._step,
            "centers": mx.array(self.centers),
            "visits": mx.array(self.visits),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore from a state dict."""
        self.dim = state["dim"]
        self.num_centers = state["num_centers"]
        self.ema = state["ema"]
        self.repeller_weight = state["repeller_weight"]
        self.min_distance = state["min_distance"]
        self._step = state["step"]
        self.centers = mx.array(state["centers"])
        self.visits = mx.array(state["visits"])

    def save_checkpoint(self, path: str | Path) -> None:
        """Save attractor state to disk as JSON-safe dict + .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save mx arrays as .npz
        npz_path = path.with_suffix(".npz")
        mx.savez(npz_path, centers=self.centers, visits=self.visits)

        # Save metadata as JSON
        meta = {
            "dim": self.dim,
            "num_centers": self.num_centers,
            "ema": self.ema,
            "repeller_weight": self.repeller_weight,
            "min_distance": self.min_distance,
            "step": self._step,
        }
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load attractor state from disk."""
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        meta_path = path.with_suffix(".json")

        with open(meta_path) as f:
            meta = json.load(f)

        self.dim = meta["dim"]
        self.num_centers = meta["num_centers"]
        self.ema = meta["ema"]
        self.repeller_weight = meta["repeller_weight"]
        self.min_distance = meta["min_distance"]
        self._step = meta["step"]

        data = mx.load(npz_path)
        self.centers = data["centers"]
        self.visits = data["visits"]
