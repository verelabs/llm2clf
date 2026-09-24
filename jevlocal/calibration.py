"""Temperature scaling (Guo et al. 2017), ported from AnyJev (Apache-2.0), fitted per model and question kind."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

EPS = 1e-12


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(np.asarray(probs, dtype=np.float64), EPS, None))
    return np.exp(log_softmax(logits / temperature))


def nll(probs: Sequence[np.ndarray], labels: Sequence[int], temperature: float = 1.0) -> float:
    return float(-np.mean([np.log(max(apply_temperature(p, temperature)[y], EPS)) for p, y in zip(probs, labels)]))


def fit_temperature(probs: Sequence[np.ndarray], labels: Sequence[int], lo: float = -3.0, hi: float = 3.0, iters: int = 60) -> float:
    """Golden-section search on log T minimizing NLL; probs may have different option counts."""
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(probs, labels, np.exp(c)), nll(probs, labels, np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(probs, labels, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(probs, labels, np.exp(d))
    return float(np.exp((a + b) / 2))


class Calibration:
    """Per-kind temperatures for one model, stored as {"noul": T, "choice": T, "score": T}."""

    def __init__(self, temperatures: dict[str, float] | None = None):
        self.temperatures = temperatures or {}

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        if not path or not Path(path).exists():
            return cls()
        return cls(json.loads(Path(path).read_text()).get("temperatures", {}))

    def apply(self, kind: str, probs: np.ndarray) -> np.ndarray:
        t = self.temperatures.get(kind)
        return probs if t is None else apply_temperature(probs, t)
