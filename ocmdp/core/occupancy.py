"""Exponentially weighted occupancy measure of the agent's own prescribing history.

The coupling operator does not see the current action. It sees

    mu_t = (1 - lam) * sum_{k >= 0} lam**k * d_{t-k}

where d_t is the action distribution realised in period t. The recursion
mu_t = lam * mu_{t-1} + (1 - lam) * d_t is exact for this sum once the history before
the first period is summarised by the initial value. A delay of D periods means d_t
enters mu only at period t + D: the lag between a change in prescribing and its
appearance as selection pressure.
"""

from collections import deque

import numpy as np


class OccupancyMeasure:
    def __init__(self, n_actions: int, lam: float, delay: int = 0, init: np.ndarray | None = None):
        if n_actions < 1:
            raise ValueError(f"n_actions must be positive, got {n_actions}")
        if not 0.0 <= lam < 1.0:
            raise ValueError(f"lam must lie in [0, 1), got {lam}")
        if delay < 0:
            raise ValueError(f"delay must be non-negative, got {delay}")
        self.n_actions = n_actions
        self.lam = lam
        self.delay = delay
        self._init = np.full(n_actions, 1.0 / n_actions) if init is None else self._check(init)
        self.reset()

    def reset(self, init: np.ndarray | None = None) -> np.ndarray:
        self._mu = (self._init if init is None else self._check(init)).copy()
        self._pending: deque[np.ndarray] = deque()
        return self.value

    def update(self, d: np.ndarray) -> np.ndarray:
        """Record one period's action distribution and return the occupancy the kernel now sees."""
        self._pending.append(self._check(d))
        if len(self._pending) > self.delay:
            self._mu = self.lam * self._mu + (1.0 - self.lam) * self._pending.popleft()
        return self.value

    @property
    def value(self) -> np.ndarray:
        return self._mu.copy()

    @property
    def effective_horizon(self) -> float:
        """Mean age, in periods, of the history mu is built from (excluding the delay)."""
        return 1.0 / (1.0 - self.lam)

    def _check(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64)
        if d.shape != (self.n_actions,):
            raise ValueError(f"expected shape ({self.n_actions},), got {d.shape}")
        if np.any(d < 0) or not np.isclose(d.sum(), 1.0):
            raise ValueError("an action distribution must be non-negative and sum to one")
        return d
