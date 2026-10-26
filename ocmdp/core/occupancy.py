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


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def scale_effective_horizon(lam: float, factor: float) -> float:
    """The lam' whose effective horizon 1 / (1 - lam') is `factor` times that of lam.

    Misspecification of lam is stated on the horizon, since lam itself cannot be doubled
    past one: factor 2 with lam = 0.8 gives 0.9, factor 1/2 gives 0.6.
    """
    if not factor > 0:  # also rejects NaN
        raise ValueError(f"factor must be positive, got {factor}")
    scaled = 1.0 - (1.0 - lam) / factor
    if not 0.0 <= scaled < 1.0:
        raise ValueError(f"factor {factor} takes lam = {lam} outside [0, 1)")
    return scaled


class OccupancyMeasure:
    def __init__(self, n_actions: int, lam: float, delay: int = 0, init: np.ndarray | None = None):
        if not _is_int(n_actions) or n_actions < 1:
            raise ValueError(f"n_actions must be a positive integer, got {n_actions!r}")
        if not 0.0 <= lam < 1.0:  # also rejects NaN
            raise ValueError(f"lam must lie in [0, 1), got {lam}")
        if not _is_int(delay) or delay < 0:
            raise ValueError(f"delay must be a non-negative integer, got {delay!r}")
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
    def pending(self) -> list[np.ndarray]:
        """Distributions recorded but not yet visible to the kernel, oldest first."""
        return [d.copy() for d in self._pending]

    def _check(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64)
        if d.shape != (self.n_actions,):
            raise ValueError(f"expected shape ({self.n_actions},), got {d.shape}")
        if not np.isfinite(d).all() or np.any(d < 0) or not np.isclose(d.sum(), 1.0):
            raise ValueError("an action distribution must be non-negative and sum to one")
        return d / d.sum()  # within tolerance is accepted, but stored exactly normalised
