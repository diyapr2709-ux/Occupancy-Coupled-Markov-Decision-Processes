"""Sliding-window UCB: the non-stationary bandit comparator. Owner: Soumitra. Not implemented yet.

Treats drift as exogenous. Arms are drugs and the reward is the negative failure. Tests
whether self-induced drift is distinguishable from exogenous drift: the window tracks
drift, but the bandit has no model that its own pulls caused it.
"""

import numpy as np

from ocmdp.baselines.heuristics import HeuristicPolicy


class SlidingWindowUCB(HeuristicPolicy):
    def __init__(self, n_drugs: int, window: int, c: float = 2.0):
        raise NotImplementedError

    def observe(self, action: int, reward: float) -> None:
        raise NotImplementedError

    def probs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        raise NotImplementedError
