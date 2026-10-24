"""C1: population belief filter over latent resistance frequencies.

Owner: Nikhila. Not implemented yet. Runs on the slow clock, once per simulated quarter,
when an antibiogram would arrive.

predict() propagates the belief one quarter under the selection pressure Phi(mu) the
agent itself caused (known to the agent, since mu is its own history). update()
conditions on the antibiogram through the explicit emission model in
ocmdp.envs.observation, so culture sampling bias is inverted rather than learned.
"""

import numpy as np

from ocmdp.envs.observation import AntibiogramEmission


class BeliefFilter:
    def __init__(self, n_mechanisms: int, emission: AntibiogramEmission):
        raise NotImplementedError

    def reset(self):
        """Prior belief at the start of an episode."""
        raise NotImplementedError

    def predict(self, belief, pressure: np.ndarray):
        raise NotImplementedError

    def update(self, belief, antibiogram: np.ndarray):
        raise NotImplementedError

    def features(self, belief) -> np.ndarray:
        """Fixed-size summary (e.g. mean and per-mechanism uncertainty) for C3 and C4."""
        raise NotImplementedError
