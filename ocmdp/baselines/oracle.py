"""Full-information oracle: the regret denominator. Owner: Soumitra. Not implemented yet.

Sees the true resistance state and dynamics and plans over the remaining horizon. On the
toy environment, ocmdp.theory.two_drug.solve_two_drug already gives the best
*stationary* policy in the mean-field limit, which is a lower bound on what the oracle
must achieve. The oracle itself must plan non-stationary rotations.
"""

import numpy as np

from ocmdp.baselines.heuristics import HeuristicPolicy


class FullInformationOracle(HeuristicPolicy):
    def __init__(self, env_config):
        raise NotImplementedError

    def probs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        raise NotImplementedError
