"""Full stewardship environment: fitness-landscape evolution coupled to ward transmission.

Owner: Nikhila. Not implemented yet; not registered with Gymnasium until it is.

Same step/quarter timing and info keys as ocmdp.envs.toy.ToyOCMDPEnv, so every
comparator runs against both unmodified. Differences from the toy:
  - dynamics: genotype frequencies on an empirical fitness landscape (evolution.py),
    spread between patients by a ward agent-based model (transmission.py);
  - observation: the agent never sees frequencies, only a lagged, aggregated
    antibiogram drawn from an explicit emission model (observation.py), filtered by C1;
  - action space: around twenty antibiotic classes with per-patient feasibility from
    patient features, enforced by C5.
"""

import gymnasium as gym


class StewardshipEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        raise NotImplementedError("StewardshipEnv: see module docstring for the contract")
