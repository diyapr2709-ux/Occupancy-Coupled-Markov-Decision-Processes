"""Observation wrappers for the toy environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MechanismObservation(gym.ObservationWrapper):
    """Observe prescribing as the selection pressure it exerts: every drug-indexed block x becomes x @ C.

    The kernel depends on prescribing only through Phi(mu) = mu @ C, and pending occupancy
    and this quarter's counts enter mu linearly, so mapping each of them through C keeps
    everything the dynamics respond to. Two prescribing mixes look the same exactly when
    they exert the same selection pressure: drugs with identical rows, and more generally
    mixes of drugs whose rows are linearly dependent. Without this, a policy that splits
    mass between interchangeable drugs differently from its training data sees occupancy
    vectors that data never contained. Proportional but unequal rows stay distinguishable,
    because C is not row-normalised.

    `mechanism_map` is (A, K) with entries in [0, 1], which keeps every output in [0, 1].
    The default is the environment's true C; pass a learned map to avoid using it. Wrap
    the raw environment: the wrapper relies on its observation layout.
    """

    def __init__(self, env: gym.Env, mechanism_map: np.ndarray | None = None):
        super().__init__(env)
        toy = env.unwrapped
        if env.observation_space != toy.observation_space:
            raise ValueError("wrap the raw toy environment: an inner wrapper has changed its observation layout")
        self.mechanism_map = toy.coupling.matrix if mechanism_map is None else np.asarray(mechanism_map, dtype=np.float64)
        m = self.mechanism_map
        if m.ndim != 2 or m.shape[0] != toy.coupling.n_drugs or not np.isfinite(m).all() or np.any(m < 0) or np.any(m > 1):
            raise ValueError(f"mechanism_map must be a ({toy.coupling.n_drugs}, K) matrix with entries in [0, 1]")
        self._n_mech, self._n_drugs = toy.coupling.n_mechanisms, toy.coupling.n_drugs
        self._markov, self._delay = toy.config.markov_observation, toy.config.delay
        k = m.shape[1]
        dim = self._n_mech + k + (1 + k * (1 + self._delay) if self._markov else 0)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(dim,), dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        m, a = self._n_mech, self._n_drugs
        parts = [obs[:m], obs[m : m + a] @ self.mechanism_map]
        if self._markov:
            counts = obs[m + a + 1 : m + 2 * a + 1]
            pending = obs[m + 2 * a + 1 :].reshape(self._delay, a)
            parts += [obs[m + a : m + a + 1], counts @ self.mechanism_map, (pending @ self.mechanism_map).ravel()]
        return np.clip(np.concatenate(parts), 0.0, 1.0).astype(np.float32)
