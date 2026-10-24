"""Reduced OC-MDP: linear selection on a handful of resistance mechanisms.

Small enough that the stationary analysis in ocmdp.theory.two_drug is exact in the
mean-field limit, and rich enough to separate action entropy from pushforward entropy
once several drugs share a mechanism. The full stewardship environment keeps this
interface and replaces the dynamics.

Timing. One step is one empiric prescribing decision. After every
`decisions_per_quarter` steps the quarter closes: its empirical action distribution
enters the occupancy measure (visible to the kernel `delay` quarters later), selection
pressure is recomputed through Phi, and resistance moves one step. The kernel is fixed
within a quarter, so no single action can influence it.

Observation: resistance frequency per mechanism, then the occupancy measure. The
pending (delayed) occupancy and in-quarter action counts are part of the Markov state
but are not observed.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ocmdp.core.coupling import CrossResistanceCoupling
from ocmdp.core.occupancy import OccupancyMeasure
from ocmdp.core.selection import check_rates, failure_probability, resistance_step


@dataclass(frozen=True)
class ToyConfig:
    cross_resistance: tuple[tuple[float, ...], ...] = ((1.0, 0.0), (0.0, 1.0))
    kappa: float = 0.10  # acquisition rate per quarter under full selection pressure
    delta: float = 0.05  # reversion rate per quarter (fitness cost of carriage)
    lam: float = 0.8  # occupancy memory per quarter
    delay: int = 1  # quarters before prescribing shows up as selection pressure
    r0: float = 0.05
    decisions_per_quarter: int = 25
    horizon_quarters: int = 40
    contraindication_rate: float = 0.0  # per drug per patient; at least one drug always stays feasible
    expected_reward: bool = False  # reward -P(fail) instead of a sampled failure

    @classmethod
    def two_drug(cls, rho: float = 0.0, **kwargs) -> "ToyConfig":
        """Two drugs, two mechanisms; each drug also selects for the other's mechanism with weight rho."""
        return cls(cross_resistance=((1.0, rho), (rho, 1.0)), **kwargs)

    @classmethod
    def paired(cls, n_mechanisms: int = 2, drugs_per_mechanism: int = 2, rho: float = 0.0, **kwargs) -> "ToyConfig":
        """Several drugs per mechanism: action entropy and pushforward entropy come apart."""
        rows = []
        for m in range(n_mechanisms):
            row = tuple(1.0 if j == m else rho for j in range(n_mechanisms))
            rows.extend([row] * drugs_per_mechanism)
        return cls(cross_resistance=tuple(rows), **kwargs)


class ToyOCMDPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ToyConfig | None = None):
        self.config = config or ToyConfig()
        check_rates(self.config.kappa, self.config.delta)
        if self.config.decisions_per_quarter < 1 or self.config.horizon_quarters < 1:
            raise ValueError("decisions_per_quarter and horizon_quarters must be positive")
        self.coupling = CrossResistanceCoupling(self.config.cross_resistance)
        self.occupancy = OccupancyMeasure(self.coupling.n_drugs, self.config.lam, self.config.delay)
        self.action_space = spaces.Discrete(self.coupling.n_drugs)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(self.coupling.n_mechanisms + self.coupling.n_drugs,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.occupancy.reset()
        self._r = np.full(self.coupling.n_mechanisms, self.config.r0)
        self._fail_prob = failure_probability(self._r, self.coupling.matrix)
        self._counts = np.zeros(self.coupling.n_drugs)
        self._quarter = 0
        self._decision = 0
        self._mask = self._sample_mask()
        return self._obs(), self._info()

    def step(self, action):
        a = int(action)
        if not self.action_space.contains(a):
            raise ValueError(f"invalid action {action!r}")
        fail_prob = float(self._fail_prob[a])
        failed = bool(self.np_random.random() < fail_prob)
        reward = -fail_prob if self.config.expected_reward else -float(failed)
        violation = not bool(self._mask[a])

        self._counts[a] += 1
        self._decision += 1
        if self._decision == self.config.decisions_per_quarter:
            self._close_quarter()
        truncated = self._quarter >= self.config.horizon_quarters

        self._mask = self._sample_mask()
        info = self._info()
        info.update(action=a, failed=failed, violation=violation)
        return self._obs(), reward, False, truncated, info

    def _close_quarter(self) -> None:
        mu = self.occupancy.update(self._counts / self._counts.sum())
        self._r = resistance_step(self._r, self.coupling(mu), self.config.kappa, self.config.delta)
        self._fail_prob = failure_probability(self._r, self.coupling.matrix)
        self._counts[:] = 0
        self._decision = 0
        self._quarter += 1

    def _sample_mask(self) -> np.ndarray:
        n = self.coupling.n_drugs
        if self.config.contraindication_rate == 0:
            return np.ones(n, dtype=bool)
        mask = self.np_random.random(n) >= self.config.contraindication_rate
        if not mask.any():
            mask[self.np_random.integers(n)] = True
        return mask

    def _obs(self) -> np.ndarray:
        return np.clip(np.concatenate([self._r, self.occupancy.value]), 0.0, 1.0).astype(np.float32)

    def _info(self) -> dict:
        return {
            "quarter": self._quarter,
            "resistance": self._r.copy(),
            "occupancy": self.occupancy.value,
            "fail_prob": self._fail_prob.copy(),
            "action_mask": self._mask.copy(),
        }
