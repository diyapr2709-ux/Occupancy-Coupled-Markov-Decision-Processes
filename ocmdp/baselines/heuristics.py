"""Non-learning comparators: fixed guideline, cycling, and stationary (incl. uniform) mixing.

All of them pass through the constraint filter when the environment supplies a feasibility mask,
so they are compared with the learned policy under the same clinical constraints.
"""

import numpy as np

from ocmdp.components.constraint_filter import sample_feasible


class HeuristicPolicy:
    name = "heuristic"

    def reset(self) -> None:
        pass

    def probs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        raise NotImplementedError

    def act(self, obs: np.ndarray, info: dict, rng: np.random.Generator) -> int:
        p = self.probs(obs, info)
        return sample_feasible(p, info.get("action_mask", np.ones(len(p), dtype=bool)), rng)


class FixedGuideline(HeuristicPolicy):
    """Current practice: one first-line drug, unchanged over the horizon."""

    def __init__(self, n_drugs: int, drug: int = 0):
        if not 0 <= drug < n_drugs:
            raise ValueError(f"drug {drug} out of range for {n_drugs} drugs")
        self.name = f"fixed[{drug}]"
        self._p = np.eye(n_drugs)[drug]

    def probs(self, obs, info):
        return self._p


class Cycling(HeuristicPolicy):
    """Rotate the first-line drug every `period` quarters: deterministic but non-stationary."""

    def __init__(self, n_drugs: int, period: int = 4, drugs: tuple[int, ...] | None = None):
        self._drugs = tuple(range(n_drugs)) if drugs is None else tuple(drugs)
        if period < 1 or not self._drugs or not all(0 <= d < n_drugs for d in self._drugs):
            raise ValueError("need period >= 1 and a non-empty rotation of valid drugs")
        self.name = f"cycling[{period}q]"
        self._eye = np.eye(n_drugs)
        self._period = period

    def probs(self, obs, info):
        return self._eye[self._drugs[(info["quarter"] // self._period) % len(self._drugs)]]


class StationaryMixing(HeuristicPolicy):
    """Blind mixing: the same distribution over drugs in every state."""

    def __init__(self, probs: np.ndarray, name: str | None = None):
        p = np.asarray(probs, dtype=np.float64)
        if np.any(p < 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("probs must be a distribution")
        self._p = p / p.sum()
        self.name = name or "mixing[" + ",".join(f"{x:.2g}" for x in self._p) + "]"

    def probs(self, obs, info):
        return self._p


def uniform_mixing(n_drugs: int) -> StationaryMixing:
    return StationaryMixing(np.full(n_drugs, 1.0 / n_drugs), name="uniform")


class ScheduledPolicy(HeuristicPolicy):
    """A fixed (Q, A) schedule of per-quarter action distributions, committed to before the episode."""

    def __init__(self, schedule: np.ndarray, name: str = "scheduled"):
        schedule = np.asarray(schedule, dtype=np.float64)
        if schedule.ndim != 2 or np.any(schedule < 0) or not np.allclose(schedule.sum(axis=1), 1.0):
            raise ValueError("schedule must be a (quarters, drugs) array of distributions")
        self._schedule = schedule
        self.name = name

    def probs(self, obs, info):
        return self._schedule[min(info["quarter"], len(self._schedule) - 1)]


class MyopicGreedy(HeuristicPolicy):
    """Give the drug with the lowest current failure probability, splitting ties evenly.

    Reacts to the state every quarter but ignores the lag between prescribing and
    selection pressure, so it switches only once the failure curves have already crossed.
    """

    name = "greedy"

    def probs(self, obs, info):
        f = info["fail_prob"]
        best = np.isclose(f, f.min())
        return best / best.sum()
