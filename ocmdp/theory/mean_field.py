"""Differentiable mean-field limit of the toy environment.

With many decisions per quarter, a quarter's realised action distribution equals the
policy's and the toy environment becomes deterministic. Given a schedule of per-quarter
action distributions, expected failures, occupancy and resistance then follow exactly.
The model calls the environment's own selection functions, so feeding it the realised
per-quarter action shares reproduces an environment trajectory to floating-point
precision (checked in ocmdp.eval.vv). Everything is differentiable in the schedule,
which is what the full-information oracle plans over, and in lam, which lets a planner
be run under a misspecified memory.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ocmdp.core.selection import failure_probability, resistance_step
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv, per_mechanism


@dataclass(frozen=True)
class MeanFieldState:
    resistance: torch.Tensor  # (..., M)
    occupancy: torch.Tensor  # (..., A)
    pending: tuple[torch.Tensor, ...]  # each (..., A), oldest first


class MeanFieldModel:
    def __init__(self, config: ToyConfig, lam: float | None = None):
        self.config = config
        self.lam = config.lam if lam is None else lam
        self.C = torch.tensor(config.cross_resistance, dtype=torch.float64)
        n_drugs, n_mech = self.C.shape
        self.kappa = torch.from_numpy(per_mechanism(config.kappa, n_mech))
        self.delta = torch.from_numpy(per_mechanism(config.delta, n_mech))
        self.r0 = torch.from_numpy(per_mechanism(config.r0, n_mech))
        self.mu0 = torch.full((n_drugs,), 1.0 / n_drugs, dtype=torch.float64) if config.mu0 is None else torch.tensor(config.mu0, dtype=torch.float64)

    def initial_state(self) -> MeanFieldState:
        return MeanFieldState(self.r0.clone(), self.mu0.clone(), ())

    @staticmethod
    def state_from_env(env: ToyOCMDPEnv) -> MeanFieldState:
        """Read the true state at the start of a quarter (the only point where the mean-field state is complete)."""
        s = env.state()
        if s["decision"] != 0:
            raise ValueError("the mean-field state is only defined at a quarter boundary")
        as_t = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float64))  # noqa: E731
        return MeanFieldState(as_t(s["resistance"]), as_t(s["occupancy"]), tuple(as_t(d) for d in s["pending"]))

    def simulate(self, schedule: torch.Tensor, state: MeanFieldState | None = None):
        """(..., Q, A) per-quarter action distributions -> expected failure rate per quarter (..., Q),
        resistance at the start of each quarter and after the last (..., Q + 1, M), and the final state."""
        state = state or self.initial_state()
        batch = schedule.shape[:-2]
        r = state.resistance.expand(*batch, -1)
        mu = state.occupancy.expand(*batch, -1)
        pending = [d.expand(*batch, -1) for d in state.pending]
        fails, rs = [], [r]
        for q in range(schedule.shape[-2]):
            p = schedule[..., q, :]
            fails.append((p * failure_probability(r, self.C)).sum(-1))
            pending.append(p)
            if len(pending) > self.config.delay:
                mu = self.lam * mu + (1.0 - self.lam) * pending.pop(0)
            r = resistance_step(r, mu @ self.C, self.kappa, self.delta)
            rs.append(r)
        return torch.stack(fails, -1), torch.stack(rs, -2), MeanFieldState(r, mu, tuple(pending))
