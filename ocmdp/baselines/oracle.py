"""Full-information oracle: the regret denominator.

Reads the simulator's true state (ToyOCMDPEnv.state) and plans the remaining horizon on
the differentiable mean-field model (ocmdp.theory.mean_field), re-planning every
`replan_every` quarters from the state actually reached. The plan is a free
per-quarter action distribution, so the oracle can be non-stationary and can rotate,
which a stationary policy cannot. Planning optimises softmax logits with Adam from
several starts. The objective is non-convex, and a symmetric start stays on the
symmetric (uniform) saddle, so the starts include cycling schedules and random ones.

Passing `lam` runs the planner under a misspecified occupancy memory while the
environment keeps its own. This is how degradation under lam misspecification is measured.

Limitation: the mean-field model ignores contraindications, so with
contraindication_rate > 0 the realised mix departs from the plan and the oracle is no
longer an upper bound.
"""

import numpy as np
import torch

from ocmdp.baselines.heuristics import HeuristicPolicy
from ocmdp.theory.mean_field import MeanFieldModel, MeanFieldState


def plan_schedule(
    model: MeanFieldModel, state: MeanFieldState, init_logits: torch.Tensor, steps: int, lr: float = 0.1
) -> tuple[torch.Tensor, float]:
    """Optimise (R, Q, A) starting logits in parallel; return the best (Q, A) logits seen and its mean failure rate.

    The best is taken over every iterate of every start, the starts included, so the result
    is never worse than the best starting schedule even if Adam overshoots.
    """
    logits = init_logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    best_value, best_logits = float("inf"), init_logits[0]
    for step in range(steps + 1):
        opt.zero_grad()
        values = model.simulate(torch.softmax(logits, -1), state)[0].mean(-1)
        i = int(values.detach().argmin())
        if values[i].item() < best_value:
            best_value, best_logits = values[i].item(), logits.detach()[i].clone()
        if step < steps:
            values.sum().backward()
            opt.step()
    return best_logits, best_value


def starting_logits(n_quarters: int, n_drugs: int, n_random: int, generator: torch.Generator, scale: float = 3.0) -> torch.Tensor:
    """Uniform, cycling with periods 1/2/4/8 quarters, and random starts, stacked to (R, Q, A)."""
    q = torch.arange(n_quarters)
    starts = [torch.zeros(n_quarters, n_drugs, dtype=torch.float64)]
    for period in (1, 2, 4, 8):
        starts.append(scale * torch.nn.functional.one_hot((q // period) % n_drugs, n_drugs).double())
    starts += list(torch.randn(n_random, n_quarters, n_drugs, generator=generator, dtype=torch.float64))
    return torch.stack(starts)


class FullInformationOracle(HeuristicPolicy):
    def __init__(
        self,
        env,
        lam: float | None = None,
        replan_every: int = 4,
        steps: int = 200,
        refine_steps: int = 50,
        n_random: int = 4,
        seed: int = 0,
    ):
        if replan_every < 1:
            raise ValueError(f"replan_every must be at least 1, got {replan_every}")
        self.env = env.unwrapped
        self.model = MeanFieldModel(self.env.config, lam=lam)
        self.name = "oracle" if lam is None else f"oracle[lam={lam:.2f}]"
        self.replan_every = replan_every
        self.steps = steps
        self.refine_steps = refine_steps
        self.n_random = n_random
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self._plan: torch.Tensor | None = None
        self._plan_start = 0
        self._generator = torch.Generator().manual_seed(self.seed)
        self.predicted_failure_rate: float | None = None  # whole episode, from the plan made at quarter 0

    def probs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        q = info["quarter"]
        if self._plan is None or q < self._plan_start:  # first call, or a new episode started without reset()
            self._plan = None
            self._replan(q)
        elif q - self._plan_start >= self.replan_every:
            self._replan(q)
        return torch.softmax(self._plan[q - self._plan_start], -1).numpy()

    def _replan(self, quarter: int) -> None:
        remaining = self.env.config.horizon_quarters - quarter
        state = self.model.state_from_env(self.env)
        n_drugs = self.env.action_space.n
        if self._plan is None:
            init, steps = starting_logits(remaining, n_drugs, self.n_random, self._generator), self.steps
        else:
            init, steps = self._plan[quarter - self._plan_start :][None], self.refine_steps
        self._plan, planned = plan_schedule(self.model, state, init, steps)
        self._plan_start = quarter
        if quarter == 0:
            self.predicted_failure_rate = planned
