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

import itertools

import numpy as np
import torch

from ocmdp.baselines.heuristics import Cycling, HeuristicPolicy
from ocmdp.envs.toy import ToyConfig
from ocmdp.theory.mean_field import MeanFieldModel, MeanFieldState


def plan_schedule(
    model: MeanFieldModel, state: MeanFieldState, init_logits: torch.Tensor, steps: int, lr: float = 0.1
) -> tuple[torch.Tensor, float]:
    """Optimise (R, Q, A) starting logits in parallel; return the best (Q, A) logits seen and its mean failure rate.

    The best is taken over every iterate of every start, the starts included, so the result
    is never worse than the best starting schedule even if Adam overshoots.
    """
    if not isinstance(steps, int) or steps < 0:
        raise ValueError(f"steps must be a non-negative integer, got {steps!r}")
    logits = init_logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    best_value, best_logits = float("inf"), init_logits[0]
    for step in range(steps + 1):
        opt.zero_grad()
        values = model.simulate(torch.softmax(logits, -1), state)[0].mean(-1)
        finite = torch.nan_to_num(values.detach(), nan=float("inf"))  # argmin would otherwise pick a NaN start
        i = int(finite.argmin())
        if finite[i].item() < best_value:
            best_value, best_logits = finite[i].item(), logits.detach()[i].clone()
        if step < steps:
            values.sum().backward()  # each start's gradient depends only on its own value, so a NaN start stays contained
            opt.step()
    if not np.isfinite(best_value):
        raise RuntimeError("planning produced no schedule with a finite failure rate")
    return best_logits, best_value


def starting_logits(n_quarters: int, n_drugs: int, n_random: int, generator: torch.Generator, scale: float = 3.0) -> torch.Tensor:
    """Uniform, cycling with periods 1/2/4/8 quarters, and random starts, stacked to (R, Q, A)."""
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be finite and positive, got {scale}")
    q = torch.arange(n_quarters)
    starts = [torch.zeros(n_quarters, n_drugs, dtype=torch.float64)]
    for period in (1, 2, 4, 8):
        starts.append(scale * torch.nn.functional.one_hot((q // period) % n_drugs, n_drugs).double())
    starts += list(torch.randn(n_random, n_quarters, n_drugs, generator=generator, dtype=torch.float64))
    return torch.stack(starts)


def tuned_cycling(config: ToyConfig, max_orders_drugs: int = 6, chunk: int = 256) -> Cycling:
    """The best blind rotation: order and period chosen on the mean-field model from the true dynamics.

    The oracle's plan is a rotation, so this is the heuristic it should be compared with.
    Like the oracle it uses full information, but only once, before the episode, and
    never looks at the state again. All cyclic orders are searched for up to
    `max_orders_drugs` drugs, and the index order beyond that.
    """
    model = MeanFieldModel(config)
    n, horizon = len(config.cross_resistance), config.horizon_quarters
    orders = [(0, *rest) for rest in itertools.permutations(range(1, n))] if n <= max_orders_drugs else [tuple(range(n))]
    candidates = [(order, period) for order in orders for period in range(1, horizon + 1)]
    q = np.arange(horizon)
    values = []
    for start in range(0, len(candidates), chunk):  # bounded memory: (n-1)! * H schedules of H x n would not fit
        schedules = np.stack([np.eye(n)[np.array(order)[(q // period) % n]] for order, period in candidates[start : start + chunk]])
        with torch.no_grad():
            values.append(model.simulate(torch.from_numpy(schedules))[0].mean(-1))
    order, period = candidates[int(torch.cat(values).argmin())]
    policy = Cycling(n, period=period, drugs=order)
    policy.name = f"tuned-cycling[{period}q]"
    return policy


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
        name: str | None = None,
    ):
        if replan_every < 1:
            raise ValueError(f"replan_every must be at least 1, got {replan_every}")
        self.env = env.unwrapped
        self.model = MeanFieldModel(self.env.config, lam=lam)
        self.name = name or ("oracle" if lam is None else f"oracle[lam={lam:.2f}]")
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
        if q >= self.env.config.horizon_quarters:
            raise ValueError("the episode is over; there is nothing left to plan")
        if self._plan is None or q < self._plan_start:  # first call, or a new episode started without reset()
            self._plan = None
            self._replan(q, info)
        elif q - self._plan_start >= self.replan_every:
            self._replan(q, info)
        return torch.softmax(self._plan[q - self._plan_start], -1).numpy()

    def _replan(self, quarter: int, info: dict) -> None:
        true = self.env.state()
        if true["quarter"] != quarter or not np.array_equal(true["resistance"], info["resistance"]):
            # e.g. constructed with one env and rolled out in another, or deep-copied with a private env copy
            raise RuntimeError("the oracle is bound to a different environment from the one it is acting in")
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
