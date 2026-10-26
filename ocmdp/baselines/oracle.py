"""Full-information oracle: the regret denominator.

Reads the simulator's true state (ToyOCMDPEnv.state) and plans the remaining horizon on
the differentiable mean-field model (ocmdp.theory.mean_field), re-planning every
`replan_every` quarters from the state actually reached. The plan is a free
per-quarter action distribution, so the oracle can be non-stationary and can rotate,
which a stationary policy cannot. Planning optimises softmax logits with Adam from
several starts. The objective is non-convex, and a symmetric start stays on the
symmetric (uniform) saddle, so every plan also starts from the best rotation found from
the current state (`best_rotation`); the first plan adds cycling and random starts.
Because the best iterate is kept, every plan is at least as good in the model as the
best rotation from the state it was made in.

Drugs with identical cross-resistance rows are interchangeable: the same failure
probability and the same selection pressure. The planner is indifferent between them
and splits mass among them arbitrarily, so the oracle averages its plan within each
such class. That leaves its failure rate unchanged and gives a canonical action
distribution, which a policy distilled from it can actually reproduce.

Passing `lam` runs the planner under a misspecified occupancy memory while the
environment keeps its own. This is how degradation under lam misspecification is measured.

It is the best schedule the planner finds, not a proven optimum. Where its model is
exact (no process noise, no contraindications) it is at least as good in the model as
every comparator, which ocmdp.eval.vv checks. With `resistance_noise` > 0 it plans on the
noise-free model, and with contraindications the realised mix departs from the plan; in
those settings regret against it can be negative.
"""

import itertools

import numpy as np
import torch

from ocmdp.baselines.heuristics import HeuristicPolicy, ScheduledPolicy
from ocmdp.core.coupling import CrossResistanceCoupling
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


def rotation_schedules(classes: list[tuple[int, ...]], n_drugs: int, n_quarters: int, max_full_orders: int = 4):
    """Yield (schedule, description) for every fixed-period rotation over classes of interchangeable drugs.

    A rotation gives each class a block of `period` quarters in turn, except that the first
    block may be shorter (phase offset 1..period). Mass is split evenly within a class.
    All orders of the classes are searched for up to `max_full_orders` classes, and the
    index order with each of its cyclic shifts beyond that.
    """
    k = len(classes)
    if k <= max_full_orders:
        orders = list(itertools.permutations(range(k)))
    else:
        orders = [tuple((i + shift) % k for i in range(k)) for shift in range(k)]
    class_rows = np.zeros((k, n_drugs))
    for c, members in enumerate(classes):
        class_rows[c, list(members)] = 1.0 / len(members)
    q = np.arange(n_quarters)
    for order in orders:
        for period in range(1, n_quarters + 1):
            for first in range(1, period + 1):
                block = np.where(q < first, 0, 1 + (q - first) // period)
                yield class_rows[np.array(order)[block % k]], (order, period, first)


def best_rotation(model: MeanFieldModel, state: MeanFieldState, classes: list[tuple[int, ...]], n_quarters: int, chunk: int = 512):
    """The rotation with the lowest mean-field failure rate from `state`: (schedule, value, description)."""
    n_drugs = model.C.shape[0]
    best = (None, float("inf"), None)
    candidates = rotation_schedules(classes, n_drugs, n_quarters)
    while batch := list(itertools.islice(candidates, chunk)):  # chunked: the candidate count grows as k! * Q^2
        schedules = torch.from_numpy(np.stack([s for s, _ in batch]))
        with torch.no_grad():
            values = model.simulate(schedules, state)[0].mean(-1)
        i = int(values.argmin())
        if values[i].item() < best[1]:
            best = (batch[i][0], values[i].item(), batch[i][1])
    return best


def tuned_cycling(config: ToyConfig) -> ScheduledPolicy:
    """The best blind rotation: order, period and phase chosen on the mean-field model from the nominal start.

    Like the oracle it uses the true dynamics, but only once, before the episode, and
    never looks at the state again.
    """
    model = MeanFieldModel(config)
    classes = CrossResistanceCoupling(config.cross_resistance).identical_row_classes()
    schedule, _, (_, period, first) = best_rotation(model, model.initial_state(), classes, config.horizon_quarters)
    return ScheduledPolicy(schedule, name=f"tuned-rotation[{period}q,first {first}q]")


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
        n_drugs = self.env.action_space.n
        self._classes = self.env.coupling.identical_row_classes()
        self._class_average = np.zeros((n_drugs, n_drugs))
        for members in self._classes:
            self._class_average[np.ix_(members, members)] = 1.0 / len(members)
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
        return torch.softmax(self._plan[q - self._plan_start], -1).numpy() @ self._class_average

    def _replan(self, quarter: int, info: dict) -> None:
        true = self.env.state()
        if true["quarter"] != quarter or not np.array_equal(true["resistance"], info["resistance"]):
            # e.g. constructed with one env and rolled out in another, or deep-copied with a private env copy
            raise RuntimeError("the oracle is bound to a different environment from the one it is acting in")
        remaining = self.env.config.horizon_quarters - quarter
        state = self.model.state_from_env(self.env)
        n_drugs = self.env.action_space.n
        rotation, _, _ = best_rotation(self.model, state, self._classes, remaining)
        # log-probabilities, so the softmax reproduces the rotation exactly (a scaled one-hot would only approximate it)
        rotation_logits = torch.from_numpy(rotation).clamp_min(1e-9).log()[None]
        if self._plan is None:
            init = torch.cat([rotation_logits, starting_logits(remaining, n_drugs, self.n_random, self._generator)])
            steps = self.steps
        else:
            init = torch.cat([self._plan[quarter - self._plan_start :][None], rotation_logits])
            steps = self.refine_steps
        self._plan, planned = plan_schedule(self.model, state, init, steps)
        self._plan_start = quarter
        if quarter == 0:
            self.predicted_failure_rate = planned
