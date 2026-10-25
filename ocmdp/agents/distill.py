"""Distil the full-information oracle into C3: a test of what the policy class can represent.

The oracle reads the simulator's hidden state and plans; C3 sees only the observation.
If C3, trained by supervised distillation on the oracle's action distributions, recovers
the oracle's failure rate, then the state-dependent behaviour that beats uniform mixing
is a function of the observation, and C3 can express it. This is a capacity check, not
the RL result: no reward is used.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from ocmdp.baselines.heuristics import HeuristicPolicy
from ocmdp.components.constraint_filter import sample_feasible
from ocmdp.components.policy_head import MixingPolicyHead
from ocmdp.eval.seeds import policy_rng


@dataclass
class Demonstrations:
    obs: torch.Tensor  # (N, S)
    probs: torch.Tensor  # (N, A) teacher action distribution before the C5 filter
    masks: torch.Tensor  # (N, A) feasibility the teacher acted under


def collect(
    env: gym.Env, teacher: HeuristicPolicy, seeds: tuple[int, ...], student: HeuristicPolicy | None = None
) -> Demonstrations:
    """Teacher-labelled states. With a student, the student acts and the teacher only labels
    the states the student visits (DAgger); the oracle can label any state since it reads
    the true simulator state and replans from wherever the student has taken it."""
    actor = student or teacher
    obs_list, probs_list, mask_list = [], [], []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        teacher.reset()
        actor.reset()
        rng = policy_rng(seed)
        truncated = terminated = False
        while not (truncated or terminated):
            p = teacher.probs(obs, info)
            obs_list.append(obs)
            probs_list.append(p)
            mask_list.append(info["action_mask"])
            act_p = p if student is None else student.probs(obs, info)
            obs, _, terminated, truncated, info = env.step(sample_feasible(act_p, info["action_mask"], rng))
    return Demonstrations(
        torch.as_tensor(np.array(obs_list), dtype=torch.float32),
        torch.as_tensor(np.array(probs_list), dtype=torch.float32),
        torch.as_tensor(np.array(mask_list), dtype=torch.bool),
    )


def dagger(
    env: gym.Env,
    teacher: HeuristicPolicy,
    head: MixingPolicyHead,
    embeddings: torch.Tensor,
    seeds: tuple[int, ...],
    rounds: int = 3,
    steps: int = 2000,
) -> "HeadPolicy":
    """Behaviour cloning, then `rounds` of DAgger: aggregate teacher labels on the student's own states and refit.

    Plain cloning fails whenever the student's states drift from the teacher's. That
    happens by construction with cross-resistant duplicates: C3 splits mass equally
    between drugs it cannot tell apart, which leaves the dynamics unchanged but moves the
    per-drug occupancy it observes off the demonstration distribution.
    """
    demos = collect(env, teacher, seeds)
    distill(head, embeddings, demos, steps=steps)
    student = HeadPolicy(head, embeddings)
    for _ in range(rounds):
        new = collect(env, teacher, seeds, student=student)
        demos = Demonstrations(*(torch.cat([old, extra]) for old, extra in zip(vars(demos).values(), vars(new).values())))
        distill(head, embeddings, demos, steps=steps)
    return student


def distill(
    head: MixingPolicyHead,
    embeddings: torch.Tensor,
    demos: Demonstrations,
    steps: int = 2000,
    lr: float = 3e-3,
    batch_size: int = 512,
    seed: int = 0,
) -> list[float]:
    """Minimise cross-entropy from the teacher's distribution to C3's, over minibatches. Returns the loss curve."""
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        idx = torch.randint(len(demos.obs), (min(batch_size, len(demos.obs)),), generator=generator)
        probs, _ = head(demos.obs[idx], embeddings)
        loss = -(demos.probs[idx] * probs.clamp_min(1e-8).log()).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


class HeadPolicy(HeuristicPolicy):
    """Run a trained C3 head as a policy; the C5 filter is applied by HeuristicPolicy.act."""

    def __init__(self, head: MixingPolicyHead, embeddings: torch.Tensor, name: str = "C3"):
        self.head = head
        self.embeddings = embeddings
        self.name = name

    def probs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        with torch.no_grad():
            p, _ = self.head(torch.as_tensor(obs, dtype=torch.float32)[None], self.embeddings)
        return p[0].double().numpy()
