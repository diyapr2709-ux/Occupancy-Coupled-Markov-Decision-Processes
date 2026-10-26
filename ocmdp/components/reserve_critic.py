"""Efficacy-reserve critic: the shadow price of depleting the set of effective drugs.

A critic separately parameterised from the reward critics and trained on its own target,
the discounted net loss of effective drugs,

    Q_res(s, a) = E[ sum_t gamma^t c_t ],   c_t = max(0, effective drugs before - after).

It enters the actor loss multiplied by the dual variable xi
(ocmdp.agents.lagrangian.DualVariable), so depleting drugs never trades against
treatment outcome inside a single scalar. c_t can only be non-zero when a quarter closes,
so the critic is meant to be trained on quarter-level transitions.
"""

import numpy as np
import torch
from torch import nn

from ocmdp.components.pairwise import PairwiseScorer


class ReserveCritic(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.scorer = PairwiseScorer(state_dim, embed_dim, hidden_dim)

    def forward(self, state: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, S) state, (A, d) embeddings -> (B, A) expected discounted option-value loss."""
        return self.scorer(state, action_embeddings)[0]


def reserve_cost(fail_prob_before: np.ndarray, fail_prob_after: np.ndarray, threshold: float) -> float:
    """c_t: the net drop in the number of effective drugs (failure probability below threshold).

    A drug regained in the same quarter offsets one lost, so the cost summed over an episode
    never exceeds the number of drugs that were effective at its start.
    """
    before, after = np.asarray(fail_prob_before, dtype=np.float64), np.asarray(fail_prob_after, dtype=np.float64)
    if before.ndim != 1 or before.shape != after.shape:
        raise ValueError(f"expected two (A,) failure-probability vectors, got shapes {before.shape} and {after.shape}")
    if not (np.isfinite(before).all() and np.isfinite(after).all()):
        raise ValueError("failure probabilities must be finite")
    return float(max(0, int((before < threshold).sum()) - int((after < threshold).sum())))


def reserve_td_target(
    cost: torch.Tensor, next_probs: torch.Tensor, next_reserve_q: torch.Tensor, gamma: float, done: torch.Tensor
) -> torch.Tensor:
    """Expected-SARSA target c + gamma (1 - done) E_{a' ~ pi}[Q_res(s', a')], for (B,) cost/done and (B, A) next-state tensors.

    gamma is per transition: on quarter-level transitions it is the per-quarter discount.
    `done` may be bool, as Gymnasium returns it.
    """
    if next_reserve_q.shape != next_probs.shape or cost.shape != next_probs.shape[:-1] or done.shape != cost.shape:
        raise ValueError(
            f"shapes cost {tuple(cost.shape)}, done {tuple(done.shape)}, next_probs {tuple(next_probs.shape)}, "
            f"next_reserve_q {tuple(next_reserve_q.shape)} do not match (B,), (B,), (B, A), (B, A)"
        )
    return cost + gamma * (1.0 - done.to(cost.dtype)) * (next_probs * next_reserve_q).sum(-1)
