"""C4: efficacy-reserve critic, the shadow price of option depletion.

Owner: Diya.

A third critic, separately parameterised from the twin reward critics and trained on a
distinct target: the discounted number of drugs that stop being effective,

    Q_res(s, a) = E[ sum_t gamma^t c_t ],   c_t = drugs crossing the efficacy threshold at step t.

It enters the actor loss multiplied by the dual variable xi
(ocmdp.agents.lagrangian.DualVariable), so option depletion never competes with
treatment outcome inside a single scalar, and xi is a number a committee can inspect and
set. Losing a drug is a slow-clock event (it only happens when a quarter closes), so c_t
is zero on every other step.
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
    """c_t: drugs that were effective (failure probability below threshold) and no longer are.

    Drugs regaining efficacy do not offset losses: the cost prices depletion, not net change.
    """
    lost = (np.asarray(fail_prob_before) < threshold) & (np.asarray(fail_prob_after) >= threshold)
    return float(lost.sum())


def reserve_td_target(
    cost: torch.Tensor, next_probs: torch.Tensor, next_reserve_q: torch.Tensor, gamma: float, done: torch.Tensor
) -> torch.Tensor:
    """Soft TD target c + gamma (1 - done) E_{a' ~ pi}[Q_res(s', a')], for (B,) cost/done and (B, A) next-state tensors.

    `done` may be bool, as Gymnasium returns it.
    """
    if next_reserve_q.shape != next_probs.shape or cost.shape != next_probs.shape[:-1] or done.shape != cost.shape:
        raise ValueError(
            f"shapes cost {tuple(cost.shape)}, done {tuple(done.shape)}, next_probs {tuple(next_probs.shape)}, "
            f"next_reserve_q {tuple(next_reserve_q.shape)} do not match (B,), (B,), (B, A), (B, A)"
        )
    return cost + gamma * (1.0 - done.to(cost.dtype)) * (next_probs * next_reserve_q).sum(-1)
