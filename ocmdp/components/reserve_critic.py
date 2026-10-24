"""C4: efficacy-reserve critic, the shadow price of option depletion.

Owner: Diya. Not implemented yet. Runs on the slow clock.

A third critic, separately parameterised from the twin reward critics and trained on a
distinct target: the discounted loss of future option value, i.e. the per-quarter drop
in the number of drugs still effective. It enters the actor loss multiplied by the dual
variable xi (ocmdp.agents.lagrangian.DualVariable), so option depletion never competes
with treatment outcome inside a single scalar, and xi is a number a committee can
inspect and set.
"""

import torch
from torch import nn


class ReserveCritic(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        raise NotImplementedError

    def forward(self, state: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, S) state, (A, d) embeddings -> (B, A) expected discounted option-value loss."""
        raise NotImplementedError
