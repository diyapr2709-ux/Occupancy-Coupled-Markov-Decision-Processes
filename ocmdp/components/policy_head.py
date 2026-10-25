"""C3: stochastic mixing policy head over the embedded action set, with state-dependent temperature.

Owner: Diya.

Each embedded action is scored against the state by a shared pairwise MLP
(ocmdp.components.pairwise), so drugs with nearby embeddings receive correlated logits.
The C5 mask is applied to logits before the softmax, which puts exactly zero mass on
infeasible drugs. A second head outputs log alpha(s), trained with
ocmdp.agents.losses.temperature_loss against ocmdp.agents.losses.entropy_target. That
state dependence is the only way a learned policy can beat uniform mixing.
"""

import torch
from torch import nn

from ocmdp.components.constraint_filter import mask_logits
from ocmdp.components.pairwise import PairwiseScorer


class MixingPolicyHead(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.scorer = PairwiseScorer(state_dim, embed_dim, hidden_dim)
        self.log_alpha_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(
        self, state: torch.Tensor, action_embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, S) state, (A, d) embeddings, optional (B, A) feasibility -> (B, A) probs, (B,) log alpha."""
        logits, h = self.scorer(state, action_embeddings)
        if mask is not None:
            logits = mask_logits(logits, mask)
        return torch.softmax(logits, dim=-1), self.log_alpha_head(h).squeeze(-1)
