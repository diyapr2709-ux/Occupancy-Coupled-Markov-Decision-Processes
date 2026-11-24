"""Stochastic mixing policy head over the embedded action set.

Each embedded action is scored against the state by a shared pairwise MLP
(ocmdp.components.pairwise), so drugs with nearby embeddings receive correlated logits
and drugs with identical embeddings receive identical ones. The feasibility mask is
applied to logits before the softmax, which puts exactly zero mass on infeasible drugs.
The temperature is not part of the head: it is a property of the agent
(ocmdp.agents.oc_sac), fixed or learned as one scalar.
"""

import torch
from torch import nn

from ocmdp.components.constraint_filter import mask_logits
from ocmdp.components.pairwise import PairwiseScorer


class MixingPolicyHead(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.scorer = PairwiseScorer(state_dim, embed_dim, hidden_dim)

    def forward(self, state: torch.Tensor, action_embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, S) state, (A, d) embeddings, optional (B, A) feasibility -> (B, A) probs."""
        logits, _ = self.scorer(state, action_embeddings)
        if mask is not None:
            logits = mask_logits(logits, mask)
        return torch.softmax(logits, dim=-1)
