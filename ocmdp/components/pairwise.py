"""Shared trunk for C3 and C4: score every (state, embedded action) pair with one MLP.

Scoring pairs rather than emitting one output per action index makes both heads
permutation-equivariant over the drug panel and ties their outputs to the embedding:
drugs at the same point of psi always get the same score, so a policy can only tell
cross-resistant drugs apart if C2 does.
"""

import torch
from torch import nn


class PairwiseScorer(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.pair_head = nn.Sequential(nn.Linear(hidden_dim + embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor, action_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, S) state, (A, d) embeddings -> (B, A) scores and the (B, H) state features."""
        h = self.state_encoder(state)
        n_batch, n_actions = h.shape[0], action_embeddings.shape[0]
        pairs = torch.cat(
            [h[:, None, :].expand(n_batch, n_actions, -1), action_embeddings[None].expand(n_batch, n_actions, -1)], dim=-1
        )
        return self.pair_head(pairs).squeeze(-1), h
