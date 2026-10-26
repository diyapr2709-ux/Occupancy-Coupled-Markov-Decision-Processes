"""Shared trunk for the policy head and the reserve critic: score every (state, embedded action) pair with one MLP.

Scoring pairs rather than emitting one output per action index ties each output to its
drug's embedding. Permuting the embeddings with the state held fixed permutes the
outputs, and drugs at the same point of the embedding always get the same score.

The full network is equivariant to relabelling the drug panel only when the state
carries no drug index, e.g. under ocmdp.envs.wrappers.MechanismObservation. The state
encoder is a plain MLP, so drug-indexed observation blocks break the symmetry.
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
