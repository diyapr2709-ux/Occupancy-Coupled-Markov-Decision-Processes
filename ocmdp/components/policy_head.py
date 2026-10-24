"""C3: stochastic mixing policy head over the embedded action set, with state-dependent temperature.

Owner: Diya. Not implemented yet.

Scores each embedded action against the state (C1 features plus ward state), so drugs
with nearby embeddings receive correlated logits. The C5 mask is applied to logits before
the softmax (ocmdp.components.constraint_filter.mask_logits). A second head outputs
log alpha(s), trained with ocmdp.agents.losses.temperature_loss against
ocmdp.agents.losses.entropy_target. That state dependence is the only way a learned
policy can beat uniform mixing.
"""

import torch
from torch import nn


class MixingPolicyHead(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        raise NotImplementedError

    def forward(
        self, state: torch.Tensor, action_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, S) state, (A, d) embeddings, (B, A) feasibility -> (B, A) probs, (B,) log alpha."""
        raise NotImplementedError
