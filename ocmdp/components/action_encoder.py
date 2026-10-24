"""C2: graph encoder embedding the drug panel over a cross-resistance graph.

Owner: Soumitra. Not implemented yet. Requires the `graph` extra (PyTorch Geometric).

The embedding psi is what makes H(psi#pi) computable: soft-assigning embedded drugs to
mechanism centroids (ocmdp.objectives.pushforward_entropy.soft_assignment) gives the
pushforward. Trained jointly with an auxiliary loss predicting measured cross-resistance
from embedding similarity, because a reward-only embedding collapses. Monitor collapse
with ocmdp.eval.metrics.effective_dimensionality.
"""

import torch
from torch import nn


class ActionEncoder(nn.Module):
    def __init__(self, drug_feature_dim: int, embed_dim: int, n_clusters: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, drug_features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """(A, F) drug features over the cross-resistance graph -> (A, d) embeddings psi."""
        raise NotImplementedError

    def assignment(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(A, K) soft assignment to learned mechanism centroids."""
        raise NotImplementedError

    def cross_resistance_loss(self, embeddings: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss: embedding similarity should predict the (A, A) measured cross-resistance."""
        raise NotImplementedError
