"""Entropy of the pushforward psi#pi: exploration measured in the space Phi acts on.

Maximum-entropy RL rewards spreading mass over actions. Phi only sees mass per
resistance mechanism, so spreading over two drugs with one mechanism is not
exploration. Each embedded action is soft-assigned to K mechanism clusters and entropy
is taken over the induced distribution on clusters. Everything here is differentiable
in the policy probabilities, the embeddings and the centroids.
"""

import torch


def soft_assignment(embeddings: torch.Tensor, centroids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """(A, d) embeddings, (K, d) centroids -> (A, K) row-stochastic assignment."""
    sq_dist = torch.cdist(embeddings, centroids).pow(2)
    return torch.softmax(-sq_dist / temperature, dim=-1)


def pushforward(probs: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """(..., A) action distribution -> (..., K) distribution over mechanism clusters."""
    return probs @ assignment


def categorical_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # the clamp keeps the gradient finite where a probability is exactly zero
    return -(probs * probs.clamp_min(eps).log()).sum(dim=-1)


def pushforward_entropy(probs: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    return categorical_entropy(pushforward(probs, assignment))
