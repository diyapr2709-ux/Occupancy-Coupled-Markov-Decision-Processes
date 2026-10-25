"""Entropy of the pushforward psi#pi: exploration measured in the space Phi acts on.

Maximum-entropy RL rewards spreading mass over actions. Phi only sees mass per
resistance mechanism, so spreading over two drugs with one mechanism is not
exploration. Two estimators are provided, both differentiable in the policy and in the
embeddings.

Cluster form: soft-assign each embedded action to K mechanism centroids and take the
entropy of the induced distribution on clusters. It needs K and learned centroids, and
centroids are where collapse enters.

Similarity form (Leinster & Cobbold 2012, order-1 similarity-sensitive diversity):

    H_Z(pi) = -sum_a pi_a log (Z pi)_a,    Z_ab = exp(-|psi_a - psi_b|^2 / bandwidth)

H_Z has no clusters and no K. It reduces to Shannon entropy when Z = I. When Z = S S^T
for a hard assignment S, it equals the cluster form exactly: (Z pi)_a is the mass of
a's cluster, so the sum regroups into -sum_k q_k log q_k. A drug duplicated at the same
embedding adds no diversity however the mass is split.
"""

import torch


def soft_assignment(embeddings: torch.Tensor, centroids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """(A, d) embeddings, (K, d) centroids -> (A, K) row-stochastic assignment."""
    sq_dist = torch.cdist(embeddings, centroids).pow(2)
    return torch.softmax(-sq_dist / temperature, dim=-1)


def pushforward(probs: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """(..., A) action distribution -> (..., K) distribution over mechanism clusters."""
    return probs @ assignment


def _floor(probs: torch.Tensor, eps: float) -> float:
    # 1e-8 underflows to zero in float16, so never go below the dtype's smallest normal number
    return max(eps, torch.finfo(probs.dtype).tiny)


def categorical_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # the clamp keeps the gradient finite where a probability is exactly zero
    return -(probs * probs.clamp_min(_floor(probs, eps)).log()).sum(dim=-1)


def pushforward_entropy(probs: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    return categorical_entropy(pushforward(probs, assignment))


def similarity_kernel(embeddings: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """(A, d) embeddings -> (A, A) symmetric similarity with unit diagonal."""
    # the matrix-multiply path of cdist leaves a nonzero self-distance at large norms
    return torch.exp(-torch.cdist(embeddings, embeddings, compute_mode="donot_use_mm_for_euclid_dist").pow(2) / bandwidth)


def similarity_sensitive_entropy(probs: torch.Tensor, similarity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """H_Z(pi) for (..., A) probs and an (A, A) similarity with unit diagonal and entries in [0, 1]."""
    return -(probs * (probs @ similarity.T).clamp_min(_floor(probs, eps)).log()).sum(dim=-1)
