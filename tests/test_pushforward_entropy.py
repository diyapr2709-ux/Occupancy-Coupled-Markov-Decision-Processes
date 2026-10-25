import math

import torch

from ocmdp.objectives.pushforward_entropy import (
    categorical_entropy,
    pushforward_entropy,
    similarity_kernel,
    similarity_sensitive_entropy,
    soft_assignment,
)

PAIRED = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])


def test_identity_assignment_recovers_action_entropy():
    probs = torch.softmax(torch.randn(16, 5), dim=-1)
    torch.testing.assert_close(pushforward_entropy(probs, torch.eye(5)), categorical_entropy(probs))


def test_drugs_sharing_a_mechanism_count_once():
    within = torch.tensor([0.5, 0.5, 0.0, 0.0])
    across = torch.tensor([0.5, 0.0, 0.5, 0.0])
    assert categorical_entropy(within).item() == categorical_entropy(across).item()
    assert pushforward_entropy(within, PAIRED).item() < 1e-6
    assert math.isclose(pushforward_entropy(across, PAIRED).item(), math.log(2), rel_tol=1e-6)


def test_soft_assignment_is_row_stochastic_and_sharpens_to_nearest_centroid():
    embeddings = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
    centroids = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    soft = soft_assignment(embeddings, centroids, temperature=10.0)
    torch.testing.assert_close(soft.sum(-1), torch.ones(3))
    hard = soft_assignment(embeddings, centroids, temperature=1e-2)
    torch.testing.assert_close(hard, torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))


def test_gradient_is_finite_at_zero_probability():
    probs = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    pushforward_entropy(probs, PAIRED).backward()
    assert torch.isfinite(probs.grad).all()


def test_gradient_reaches_embeddings_and_centroids():
    embeddings = torch.randn(4, 3, requires_grad=True)
    centroids = torch.randn(2, 3, requires_grad=True)
    probs = torch.softmax(torch.randn(8, 4), dim=-1)
    pushforward_entropy(probs, soft_assignment(embeddings, centroids)).sum().backward()
    assert embeddings.grad.abs().sum() > 0 and centroids.grad.abs().sum() > 0


def test_similarity_entropy_ignores_how_mass_splits_between_identical_drugs():
    psi = torch.tensor([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]])  # drugs 0 and 1 share an embedding
    Z = similarity_kernel(psi, bandwidth=1.0)
    a = similarity_sensitive_entropy(torch.tensor([0.5, 0.0, 0.5]), Z)
    b = similarity_sensitive_entropy(torch.tensor([0.1, 0.4, 0.5]), Z)
    torch.testing.assert_close(a, b)


def test_similarity_entropy_moves_smoothly_between_shannon_and_merged():
    probs = torch.tensor([0.5, 0.5])
    far = similarity_sensitive_entropy(probs, similarity_kernel(torch.tensor([[0.0], [10.0]])))
    near = similarity_sensitive_entropy(probs, similarity_kernel(torch.tensor([[0.0], [0.5]])))
    same = similarity_sensitive_entropy(probs, similarity_kernel(torch.tensor([[0.0], [0.0]])))
    assert math.isclose(far.item(), math.log(2), rel_tol=1e-6)
    assert same.abs().item() < 1e-6
    assert same < near < far


def test_similarity_entropy_gradient_reaches_embeddings():
    psi = torch.randn(5, 3, requires_grad=True)
    probs = torch.softmax(torch.randn(8, 5), dim=-1)
    similarity_sensitive_entropy(probs, similarity_kernel(psi)).sum().backward()
    assert torch.isfinite(psi.grad).all() and psi.grad.abs().sum() > 0
